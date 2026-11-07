"""Command line of `python -m tools.convert.gemma4_build`: build, check, setup-gptq, launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import gemma4_build_launcher as launcher_tools
from .gemma4_build import (
    CALIBRATION_REPO,
    CALIBRATION_REVISION,
    FORK_ROOT,
    PRESETS,
    SERVE_CHECK_ACTIONS,
    BuildConfig,
    BuildError,
    Builder,
    ServeSettings,
    build_identity,
    build_plan,
    default_name,
    describe_plan,
    fingerprint_of,
    safe_name,
    serve_check,
)
from .gemma4_build_checks import GPTQ_PINS, Probes, find_llama_imatrix, parse_pins, preflight, versions_text

SUBCOMMANDS = ("setup-gptq", "launcher", "serve-check")

METHODS = {
    "best-small": "GPTQ g256",
    "best-medium": "GPTQ g32",
    "lite-medium": "RTN g32, 6-bit scales",
    "lite-small": "RTN g256",
    "draft": "absmax g32, no calibration",
}
NEEDS = {
    "gptq": "GPU with 12 GiB free, 64 GiB RAM, about 75 GB disk, the GPTQ environment (setup-gptq)",
    "rtn": "llama.cpp (convert_hf_to_gguf.py + llama-imatrix), about 35 GB scratch disk",
    "plain": "nothing extra",
}
EPILOG = f"""\
presets (measured on one model, text only, W4A8, FP8 MTP assistant; KL is mean KL from the near-lossless reference,
lower is better):

{{table}}

Recommended: best-small for most people (best quality per byte); best-medium for the highest quality; lite-medium or
lite-small only if you cannot run GPTQ (less RAM and VRAM, no GPTQ environment): they score lower. draft is a quick
test build only.

other commands:
  python -m tools.convert.gemma4_build setup-gptq --venv DIR     create the pinned GPTQ environment
  python -m tools.convert.gemma4_build launcher --artifact F --out FILE.cmd   write a launcher for an existing file
  python -m tools.convert.gemma4_build serve-check --launcher F.cmd   start the launcher's server, probe the OpenAI
                                       endpoints, stop it again and write a serving receipt JSON

Calibration text defaults to the pinned public corpus {CALIBRATION_REPO}@{CALIBRATION_REVISION[:12]}.
The serve launcher uses the qualified settings: 131072 context, FP8 KV, one lane, prefill chunk 128, MTP when the
file has it. The server accepts any API key. With --model-match any (the default here) it also accepts any model
name; with exact, OpenAI-style requests must use the --alias model name.
"""


def preset_table() -> str:
    rows = [f"  {'id':<12} {'method':<28} {'size':<7} {'file':<9} {'KL':<7} {'wait':<13} needs"]
    for preset in PRESETS.values():
        rows.append(
            f"  {preset.id:<12} {METHODS[preset.id]:<28} {preset.size_class:<7} {f'{preset.file_gb:g} GB':<9} "
            f"{preset.kl:<7.4f} {preset.wait:<13} {NEEDS[preset.method]}"
        )
    return "\n".join(rows)


def default_venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build",
        description="Build a Gemma 4 31B 4-bit .ninfer from a BF16 checkpoint in one command: checks, plan, "
        "calibration, conversion, receipts, launcher.",
        epilog=EPILOG.format(table=preset_table()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add = parser.add_argument
    add("--preset", choices=tuple(PRESETS), help="what to build (see the table below); required for a build")
    add("--list-presets", action="store_true", help="print the preset table and exit")
    add("--model", type=Path, help="BF16 Hugging Face directory of Gemma 4 31B dense or a fine-tune (only read)")
    add("--assistant", type=Path, help="google/gemma-4-31B-it-assistant directory, embedded as the MTP component")
    add("--no-mtp", action="store_true", help="build without the MTP assistant (pass this or --assistant)")
    add("--out-dir", type=Path, help="where the deliverables go; the artifact is never overwritten")
    add("--work-dir", type=Path, help="scratch and resume state (default: <out-dir>/work-<name>); put it on a "
        "big fast drive; deleted after success unless --keep-work")
    add("--name", help="artifact name, used for the file names (default: <model dir>-<preset>)")
    add("--with-vision", action="store_true", help="keep the vision tower (default: text only, as measured)")
    add("--calibration-text", type=Path, help="your own public UTF-8 calibration text, disjoint from anything you "
        "evaluate on (default: download the pinned public English corpus)")
    add("--gptq-python", type=Path, help="python.exe of the GPTQ environment (default: <checkout>/.venv-gptq)")
    add("--llama-cpp-dir", type=Path, help="llama.cpp source checkout holding convert_hf_to_gguf.py (lite presets)")
    add("--llama-imatrix", type=Path, help="llama-imatrix executable (default: searched in --llama-cpp-dir, "
        "its build/bin, and PATH)")
    add("--llama-python", default=sys.executable, help="Python that runs convert_hf_to_gguf.py (needs gguf, "
        "transformers, torch, numpy; default: this one)")
    add("--imatrix-gpu-layers", default="auto", help="llama-imatrix --gpu-layers value (default auto: fit what the GPU "
        "holds; the measured run used auto)")
    add("--context", type=int, default=launcher_tools.DEFAULT_CONTEXT, help="launcher --max-context (default 131072)")
    add("--alias", help="launcher --model-id, the name /v1/models reports and requests must use (default <context/1000>k, "
        "for example 131k)")
    add("--host", default=launcher_tools.DEFAULT_HOST, help="launcher --host (default 127.0.0.1; 0.0.0.0 for the LAN)")
    add("--port", type=int, default=launcher_tools.DEFAULT_PORT, help="launcher --port (default 8080)")
    add("--kv-dtype", choices=launcher_tools.KV_DTYPES, default="fp8", help="launcher KV storage (default fp8, qualified)")
    add("--model-match", choices=launcher_tools.MODEL_MATCHES, default=launcher_tools.DEFAULT_MODEL_MATCH,
        help="launcher --model-match: any (default) lets OpenAI requests send any model name, as a gateway that "
        "forwards several names needs; exact accepts only --alias and emits no flag")
    add("--server-exe", type=Path, help="ninfer-spo-serve executable (default: newest build*/apps of the checkout "
        "or ../build-spo-*/apps that knows the preset's format)")
    add("--cuda-bin", type=Path, help="CUDA bin directory the launcher puts on PATH (if the DLLs are not beside the exe)")
    add("--no-launcher", action="store_true", help="do not generate a launcher (and do not need the server binary)")
    add("--score", action="store_true", help="after the build, run the paired-KL quality check of section 6 of "
        "docs/gemma4-31b-conversion-technical.md (about 1 h and 60 GB extra; needs ninfer-spo-perplexity beside "
        "the server exe)")
    add("--keep-work", action="store_true", help="keep the work dir (logs, intermediates) after success")
    add("--yes", action="store_true", help="do not ask for confirmation")
    add("--dry-run", action="store_true", help="print the checks and the exact plan, write nothing, exit 0")
    add("--check", action="store_true", help="run only the preflight checks; exit 1 if any fails")
    return parser


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build setup-gptq",
        description="Create the pinned GPTQ environment (uv venv + tools/gptq/requirements.txt, torch cu130) and "
        "verify its versions. Needs uv (https://docs.astral.sh/uv/) and about 10 GB.",
    )
    parser.add_argument("--venv", type=Path, default=FORK_ROOT / ".venv-gptq", help="environment directory "
                        "(default: <checkout>/.venv-gptq); an existing environment is reused")
    parser.add_argument("--python", default="3.13", help="Python version for uv venv (default 3.13, the measured one)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    return parser


def launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build launcher",
        description="Write a Windows .cmd that starts ninfer-spo-serve on a .ninfer with the qualified settings. "
        "MTP and vision are detected from the artifact.",
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new .cmd file (never overwritten)")
    parser.add_argument("--server-exe", type=Path, help="default: newest server build found")
    parser.add_argument("--context", type=int, default=launcher_tools.DEFAULT_CONTEXT)
    parser.add_argument("--alias", help="--model-id (default <context/1000>k)")
    parser.add_argument("--host", default=launcher_tools.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=launcher_tools.DEFAULT_PORT)
    parser.add_argument("--kv-dtype", choices=launcher_tools.KV_DTYPES, default="fp8")
    parser.add_argument("--model-match", choices=launcher_tools.MODEL_MATCHES, default=launcher_tools.DEFAULT_MODEL_MATCH,
                        help="any (default): OpenAI requests may send any model name; exact: only --alias, no flag")
    parser.add_argument("--cuda-bin", type=Path)
    parser.add_argument("--no-mtp", action="store_true", help="leave speculative decoding off even if the file has MTP")
    return parser


def serve_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build serve-check",
        description="Start what a generated serve launcher runs, probe the OpenAI endpoints (/v1/models and chat "
        "completions: the launcher model id, another name under the launcher's --model-match, a missing model field, "
        "an arbitrary bearer key), stop the child again and write a serving receipt JSON. Exit 0 only if every "
        "check passes.",
    )
    parser.add_argument("--launcher", required=True, type=Path, help="serve-<name>.cmd written by the build or the "
                        "launcher command")
    parser.add_argument("--out", type=Path, help="receipt JSON path (default: serve-check-<utcstamp>.json beside "
                        "the launcher)")
    parser.add_argument("--host", help="override the launcher --host for the probe (default: the launcher's)")
    parser.add_argument("--port", type=int, help="override the launcher --port for the probe (default: the launcher's)")
    return parser


# ---------------------------------------------------------------- build


def resolve_config(args: argparse.Namespace) -> tuple[BuildConfig, ServeSettings, Path | None]:
    """Turn flags into a config; every argument-only mistake is refused here, before any probe."""
    if args.preset is None:
        raise ValueError("--preset is required (see --list-presets)")
    for flag in ("model", "out_dir"):
        if getattr(args, flag) is None:
            raise ValueError(f"--{flag.replace('_', '-')} is required")
    if args.no_mtp == (args.assistant is not None):
        raise ValueError("Pass --assistant <dir> (hf download google/gemma-4-31B-it-assistant --local-dir <DIR>) "
                         "or explicitly --no-mtp")
    preset = PRESETS[args.preset]
    model, out_dir = args.model.resolve(), args.out_dir.resolve()
    name = safe_name(args.name) if args.name else default_name(model, preset.id)
    work_dir = (args.work_dir or out_dir / f"work-{name}").resolve()
    if work_dir == out_dir:
        raise ValueError("--work-dir must not be the output directory itself")
    alias = args.alias or launcher_tools.default_alias(args.context)
    launcher_tools.check_serve_settings(host=args.host, port=args.port, alias=alias, context=args.context,
                                        kv_dtype=args.kv_dtype, model_match=args.model_match)
    if args.score and args.no_launcher and args.server_exe is None:
        raise ValueError("--score needs ninfer-spo-perplexity beside the server: pass --server-exe")
    server = args.server_exe.resolve() if args.server_exe else (
        None if args.no_launcher and not args.score
        else launcher_tools.find_server_exe(FORK_ROOT, (preset.format_name,))
    )
    gptq_python = None
    if preset.method == "gptq":
        gptq_python = (args.gptq_python or default_venv_python(FORK_ROOT / ".venv-gptq")).absolute()
    llama_dir = args.llama_cpp_dir.resolve() if args.llama_cpp_dir else None
    cfg = BuildConfig(
        preset=preset, model=model, assistant=args.assistant.resolve() if args.assistant else None,
        with_vision=args.with_vision, out_dir=out_dir, work_dir=work_dir, name=name,
        gptq_python=gptq_python, llama_cpp_dir=llama_dir,
        llama_imatrix=find_llama_imatrix(llama_dir, args.llama_imatrix.resolve() if args.llama_imatrix else None)
        if preset.method == "rtn" else None,
        llama_python=args.llama_python, imatrix_gpu_layers=args.imatrix_gpu_layers,
        calibration_text=args.calibration_text.resolve() if args.calibration_text else None,
        score=args.score, perplexity_exe=launcher_tools.sibling_binary(server, launcher_tools.PERPLEXITY_NAME)
        if args.score and server else None, keep_work=args.keep_work,
    )
    serve = ServeSettings(host=args.host, port=args.port, alias=args.alias, context=args.context,
                          kv_dtype=args.kv_dtype, model_match=args.model_match, server_exe=server,
                          cuda_bin=args.cuda_bin,
                          launcher=not args.no_launcher)
    return cfg, serve, server


def confirm(args: argparse.Namespace) -> bool:
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print("ninfer.spo: not a terminal; pass --yes to start without a prompt", file=sys.stderr)
        return False
    return input("Start the build? [y/N] ").strip().lower() in ("y", "yes")


def run_build(args: argparse.Namespace, probes: Probes) -> int:
    cfg, serve, server = resolve_config(args)
    identity = build_identity(cfg)
    fingerprint = fingerprint_of(identity)
    report = preflight(cfg, serve, server, probes, fingerprint, identity)
    print("Preflight")
    print("\n".join(report.render()))
    print()
    if args.check:
        print(f"{len(report.failed)} check(s) failed." if report.failed else "All checks passed.")
        return 1 if report.failed else 0
    steps = build_plan(cfg)
    print("\n".join(describe_plan(cfg, steps)))
    print()
    if args.dry_run:
        if report.failed:
            print(f"Dry run: nothing was written. {len(report.failed)} preflight check(s) above would stop a real run.")
        else:
            print("Dry run: nothing was written.")
        return 0
    if report.failed:
        print(f"Not starting: {len(report.failed)} preflight check(s) failed (see the lines marked [FAIL]).",
              file=sys.stderr)
        return 1
    if not confirm(args):
        print("Cancelled; nothing was written.")
        return 1
    builder = Builder(cfg, steps, identity, serve, server_exe=server, tool_versions=report.tool_versions)
    builder.run()
    removed = builder.cleanup()
    print()
    print(f"Done. Artifact: {cfg.artifact}")
    print(f"      Summary:  {cfg.out_dir / 'BUILD-SUMMARY.md'}")
    if serve.launcher and server is not None:
        print(f"      Launcher: {cfg.out_dir / f'serve-{cfg.name}.cmd'}")
    print("      Work dir removed." if removed else f"      Work dir kept: {cfg.work_dir}")
    return 0


# ---------------------------------------------------------------- setup-gptq and launcher


def setup_gptq(args: argparse.Namespace, probes: Probes) -> int:
    uv = shutil.which("uv")
    venv = args.venv.resolve()
    python = default_venv_python(venv)
    requirements = FORK_ROOT / "tools" / "gptq" / "requirements.txt"
    create = [uv or "uv", "venv", "--python", args.python, str(venv)]
    install = [uv or "uv", "pip", "install", "--python", str(python), "--index-url",
               "https://download.pytorch.org/whl/cu130", "--extra-index-url", "https://pypi.org/simple",
               "--index-strategy", "unsafe-best-match", "-r", str(requirements)]
    reuse = (venv / "pyvenv.cfg").is_file()
    commands = ([] if reuse else [create]) + [install]
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command))
        print("Dry run: nothing was changed.")
        return 0
    if uv is None:
        raise ValueError("uv is not installed: see https://docs.astral.sh/uv/getting-started/installation/ "
                         "(for example: winget install astral-sh.uv), then rerun")
    if venv.exists() and any(venv.iterdir()) and not reuse:
        raise ValueError(f"{venv} exists and is not a virtual environment; choose another --venv")
    for command in commands:
        print("+ " + (subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)), flush=True)
        if subprocess.run(command).returncode != 0:
            raise ValueError(f"{command[1]} failed; fix the error above and rerun (an existing venv is reused)")
    pins = parse_pins(requirements, GPTQ_PINS)
    info = probes.python_info(str(python), GPTQ_PINS, cuda=True)
    if info is None or "error" in info:
        raise ValueError(f"the new environment's interpreter {python} does not run: {(info or {}).get('error')}")
    wrong = [f"{name} {info['packages'].get(name)} (pinned {pins[name]})" for name in pins
             if (info["packages"].get(name) or "").split("+")[0] != pins[name].split("+")[0]]
    print(f"{python}: Python {info['python']}; {versions_text(info['packages'])}; CUDA "
          f"{'visible' if info.get('cuda') else 'NOT visible'}")
    if wrong:
        raise ValueError("pinned versions differ: " + "; ".join(wrong))
    if not info.get("cuda"):
        raise ValueError("the environment installed but torch cannot see CUDA: update the NVIDIA driver (CUDA 13 "
                         "builds need a recent one) and rerun")
    print(f"Ready. GPTQ presets pick it up automatically when it is {default_venv_python(FORK_ROOT / '.venv-gptq')}; "
          f"otherwise pass --gptq-python {python}")
    return 0


def run_launcher(args: argparse.Namespace) -> int:
    server = args.server_exe.resolve() if args.server_exe else launcher_tools.find_server_exe(FORK_ROOT)
    if server is None:
        raise ValueError("no ninfer-spo-serve found in build*/apps; pass --server-exe")
    alias = args.alias or launcher_tools.default_alias(args.context)
    info = launcher_tools.write_launcher(
        args.out.resolve(), args.artifact.resolve(), server, host=args.host, port=args.port, alias=alias,
        context=args.context, kv_dtype=args.kv_dtype, model_match=args.model_match, no_mtp=args.no_mtp,
        cuda_bin=args.cuda_bin,
    )
    print(f"Wrote {args.out}: server {server}; MTP {'on' if info.mtp and not args.no_mtp else 'off'}; "
          f"vision {'on' if info.vision else 'off'}; --model-id {alias}; context {args.context}; "
          f"http://{args.host}:{args.port}")
    return 0


def run_serve_check(args: argparse.Namespace) -> int:
    checks, receipt = serve_check(args.launcher, args.out, host=args.host, port=args.port)
    failed = [item for item in checks if item["status"] != "pass"]
    for item in checks:
        head = "PASS" if item["status"] == "pass" else "FAIL"
        line = f"{head} {item['name']}" + (f": {item['detail']}" if item["detail"] else "")
        if item["status"] != "pass":
            action = SERVE_CHECK_ACTIONS.get(item["name"], "rerun serve-check after fixing the above")
            line += f"\n     -> {action}"
        print(line)
    print(f"Receipt: {receipt}")
    if failed:
        action = SERVE_CHECK_ACTIONS.get(failed[0]["name"], "rerun serve-check after fixing the above")
        print(f"ninfer.spo: serve-check failed: {failed[0]['name']}; {action}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None, probes: Probes | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    probes = probes or Probes()
    try:
        if argv and argv[0] == "setup-gptq":
            return setup_gptq(setup_parser().parse_args(argv[1:]), probes)
        if argv and argv[0] == "serve-check":
            return run_serve_check(serve_check_parser().parse_args(argv[1:]))
        if argv and argv[0] == "launcher":
            return run_launcher(launcher_parser().parse_args(argv[1:]))
        args = build_parser().parse_args(argv)
        if args.list_presets:
            print(preset_table())
            return 0
        return run_build(args, probes)
    except (ValueError, OSError, BuildError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ninfer.spo: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ninfer.spo: interrupted; rerun the same command to resume", file=sys.stderr)
        return 130
