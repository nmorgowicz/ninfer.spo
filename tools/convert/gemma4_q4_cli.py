"""Convert Gemma 4 31B BF16 into a Q4/FP8 artifact, verify it, and render calibration text.

`python -m tools.convert.gemma4_q4 <command>`; run it from this checkout. `convert` uses the converter's
Q4 group-32 kernels with an FP8 vocabulary, not QAT recovery or NVFP4 calibration. The MTP assistant
(google/gemma-4-31B-it-assistant) is embedded as the `mtp` component unless --no-mtp is given. Source
files are read-only. Conversion receipts attest storage and identity, not model quality.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys

from .gemma4_q4 import (
    ASSISTANT_HINT,
    MTP_WEIGHTS,
    ASSISTANT_REPO,
    Q4_REGIONS,
    Q4_WEIGHTS,
    SCALE_SEARCHES,
    Q4Options,
    build_recipe,
    check_assistant,
    normalize_config,
    render,
    select_components,
    sha256_file,
)

FORK_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCHEMA = "ninfer.gemma4-q4-build/v1"
CALIBRATION_SCHEMA = "ninfer.templated-calibration/v1"


def fork_revision() -> str:
    """HEAD of this checkout; tracked modifications are refused so a receipt names the code that ran."""
    revision = subprocess.check_output(["git", "-C", str(FORK_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(FORK_ROOT), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        raise ValueError("ninfer.spo has tracked modifications; reconcile them before conversion")
    return revision


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_artifact(path: Path) -> dict:
    from tools.artifact.inspect import artifact_summary
    from tools.artifact.reader import Artifact

    with Artifact(path) as artifact:
        summary = artifact_summary(artifact)
        config = artifact.directory.components["text"]["config"]
        if config.get("architectures") != ["Gemma4ForCausalLM"]:
            raise ValueError("Artifact is not a Gemma 4 text model")
        # Iterating every object checks continuation headers, lengths and encodings.
        for obj in artifact.objects:
            for _ in artifact.iter_object(obj.id):
                pass
        files = [
            path if index == 0 else path.parent / record.path
            for index, record in enumerate(artifact.directory.files)
        ]
    summary["file_sha256"] = {file.name: sha256_file(file) for file in files}
    return summary


def options_from(args: argparse.Namespace) -> Q4Options:
    return Q4Options(
        weights=args.weights,
        group_size=args.group_size,
        region_groups=tuple(args.region_group),
        scale_search=args.scale_search,
        native_groups=args.native_groups,
        imatrix=args.imatrix,
        mtp_weights=args.mtp_weights,
    )


def check_convert_arguments(args: argparse.Namespace) -> tuple[Q4Options, Path]:
    """Refuse every argument-only mistake before any checkpoint is opened; return options and output."""
    if args.rows_per_chunk < 1:
        raise ValueError("--rows-per-chunk must be positive")
    output = args.out.resolve()
    if output.suffix != ".ninfer":
        raise ValueError("--out must end in .ninfer")
    if (
        output.exists()
        or Path(str(output) + ".build.json").exists()
        or Path(str(output) + ".conversion.json").exists()
    ):
        raise FileExistsError(f"Output already exists: {output}; use a fresh run directory")
    if output.is_relative_to(args.model.resolve()):
        raise ValueError("Output must be outside the source checkpoint")
    if args.no_mtp == (args.mtp_assistant is not None):
        raise ValueError(f"Pass --mtp-assistant <dir> ({ASSISTANT_HINT}) or explicitly --no-mtp")
    if args.no_mtp and args.mtp_weights != "bf16":
        raise ValueError("--mtp-weights fp8|q4 needs the MTP assistant; it cannot be combined with --no-mtp")
    options = options_from(args)
    if args.mtp_assistant is not None:
        check_assistant(args.mtp_assistant)
    return options, output


def convert_model(args: argparse.Namespace) -> None:
    from tools.convert.gemma4 import build_model
    from tools.convert.pipeline import convert
    from tools.convert.sources.safetensors import SafetensorsSource

    revision = fork_revision()
    options, output = check_convert_arguments(args)
    receipt_path = Path(str(output) + ".build.json")
    with ExitStack() as stack:
        source = stack.enter_context(SafetensorsSource(args.model))
        assistant = (
            stack.enter_context(SafetensorsSource(args.mtp_assistant))
            if args.mtp_assistant is not None
            else None
        )
        original_config = source.config
        source.config = normalize_config(original_config)
        components = select_components(
            bool(original_config.get("vision_config")), args.text_only, assistant is not None
        )
        model = build_model(
            source, components=components, companions={"mtp": assistant} if assistant else None
        )
        # A bounded read forces the adapter to check every selected source shape.
        for parameter in model.parameters.values():
            parameter.source.values(0, 1)
        sources = [source] + ([assistant] if assistant else [])
        if any(src.describe(name).dtype not in ("BF16", "F32") for src in sources for name in src.weight_map):
            raise ValueError("Expected BF16 weights (F32 scalars allowed)")
        built = build_recipe(
            model, source, options, device=args.device, rows_per_chunk=args.rows_per_chunk
        )
        print(
            json.dumps(
                {
                    "fork_revision": revision,
                    "source": str(args.model),
                    "components": components,
                    "recipe": built.name,
                    "weight_bytes": built.payload,
                    "weight_gib": built.payload / 2**30,
                    "q4_groups": built.q4_groups,
                    "global_head_dim": model.config["global_head_dim"],
                    "num_global_key_value_heads": model.config["num_global_key_value_heads"],
                },
                indent=2,
            ),
            flush=True,
        )
        if args.dry_run:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        input_files = sorted(
            set(source.weight_map.values())
            | {
                args.model / "config.json",
                args.model / "model.safetensors.index.json",
                *(
                    args.model / role
                    for role in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                        "generation_config.json",
                    )
                ),
            }
        )
        if "vision" in components:
            input_files.append(args.model / "processor_config.json")
        print("Hashing source inputs (read-only)", flush=True)
        provenance = {
            "fork": "https://github.com/spongioblast/ninfer.spo",
            "revision": revision,
            "tool_sha256": sha256_file(Path(__file__)),
            "source": str(args.model.resolve()),
            "source_sha256": {file.name: sha256_file(file) for file in input_files},
            "normalized_text_config": source.config["text_config"],
            "recipe": built.name,
            "q4_groups": built.q4_groups,
            "scale_search": built.scale_search,
            "mtp": (
                {
                    "source": str(args.mtp_assistant.resolve()),
                    "sha256": {
                        file.name: sha256_file(file)
                        for file in sorted(
                            set(assistant.weight_map.values()) | {args.mtp_assistant / "config.json"}
                        )
                    },
                    "recipe": "fp8 head, direct other parameters (fork official)",
                }
                if assistant
                else "omitted (--no-mtp)"
            ),
        }
        receipt = {"schema": BUILD_SCHEMA, "status": "converting", **provenance}
        write_json(receipt_path, receipt)

        def progress(index, total, job):
            print(f"[{index + 1}/{total}] {job.parameters[0]}: {job.spec.format}", flush=True)

        try:
            report = convert(
                model,
                built.recipe,
                output,
                name=args.name or args.model.name,
                provenance=provenance,
                device=args.device,
                rows_per_chunk=args.rows_per_chunk,
                progress=progress,
            )
            receipt["artifact"] = verify_artifact(output)
            receipt["seconds"] = report["seconds"]
            receipt["status"] = "storage-verified; runtime-and-quality-unverified"
        except Exception as error:
            receipt.update(status="failed", error=str(error))
            raise
        finally:
            write_json(receipt_path, receipt)
        print(f"Verified storage: {output}", flush=True)


def calibration_text(args: argparse.Namespace) -> int:
    receipt_path = args.out.with_suffix(".json")
    if args.out.exists() or receipt_path.exists():
        print(f"error: {args.out} or its receipt already exists; use a fresh path", file=sys.stderr)
        return 2
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("calibration-text requires the tokenizers package: pip install tokenizers") from error
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prefix_ids = [int(item) for item in args.prefix_ids.split(",")]
    text = args.source.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    try:
        segments, skipped, stride = render(tokenizer, text, prefix_ids, args.ctx_size, args.chunks)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(segments), encoding="utf-8", newline="")
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "out": str(args.out.resolve()),
        "out_sha256": sha256_file(args.out),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "prefix_ids": prefix_ids,
        "ctx_size": args.ctx_size,
        "chunks": len(segments),
        "skipped": skipped,
        "stride_tokens": stride,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.convert.gemma4_q4", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("convert", help="Read BF16 source and write Q4/FP8 V3 artifact plus receipts")
    build.add_argument("--model", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--name", default="")
    build.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    build.add_argument("--rows-per-chunk", type=int, default=512)
    build.add_argument("--text-only", action="store_true", help="Omit source vision tower")
    build.add_argument(
        "--weights",
        choices=Q4_WEIGHTS,
        default="q4_g32",
        help="q4_g32: shipping Q4/W4A8 recipe (default). q4_g32_a16: same Q4 weights with no "
        "A8 prefill permission (weight-only quality arm). fp8_row: A16 FP8-row trunk used "
        "only as the near-BF16 KL reference; never a deliverable",
    )
    build.add_argument(
        "--group-size",
        type=int,
        choices=(32, 64, 128, 256),
        default=32,
        help="Q4 logical absmax group for projections not matched by --region-group. Values above 32 "
        "use the emulated method: g32 words with the coarse scale repeated, so quality is "
        "measured but the file does not shrink. Ignored for --weights fp8_row",
    )
    build.add_argument(
        "--region-group",
        action="append",
        default=[],
        metavar="REGION=G",
        help="Per-region logical group; repeat. Regions: "
        + ", ".join(Q4_REGIONS)
        + ". The first matching flag wins, in command order",
    )
    build.add_argument(
        "--scale-search",
        choices=SCALE_SEARCHES,
        default="absmax",
        help="Q4 scale choice. absmax: existing control. mse: exact squared-error scale search "
        "(unweighted). weighted: the same search with imatrix importance (needs --imatrix)",
    )
    build.add_argument(
        "--native-groups",
        action="store_true",
        help="Store logical groups above 32 in the real q4_g128_fp16/q4_g256_fp16 format (smaller file) "
        "instead of repeated g32 scales. Works with A16 (q4_g32_a16) and W4A8 (q4_g32). Group 64 "
        "has no Gemma kernels and is refused",
    )
    build.add_argument(
        "--imatrix",
        type=Path,
        help="GGUF imatrix for --scale-search weighted (needs `pip install gguf`). A file from a different "
        "model is a sensitivity arm, recorded as such; production statistics come from the target model",
    )
    build.add_argument(
        "--mtp-assistant",
        type=Path,
        help=f"Local {ASSISTANT_REPO} directory embedded as the mtp component (required unless --no-mtp)",
    )
    build.add_argument("--no-mtp", action="store_true", help="Explicitly build without the MTP assistant")
    build.add_argument(
        "--mtp-weights",
        choices=MTP_WEIGHTS,
        default="bf16",
        help="Storage of the MTP assistant's layer projections (query, output, gate, up, down): bf16 (default, "
        "unchanged), fp8 (FP8 per-row) or q4 (q4_g32, absmax). They always run on 16-bit activations; the "
        "assistant's FP8 head and outer projections are unchanged. Draft quality, not target output, is at risk: "
        "measure MTP acceptance before shipping",
    )
    build.add_argument(
        "--dry-run", action="store_true", help="Read-only shape/recipe preflight; writes nothing"
    )
    verify = subs.add_parser("verify", help="Read all artifact objects, validate storage and hash files")
    verify.add_argument("--artifact", required=True, type=Path)
    calibration = subs.add_parser(
        "calibration-text",
        help="Render public text as fixed-length chunks that each start with a chat prefix, for llama-imatrix",
        description="A chat fine-tune scores prose badly without its turn prefix, so statistics collected on "
        "bare text describe activations the model never sees at evaluation. This writes one text file in which "
        "every --ctx-size-token chunk is `prefix + body`; collect it with llama-imatrix --parse-special "
        "--ctx-size N --chunks K. llama-imatrix adds one BOS to the whole file, so the first chunk's literal BOS "
        "is dropped; every later chunk keeps its own. Bodies are spread evenly across the source; a chunk that "
        "does not re-encode to exactly the chunk length is skipped and counted. The source text must be disjoint "
        "from the evaluation corpus: the receipt records its hash. Needs the tokenizers package.",
    )
    calibration.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json of the target model")
    calibration.add_argument(
        "--source", required=True, type=Path, help="public UTF-8 text, disjoint from the evaluation corpus"
    )
    calibration.add_argument(
        "--prefix-ids", required=True, help="comma-separated prefix token ids, starting with BOS"
    )
    calibration.add_argument("--chunks", required=True, type=int)
    calibration.add_argument("--ctx-size", type=int, default=512)
    calibration.add_argument(
        "--out", required=True, type=Path, help="new text file; a .json receipt is written beside it"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "convert":
            convert_model(args)
        elif args.command == "verify":
            print(json.dumps(verify_artifact(args.artifact), indent=2))
        else:
            return calibration_text(args)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"ninfer.spo: {error}", file=sys.stderr)
        return 1
