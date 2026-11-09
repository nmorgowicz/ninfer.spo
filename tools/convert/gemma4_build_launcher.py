"""Windows launcher for the Gemma 4 31B server, the server binary search, and binary checks.

The serve flags are the qualified ones from the conversion guide: 131,072 context, FP8 global KV, one lane,
prefill chunk 128, MTP when the artifact carries the assistant, vision only when it carries the tower. Two
opt-in settings change them: `--gemma4-qk fp8` (FP8 QK, needs FP8 KV; the default emits no flag so the engine
default, BF16 QK, applies) and `--prefill-chunk` (a multiple of 128 or `auto`). This module imports nothing from
the other `gemma4_build*` modules.
"""

from __future__ import annotations

import glob
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

SERVER_NAME = "ninfer-spo-serve"
PERPLEXITY_NAME = "ninfer-spo-perplexity"
DEFAULT_CONTEXT = 131072
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
KV_DTYPES = ("fp8", "bf16")
MODEL_MATCHES = ("any", "exact")
DEFAULT_MODEL_MATCH = "any"
GEMMA4_QKS = ("bf16", "fp8")
DEFAULT_GEMMA4_QK = "bf16"
DEFAULT_PREFILL_CHUNK = "128"
# Libraries the fork's binaries load from their own directory (docs/building.md): FFmpeg and the CUDA runtime.
REQUIRED_DLLS = (
    "avcodec-*.dll",
    "avformat-*.dll",
    "avutil-*.dll",
    "swscale-*.dll",
    "swresample-*.dll",
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
)
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]*")


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """What the launcher needs to know about a `.ninfer` file."""

    mtp: bool
    vision: bool
    max_context: int


def exe_name(base: str) -> str:
    return base + (".exe" if os.name == "nt" else "")


def cmd_path(value: Path) -> str:
    """Absolute path safe inside a double-quoted CMD word; refuses quotes and line breaks like the other tools."""
    text = str(Path(value).resolve())
    if any(char in text for char in ('"', "\r", "\n")):
        raise ValueError(f"CMD paths cannot contain quotes or line breaks: {text!r}")
    return text.replace("%", "%%")


def default_alias(context: int) -> str:
    return f"{round(context / 1000)}k"


def check_serve_settings(
    *, host: str, port: int, alias: str, context: int, kv_dtype: str, model_match: str = DEFAULT_MODEL_MATCH,
    gemma4_qk: str = DEFAULT_GEMMA4_QK, prefill_chunk: str = DEFAULT_PREFILL_CHUNK,
) -> None:
    if not _HOST.fullmatch(host):
        raise ValueError(f"--host {host!r}: use an IPv4/IPv6 address or host name (letters, digits, '.', ':', '-')")
    if not 1 <= port <= 65535:
        raise ValueError("--port must be 1..65535")
    if not _ALIAS.fullmatch(alias):
        raise ValueError(f"--alias {alias!r}: use letters, digits, '.', '_' and '-' only (no spaces, quotes or %)")
    if context < 1:
        raise ValueError("--context must be positive")
    if kv_dtype not in KV_DTYPES:
        raise ValueError(f"--kv-dtype must be one of {KV_DTYPES} (fp8 is the qualified setting)")
    if model_match not in MODEL_MATCHES:
        raise ValueError(f"--model-match must be one of {MODEL_MATCHES}")
    if gemma4_qk not in GEMMA4_QKS:
        raise ValueError(f"--gemma4-qk must be bf16 or fp8, got {gemma4_qk!r}")
    if gemma4_qk == "fp8" and kv_dtype != "fp8":
        raise ValueError("--gemma4-qk fp8 needs --kv-dtype fp8 (the server rejects it with "
                         f"{kv_dtype} KV); drop --gemma4-qk or use --kv-dtype fp8")
    if prefill_chunk != "auto" and not (
        prefill_chunk.isdigit() and 128 <= int(prefill_chunk) <= 16384 and int(prefill_chunk) % 128 == 0
    ):
        raise ValueError(f"--prefill-chunk must be 'auto' or a multiple of 128 in 128..16384, got {prefill_chunk!r}")


def serve_groups(
    *, host: str, port: int, alias: str, context: int, kv_dtype: str, mtp: bool, vision: bool,
    model_match: str = DEFAULT_MODEL_MATCH, gemma4_qk: str = DEFAULT_GEMMA4_QK,
    prefill_chunk: str = DEFAULT_PREFILL_CHUNK,
) -> list[list[str]]:
    """The recommended `ninfer-spo-serve` flags (after the artifact path), grouped one launcher line each."""
    groups = [
        ["--host", host, "--port", str(port), "--model-id", alias]
        + (["--model-match", "any"] if model_match == "any" else []),
        ["--max-context", str(context), "--kv-capacity", "auto", "--kv-headroom-mib", "1024", "--kv-dtype", kv_dtype]
        + (["--gemma4-qk", "fp8"] if gemma4_qk == "fp8" else []),
        ["--max-concurrency", "1", "--prefill-chunk", prefill_chunk],
    ]
    if mtp:
        groups.append(["--spec", "mtp", "--draft-tokens", "auto"])
    groups.append(["--device-state-slots", "0", "--host-context-mib", "8192"] + (["--vision"] if vision else []))
    return groups


def serve_argv(server: Path, artifact: Path, **settings) -> list[str]:
    """Flat argv of the recommended server command, for the build summary."""
    return [str(server), str(artifact)] + [flag for group in serve_groups(**settings) for flag in group]


def launcher_lines(
    server: str,
    artifact: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    alias: str | None = None,
    context: int = DEFAULT_CONTEXT,
    kv_dtype: str = "fp8",
    model_match: str = DEFAULT_MODEL_MATCH,
    gemma4_qk: str = DEFAULT_GEMMA4_QK,
    prefill_chunk: str = DEFAULT_PREFILL_CHUNK,
    mtp: bool,
    vision: bool,
    cuda_bin: str | None = None,
) -> list[str]:
    """CMD lines; `server`, `artifact` and `cuda_bin` are already `cmd_path` strings."""
    alias = alias or default_alias(context)
    check_serve_settings(
        host=host, port=port, alias=alias, context=context, kv_dtype=kv_dtype, model_match=model_match,
        gemma4_qk=gemma4_qk, prefill_chunk=prefill_chunk,
    )
    groups = serve_groups(
        host=host, port=port, alias=alias, context=context, kv_dtype=kv_dtype, mtp=mtp, vision=vision,
        model_match=model_match, gemma4_qk=gemma4_qk, prefill_chunk=prefill_chunk,
    )
    lines = [
        "@echo off",
        "setlocal DisableDelayedExpansion",
        "rem Generated by python -m tools.convert.gemma4_build; qualified serve settings of the Gemma 4 31B guide"
        + ("; MTP assistant enabled." if mtp else "; no MTP assistant in this artifact."),
        "rem The server accepts any API key (none is configured). "
        + (
            "--model-match any: OpenAI requests may send any model name."
            if model_match == "any"
            else "OpenAI endpoints answer only to --model-id."
        ),
    ]
    if gemma4_qk == "fp8":
        lines.append("rem Gemma attention QK in FP8: opt-in, faster long prompts, measurably lower fidelity "
                     "(ninfer.spo docs/configuration.md, Choosing --gemma4-qk).")
    if cuda_bin:
        lines.append(f'set "PATH={cuda_bin};%PATH%"')
    lines.append(f'"{server}" "{artifact}" ^')
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        lines.append("  " + " ".join(group) + ("" if last else " ^"))
    lines.extend(["exit /b %errorlevel%", ""])
    return lines


def artifact_info(path: Path) -> ArtifactInfo:
    """Components and context limit of a Gemma 4 artifact (reads the directory only)."""
    from tools.artifact.reader import Artifact

    with Artifact(path) as artifact:
        components = artifact.directory.components
        config = components["text"]["config"]
        if config.get("architectures") != ["Gemma4ForCausalLM"]:
            raise ValueError(f"{path} is not a Gemma 4 text artifact")
        return ArtifactInfo(
            mtp="mtp" in components,
            vision="vision" in components,
            max_context=int(config["max_position_embeddings"]),
        )


def write_launcher(
    out: Path,
    artifact: Path,
    server: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    alias: str | None = None,
    context: int = DEFAULT_CONTEXT,
    kv_dtype: str = "fp8",
    model_match: str = DEFAULT_MODEL_MATCH,
    gemma4_qk: str = DEFAULT_GEMMA4_QK,
    prefill_chunk: str = DEFAULT_PREFILL_CHUNK,
    no_mtp: bool = False,
    cuda_bin: Path | None = None,
    info: ArtifactInfo | None = None,
    overwrite: bool = False,
) -> ArtifactInfo:
    """Write `out` (CRLF, refuses an existing file unless `overwrite`) and return what it detected."""
    if not server.is_file():
        raise ValueError(f"server binary not found: {server}")
    if not artifact.is_file():
        raise ValueError(f"artifact not found: {artifact}")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Launcher already exists: {out}")
    info = info or artifact_info(artifact)
    if context > info.max_context:
        raise ValueError(f"--context {context} exceeds the artifact's position limit {info.max_context}")
    lines = launcher_lines(
        cmd_path(server),
        cmd_path(artifact),
        host=host,
        port=port,
        alias=alias,
        context=context,
        kv_dtype=kv_dtype,
        model_match=model_match,
        gemma4_qk=gemma4_qk,
        prefill_chunk=prefill_chunk,
        mtp=info.mtp and not no_mtp,
        vision=info.vision,
        cuda_bin=cmd_path(cuda_bin) if cuda_bin else None,
    )
    if any(ord(char) > 127 for line in lines for char in line):
        lines.insert(1, "chcp 65001 >nul")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\r\n".join(lines), encoding="utf-8", newline="")
    return info


def server_candidates(fork_root: Path) -> list[Path]:
    """Server binaries of the fork's build trees, newest first: `<fork>/build*/apps` and `<fork>/../build-spo-*/apps`."""
    patterns = [
        str(fork_root / "build*" / "apps" / exe_name(SERVER_NAME)),
        str(fork_root.parent / "build-spo-*" / "apps" / exe_name(SERVER_NAME)),
    ]
    found = {Path(item) for pattern in patterns for item in glob.glob(pattern)}
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def scan_binary(path: Path, needles: tuple[str, ...]) -> dict[str, bool]:
    """Whether each ASCII needle occurs in the binary (memory-mapped, read-only)."""
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return {needle: data.find(needle.encode("ascii")) >= 0 for needle in needles}


def find_server_exe(fork_root: Path, required: tuple[str, ...] = ()) -> Path | None:
    """The newest server build that contains every `required` format name; else the newest build; else None."""
    candidates = server_candidates(fork_root)
    for path in candidates:
        if all(scan_binary(path, required).values()):
            return path
    return candidates[0] if candidates else None


def missing_dlls(exe: Path) -> list[str]:
    """DLL patterns found neither next to `exe` nor in a PATH directory."""
    folders = [exe.parent] + [Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    missing = []
    for pattern in REQUIRED_DLLS:
        if not any(glob.glob(str(folder / pattern)) for folder in folders):
            missing.append(pattern)
    return missing


def sibling_binary(server: Path, base: str) -> Path | None:
    path = server.parent / exe_name(base)
    return path if path.is_file() else None


def which_any(names: tuple[str, ...], folders: list[Path]) -> Path | None:
    for folder in folders:
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None
