"""Opt-in native Gemma MLP input collection and identity-checked A4 artifact handoff.

collect executes the native eager calibration target; candidate is a CPU-only encoded-object
copy, not a weight reconversion. It never changes its source or existing runtime defaults.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from .gemma4_q4 import sha256_file

SCHEMA = "ninfer.gemma4.mlp-calibration/v1"
NATIVE_SCHEMA = "ninfer.gemma4.mlp-input-absmax/v1"
BOUNDARIES = {
    "gate_up": "post-attention residual plus pre_feedforward_layernorm BF16 output; shared gate/up input",
    "down": "BF16 GEGLU(gate,up) product before down projection; not gate/up outputs or residual",
}


def save_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def artifact_identity(artifact: Artifact) -> dict:
    paths = [artifact.path] + [artifact.path.parent / f.path for f in artifact.directory.files[1:]]
    return {"artifact_id": artifact.artifact_id.hex(), "files": [identity(p) for p in paths]}


def require_identity(actual: dict, expected: dict) -> None:
    # Location is not identity: an exact copy can be collected/converted from another path.
    def content(value):
        return value["artifact_id"], [(f["bytes"], f["sha256"]) for f in value["files"]]
    if content(actual) != content(expected):
        raise ValueError("calibration source artifact identity mismatch")


def config_for(artifact: Artifact) -> dict:
    config = artifact.directory.components.get("text", {}).get("config", {})
    if config.get("model_type") != "gemma4_text" or config.get("architectures") != ["Gemma4ForCausalLM"]:
        raise ValueError("requires a native Gemma 4 text artifact")
    for key in ("num_hidden_layers", "hidden_size", "intermediate_size"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"invalid Gemma geometry: {key}")
    if any(u.get("activation_policy", "A16Only") != "A16Only" for u in artifact.directory.uses):
        raise ValueError("calibration source must be entirely A16Only, including protected components")
    return config


def validate_statistics(raw: dict, config: dict) -> dict[str, float]:
    """Reject incomplete boundaries/counts; return one FP32 divisor per logical input."""
    if raw.get("schema") != NATIVE_SCHEMA or raw.get("kv_dtype") != "bf16" or raw.get("qk") != "bf16":
        raise ValueError("unsupported native collection protocol")
    rows, context = raw.get("rows"), raw.get("context")
    if type(rows) is not int or type(context) is not int or rows < 1 or not 1 <= context <= 16384:
        raise ValueError("invalid calibration extent")
    expected = {}
    for layer in range(config["num_hidden_layers"]):
        for boundary, width, logical in (
            ("gate_up", config["hidden_size"], f"text/layers/{layer}/ffn_input"),
            ("down", config["intermediate_size"], f"text/layers/{layer}/mlp/product"),
        ):
            expected[f"text/layers/{layer}/mlp/{boundary}"] = (width, logical)
    statistics = raw.get("statistics", {})
    if set(statistics) != set(expected):
        raise ValueError("statistics must cover exactly both logical MLP inputs in every layer")
    divisors = {}
    for name, (width, logical) in expected.items():
        stat = statistics[name]
        if (stat.get("width"), stat.get("tokens"), stat.get("calls")) != (width, rows * context, rows):
            raise ValueError(f"{name}: incomplete or wrong-width logical input coverage")
        amax = stat.get("absmax")
        if type(amax) not in (int, float) or not math.isfinite(amax) or amax <= 0:
            raise ValueError(f"{name}: absmax must be finite and positive; no fabricated zero fallback")
        # Actual consumer: block_scale = E4M3(divisor * block_absmax / 6), codes =
        # E2M1(input * divisor / block_scale), epilogue alpha = 1/(weight_divisor*divisor).
        # Match official gemma4_nvfp4_a4_mlp's /6 activation rule, not the /4,/6 weight search.
        try:
            divisor = struct.unpack("<f", struct.pack("<f", 448.0 * 6.0 / amax))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name}: activation divisor is not representable in FP32") from error
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError(f"{name}: invalid FP32 activation divisor")
        divisors[logical] = divisor
    return divisors


def a4_uses(artifact: Artifact, config: dict, divisors: dict) -> tuple[list, dict]:
    """Keep all non-NVFP4/protected uses unchanged, enforce native packed-parent semantics."""
    uses = deepcopy(list(artifact.directory.uses))
    selected = {}
    parent_inputs = {}
    expected_uses = set()
    for layer in range(config["num_hidden_layers"]):
        for role in ("gate", "up", "down"):
            expected_uses.add((f"text/layers/{layer}/mlp/{role}",
                               f"text/layers/{layer}/mlp/product" if role == "down" else f"text/layers/{layer}/ffn_input"))
    found = set()
    for use in uses:
        name, logical = use["parameter"], use["input"]
        if (name, logical) not in expected_uses:
            if name.startswith("text/layers/") and "/mlp/" in name:
                raise ValueError(f"unexpected mathematical MLP use: {name}@{logical}")
            continue
        if (name, logical) in found:
            raise ValueError("duplicate logical MLP use")
        found.add((name, logical))
        binding = artifact.directory.bindings[name]
        parts = binding.get("parts", [{"object": binding.get("object")}])
        objects = [artifact.object(p["object"]) for p in parts]
        formats = {getattr(o, "format", None) for o in objects}
        if formats != {"nvfp4"}:
            if "nvfp4" in formats:
                raise ValueError(f"{name}: mixed-format logical projection cannot be partially A4")
            continue
        width = config["intermediate_size"] if name.endswith("/down") else config["hidden_size"]
        rows = config["hidden_size"] if name.endswith("/down") else config["intermediate_size"]
        count = 0
        for part, obj in zip(parts, objects):
            if len(obj.shape) != 2 or obj.shape[1] != width:
                raise ValueError(f"{name}: packed parent input width mismatch")
            begin, end = part.get("range", (0, math.prod(obj.shape)))
            if begin % width or end % width or not 0 <= begin < end <= math.prod(obj.shape):
                raise ValueError(f"{name}: parent slice is not complete logical rows")
            count += end - begin
            previous = parent_inputs.setdefault(obj.id, logical)
            if previous != logical:
                raise ValueError("shared NVFP4 parent has different logical inputs/divisors")
        if count != rows * width:
            raise ValueError(f"{name}: incomplete logical projection binding")
        aux_id = "calibration/a4/" + logical
        if aux_id in artifact.by_id:
            raise ValueError("source already contains A4 calibration objects")
        use["activation_policy"] = "AllowA4"
        use.setdefault("auxiliaries", {})["activation_input_divisor"] = {"object": aux_id}
        selected[aux_id] = divisors[logical]
    if found != expected_uses:
        raise ValueError("artifact is missing logical MLP uses")
    by_parameter = {u["parameter"]: u for u in uses}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"text/layers/{layer}/mlp/"
        gate, up = (by_parameter[prefix + role] for role in ("gate", "up"))
        if gate["activation_policy"] != up["activation_policy"]:
            raise ValueError("native packed gate/up must have the same format and activation policy")
        for roles in (("gate", "up"), ("down",)):
            if by_parameter[prefix + roles[0]]["activation_policy"] != "AllowA4":
                continue
            parent = None
            cursor = 0
            for role in roles:
                binding = artifact.directory.bindings[prefix + role]
                parts = binding.get("parts", [{"object": binding.get("object")}])
                if len(parts) != 1:
                    raise ValueError("native NVFP4 projection requires one complete packed parent")
                part = parts[0]
                obj = artifact.object(part["object"])
                if parent is not None and parent != obj.id:
                    raise ValueError("native gate/up must occupy the same packed parent")
                parent = obj.id
                begin, end = part.get("range", (0, math.prod(obj.shape)))
                if begin != cursor:
                    raise ValueError("native packed rows must be contiguous in gate-then-up order")
                cursor = end
            if cursor != math.prod(artifact.object(parent).shape):
                raise ValueError("native NVFP4 projection must fill its packed parent")
    if not selected:
        raise ValueError("source has no NVFP4 MLP projections; no candidate produced")
    # A parent may not also feed an unmodified attention/assistant use. The runtime consumes
    # one policy/divisor per packed parent, not a different scale for each output-row slice.
    for use in uses:
        binding = artifact.directory.bindings[use["parameter"]]
        for part in binding.get("parts", [{"object": binding.get("object")}]):
            if part["object"] in parent_inputs and use["input"] != parent_inputs[part["object"]]:
                raise ValueError("A4 parent also serves an unrelated logical input")
    return uses, selected


def receipt_statistics(receipt: dict) -> dict:
    """Bind the embedded stats to the retained native output and exact selected token rows."""
    for record in (receipt["statistics_file"], receipt["calibration"]["tokens"]):
        actual = identity(Path(record["path"]))
        if (actual["bytes"], actual["sha256"]) != (record["bytes"], record["sha256"]):
            raise ValueError("calibration evidence changed since collection")
    raw = json.loads(Path(receipt["statistics_file"]["path"]).read_text(encoding="utf-8"))
    if raw != receipt["statistics"]:
        raise ValueError("embedded statistics differ from native collector output")
    rows = json.loads(Path(receipt["calibration"]["tokens"]["path"]).read_text(encoding="utf-8"))
    prefix = receipt["calibration"]["prefix_ids"]
    if not isinstance(prefix, list) or not prefix or any(type(t) is not int or t < 0 for t in prefix):
        raise ValueError("invalid recorded calibration prefix")
    if (not isinstance(rows, list) or len(rows) != raw["rows"] or
            len(rows) != receipt["calibration"]["selected_rows"] or
            any(not isinstance(row, list) or len(row) != raw["context"] or
                row[:len(prefix)] != prefix or any(type(t) is not int or t < 0 for t in row)
                for row in rows)):
        raise ValueError("calibration token coverage/prefix differs from recorded protocol")
    return raw


def collect(args) -> None:
    from tokenizers import Tokenizer
    from tools.gptq.make_calib import GEMMA4_CHAT_PREFIX, calibration_rows

    args.out.mkdir(parents=True, exist_ok=False)
    prefix = [int(x) for x in (args.prefix_ids or GEMMA4_CHAT_PREFIX).split(",")]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = calibration_rows(args.text.read_text(encoding="utf-8"), tokenizer, args.context, prefix)
    available = len(rows)
    if args.rows is not None:
        if not 1 <= args.rows <= available:
            raise ValueError("--rows exceeds actual calibration material")
        rows = rows[:args.rows]
    with Artifact(args.artifact) as artifact:
        config = config_for(artifact)
        resources = artifact.directory.components["text"].get("resources", {})
        embedded = artifact.read_object(resources["tokenizer.json"])
        if hashlib.sha256(embedded).hexdigest() != sha256_file(args.tokenizer):
            raise ValueError("external tokenizer differs from the collected artifact's tokenizer")
        before = artifact_identity(artifact)
    binary = identity(args.collector)
    text_id, tokenizer_id = identity(args.text), identity(args.tokenizer)
    tokens_path, stats_path = args.out / "tokens.json", args.out / "statistics.json"
    save_new(tokens_path, rows)
    tokens_id = identity(tokens_path)
    command = [str(args.collector.resolve()), str(args.artifact.resolve()), str(tokens_path.resolve()), str(stats_path.resolve())]
    save_new(args.out / "invocation.json", {"argv": command, "source": before, "binary": binary})
    with (args.out / "collector.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    validate_statistics(raw, config)
    if (raw["rows"], raw["context"]) != (len(rows), args.context):
        raise ValueError("native statistics extent differs from supplied tokens")
    with Artifact(args.artifact) as artifact:
        require_identity(artifact_identity(artifact), before)
    for path, expected in ((args.collector, binary), (args.text, text_id),
                           (args.tokenizer, tokenizer_id), (tokens_path, tokens_id)):
        if identity(path) != expected:
            raise ValueError(f"collection input changed: {path}")
    save_new(args.out / "receipt.json", {
        "schema": SCHEMA, "utc": datetime.now(timezone.utc).isoformat(), "source": before,
        "collector": binary, "handoff": identity(Path(__file__)), "argv": command,
        "calibration": {"text": text_id, "tokenizer": tokenizer_id, "prefix_ids": prefix,
                        "available_rows": available, "selected_rows": len(rows), "selection": "leading rows",
                        "tokens": tokens_id},
        "boundaries": BOUNDARIES, "statistics_file": identity(stats_path), "statistics": raw,
        "scale_rule": "FP32(448*6/absmax); native E4M3 /6 block scales; no clipping search",
        "qualification": "unqualified; BF16 KV/QK eager A16 source; MTP/vision not executed",
    })


def candidate(args) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA or receipt.get("boundaries") != BOUNDARIES:
        raise ValueError("unsupported calibration receipt or logical input boundaries")
    if receipt.get("handoff", {}).get("sha256") != sha256_file(Path(__file__)):
        raise ValueError("calibration receipt belongs to a different collector handoff implementation")
    raw = receipt_statistics(receipt)
    with Artifact(args.artifact) as source:
        config = config_for(source)
        before = artifact_identity(source)
        require_identity(before, receipt["source"])
        divisors = validate_statistics(raw, config)
        uses, auxiliaries = a4_uses(source, config, divisors)
        specs = [TensorSpec(o.id, o.shape, o.format, o.layout) if o.kind == "tensor"
                 else ResourceSpec(o.id, o.bytes, o.encoding) for o in source.objects]
        specs += [TensorSpec(name, (), "fp32", "contiguous_le_v1") for name in auxiliaries]
        provenance = deepcopy(source.directory.provenance)
        provenance["gemma4_a4_mlp"] = {"receipt": identity(args.receipt), "source": before,
                                      "calibration": receipt["calibration"], "boundaries": BOUNDARIES,
                                      "scale_rule": receipt["scale_rule"], "handoff": identity(Path(__file__))}
        # Writer publishes only after complete coverage; no source file is opened for writing.
        with ArtifactWriter(args.output, specs, components=source.directory.components,
                            bindings=source.directory.bindings, uses=uses,
                            metadata=source.directory.metadata, provenance=provenance) as writer:
            for obj in source.objects:
                writer.write_object(obj.id, source.iter_object(obj.id))
            for name, divisor in auxiliaries.items():
                writer.write_object(name, struct.pack("<f", divisor))
            require_identity(artifact_identity(source), before)
        with Artifact(args.output) as output:
            output_identity = artifact_identity(output)
        save_new(args.output.with_suffix(args.output.suffix + ".a4.json"), {
            "schema": "ninfer.gemma4.a4-mlp-candidate/v1", "source": before,
            "receipt": identity(args.receipt), "output": output_identity,
            "changed_uses": [u for u in uses if u["activation_policy"] == "AllowA4"],
            "divisors": auxiliaries, "qualification": "not measured; no promotion implied",
        })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    p = actions.add_parser("collect", help="run eager native collector with exact calibration text")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new collection directory")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--prefix-ids")
    p.add_argument("--rows", type=int, help="explicit leading-row smoke subset; default all rows")
    p = actions.add_parser("candidate", help="CPU copy with calibrated A4 on NVFP4 MLP uses only")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "candidate":
        if args.output.exists() or args.output.with_suffix(args.output.suffix + ".a4.json").exists():
            raise FileExistsError(args.output)
        candidate(args)
    else:
        collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
