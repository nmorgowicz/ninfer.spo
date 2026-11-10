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
SAMPLED_SCHEMA = "ninfer.gemma4.mlp-input-k16/v2"
SAMPLING = {"method": "stratified-k16-reservoir-splitmix64/v1", "block_size": 16,
            "blocks_per_call": 8, "capacity": 1024, "probe_divisors": [1.0, 256.0, 1.2000000476837158]}
SEARCH_STEPS = tuple(range(-16, 49))
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
    if (raw.get("schema") not in (NATIVE_SCHEMA, SAMPLED_SCHEMA) or
            raw.get("kv_dtype") != "bf16" or raw.get("qk") != "bf16"):
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


def nearest_codes(values, levels):
    """Positive finite SATFINITE/RN-even; level indices are the actual magnitude codes."""
    import numpy as np

    upper = np.minimum(np.searchsorted(levels, values), len(levels) - 1)
    lower = np.maximum(upper - 1, 0)
    low_distance, high_distance = values - levels[lower], levels[upper] - values
    take_lower = (low_distance < high_distance) | ((low_distance == high_distance) & ((lower & 1) == 0))
    return np.where(take_lower, lower, upper).astype(np.uint8)


def quantize_blocks(blocks, divisor):
    """Emulate nvfp4_codec.cuh's separate FP32 mul/div, E4M3 and E2M1 RN-even.

    Return raw serving codes/scales as well as input-domain reconstruction. The latter
    excludes weight/accumulator/epilogue rounding and is not a model-quality estimate.
    """
    import numpy as np

    x = np.asarray(blocks, dtype=np.float32)
    d = np.float32(divisor)
    e4m3 = np.asarray([(i & 7) * 2.0 ** -9 if i < 8 else
                       (1 + (i & 7) / 8) * 2.0 ** ((i >> 3) - 7)
                       for i in range(127)], dtype=np.float32)
    e2m1 = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    unencoded = (np.max(np.abs(x), axis=1) * d) / np.float32(6)
    scales = nearest_codes(unencoded, e4m3)
    decoded = e4m3[scales]
    # Zero encoded scales return zero-initialized codes in the native codec.
    scaled = (x * d) / np.where(decoded == 0, np.float32(1), decoded)[:, None]
    magnitudes = nearest_codes(np.abs(scaled), e2m1)
    codes = magnitudes | (np.signbit(scaled).astype(np.uint8) << 3)
    codes[decoded == 0] = 0
    reconstruction = (np.where(codes & 8, -e2m1[codes & 7], e2m1[codes & 7]) *
                      decoded[:, None]) / d
    words = np.zeros((len(x), 2), dtype=np.uint32)
    for i in range(16):
        words[:, i // 8] |= codes[:, i].astype(np.uint32) << np.uint32(4 * (i % 8))
    return reconstruction, scales, words, unencoded


def sample_coordinates(rows: int, blocks: int) -> list[tuple[int, int]]:
    """Reproduce the native stratification and bounded reservoir, including slot order."""
    mask = (1 << 64) - 1

    def mix(x):
        x = (x + 0x9e3779b97f4a7c15) & mask
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)

    count = min(SAMPLING["blocks_per_call"], blocks)
    reservoir = []
    seen = 0
    for call in range(rows):
        for slot in range(count):
            begin, end = blocks * slot // count, blocks * (slot + 1) // count
            block = begin + mix(call * SAMPLING["blocks_per_call"] + slot) % (end - begin)
            seen += 1
            if seen <= SAMPLING["capacity"]:
                reservoir.append((call, block))
            else:
                replacement = mix(seen) % seen
                if replacement < SAMPLING["capacity"]:
                    reservoir[replacement] = (call, block)
    return reservoir


def sampled_blocks(raw: dict, config: dict) -> dict:
    """Validate real bounded BF16 blocks and match retained native codec probes exactly."""
    import numpy as np

    if raw.get("schema") != SAMPLED_SCHEMA or raw.get("sampling") != SAMPLING:
        raise ValueError("MSE calibration requires actual v2 K16 samples; absmax cannot supply percentiles/errors")
    validate_statistics(raw, config)
    if any(config[key] % 16 for key in ("hidden_size", "intermediate_size")):
        raise ValueError("K16 collection requires widths divisible by 16")
    result = {}
    for name, stat in raw["statistics"].items():
        count = min(SAMPLING["blocks_per_call"], raw["context"] * stat["width"] // 16)
        seen = raw["rows"] * count
        samples = stat.get("samples", [])
        if (not isinstance(samples, list) or any(not isinstance(s, dict) for s in samples) or
                stat.get("samples_seen") != seen or len(samples) != min(seen, SAMPLING["capacity"])):
            raise ValueError(f"{name}: incomplete bounded K16 sample coverage")
        if [(s.get("call"), s.get("block")) for s in samples] != sample_coordinates(
                raw["rows"], raw["context"] * stat["width"] // 16):
            raise ValueError(f"{name}: K16 coordinates differ from recorded sampling scheme")
        coordinates = set()
        for sample in samples:
            call, block = sample.get("call"), sample.get("block")
            if (type(call) is not int or not 0 <= call < raw["rows"] or type(block) is not int or
                    not 0 <= block < raw["context"] * stat["width"] // 16 or
                    (call, block) in coordinates or len(sample.get("values", [])) != 16):
                raise ValueError(f"{name}: invalid K16 sample coordinates/width")
            coordinates.add((call, block))
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in sample["values"]):
                raise ValueError(f"{name}: nonfinite/non-numeric K16 sample")
        blocks = np.asarray([s["values"] for s in samples], dtype=np.float32)
        bits = blocks.view(np.uint32)
        if (not np.all(np.isfinite(blocks)) or np.any(bits & 0xffff) or
                float(np.max(np.abs(blocks))) > stat["absmax"]):
            raise ValueError(f"{name}: samples must be exact BF16 values within observed absmax")
        for p, divisor in enumerate(SAMPLING["probe_divisors"]):
            _, scales, words, _ = quantize_blocks(blocks, divisor)
            expected = [{"scale": int(scale), "codes": [int(word) for word in pair]}
                        for scale, pair in zip(scales, words)]
            if any(not isinstance(s.get("probes"), list) or len(s["probes"]) != len(SAMPLING["probe_divisors"]) or
                   s["probes"][p] != probe for s, probe in zip(samples, expected)):
                raise ValueError(f"{name}: native consumer-codec probe mismatch")
        result[name] = blocks
    return result


def calibrate(raw: dict, config: dict, rule: str) -> tuple[dict, dict]:
    divisors = validate_statistics(raw, config)
    samples = sampled_blocks(raw, config) if raw.get("schema") == SAMPLED_SCHEMA else None
    if rule == "absmax":
        return divisors, {"rule": "absmax", "formula": "FP32(448*6/absmax); no clipping search",
                          "native_probe_match": samples is not None}
    if rule != "mse":
        raise ValueError("unknown activation scale rule")
    import numpy as np

    if samples is None:
        raise ValueError("MSE calibration requires actual v2 K16 samples; absmax cannot supply percentiles/errors")
    fits = {}
    for name, blocks in samples.items():
        layer, role = name.split("/")[2], name.rsplit("/", 1)[1]
        logical = f"text/layers/{layer}/" + ("ffn_input" if role == "gate_up" else "mlp/product")
        baseline = divisors[logical]
        energy = float(np.sum(blocks.astype(np.float64) ** 2))
        if not energy > 0:
            raise ValueError(f"{name}: sampled energy is zero; cannot fit a reconstruction scale")
        grid = []
        for step in SEARCH_STEPS:
            divisor = float(np.float32(baseline) * np.float32(2.0 ** (step / 16)))
            if not math.isfinite(divisor) or divisor <= 0:
                raise ValueError(f"{name}: search divisor is not finite positive FP32")
            reconstructed, scales, _, unencoded = quantize_blocks(blocks, divisor)
            error = reconstructed.astype(np.float64) - blocks
            nmse = float(np.sum(error * error) / energy)
            if not math.isfinite(nmse):
                raise ValueError(f"{name}: nonfinite reconstruction error")
            grid.append({"step": step, "divisor": divisor, "nmse": nmse,
                         "saturated_blocks": int(np.count_nonzero(unencoded > 448)),
                         "zero_scale_blocks": int(np.count_nonzero(scales == 0))})
        # Exact ties prefer the baseline, then the closest phase/range perturbation.
        best = min(grid, key=lambda item: (item["nmse"], abs(item["step"]), item["step"]))
        divisors[logical] = best["divisor"]
        fits[logical] = {"blocks": len(blocks), "energy": energy, "selected": best,
                         "baseline": next(item for item in grid if item["step"] == 0), "grid": grid}
    return divisors, {"rule": "mse", "objective": "sum((x-E2M1(x*d/s)*s/d)^2)/sum(x^2)",
                      "consumer": "nvfp4_codec.cuh FP32 mul/div; s=E4M3(d*K16_absmax/6); RN-even SATFINITE",
                      "search": "FP32(absmax_divisor)*FP32(2^(step/16)); step=-16..48",
                      "sampling": SAMPLING, "native_probe_match": True, "fits": fits,
                      "limits": "activation reconstruction only, not weight-aware/model quality; bounded A16 taps"}


def layer_selection(value: str, count: int) -> set[int]:
    if value == "all":
        return set(range(count))
    selected = set()
    try:
        for item in value.split(","):
            ends = item.split("-")
            if len(ends) not in (1, 2) or any(not end.isdecimal() for end in ends):
                raise ValueError
            begin, end = int(ends[0]), int(ends[-1])
            if not 0 <= begin <= end < count:
                raise ValueError
            selected.update(range(begin, end + 1))
    except (ValueError, AttributeError) as error:
        raise ValueError("--layers requires all or comma-separated zero-based layers/inclusive ranges") from error
    if not selected:
        raise ValueError("empty layer policy")
    return selected


def a4_uses(artifact: Artifact, config: dict, divisors: dict, *, mlp="full", layers=None) -> tuple[list, dict]:
    """Keep all non-NVFP4/protected uses unchanged, enforce native packed-parent semantics."""
    if mlp not in ("full", "gate-up", "down"):
        raise ValueError("MLP policy must be full, gate-up or down")
    layers = set(range(config["num_hidden_layers"])) if layers is None else set(layers)
    if not layers or any(type(layer) is not int or not 0 <= layer < config["num_hidden_layers"] for layer in layers):
        raise ValueError("invalid/empty layer policy")
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
        layer = int(name.split("/")[2])
        if layer not in layers or (mlp == "gate-up" and name.endswith("/down")) or (
                mlp == "down" and not name.endswith("/down")):
            continue
        aux_id = "calibration/a4/" + logical
        if aux_id in artifact.by_id:
            raise ValueError("source already contains A4 calibration objects")
        use["activation_policy"] = "AllowA4"
        use.setdefault("auxiliaries", {})["activation_input_divisor"] = {"object": aux_id}
        divisor = divisors[logical]
        try:
            represented = struct.unpack("<f", struct.pack("<f", divisor))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{logical}: divisor is not representable in FP32") from error
        if type(divisor) not in (int, float) or not math.isfinite(divisor) or divisor <= 0 or represented != divisor:
            raise ValueError(f"{logical}: divisor must be finite positive exact FP32")
        selected[aux_id] = divisor
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
    selected_parents = {}
    for use in uses:
        if use["activation_policy"] == "AllowA4":
            binding = artifact.directory.bindings[use["parameter"]]
            for part in binding.get("parts", [{"object": binding.get("object")}]):
                selected_parents[part["object"]] = (use["input"], use["auxiliaries"]["activation_input_divisor"])
    for use in uses:
        binding = artifact.directory.bindings[use["parameter"]]
        for part in binding.get("parts", [{"object": binding.get("object")}]):
            if part["object"] in parent_inputs and use["input"] != parent_inputs[part["object"]]:
                raise ValueError("A4 parent also serves an unrelated logical input")
            expected = selected_parents.get(part["object"])
            if expected is not None and (use["activation_policy"] != "AllowA4" or
                    (use["input"], use.get("auxiliaries", {}).get("activation_input_divisor")) != expected):
                raise ValueError("A4 packed parent also serves a protected/unselected use")
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
    if raw.get("schema") == SAMPLED_SCHEMA:
        sampled_blocks(raw, config)
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
        "scale_rule": "absmax baseline retained; optional native-probe-checked K16 reconstruction MSE search",
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
        rule = args.scale_rule
        mlp = args.mlp
        layers = layer_selection(args.layers, config["num_hidden_layers"])
        divisors, fit = calibrate(raw, config, rule)
        uses, auxiliaries = a4_uses(source, config, divisors, mlp=mlp, layers=layers)
        policy = {"mlp": mlp, "layers": sorted(layers)}
        specs = [TensorSpec(o.id, o.shape, o.format, o.layout) if o.kind == "tensor"
                 else ResourceSpec(o.id, o.bytes, o.encoding) for o in source.objects]
        specs += [TensorSpec(name, (), "fp32", "contiguous_le_v1") for name in auxiliaries]
        provenance = deepcopy(source.directory.provenance)
        provenance["gemma4_a4_mlp"] = {"receipt": identity(args.receipt), "source": before,
                                      "calibration": receipt["calibration"], "boundaries": BOUNDARIES,
                                      "scale_rule": fit, "policy": policy, "handoff": identity(Path(__file__))}
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
            "divisors": auxiliaries, "policy": policy, "scale_rule": fit,
            "qualification": "not measured; quality gate required before any speed claim",
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
    p.add_argument("--scale-rule", choices=("absmax", "mse"), default="absmax",
                   help="mse requires new real K16 samples and exact native codec-probe agreement")
    p.add_argument("--mlp", choices=("full", "gate-up", "down"), default="full",
                   help="only native NVFP4 projections; gate/up always share one policy and divisor")
    p.add_argument("--layers", default="all", help="all or zero-based comma-separated layers/inclusive ranges")
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
