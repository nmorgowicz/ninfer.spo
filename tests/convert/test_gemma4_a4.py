"""Uncertain A4 boundaries: packed-parent inputs, identity, coverage and lossless handoff.

No model execution or fake activation collector: synthetic V3 objects exercise actual reader/writer
and converter behavior. Native numerical/GPU qualification is deliberately a separate gate.
"""
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert import gemma4_a4 as a4

CONFIG = {"model_type": "gemma4_text", "architectures": ["Gemma4ForCausalLM"],
          "num_hidden_layers": 1, "hidden_size": 128, "intermediate_size": 128}


def statistics():
    return {"schema": a4.NATIVE_SCHEMA, "rows": 2, "context": 4, "kv_dtype": "bf16", "qk": "bf16",
            "statistics": {f"text/layers/0/mlp/{role}":
                           {"width": 128, "tokens": 8, "calls": 2, "absmax": maximum}
                           for role, maximum in (("gate_up", 12.0), ("down", 21.0))}}


def source_artifact(path, *, shared_down=False, bad_slice=False, fp8_gate_up=False, reverse=False):
    specs = [TensorSpec("packed", (384 if shared_down else 256, 128),
                        "fp8_e4m3fn_row_bf16" if fp8_gate_up else "nvfp4",
                        "row_scale_v1" if fp8_gate_up else "block_scale_k16_m128x4_v1"),
             TensorSpec("head", (128, 128), "bf16", "contiguous_le_v1")]
    if not shared_down:
        specs.append(TensorSpec("down", (128, 128), "nvfp4", "block_scale_k16_m128x4_v1"))
    bindings = {f"text/layers/0/mlp/{role}": {"parts": [{"object": "packed",
                 "range": [i * 16384 + (1 if bad_slice and i == 0 else 0),
                           (i + 1) * 16384 + (1 if bad_slice and i == 0 else 0)]}]}
                for i, role in enumerate(("gate", "up"))}
    if reverse:
        bindings["text/layers/0/mlp/gate"], bindings["text/layers/0/mlp/up"] = (
            bindings["text/layers/0/mlp/up"], bindings["text/layers/0/mlp/gate"])
    bindings["text/layers/0/mlp/down"] = ({"parts": [{"object": "packed", "range": [32768, 49152]}]}
                                                 if shared_down else {"object": "down"})
    bindings["text/output_head"] = {"object": "head"}
    uses = [{"parameter": f"text/layers/0/mlp/{role}", "activation_policy": "A16Only",
             "input": "text/layers/0/mlp/product" if role == "down" else "text/layers/0/ffn_input"}
            for role in ("gate", "up", "down")]
    uses.append({"parameter": "text/output_head", "input": "text/final_norm", "activation_policy": "A16Only"})
    with ArtifactWriter(path, specs, components={"text": {"config": CONFIG}}, bindings=bindings, uses=uses) as writer:
        for index, obj in enumerate(writer.objects):
            writer.write_object(obj.id, bytes((i + index) % 256 for i in range(obj.bytes)))


def receipt(tmp_path, source):
    raw = statistics()
    stats_path, tokens_path = tmp_path / "statistics.json", tmp_path / "tokens.json"
    a4.save_new(stats_path, raw)
    a4.save_new(tokens_path, [[2, 105, 9, 10], [2, 105, 11, 12]])
    with Artifact(source) as artifact:
        source_id = a4.artifact_identity(artifact)
    value = {"schema": a4.SCHEMA, "boundaries": a4.BOUNDARIES, "source": source_id,
             "handoff": a4.identity(Path(a4.__file__)), "statistics": raw,
             "statistics_file": a4.identity(stats_path), "scale_rule": "FP32(448*6/absmax)",
             "calibration": {"tokens": a4.identity(tokens_path), "prefix_ids": [2, 105], "selected_rows": 2}}
    path = tmp_path / "receipt.json"
    a4.save_new(path, value)
    return path, value


def test_candidate_preserves_objects_and_shares_exact_gate_up_divisor(tmp_path):
    source, output = tmp_path / "a16.ninfer", tmp_path / "a4.ninfer"
    source_artifact(source)
    receipt_path, _ = receipt(tmp_path, source)
    original_hash = a4.sha256_file(source)
    a4.candidate(SimpleNamespace(artifact=source, output=output, receipt=receipt_path))
    with Artifact(source) as old, Artifact(output) as new:
        assert old.artifact_id != new.artifact_id
        assert old.directory.bindings == new.directory.bindings
        for obj in old.objects:
            assert old.read_object(obj.id) == new.read_object(obj.id)
        gate, up, down, head = new.directory.uses
        assert head == old.directory.uses[-1]
        gate_aux = gate["auxiliaries"]["activation_input_divisor"]
        assert gate_aux == up["auxiliaries"]["activation_input_divisor"]
        assert new.read_object(gate_aux["object"]) == struct.pack("<f", 224.0)
        assert new.read_object(down["auxiliaries"]["activation_input_divisor"]["object"]) == struct.pack("<f", 128.0)
        assert [u["activation_policy"] for u in new.directory.uses] == ["AllowA4"] * 3 + ["A16Only"]
    assert a4.sha256_file(source) == original_hash
    with pytest.raises(FileExistsError):
        a4.main(["candidate", "--artifact", str(source), "--receipt", str(receipt_path), "--output", str(output)])


@pytest.mark.parametrize("maximum", [0, -1, float("nan"), float("inf"), 1e-320])
def test_nonpositive_nonfinite_or_unrepresentable_scales_are_not_filled_in(maximum):
    raw = statistics()
    raw["statistics"]["text/layers/0/mlp/down"]["absmax"] = maximum
    with pytest.raises(ValueError):
        a4.validate_statistics(raw, CONFIG)


@pytest.mark.parametrize("field,value", [("width", 256), ("tokens", 7), ("calls", 1)])
def test_actual_boundary_coverage_must_match(field, value):
    raw = statistics()
    raw["statistics"]["text/layers/0/mlp/down"][field] = value
    with pytest.raises(ValueError, match="coverage"):
        a4.validate_statistics(raw, CONFIG)


def test_missing_boundary_is_not_substituted_by_other_boundary():
    raw = statistics()
    del raw["statistics"]["text/layers/0/mlp/down"]
    with pytest.raises(ValueError, match="exactly"):
        a4.validate_statistics(raw, CONFIG)


@pytest.mark.parametrize("options,message", [({"shared_down": True}, "different logical inputs"),
                                             ({"bad_slice": True}, "complete logical rows"),
                                             ({"reverse": True}, "gate-then-up order")])
def test_parent_semantics_reject_incompatible_packing(tmp_path, options, message):
    source = tmp_path / "a16.ninfer"
    source_artifact(source, **options)
    with Artifact(source) as artifact, pytest.raises(ValueError, match=message):
        a4.a4_uses(artifact, CONFIG, a4.validate_statistics(statistics(), CONFIG))


def test_same_native_id_with_changed_payload_is_not_same_calibration_identity(tmp_path):
    source = tmp_path / "a16.ninfer"
    source_artifact(source)
    with Artifact(source) as artifact:
        before = a4.artifact_identity(artifact)
        offset = artifact.payload_offset + artifact.objects[0].offset
    with source.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"\xff")
    with Artifact(source) as artifact:
        assert artifact.artifact_id.hex() == before["artifact_id"]
        with pytest.raises(ValueError, match="identity mismatch"):
            a4.require_identity(a4.artifact_identity(artifact), before)


@pytest.mark.parametrize("target", ["statistics_file", "tokens"])
def test_changed_retained_evidence_is_rejected(tmp_path, target):
    source = tmp_path / "a16.ninfer"
    source_artifact(source)
    _, value = receipt(tmp_path, source)
    record = value["statistics_file"] if target == "statistics_file" else value["calibration"]["tokens"]
    Path(record["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        a4.receipt_statistics(value)


def test_recorded_prefix_cannot_disagree_with_exact_tokens(tmp_path):
    source = tmp_path / "a16.ninfer"
    source_artifact(source)
    _, value = receipt(tmp_path, source)
    value["calibration"]["prefix_ids"] = [7]
    with pytest.raises(ValueError, match="prefix"):
        a4.receipt_statistics(value)


def test_fp8_sensitive_gate_up_permissions_are_not_changed(tmp_path):
    source = tmp_path / "mixed.ninfer"
    source_artifact(source, fp8_gate_up=True)
    with Artifact(source) as artifact:
        uses, auxiliaries = a4.a4_uses(artifact, CONFIG, a4.validate_statistics(statistics(), CONFIG))
        assert uses[:2] == list(artifact.directory.uses[:2])
        assert uses[2]["activation_policy"] == "AllowA4"
        assert set(auxiliaries) == {"calibration/a4/text/layers/0/mlp/product"}
