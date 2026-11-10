"""Uncertain A4 boundaries: packed-parent inputs, identity, coverage and lossless handoff.

No model execution or fake activation collector: synthetic V3 objects exercise actual reader/writer
and converter behavior. Native numerical/GPU qualification is deliberately a separate gate.
"""
from pathlib import Path
import struct
from copy import deepcopy
import json
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


def source_artifact(path, *, shared_down=False, bad_slice=False, fp8_gate_up=False, reverse=False,
                    layers=1, protected=False, protected_alias=False):
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
    for layer in range(1, layers):
        for role, shape in (("packed", (256, 128)), ("down", (128, 128))):
            specs.append(TensorSpec(f"{role}{layer}", shape, "nvfp4", "block_scale_k16_m128x4_v1"))
        for i, role in enumerate(("gate", "up", "down")):
            name = f"text/layers/{layer}/mlp/{role}"
            bindings[name] = ({"object": f"down{layer}"} if role == "down" else
                              {"parts": [{"object": f"packed{layer}", "range": [i * 16384, (i + 1) * 16384]}]})
            uses.append({"parameter": name, "activation_policy": "A16Only",
                         "input": f"text/layers/{layer}/" + ("mlp/product" if role == "down" else "ffn_input")})
    if protected:
        for name, logical in (("text/layers/0/attention/o", "text/layers/0/attention/output"),
                              ("mtp/layers/0/mlp/gate", "mtp/layers/0/ffn_input")):
            bindings[name] = {"object": "head"}
            uses.append({"parameter": name, "input": logical, "activation_policy": "A16Only"})
    if protected_alias:
        bindings["mtp/protected_alias"] = {"parts": [{"object": "packed", "range": [0, 16384]}]}
        uses.append({"parameter": "mtp/protected_alias", "input": "text/layers/0/ffn_input",
                     "activation_policy": "A16Only"})
    config = {**CONFIG, "num_hidden_layers": layers}
    with ArtifactWriter(path, specs, components={"text": {"config": config}}, bindings=bindings, uses=uses) as writer:
        for index, obj in enumerate(writer.objects):
            writer.write_object(obj.id, bytes((i + index) % 256 for i in range(obj.bytes)))


def receipt(tmp_path, source, raw=None):
    raw = statistics() if raw is None else raw
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
    a4.candidate(SimpleNamespace(artifact=source, output=output, receipt=receipt_path,
                                 scale_rule="absmax", mlp="full", layers="all"))
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


# Analytic codec vectors, not purported model activations or collected GPU evidence.
TIE_BLOCK = [6, -6, .25, -.25, .75, -.75, 1.25, -1.25,
             1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5, -5]
TIE_WORDS = [0xA2A280F7, 0xE6E6C4C4]


def sampled_statistics():
    raw = statistics()
    raw["schema"] = a4.SAMPLED_SCHEMA
    raw["sampling"] = deepcopy(a4.SAMPLING)
    for stat in raw["statistics"].values():
        coordinates = a4.sample_coordinates(raw["rows"], raw["context"] * stat["width"] // 16)
        stat["samples_seen"] = raw["rows"] * a4.SAMPLING["blocks_per_call"]
        stat["samples"] = [{"call": call, "block": block, "values": list(TIE_BLOCK),
                            "probes": [{"scale": scale, "codes": list(words)} for scale, words in (
                                (0x38, TIE_WORDS), (0x78, TIE_WORDS),
                                (0x3A, [0xA29180F7, 0xE6D5C4B3]))]}
                           for call, block in coordinates]
    return raw


def test_native_codec_rounding_ties_sign_nibble_order_and_power_of_two_invariance():
    import numpy as np

    for divisor, scale in ((1, 0x38), (256, 0x78)):
        reconstructed, scales, words, _ = a4.quantize_blocks([TIE_BLOCK], divisor)
        assert scales.tolist() == [scale]
        assert words.tolist() == [TIE_WORDS]
        np.testing.assert_array_equal(reconstructed, [[6, -6, 0, 0, 1, -1, 1, -1,
                                                      2, -2, 2, -2, 4, -4, 4, -4]])


def test_non_power_of_two_divisor_preserves_fp32_multiply_before_divide_rounding():
    # BF16 x=3.125, amax=7.5: RN(x*d)/1.5 rounds above2.5, but RN(x/1.5)*d
    # rounds to2.5. The real codec therefore selects3(code5), not2(code4).
    _, scales, words, _ = a4.quantize_blocks([[7.5, 3.125] + [0] * 14], 1.2000000476837158)
    assert scales.tolist() == [0x3C]
    assert words.tolist() == [[0x57, 0]]


def test_e4m3_rn_even_zero_scale_and_saturation_are_actual_range_effects():
    import numpy as np

    # E4M3 halfway: 1.0625 -> 1 (even code), 1.1875 -> 1.25 (even code).
    blocks = [[6.375] * 16, [7.125] * 16, [2 ** -10] * 16, [3072] * 16]
    reconstructed, scales, words, unencoded = a4.quantize_blocks(blocks, 1)
    assert scales.tolist() == [0x38, 0x3A, 0, 0x7E]
    assert words[2].tolist() == [0, 0]
    assert unencoded[-1] == 512
    np.testing.assert_array_equal(reconstructed[2], np.zeros(16))
    np.testing.assert_array_equal(reconstructed[3], np.full(16, 2688))


def test_error_search_uses_real_blocks_and_retains_complete_metrics():
    raw = sampled_statistics()
    divisors, fit = a4.calibrate(raw, CONFIG, "mse")
    assert fit["native_probe_match"] is True
    for logical, stat in fit["fits"].items():
        assert stat["blocks"] == 16
        assert stat["selected"]["nmse"] <= stat["baseline"]["nmse"]
        assert stat["selected"]["divisor"] == divisors[logical]
        assert [point["step"] for point in stat["grid"]] == list(a4.SEARCH_STEPS)
        assert all(set(point) == {"step", "divisor", "nmse", "saturated_blocks", "zero_scale_blocks"}
                   for point in stat["grid"])
    with pytest.raises(ValueError, match="actual v2"):
        a4.calibrate(statistics(), CONFIG, "mse")


@pytest.mark.parametrize("kind", ["missing", "count", "coordinates", "non-bf16", "outside-absmax",
                                 "nonfinite", "probe", "probe-divisors", "sampling-method"])
def test_malformed_samples_or_consumer_probes_cannot_produce_scales(kind):
    raw = sampled_statistics()
    stat = raw["statistics"]["text/layers/0/mlp/down"]
    if kind == "missing":
        stat["samples"].pop()
    elif kind == "count":
        stat["samples_seen"] -= 1
    elif kind == "coordinates":
        stat["samples"][0]["block"] = stat["samples"][1]["block"]
    elif kind == "non-bf16":
        stat["samples"][0]["values"][0] = 1.001
    elif kind == "outside-absmax":
        stat["samples"][0]["values"][0] = 32
    elif kind == "nonfinite":
        stat["samples"][0]["values"][0] = float("nan")
    elif kind == "probe":
        stat["samples"][0]["probes"][0]["codes"][0] ^= 1
    elif kind == "probe-divisors":
        raw["sampling"]["probe_divisors"] = [1, 128]
    else:
        raw["sampling"]["method"] = "fabricated-percentiles"
    with pytest.raises(ValueError):
        a4.calibrate(raw, CONFIG, "mse")


def test_reservoir_is_bounded_unique_and_covers_late_collection_rows():
    coordinates = a4.sample_coordinates(400, 512 * 5120 // 16)
    assert len(coordinates) == a4.SAMPLING["capacity"]
    assert len(set(coordinates)) == len(coordinates)
    assert min(call for call, _ in coordinates) < 20
    assert max(call for call, _ in coordinates) > 380
    assert coordinates == a4.sample_coordinates(400, 512 * 5120 // 16)


@pytest.mark.parametrize("mlp,policies", [
    ("gate-up", ["AllowA4", "AllowA4", "A16Only", "A16Only"]),
    ("down", ["A16Only", "A16Only", "AllowA4", "A16Only"]),
    ("full", ["AllowA4", "AllowA4", "AllowA4", "A16Only"]),
])
def test_mse_selective_candidates_are_consumable_and_encoded_copies(tmp_path, mlp, policies):
    source, output = tmp_path / "source.ninfer", tmp_path / f"{mlp}.ninfer"
    source_artifact(source, protected=True)
    receipt_path, _ = receipt(tmp_path, source, sampled_statistics())
    a4.candidate(SimpleNamespace(artifact=source, output=output, receipt=receipt_path,
                                 scale_rule="mse", mlp=mlp, layers="0"))
    report = json.loads(output.with_suffix(".ninfer.a4.json").read_text(encoding="utf-8"))
    assert report["policy"] == {"mlp": mlp, "layers": [0]}
    assert report["scale_rule"]["native_probe_match"] is True
    with Artifact(source) as old, Artifact(output) as new:
        assert [u["activation_policy"] for u in new.directory.uses[:4]] == policies
        assert new.directory.uses[-2:] == old.directory.uses[-2:]  # Attention and MTP.
        assert old.directory.bindings == new.directory.bindings
        for obj in old.objects:
            assert old.read_object(obj.id) == new.read_object(obj.id)
        if mlp != "down":
            assert new.directory.uses[0]["auxiliaries"] == new.directory.uses[1]["auxiliaries"]
        for use in new.directory.uses:
            if use["activation_policy"] == "AllowA4":
                name = use["auxiliaries"]["activation_input_divisor"]["object"]
                assert new.read_object(name) == struct.pack("<f", report["divisors"][name])
        assert new.directory.provenance["gemma4_a4_mlp"]["policy"] == report["policy"]


def test_layer_policy_preserves_unselected_layer_and_sensitive_projections(tmp_path):
    source = tmp_path / "layers.ninfer"
    source_artifact(source, layers=2, protected=True)
    config = {**CONFIG, "num_hidden_layers": 2}
    divisors = {f"text/layers/{layer}/{logical}": 128.0 for layer in range(2)
                for logical in ("ffn_input", "mlp/product")}
    with Artifact(source) as artifact:
        uses, _ = a4.a4_uses(artifact, config, divisors, mlp="gate-up", layers={1})
        for before, after in zip(artifact.directory.uses, uses):
            if before["parameter"] in ("text/layers/1/mlp/gate", "text/layers/1/mlp/up"):
                assert after["activation_policy"] == "AllowA4"
            else:
                assert after == before
    assert a4.layer_selection("0,2-4", 5) == {0, 2, 3, 4}


@pytest.mark.parametrize("layers", ["", "-1", "1-0", "0-2", "0,,1", "mtp", "all,0"])
def test_invalid_layer_combinations_reject(layers):
    with pytest.raises(ValueError, match="layers"):
        a4.layer_selection(layers, 2)


def test_selected_non_nvfp4_group_is_not_silently_replaced_by_other_group(tmp_path):
    source = tmp_path / "mixed.ninfer"
    source_artifact(source, fp8_gate_up=True)
    with Artifact(source) as artifact, pytest.raises(ValueError, match="no candidate"):
        a4.a4_uses(artifact, CONFIG, a4.validate_statistics(statistics(), CONFIG), mlp="gate-up")


@pytest.mark.parametrize("rule", ["mse", "absmax"])
def test_native_probe_failure_precedes_candidate_publication(tmp_path, rule):
    source, output = tmp_path / "source.ninfer", tmp_path / "bad.ninfer"
    source_artifact(source)
    raw = sampled_statistics()
    raw["statistics"]["text/layers/0/mlp/down"]["samples"][0]["probes"][0]["scale"] ^= 1
    receipt_path, _ = receipt(tmp_path, source, raw)
    with pytest.raises(ValueError, match="probe mismatch"):
        a4.candidate(SimpleNamespace(artifact=source, output=output, receipt=receipt_path,
                                     scale_rule=rule, mlp="full", layers="all"))
    assert not output.exists()


@pytest.mark.parametrize("divisor", [0.0, -1.0, float("inf"), float("nan"), 1e40, 1.00000000001])
def test_invalid_runtime_fp32_divisors_are_rejected(tmp_path, divisor):
    source = tmp_path / "source.ninfer"
    source_artifact(source)
    divisors = a4.validate_statistics(statistics(), CONFIG)
    divisors["text/layers/0/ffn_input"] = divisor
    with Artifact(source) as artifact, pytest.raises(ValueError, match="FP32"):
        a4.a4_uses(artifact, CONFIG, divisors)


def test_protected_alias_cannot_inherit_a4_even_when_logical_input_matches(tmp_path):
    source = tmp_path / "source.ninfer"
    source_artifact(source, protected_alias=True)
    with Artifact(source) as artifact, pytest.raises(ValueError, match="protected/unselected"):
        a4.a4_uses(artifact, CONFIG, a4.validate_statistics(statistics(), CONFIG))
