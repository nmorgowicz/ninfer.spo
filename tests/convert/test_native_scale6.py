"""Native q4_g32_s6_fp16: the two-level scale plane decodes bit-identically to the emulated g32 words."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import decode_row_split_codes, dequantize_row_split, encode_row_split
from tools.artifact.codecs.scale_s6 import decode_s6_scales, pack_s6_scales
from tools.artifact.layouts import encoded_size
from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert import gemma4_q4_cli as cli
from tools.convert.gemma4_q4 import Q4Options, build_recipe
from tools.convert.methods import grouped_absmax, grouped_absmax_emulated, grouped_mse_search
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import Q4_A8_FORMATS
from tools.convert.quantization.groupwise import (
    QuantizedMatrix,
    quantize_q4_scale_search,
    to_native_scale6,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

from .test_gemma4_q4 import allow_a8, build, gemma_model, projections, selection

S6 = "q4_g32_s6_fp16"
GOLDEN_BITS = (0x0000, 0x3555, 0x3955, 0x3C00, 0x4D40, 0x4D2A, 0x492A, 0x45AA)


def numpy_decode_bits(plane: np.ndarray) -> np.ndarray:
    """Independent decode of S6 blocks: binary32 product of S and m, then one rounding to binary16."""
    rows, size = plane.shape
    out = np.zeros((rows, size), dtype=np.uint16)
    for row in range(rows):
        for block in range(size // 8):
            word = int.from_bytes(bytes(plane[row, block * 8 : block * 8 + 8].tolist()), "little")
            super_scale = np.array([word & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
            for j in range(8):
                multiplier = (word >> (16 + 6 * j)) & 63
                product = np.float32(super_scale) * np.float32(multiplier)
                out[row, block * 8 + j] = np.array([product], dtype=np.float32).astype(np.float16).view(np.uint16)[0]
    return out


def _words(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64).to(torch.int16).view(torch.float16)


def _values(k: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9)
    values = torch.randn(7, k, generator=generator) * 0.03
    values[0] = 0.0  # all-zero row
    values[1] *= 1e-6  # scales underflow to the smallest binary16 subnormal
    values[2] *= 2e4  # large scales
    values[3, 10] = 40.0  # outlier
    values[4, : k // 2] = 0.0  # all-zero 256-column blocks next to live ones
    return values


@pytest.mark.parametrize("k", [256, 512, 768])
@pytest.mark.parametrize("weighted", [False, True])
def test_native_decodes_bit_identically_to_the_emulated_g32_words(k, weighted):
    values = _values(k)
    importance = torch.rand(k) if weighted else None
    emulated = quantize_q4_scale_search(values, 32, importance, device="cpu", scale_bits=6)
    native = to_native_scale6(emulated)
    assert native.scales.dtype == torch.uint8 and tuple(native.scales.shape) == (7, k // 32)
    assert native.codes is emulated.codes
    want = emulated.scales.view(torch.int16).numpy().astype(np.uint16)
    assert np.array_equal(numpy_decode_bits(native.scales.numpy()), want)
    assert torch.equal(decode_s6_scales(native.scales).view(torch.int16), emulated.scales.view(torch.int16))
    shape = tuple(values.shape)
    got = dequantize_row_split(encode_row_split(native.codes, native.scales, S6, shape), S6, shape, dtype=torch.float32)
    ref = dequantize_row_split(
        encode_row_split(emulated.codes, emulated.scales, "q4_g32_fp16", shape), "q4_g32_fp16", shape, dtype=torch.float32
    )
    assert torch.equal(got, ref)


def test_native_handles_padded_k_with_zero_padding_groups():
    values = _values(500)  # K padded to 512
    emulated = quantize_q4_scale_search(values, 32, None, device="cpu", scale_bits=6)
    native = to_native_scale6(emulated)
    shape = tuple(values.shape)
    assert tuple(native.codes.shape) == (7, 16, 32)
    assert not bool(native.codes[:, 15, 20:].any())  # columns >= 500 are zero codes
    got = dequantize_row_split(encode_row_split(native.codes, native.scales, S6, shape), S6, shape, dtype=torch.float32)
    ref = dequantize_row_split(
        encode_row_split(emulated.codes, emulated.scales, "q4_g32_fp16", shape), "q4_g32_fp16", shape, dtype=torch.float32
    )
    assert got.shape == (7, 500) and torch.equal(got, ref)


def test_golden_block_through_the_converter_and_a_zero_block():
    words = _words(list(GOLDEN_BITS) + [0] * 8)
    matrix = QuantizedMatrix(codes=torch.zeros(1, 16, 32, dtype=torch.int8), scales=words.reshape(1, 16))
    plane = to_native_scale6(matrix).scales
    assert plane[0, 8:].tolist() == [0] * 8  # an all-zero block is S = 0, m = 0
    decoded = decode_s6_scales(plane).view(torch.int16).reshape(-1).tolist()
    assert decoded == words.view(torch.int16).reshape(-1).tolist()
    # The golden words are reproduced by the block that is spelled out in the format document.
    block = torch.tensor([[0x55, 0x35, 0x40, 0x20, 0x0C, 0xBF, 0xFF, 0x45]], dtype=torch.uint8)
    golden = decode_s6_scales(block).view(torch.uint8).reshape(-1)
    assert bytes(golden.tolist()) == b"".join(int(w).to_bytes(2, "little") for w in GOLDEN_BITS)


def test_pack_recovers_adversarial_blocks_exactly():
    rng = np.random.default_rng(4)
    count = 6000
    super_bits = np.concatenate(
        (
            rng.integers(0x0001, 0x7C00, size=count // 2),  # any finite positive binary16
            rng.integers(0x0001, 0x0400, size=count // 2),  # subnormal super scales
        )
    ).astype(np.uint16)
    multipliers = rng.integers(0, 64, size=(count, 8))
    multipliers[::7] = 0  # blocks with an all-zero multiplier set (and zero words)
    multipliers[1::7, :] = 63
    multipliers[2::7, 1:] = 0  # a single live group
    multipliers[3::7] = 1
    for column in range(8):
        multipliers[4 + column :: 29, column] = 0  # a zero group inside a live block
    with np.errstate(over="ignore"):
        product = super_bits.view(np.float16).astype(np.float32)[:, None] * multipliers.astype(np.float32)
        words = product.astype(np.float16)
    finite = np.isfinite(words).all(axis=1)
    words = words[finite]
    assert words.shape[0] > count // 2
    tensor = torch.from_numpy(np.ascontiguousarray(words))
    plane = pack_s6_scales(tensor)
    assert np.array_equal(numpy_decode_bits(plane.numpy()), tensor.view(torch.int16).numpy().astype(np.uint16))
    # A super scale of 65504 admits only m = 1; S subnormal covers m up to 63.
    edge = _words([0x7BFF, 0, 0, 0, 0, 0, 0, 0, 0x0001, 0x0003, 0x003F, 0x0001, 0, 0x0020, 0, 0x0002])
    plane = pack_s6_scales(edge.reshape(1, 16))
    assert torch.equal(decode_s6_scales(plane).view(torch.int16), edge.view(torch.int16).reshape(1, 16))


def test_pack_refuses_what_it_cannot_reproduce_exactly():
    # 1.0 and 1.0009765625 (one binary16 ulp apart) cannot both be a 6-bit multiple of one super scale.
    close = _words([0x3C00, 0x3C01, 0, 0, 0, 0, 0, 0]).reshape(1, 8)
    with pytest.raises(ValueError, match="no binary16 super"):
        pack_s6_scales(close)
    for bad in ([0x8000] + [0] * 7, [0xBC00] + [0] * 7, [0x7C00] + [0] * 7, [0x7E00] + [0] * 7):
        with pytest.raises(ValueError, match="finite and non-negative"):
            pack_s6_scales(_words(bad).reshape(1, 8))
    with pytest.raises(TypeError):
        pack_s6_scales(torch.zeros(1, 12, dtype=torch.float16))
    matrix = QuantizedMatrix(codes=torch.zeros(1, 12, 32, dtype=torch.int8), scales=torch.zeros(1, 12, dtype=torch.float16))
    with pytest.raises(ValueError, match="cannot form S6 blocks"):
        to_native_scale6(matrix)


# --- recipes, artifacts and receipts ----------------------------------------------------------------


def _model(k: int, rows: int = 4) -> Model:
    model = Model({"text": {"config": {}}})
    generator = torch.Generator().manual_seed(2)
    values = (torch.randn(rows, k, generator=generator) * 0.02).to(torch.bfloat16)
    model.add(Parameter("w", (rows, k), array_source(values, "w"), inputs=("input",)))
    model.packing_groups = []
    return model


def _written_object(tmp_path, format_name, k, parameters):
    model = _model(k)
    recipe = Recipe(model)
    recipe.assign("w", format=format_name, method=grouped_mse_search, parameters=parameters)
    prepared = recipe.prepare(device="cpu", rows_per_chunk=3)
    path = tmp_path / f"{format_name}.ninfer"
    specs = [job.spec for job in prepared.weights]
    with ArtifactWriter(
        path, specs, components=model.components, bindings=prepared.bindings, uses=prepared.uses
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    with Artifact(path) as artifact:
        (obj,) = [o for o in artifact.objects if o.format == format_name]
        payload = artifact.read_object(obj.id)
    return payload, obj.shape


@pytest.mark.parametrize("k", [512, 500, 200])
def test_written_artifact_decodes_like_the_emulated_artifact_and_is_smaller(tmp_path, k):
    parameters = {"group_size": 32, "weighted": False, "scale_bits": 6}
    native, shape = _written_object(tmp_path, S6, k, parameters)
    emulated, _ = _written_object(tmp_path, "q4_g32_fp16", k, parameters)
    assert len(native) == encoded_size("row_split_k128_v1", S6, shape)
    assert len(emulated) - len(native) == shape[0] * (-(-k // 128) * 128) // 32
    got = dequantize_row_split(native, S6, shape, dtype=torch.float32)
    want = dequantize_row_split(emulated, "q4_g32_fp16", shape, dtype=torch.float32)
    assert torch.equal(got, want)
    # Complete physical padding groups (K = 200 pads to 256: group 7) carry scale 0, as the layout requires.
    scales, codes = decode_row_split_codes(native, S6, shape)
    logical_groups = -(-k // 32)
    assert not bool(scales[:, logical_groups:].view(torch.int16).any())
    assert not bool(codes[:, logical_groups:].any())


def test_methods_accept_s6_only_with_group_32_and_scale_bits_6():
    parameters = {"group_size": 32, "weighted": False, "scale_bits": 6}
    recipe = Recipe(_model(512))
    recipe.assign("w", format=S6, method=grouped_mse_search, parameters=parameters)
    assert recipe.prepare(device="cpu").weights[0].spec.format == S6
    for bad in (
        {"group_size": 32},
        {"group_size": 32, "scale_bits": 8},
        {"group_size": 128, "scale_bits": 6},
        {"group_size": 256, "scale_bits": 6},
    ):
        recipe = Recipe(_model(512))
        recipe.assign("w", format=S6, method=grouped_mse_search, parameters=bad)
        with pytest.raises(ValueError, match="scale_bits 6"):
            recipe.prepare(device="cpu")
    recipe = Recipe(_model(512))
    recipe.assign("w", format=S6, method=grouped_absmax)
    with pytest.raises(ValueError, match="grouped_mse_search with scale_bits 6"):
        recipe.prepare(device="cpu")
    recipe = Recipe(_model(512))
    recipe.assign("w", format=S6, method=grouped_absmax_emulated, parameters={"group_size": 32})
    with pytest.raises(ValueError, match="requires target format"):
        recipe.prepare(device="cpu")


def test_s6_target_refuses_k_that_is_not_a_multiple_of_256_at_prepare():
    recipe = Recipe(_model(640))
    recipe.assign(
        "w", format=S6, method=grouped_mse_search, parameters={"group_size": 32, "weighted": False, "scale_bits": 6}
    )
    with pytest.raises(ValueError, match="multiple of 256"):
        recipe.prepare(device="cpu")


def test_native_scale6_recipe_names_receipt_and_sizes():
    options = Q4Options(weights="q4_g32_a16", scale_search="mse", scale_bits=6, native_groups=True)
    assert options.native_scale6
    model, _, native = build(options)
    emulated_options = Q4Options(weights="q4_g32_a16", scale_search="mse", scale_bits=6)
    _, _, emulated = build(emulated_options)
    assert native.name == "q4_g32_mse_search_fp8_vocab_a16_native_scale6"
    assert emulated.name == "q4_g32_mse_search_fp8_vocab_a16_emulated_scale6"
    for name in projections(model):
        chosen = selection(native, name)
        assert chosen.format == S6 and chosen.method is grouped_mse_search
        assert chosen.parameters == {"group_size": 32, "weighted": False, "scale_bits": 6}
        assert selection(emulated, name).format == "q4_g32_fp16"
    assert native.scale_search["scale_bits"] == 6 and native.scale_search["scale_storage"].startswith(S6)
    assert "scale_storage" not in emulated.scale_search
    # The receipt reports the real size; the emulated build still only projects it.
    assert native.q4_groups["projected_real_weight_bytes"] == native.payload
    assert "real two-level" in native.q4_groups["projection_note"] and S6 in native.q4_groups["projection_note"]
    assert emulated.q4_groups["projection_note"].startswith("scale-plane bytes only")
    assert emulated.q4_groups["projected_real_weight_bytes"] < emulated.payload
    saved = sum(
        model.parameters[name].shape[0] * model.parameters[name].shape[1] // 32 for name in projections(model)
    )
    assert emulated.payload - native.payload == saved
    # The emulated projection (6 bits per 32 weights + a binary16 per 256) is exactly the real format's size.
    assert emulated.q4_groups["projected_real_weight_bytes"] == native.payload


def test_w4a8_and_a16_both_accept_native_scale6_and_the_a8_override_covers_it():
    assert S6 in Q4_A8_FORMATS
    for weights, a8 in (("q4_g32", True), ("q4_g32_a16", False)):
        options = Q4Options(weights=weights, scale_search="mse", scale_bits=6, native_groups=True)
        model, _, built = build(options)
        assert (allow_a8(built) == set(projections(model))) is a8
        assert built.name.endswith("_native_scale6")
        assert built.name.startswith(f"q4_g32_mse_search_fp8_vocab_{'a8' if a8 else 'a16'}")
    options = Q4Options(
        weights="q4_g32_a16", scale_search="weighted", scale_bits=6, native_groups=True, imatrix=Path("m.gguf")
    )
    assert options.native_scale6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale_bits": 5, "native_groups": True, "scale_search": "mse"}, "only for --scale-bits 6"),
        ({"scale_bits": 8, "native_groups": True, "scale_search": "mse"}, "only for --scale-bits 6"),
        ({"scale_bits": 6, "native_groups": True}, "--scale-search mse"),
        ({"scale_bits": 6, "native_groups": True, "scale_search": "mse", "group_size": 128}, "--group-size 32"),
        ({"scale_bits": 6, "native_groups": True, "scale_search": "mse", "region_groups": ("mlp_down=128",)}, "region-group"),
        ({"scale_bits": 3, "scale_search": "mse"}, "4..8"),
    ],
)
def test_options_refuse_incoherent_two_level_scale_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Q4Options(weights="q4_g32_a16", **kwargs)
    with pytest.raises(ValueError, match="scale-bits|apply only"):
        Q4Options(weights="fp8_row", scale_bits=6, native_groups=True)


def test_scale_bits_without_native_groups_stays_the_emulated_screening_path():
    for bits in (4, 6, 8):
        options = Q4Options(weights="q4_g32_a16", scale_search="mse", scale_bits=bits)
        assert not options.native_scale6
        model, _, built = build(options)
        assert built.name.endswith(f"_emulated_scale{bits}")
        assert {selection(built, name).format for name in projections(model)} == {"q4_g32_fp16"}
        assert built.scale_search == {"method": "mse", "scale_bits": bits}


def test_cli_accepts_native_scale6_and_refuses_the_rest(tmp_path):
    parser = cli.build_parser()
    model = tmp_path / "model"
    model.mkdir()
    base = ["convert", "--model", str(model), "--out", str(tmp_path / "out" / "g.ninfer"), "--no-mtp"]
    for weights in ("q4_g32", "q4_g32_a16"):
        args = parser.parse_args(
            base + ["--weights", weights, "--scale-search", "mse", "--scale-bits", "6", "--native-groups"]
        )
        options, _ = cli.check_convert_arguments(args)
        assert options.native_scale6 and options.scale_bits == 6 and options.weights == weights
    for extra, message in (
        (["--scale-bits", "5", "--native-groups", "--scale-search", "mse"], "only for --scale-bits 6"),
        (["--scale-bits", "6", "--native-groups"], "--scale-search mse"),
        (["--scale-bits", "6", "--native-groups", "--scale-search", "mse", "--group-size", "256"], "--group-size 32"),
    ):
        with pytest.raises(ValueError, match=message):
            cli.check_convert_arguments(parser.parse_args(base + extra))
