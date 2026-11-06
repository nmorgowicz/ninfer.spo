from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    block_scale_geometry,
)
from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "bf16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "fp32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "int32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "bf16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


@pytest.mark.parametrize(
    (
        "format_name",
        "k",
        "group_size",
        "k_pad",
        "scale_offset",
        "prefix",
        "base_prefix",
        "high_prefix",
    ),
    [
        pytest.param(
            "q4_g64_fp16",
            65,
            64,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4",
        ),
        pytest.param(
            "q5_g64_fp16",
            130,
            64,
            256,
            512,
            (-16, -15, -1, 0, 1, 15),
            b"\x10\x0f\xf1",
            b"\x07",
            id="q5",
        ),
        pytest.param(
            "q6_g64_fp16",
            65,
            64,
            128,
            512,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
            id="q6",
        ),
        pytest.param(
            "q4_g32_fp16",
            33,
            32,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4-g32",
        ),
        pytest.param(
            "q4_g128_fp16",
            129,
            128,
            256,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4-g128",
        ),
        pytest.param(
            "q4_g256_fp16",
            512,
            256,
            512,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4-g256",
        ),
        pytest.param(
            "q8_g32_fp16",
            33,
            32,
            128,
            256,
            (-127, -1, 0, 1, 127),
            b"\x81\xff\x00\x01\x7f",
            b"",
            id="q8",
        ),
    ],
)
def test_row_split_matches_known_packed_bytes(
    format_name, k, group_size, k_pad, scale_offset, prefix, base_prefix, high_prefix
):
    groups = k_pad // group_size
    codes = torch.zeros((1, groups, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, groups), dtype=torch.float16)
    scales[0, :2] = torch.tensor([1.5, 0.25], dtype=torch.float16)

    # Literal wire positions and words come from the format, independently of layout helpers.
    expected = bytearray(scale_offset + groups * 2)
    expected[: len(base_prefix)] = base_prefix
    expected[256 : 256 + len(high_prefix)] = high_prefix
    expected[scale_offset:] = struct.pack(
        "<" + "H" * groups, 0x3E00, 0x3400, *([0] * (groups - 2))
    )
    assert encode_row_split(codes, scales, format_name, (1, k)) == expected
    decoded_scales, decoded_codes = decode_row_split_codes(
        expected, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)


def test_nvfp4_known_vector_geometry_swizzle_tail_and_round_trip():
    shape = (128, 64)
    geometry = block_scale_geometry("nvfp4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.payload_bytes,
    ) == (4096, 4096, 512, 4608, 4612)

    packed = (
        torch.arange(geometry.code_plane_bytes, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(128, 32)
    )
    packed[0, 0] = 0x10
    scales = (
        torch.arange(128 * 4, dtype=torch.int64)
        .remainder(0x7F)
        .to(torch.uint8)
        .reshape(128, 4)
    )
    divisor = struct.pack("<f", 2.5)
    payload = encode_nvfp4(packed, scales, divisor, shape)

    assert len(payload) == 4612
    assert payload[0] == 0x10  # low nibble is K=0; high nibble is K=1.
    for row, lane in ((0, 0), (31, 3), (32, 0), (127, 3)):
        offset = geometry.scale_plane_offset + (row % 32) * 16 + (row // 32) * 4 + lane
        assert payload[offset] == int(scales[row, lane])
    assert payload[geometry.weight_divisor_offset :] == divisor

    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(payload, shape)
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) == divisor


@pytest.mark.parametrize("k", [100, 128, 384, 640])
def test_g256_refuses_k_whose_padded_width_is_not_a_multiple_of_256(k):
    from tools.artifact.layouts import row_split_geometry

    with pytest.raises(ValueError, match="multiple of its group size"):
        row_split_geometry("q4_g256_fp16", (4, k))
    assert row_split_geometry("q4_g128_fp16", (4, k)).groups_per_row == -(-k // 128)


# --- q4_g32_s6_fp16: two-level scale plane -----------------------------------------------------------

GOLDEN_BLOCK = bytes([0x55, 0x35, 0x40, 0x20, 0x0C, 0xBF, 0xFF, 0x45])  # S = 0x3555, m = 0,1,2,3,63,62,31,17
GOLDEN_SCALE_BITS = (0x0000, 0x3555, 0x3955, 0x3C00, 0x4D40, 0x4D2A, 0x492A, 0x45AA)


def _golden_plane(rows: int = 1) -> torch.Tensor:
    return torch.frombuffer(bytearray(GOLDEN_BLOCK * rows), dtype=torch.uint8).reshape(rows, 8)


def test_s6_golden_block_decodes_to_the_pinned_binary16_scales():
    from tools.artifact.codecs.scale_s6 import decode_s6_scales, pack_s6_scales

    decoded = decode_s6_scales(_golden_plane())
    assert decoded.dtype == torch.float16
    assert decoded.view(torch.int16).reshape(-1).tolist() == [
        _signed_word(word, 16) for word in GOLDEN_SCALE_BITS
    ]
    # j = 3: 3 * S = 0.99975 rounds to exactly 1.0 in binary16 (the single rounding of the rule).
    assert float(decoded[0, 3]) == 1.0
    # The bit fields are as specified: S = W & 0xffff, m_j = (W >> (16 + 6 j)) & 63.
    word = int.from_bytes(GOLDEN_BLOCK, "little")
    assert word & 0xFFFF == 0x3555
    assert [(word >> (16 + 6 * j)) & 63 for j in range(8)] == [0, 1, 2, 3, 63, 62, 31, 17]
    # Packing the decoded words recovers a block that decodes to the same words.
    repacked = pack_s6_scales(decoded)
    assert torch.equal(decode_s6_scales(repacked).view(torch.int16), decoded.view(torch.int16))


def test_s6_row_split_wire_positions_and_round_trip():
    from tools.artifact.codecs.scale_s6 import decode_s6_scales

    rows, k = 2, 256
    codes = torch.zeros((rows, 8, 32), dtype=torch.int8)
    codes[0, 0, :6] = torch.tensor((-8, -7, -1, 0, 1, 7), dtype=torch.int8)
    other = bytes([0x00, 0x3C, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])  # S = 1.0, m_0 = 1, the rest 0
    plane = torch.cat((_golden_plane(), torch.frombuffer(bytearray(other), dtype=torch.uint8).reshape(1, 8)))
    payload = encode_row_split(codes, plane, "q4_g32_s6_fp16", (rows, k))
    # Code plane 2 * 128 bytes; no high plane; the scale plane starts 256-aligned and is K/32 bytes per row.
    expected = bytearray(256 + 2 * 8)
    expected[:3] = b"\x98\x0f\x71"
    expected[256:264] = GOLDEN_BLOCK
    expected[264:272] = other
    assert payload == bytes(expected)
    scales, decoded_codes = decode_row_split_codes(payload, "q4_g32_s6_fp16", (rows, k))
    assert torch.equal(decoded_codes, codes)
    assert scales.dtype == torch.float16 and tuple(scales.shape) == (rows, 8)
    assert torch.equal(scales, decode_s6_scales(plane))
    assert scales[1].view(torch.int16).tolist() == [0x3C00] + [0] * 7


def test_s6_dequantize_split_gather_and_assemble_agree_with_the_decoded_scales():
    from tools.artifact.codecs.row_split import (
        assemble_row_planes,
        dequantize_row_split,
        gather_row_planes,
        split_row_planes,
    )
    from tools.artifact.layouts import row_split_geometry

    rows, k = 5, 512
    generator = torch.Generator().manual_seed(3)
    codes = torch.randint(-8, 8, (rows, 16, 32), generator=generator, dtype=torch.int8)
    plane = torch.randint(0, 256, (rows, 16), generator=generator, dtype=torch.uint8)
    plane[:, 1::8] = plane[:, 1::8] & 0x3B  # keep every super scale finite and small
    shape = (rows, k)
    payload = encode_row_split(codes, plane, "q4_g32_s6_fp16", shape)
    geometry = row_split_geometry("q4_g32_s6_fp16", shape)
    assert geometry.scale_row_bytes == 16 and len(payload) == geometry.payload_bytes

    scales, _ = decode_row_split_codes(payload, "q4_g32_s6_fp16", shape)
    reference = (codes.float() * scales.float().unsqueeze(-1)).reshape(rows, k)
    for dtype in (torch.float32, torch.bfloat16):
        got = dequantize_row_split(payload, "q4_g32_s6_fp16", shape, dtype=dtype)
        assert torch.equal(got, reference.to(dtype))

    planes = split_row_planes(payload, geometry, 1, 3)
    assert bytes(planes.scale) == payload[geometry.scale_offset + 16 : geometry.scale_offset + 64]
    gathered = gather_row_planes(payload, geometry, [4, 0, 2])
    for index, row in enumerate((4, 0, 2)):
        assert bytes(gathered.scale[index * 16 : (index + 1) * 16]) == bytes(plane[row].numpy())
    whole = split_row_planes(payload, geometry)
    assert assemble_row_planes(whole, "q4_g32_s6_fp16", k) == payload
    on_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    assert torch.equal(
        dequantize_row_split(on_tensor, "q4_g32_s6_fp16", shape, dtype=torch.float32),
        reference,
    )


def test_s6_encode_takes_the_uint8_plane_not_group_words():
    codes = torch.zeros((1, 8, 32), dtype=torch.int8)
    with pytest.raises(TypeError, match="uint8"):
        encode_row_split(codes, torch.zeros((1, 8), dtype=torch.float16), "q4_g32_s6_fp16", (1, 256))
    with pytest.raises(TypeError, match="float16"):
        encode_row_split(codes, torch.zeros((1, 8), dtype=torch.uint8), "q4_g32_fp16", (1, 256))


def test_s6_geometry_and_encoded_size_for_the_gemma_shapes():
    from tools.artifact.layouts import encoded_size, row_split_geometry

    for n, k in ((21504, 5376), (5376, 21504), (16384, 5376), (8192, 5376)):
        s6 = row_split_geometry("q4_g32_s6_fp16", (n, k))
        g32 = row_split_geometry("q4_g32_fp16", (n, k))
        assert (s6.groups_per_row, s6.scale_row_bytes, s6.scale_bytes) == (k // 32, k // 32, n * k // 32)
        assert s6.base_bytes == g32.base_bytes and s6.high_bytes == 0
        assert s6.scale_offset == g32.scale_offset and s6.scale_offset % 256 == 0
        assert s6.payload_bytes == s6.scale_offset + n * (k // 32)
        assert g32.payload_bytes - s6.payload_bytes == n * (k // 32)
        assert encoded_size("row_split_k128_v1", "q4_g32_s6_fp16", (n, k)) == s6.payload_bytes
    assert row_split_geometry("q4_g32_s6_fp16", (21504, 5376)).scale_row_bytes == 168  # not a multiple of 16
    assert row_split_geometry("q4_g32_s6_fp16", (4, 256)).payload_bytes == 4 * 128 + 4 * 8


@pytest.mark.parametrize("k", [100, 128, 384, 640, 5376 + 128])
def test_s6_refuses_k_whose_padded_width_is_not_a_multiple_of_256(k):
    from tools.artifact.layouts import encoded_size, row_split_geometry

    with pytest.raises(ValueError, match="multiple of 256"):
        row_split_geometry("q4_g32_s6_fp16", (4, k))
    with pytest.raises(ValueError, match="multiple of 256"):
        encoded_size("row_split_k128_v1", "q4_g32_s6_fp16", (4, k))
    assert row_split_geometry("q4_g32_fp16", (4, k)).groups_per_row == -(-k // 128) * 4


def test_existing_quant_formats_keep_their_scale_row_bytes():
    from tools.artifact.layouts import row_split_geometry

    for name, k in (("q4_g32_fp16", 5376), ("q4_g64_fp16", 5376), ("q4_g128_fp16", 5376), ("q4_g256_fp16", 5376 + 128 * 2)):
        geometry = row_split_geometry(name, (3, k))
        assert geometry.scale_row_bytes == geometry.groups_per_row * 2
