"""S6 two-level scale plane of ``q4_g32_s6_fp16``.

Per row and per 256 consecutive K columns the scale plane stores one 8-byte little-endian block ``W``:
``S = W & 0xffff`` is a binary16 super scale and ``m_j = (W >> (16 + 6*j)) & 63`` (``j`` = 0..7) are the
eight 6-bit multipliers of the block's eight consecutive group-32 columns. The scale of group ``j`` is

    s_j = binary16_round_nearest_even(float32(S) * float32(m_j))

The binary32 product is exact (11-bit by 6-bit significand), so the single rounding to binary16 is the
only one. The group is then decoded like an ordinary ``q4_g32_fp16`` group: code times ``s_j``. A row's
scale plane is ``K_pad / 32`` bytes, block ``b`` of row ``r`` at byte ``(r * (K_pad / 256) + b) * 8``.

The same rule exists once in the engine (``src/ops/common/q4_scale_mode.h``); the golden vector in
``tests/artifact/test_codecs.py`` pins both implementations.
"""

from __future__ import annotations

import numpy as np
import torch

BLOCK_BYTES = 8
GROUPS_PER_BLOCK = 8
MULTIPLIER_BITS = 6
MULTIPLIER_MAX = (1 << MULTIPLIER_BITS) - 1
_FP16_INF_BITS = 0x7C00


def decode_s6_scales(plane: torch.Tensor) -> torch.Tensor:
    """Decode S6 blocks to binary16 group scales.

    ``plane`` is a uint8 tensor whose last dimension is a multiple of 8 (``[N, K/32]`` or flat); the
    result is a float16 tensor of the same shape with one scale per byte of the plane.
    """

    if plane.dtype != torch.uint8:
        raise TypeError("S6 scale planes must be uint8")
    if plane.shape[-1] % BLOCK_BYTES:
        raise ValueError(f"S6 scale rows must be a multiple of {BLOCK_BYTES} bytes")
    blocks = plane.contiguous().reshape(-1, BLOCK_BYTES)
    super_scale = blocks[:, :2].contiguous().view(torch.float16).reshape(-1, 1).float()
    field = torch.zeros(blocks.shape[0], dtype=torch.int64, device=blocks.device)
    for index in range(6):
        field |= blocks[:, 2 + index].to(torch.int64) << (8 * index)
    shifts = MULTIPLIER_BITS * torch.arange(GROUPS_PER_BLOCK, device=blocks.device, dtype=torch.int64)
    multiplier = ((field.unsqueeze(1) >> shifts) & MULTIPLIER_MAX).float()
    return (super_scale * multiplier).to(torch.float16).reshape(plane.shape)


def _encode_blocks(super_bits: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    """Pack ``[R]`` binary16 super-scale words and ``[R, 8]`` multipliers into ``[R, 8]`` bytes."""

    blocks = np.zeros((super_bits.shape[0], BLOCK_BYTES), dtype=np.uint8)
    blocks[:, 0] = super_bits & 0xFF
    blocks[:, 1] = super_bits >> 8
    field = np.zeros(super_bits.shape[0], dtype=np.uint64)
    for j in range(GROUPS_PER_BLOCK):
        field |= multiplier[:, j].astype(np.uint64) << np.uint64(MULTIPLIER_BITS * j)
    for index in range(6):
        blocks[:, 2 + index] = (field >> np.uint64(8 * index)) & np.uint64(0xFF)
    return blocks


def pack_s6_scales(words: torch.Tensor) -> torch.Tensor:
    """Derive the S6 plane that decodes exactly to the binary16 group scales ``words``.

    ``words`` is float16 ``[N, groups]`` with ``groups`` a multiple of 8 (the groups of a K-padded row).
    For every run of eight consecutive groups a super scale ``S`` and multipliers ``m_j`` in 0..63 are
    found so that ``fp16(float32(S) * m_j)`` equals the word bit for bit. Candidates come from the
    block's largest word ``w`` (the largest multiplier ``m`` gives ``S ~ w/m``, tried from 63 down with
    the two neighbouring binary16 values), so the scales an emulated two-level search produced are
    recovered without a new search. An all-zero block is ``S = 0``, all ``m = 0``. A block that no
    ``(S, m)`` reproduces exactly, a negative, infinite or NaN word raises ``ValueError``.

    Returns uint8 ``[N, groups]``: the row-major scale plane (``groups / 8`` blocks of 8 bytes per row).
    """

    if words.dtype != torch.float16 or words.dim() != 2 or words.shape[1] % GROUPS_PER_BLOCK:
        raise TypeError("S6 packing needs float16 words of shape [N, groups] with groups a multiple of 8")
    rows, groups = words.shape
    host = words.detach().cpu().contiguous().numpy().reshape(-1, GROUPS_PER_BLOCK)
    bits = host.view(np.uint16)
    if bool(((bits & 0x8000) != 0).any()) or bool(((bits & 0x7FFF) >= _FP16_INF_BITS).any()):
        raise ValueError("S6 scales must be finite and non-negative binary16 words (no -0, inf or NaN)")
    values = host.astype(np.float32)
    top = values.max(axis=1)
    count = values.shape[0]
    super_bits = np.zeros(count, dtype=np.uint16)
    multiplier = np.zeros((count, GROUPS_PER_BLOCK), dtype=np.uint8)
    unresolved = top > 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for top_multiplier in range(MULTIPLIER_MAX, 0, -1):
            pending = np.flatnonzero(unresolved)
            if pending.size == 0:
                break
            guess = (top[pending] / np.float32(top_multiplier)).astype(np.float16).view(np.uint16)
            for delta in (0, 1, -1):
                candidate = guess.astype(np.int32) + delta
                valid = (candidate >= 1) & (candidate < _FP16_INF_BITS)
                candidate = np.where(valid, candidate, 1).astype(np.uint16)
                scale = candidate.view(np.float16).astype(np.float32)
                chosen = np.clip(np.rint(values[pending] / scale[:, None]), 0, MULTIPLIER_MAX)
                decoded = (scale[:, None] * chosen).astype(np.float16).view(np.uint16)
                exact = valid & (decoded == bits[pending]).all(axis=1)
                hit = pending[exact]
                super_bits[hit] = candidate[exact]
                multiplier[hit] = chosen[exact].astype(np.uint8)
                unresolved[hit] = False
                keep = ~exact
                pending, guess = pending[keep], guess[keep]
                if pending.size == 0:
                    break
    if bool(unresolved.any()):
        raise ValueError(
            f"{int(unresolved.sum())} of {count} S6 blocks have group scales that no binary16 super "
            "scale times 6-bit multipliers reproduces exactly"
        )
    plane = torch.from_numpy(_encode_blocks(super_bits, multiplier)).reshape(rows, groups)
    if not torch.equal(decode_s6_scales(plane).view(torch.int16), words.detach().cpu().view(torch.int16)):
        raise ValueError("S6 plane does not decode to the requested scales")
    return plane
