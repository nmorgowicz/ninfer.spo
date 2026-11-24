"""Pre-solved NVFP4 words for research arms: codes, E4M3 words and the parent divisor per logical parameter.

A trusted caller registers one safetensors file per logical parameter id before ``Recipe.prepare``. Each file holds, for the
parameter ``P`` (for example ``text/layers/3/attention/query``)::

    P.codes    uint8 [N, K/2]   two E2M1 codes per byte, even element in the low nibble, in the converter's LOGICAL order
    P.scales   uint8 [N, K/16]  E4M3 words, one per contiguous 16-column group of the logical order
    P.divisor  float32 [1]      the packed parent's FP32 divisor (identical for every member of one packed parent)

The words are produced by an external solver; the converter only stores them, after proving that they decode to the matrix
they claim to be (relative squared error against the logical source values) and that every member of a packed parent agrees on
its divisor.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from .nvfp4 import BLOCK, E2M1

_WORDS: dict[str, Path] = {}


def set_words(index: dict[str, str | Path]) -> None:
    _WORDS.clear()
    _WORDS.update({name: Path(path) for name, path in index.items()})


def clear_words() -> None:
    _WORDS.clear()


def _file(parameter: str) -> Path:
    if parameter not in _WORDS:
        raise ValueError(f"{parameter}: no pre-solved NVFP4 words registered")
    return _WORDS[parameter]


def read_divisor(parameter: str) -> bytes:
    with safe_open(str(_file(parameter)), framework="pt") as handle:
        tensor = handle.get_tensor(f"{parameter}.divisor")
    if tensor.dtype != torch.float32 or tensor.numel() != 1 or not bool(tensor.isfinite().all()) or float(tensor) <= 0:
        raise ValueError(f"{parameter}: divisor must be one finite positive float32")
    return tensor.numpy().tobytes()


def read_words(parameter: str, begin: int, end: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(_file(parameter)), framework="pt") as handle:
        codes = handle.get_slice(f"{parameter}.codes")
        scales = handle.get_slice(f"{parameter}.scales")
        if list(codes.get_shape())[1:] != [k // 2] or list(scales.get_shape())[1:] != [k // BLOCK]:
            raise ValueError(f"{parameter}: word shapes do not match K={k}")
        c, s = codes[begin:end], scales[begin:end]
    if c.dtype != torch.uint8 or s.dtype != torch.uint8:
        raise ValueError(f"{parameter}: codes and scales must be uint8")
    if bool((s > 0x7E).any()):
        raise ValueError(f"{parameter}: scale words must be nonnegative finite E4M3")
    return c.contiguous(), s.contiguous()


def decode_packed(codes: torch.Tensor, scales: torch.Tensor, divisor: float) -> torch.Tensor:
    """FP32 decode of packed words ``[N, K/2]``/``[N, K/16]`` to ``[N, K]``."""
    n = codes.shape[0]
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(n, -1)
    table = torch.tensor(E2M1, device=codes.device, dtype=torch.float32)
    magnitude = table[(nibbles & 7).long()]
    signed = torch.where(nibbles >= 8, -magnitude, magnitude)
    scale = scales.view(torch.float8_e4m3fn).float().repeat_interleave(BLOCK, dim=1)
    return signed * scale / divisor
