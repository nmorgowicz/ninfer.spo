"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import (
    row_split_geometry,
)
from tools.artifact.formats import QuantFormat, get_format

_FP16_MIN_SUBNORMAL = 2.0**-24


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def quantize_q4_g32_qat(
    weight: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Recover the 4-bit group-32 grid already present in a QAT bf16 matrix.

    Each group of 32 values is an integer code in [-8, 7] times one binary16
    scale. The scale is the group's absolute maximum divided by the largest
    code it uses (8 when code -8 is present, otherwise 7 or less). Picking
    that divisor keeps the stored values; dividing by 7 for every group
    quantizes them a second time.
    """

    if weight.dim() != 2 or not weight.dtype.is_floating_point:
        raise ValueError("q4_g32_qat requires a floating-point rank-2 matrix")
    geometry = row_split_geometry(Q4_G32_NAME, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical
    grouped = logical.reshape(geometry.n, geometry.groups_per_row, 32)
    amax = grouped.abs().amax(dim=2)
    best_codes = best_scales = best_error = None
    for divisor in range(1, 9):
        host_scales, host_reciprocal = _canonical_scale_words(amax / divisor, 1)
        scales = host_scales.to(target)
        reciprocal = host_reciprocal.to(target)
        codes = torch.clamp(
            torch.round(grouped * reciprocal.unsqueeze(-1)), -8, 7
        ).to(torch.int8)
        error = (codes.float() * scales.unsqueeze(-1) - grouped).square().sum(dim=2)
        if best_error is None:
            best_codes, best_scales, best_error = codes, scales, error
        else:
            take = error < best_error
            best_codes = torch.where(take.unsqueeze(-1), codes, best_codes)
            best_scales = torch.where(take, scales, best_scales)
            best_error = torch.where(take, error, best_error)
    return QuantizedMatrix(codes=best_codes, scales=best_scales)


Q4_G32_NAME = "q4_g32_fp16"


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)


VIRTUAL_GROUP_SIZES = (32, 64, 128, 256)


def quantize_q4_virtual_groups(
    weight: torch.Tensor,
    group_size: int,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Absmax-quantize logical groups of ``group_size`` into ordinary q4_g32 words.

    One binary16 scale and one reciprocal are derived from the absolute maximum
    of each logical group (the last group of a row may be partial; registered K
    padding is zero and adds nothing).  The same scale word is stored for each
    of the group's 32-wide physical subgroups, so the g32 decoder reproduces the
    coarse-group dequantization exactly.  ``group_size`` 32 equals
    ``quantize_matrix`` for q4_g32_fp16.
    """

    if group_size not in VIRTUAL_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {VIRTUAL_GROUP_SIZES}")
    if weight.dim() != 2 or not weight.dtype.is_floating_point:
        raise ValueError("virtual groups require a floating-point rank-2 matrix")
    spec = get_format(Q4_G32_NAME)
    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    sub_max = grouped.abs().amax(dim=2)
    ratio = group_size // spec.group_size
    coarse = -(-geometry.groups_per_row // ratio)
    if coarse * ratio != geometry.groups_per_row:
        padded = torch.zeros((geometry.n, coarse * ratio), dtype=sub_max.dtype, device=target)
        padded[:, : geometry.groups_per_row].copy_(sub_max)
        sub_max = padded
    max_abs = sub_max.reshape(geometry.n, coarse, ratio).amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    keep = geometry.groups_per_row
    scales = host_scales.to(target).repeat_interleave(ratio, dim=1)[:, :keep]
    reciprocal = host_reciprocal.to(target).repeat_interleave(ratio, dim=1)[:, :keep]
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales.contiguous())


SEARCH_DIVISORS = (7.0,) + tuple(float(d) for d in np.arange(5.0, 9.0001, 0.25) if d != 7.0)
REFINE_PASSES = 2


def quantize_q4_scale_search(
    weight: torch.Tensor,
    group_size: int,
    importance: torch.Tensor | None = None,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Choose each logical group's binary16 scale by minimizing the (weighted) squared error.

    Candidates are ``absmax/d`` for ``d`` in ``SEARCH_DIVISORS`` (``d == 7`` is the ordinary
    absmax scale and wins every tie), then ``REFINE_PASSES`` rounds of the weighted
    least-squares scale for the current codes, rounded to binary16. Codes are always the nearest
    in [-8, 7] for the exact stored scale, and the error that is compared is evaluated with
    that stored scale and those codes. ``importance`` is one non-negative weight per logical
    column (None means every weight is 1, the unweighted MSE arm). The result uses the same
    q4_g32 words as ``quantize_q4_virtual_groups``.
    """

    if group_size not in VIRTUAL_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {VIRTUAL_GROUP_SIZES}")
    if weight.dim() != 2 or not weight.dtype.is_floating_point:
        raise ValueError("scale search requires a floating-point rank-2 matrix")
    spec = get_format(Q4_G32_NAME)
    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    if importance is not None:
        if tuple(importance.shape) != (geometry.k,):
            raise ValueError(
                f"importance has shape {tuple(importance.shape)}; expected ({geometry.k},)"
            )
        if not bool(torch.isfinite(importance).all()) or bool((importance < 0).any()):
            raise ValueError("importance must be finite and non-negative")
    ratio = group_size // spec.group_size
    coarse = -(-geometry.k_pad // group_size)
    columns = coarse * group_size
    grouped = torch.zeros((geometry.n, columns), dtype=torch.float32, device=target)
    grouped[:, : geometry.k].copy_(weight.detach().to(device=target, dtype=torch.float32))
    grouped = grouped.reshape(geometry.n, coarse, group_size)
    weights = torch.zeros(columns, dtype=torch.float32, device=target)
    weights[: geometry.k] = 1.0 if importance is None else importance.to(device=target, dtype=torch.float32)
    weights = weights.reshape(1, coarse, group_size)
    max_abs = grouped.abs().amax(dim=2)

    def evaluate(words: torch.Tensor, reciprocal: torch.Tensor):
        scale = words.to(target)
        codes = torch.clamp(
            torch.round(grouped * reciprocal.to(target).unsqueeze(-1)), spec.qmin, spec.qmax
        ).to(torch.int8)
        residual = codes.float() * scale.float().unsqueeze(-1) - grouped
        return codes, scale, (weights * residual.square()).sum(dim=2)

    best_codes = best_scale = best_error = None
    for divisor in SEARCH_DIVISORS:
        words, reciprocal = _canonical_scale_words(max_abs, divisor)
        codes, scale, error = evaluate(words, reciprocal)
        if best_error is None:
            best_codes, best_scale, best_error = codes, scale, error
            continue
        take = error < best_error
        best_codes = torch.where(take.unsqueeze(-1), codes, best_codes)
        best_scale = torch.where(take, scale, best_scale)
        best_error = torch.where(take, error, best_error)
    for _ in range(REFINE_PASSES):
        q = best_codes.float()
        numerator = (weights * q * grouped).sum(dim=2)
        denominator = (weights * q.square()).sum(dim=2)
        ideal = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30), torch.zeros_like(numerator))
        words, reciprocal = _canonical_scale_words(ideal.clamp_min(0), 1.0)
        codes, scale, error = evaluate(words, reciprocal)
        take = error < best_error
        best_codes = torch.where(take.unsqueeze(-1), codes, best_codes)
        best_scale = torch.where(take, scale, best_scale)
        best_error = torch.where(take, error, best_error)
    keep = geometry.groups_per_row
    codes = best_codes.reshape(geometry.n, columns)[:, : geometry.k_pad].reshape(
        geometry.n, keep, spec.group_size
    )
    scales = best_scale.repeat_interleave(ratio, dim=1)[:, :keep].contiguous()
    return QuantizedMatrix(codes=codes.contiguous(), scales=scales)


def to_native_groups(encoded: QuantizedMatrix, group_size: int) -> QuantizedMatrix:
    """Re-shape g32-emulated coarse-group words into a native q4_g{G}_fp16 matrix.

    The emulated matrix repeats each coarse scale ``G/32`` times; the native matrix keeps one. The
    codes are untouched (they are already the coarse-group codes), so decoded values are identical.
    """

    n, subgroups, width = encoded.codes.shape
    ratio = group_size // 32
    if width != 32 or group_size not in VIRTUAL_GROUP_SIZES or subgroups % ratio:
        raise ValueError(f"cannot form q4_g{group_size} groups from {subgroups} g32 subgroups")
    words = encoded.scales.view(torch.int16).reshape(n, subgroups // ratio, ratio)
    if not bool((words == words[:, :, :1]).all()):
        raise ValueError("emulated scales are not constant inside a coarse group")
    return QuantizedMatrix(
        codes=encoded.codes.reshape(n, subgroups // ratio, group_size).contiguous(),
        scales=encoded.scales.reshape(n, subgroups // ratio, ratio)[:, :, 0].contiguous(),
    )
