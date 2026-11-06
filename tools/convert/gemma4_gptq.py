"""Encoded sources for the Gemma 4 text projections of an external compressed-tensors INT4 checkpoint.

The checkpoint is a text-only (`model.layers.N...`) or full (`model.language_model.layers.N...`) Hugging Face tree in
the original head order. Global-layer q/k rows and o columns are permuted by the converter (`prope_perm`); the same
permutation is applied here to the encoded words, so the stored codes and scales are the ones the external tool chose.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .gemma4 import prope_perm
from .sources.logical import LogicalSource
from .sources.packed_int4 import GROUP_SIZES, packed_int4_matrix_source
from .sources.safetensors import SafetensorsSource

_HF_ROLES = {
    "attention/query": "self_attn.q_proj",
    "attention/key": "self_attn.k_proj",
    "attention/value": "self_attn.v_proj",
    "attention/output": "self_attn.o_proj",
    "mlp/gate": "mlp.gate_proj",
    "mlp/up": "mlp.up_proj",
    "mlp/down": "mlp.down_proj",
}


def checkpoint_group_size(config: dict) -> int:
    """Validate the checkpoint's quantization_config for the supported layout and return its group size."""
    quant = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")
    if not quant or quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "pack-quantized":
        raise ValueError("the GPTQ source must be a compressed-tensors pack-quantized checkpoint")
    groups = quant.get("config_groups") or {}
    if len(groups) != 1:
        raise ValueError(f"expected exactly one quantization config group, got {sorted(groups)}")
    weights = next(iter(groups.values())).get("weights") or {}
    group_size = weights.get("group_size")
    if (
        weights.get("num_bits") != 4
        or weights.get("type") != "int"
        or weights.get("symmetric") is not True
        or weights.get("strategy") != "group"
        or weights.get("actorder") not in (None, "null")
        or group_size not in GROUP_SIZES
    ):
        raise ValueError(
            f"unsupported weight scheme {json.dumps(weights, sort_keys=True)}; "
            f"need 4-bit symmetric int, strategy group, group_size in {GROUP_SIZES}, no activation order"
        )
    return group_size


def _head_index(perm: torch.Tensor, heads: int) -> torch.Tensor:
    """Fork index -> source index when every head of `perm.numel()` entries is permuted by `perm`."""
    dim = perm.numel()
    return (torch.arange(heads, dtype=torch.int64)[:, None] * dim + perm[None, :]).reshape(-1)


def gptq_projection_sources(model, root: Path) -> tuple[dict[str, LogicalSource], int]:
    """Encoded `q4_g32_fp16` sources for every text-layer projection of `model`, read from `root`.

    Returns (sources by fork parameter name, the checkpoint group size). The sources reopen files on demand.
    """
    store = SafetensorsSource(root)
    try:
        group_size = checkpoint_group_size(store.config)
        probe = "layers.0.mlp.down_proj.weight_packed"
        stem = next(
            (p for p in ("model.layers.", "model.language_model.layers.") if store.has(p[: -len("layers.")] + probe)),
            None,
        )
        if stem is None:
            raise ValueError(f"{root}: no packed Gemma 4 text projections (looked for model[.language_model].{probe})")
        config = model.config
        perm = prope_perm(config["global_head_dim"], config["partial_rotary_factor"])
        heads, kv = config["num_attention_heads"], config["num_global_key_value_heads"]
        sources = {}
        for name, parameter in model.parameters.items():
            if not (name.startswith("text/layers/") and parameter.projection):
                continue
            _, _, layer, role = name.split("/", 3)
            index = int(layer)
            global_layer = config["layer_types"][index] == "full_attention"
            row_index = col_index = None
            if global_layer and role in ("attention/query", "attention/key"):
                row_index = _head_index(perm, heads if role == "attention/query" else kv)
            elif global_layer and role == "attention/output":
                col_index = _head_index(perm, heads)
            sources[name] = packed_int4_matrix_source(
                store,
                f"{stem}{index}.{_HF_ROLES[role]}",
                parameter.shape,
                group_size,
                row_index=row_index,
                col_index=col_index,
            )
        if not sources:
            raise ValueError("the model has no text-layer projections")
        return sources, group_size
    finally:
        store.close()
