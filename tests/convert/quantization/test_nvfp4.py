from __future__ import annotations

import json

import numpy as np
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.quantization.nvfp4 import nvfp4_divisor, quantize_nvfp4_rows
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource, tensor_source

TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def decode(codes: torch.Tensor, scales: torch.Tensor, divisor: float, k: int) -> np.ndarray:
    """FP64 decode with the compressed-tensors rule value = code * scale / divisor."""
    packed = codes.numpy().astype(np.int64)
    nibbles = np.stack((packed & 15, packed >> 4), axis=-1).reshape(packed.shape[0], k)
    magnitude = TABLE[nibbles & 7]
    signed = np.where(nibbles >= 8, -magnitude, magnitude)
    scale = scales.view(torch.float8_e4m3fn).double().numpy().repeat(16, axis=1)
    return signed * scale / divisor


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def matrices() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    gaussian = torch.randn(256, 512, generator=generator)
    heavy = torch.distributions.StudentT(2.5).sample((256, 512)) * 0.02
    outliers = gaussian.clone()
    outliers[::17, ::31] *= 40.0
    return {"gaussian": gaussian, "student_t": heavy, "outliers": outliers}


def test_low_nibble_holds_the_even_element() -> None:
    row = torch.zeros(1, 16)
    row[0, 0] = 6.0
    row[0, 1] = -0.5
    divisor = nvfp4_divisor(6.0)
    encoded = quantize_nvfp4_rows(row, divisor, device="cpu")
    assert encoded.codes[0, 0].item() == 0x7 | (0x9 << 4)
    values = decode(encoded.codes, encoded.scales, divisor, 16)
    assert values[0, 0] == 6.0 and values[0, 1] == -0.5 and not values[0, 2:].any()


def test_mse46_never_loses_to_absmax_and_codes_are_nearest() -> None:
    for name, values in matrices().items():
        host = values.double().numpy()
        divisor = nvfp4_divisor(float(values.abs().max()))
        absmax = quantize_nvfp4_rows(values, divisor, mse_46=False, device="cpu")
        best = quantize_nvfp4_rows(values, divisor, mse_46=True, device="cpu")
        plain = decode(absmax.codes, absmax.scales, divisor, values.shape[1])
        chosen = decode(best.codes, best.scales, divisor, values.shape[1])

        blocks = host.reshape(host.shape[0], -1, 16)
        plain_err = ((plain.reshape(blocks.shape) - blocks) ** 2).sum(-1)
        chosen_err = ((chosen.reshape(blocks.shape) - blocks) ** 2).sum(-1)
        assert (chosen_err <= plain_err + 1e-18).all(), name
        assert mse(chosen, host) < mse(plain, host), name
        relative = mse(chosen, host) / float(np.mean(host**2))
        assert relative < 0.02, (name, relative)

        # Every element decodes to the nearest representable value of its block scale.
        scale = best.scales.view(torch.float8_e4m3fn).double().numpy().repeat(16, axis=1)
        grid = np.concatenate((TABLE, -TABLE[1:]))
        candidates = grid[None, None, :] * scale[..., None] / divisor
        nearest = np.abs(candidates - host[..., None]).min(-1)
        assert (np.abs(chosen - host) <= nearest + 1e-12 * np.abs(host).max()).all(), name


def test_streamed_nvfp4_conversion_rereads_the_same_values(tmp_path) -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    source = matrices()["outliers"].to(torch.bfloat16)
    shard = "model.safetensors"
    save_file({name: source}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard}}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text("{}")
    model = Model({"text": {"config": {}}})
    shape = tuple(source.shape)
    with SafetensorsSource(tmp_path) as reader:
        model.add(
            Parameter("down", shape, tensor_source(reader, name, shape), inputs=("input",))
        )
        recipe = Recipe(model)
        recipe.assign("down", format="nvfp4", method="nvfp4_mse46")
        path = tmp_path / "down.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=128)
    with Artifact(path) as artifact:
        object_id = artifact.directory.bindings["down"]["object"]
        codes, scales, stored = decode_nvfp4_words(artifact.read_object(object_id), shape)
    divisor = float(stored)
    assert divisor == nvfp4_divisor(float(source.float().abs().max()))
    expected = quantize_nvfp4_rows(source.float(), divisor, device="cpu")
    assert torch.equal(codes, expected.codes)
    assert torch.equal(scales, expected.scales)


def test_exhaustive_search_never_loses_to_mse46_and_importance_one_is_unweighted() -> None:
    from tools.convert.quantization.nvfp4 import quantize_nvfp4_rows_search

    for name, values in matrices().items():
        host = values.double().numpy()
        divisor = nvfp4_divisor(float(values.abs().max()))
        best = quantize_nvfp4_rows(values, divisor, mse_46=True, device="cpu")
        every = quantize_nvfp4_rows_search(values, divisor, device="cpu")
        ones = quantize_nvfp4_rows_search(values, divisor, importance=torch.ones(values.shape[1]), device="cpu")
        assert torch.equal(every.codes, ones.codes) and torch.equal(every.scales, ones.scales), name
        blocks = host.reshape(host.shape[0], -1, 16)
        old = decode(best.codes, best.scales, divisor, values.shape[1]).reshape(blocks.shape)
        new = decode(every.codes, every.scales, divisor, values.shape[1]).reshape(blocks.shape)
        assert (((new - blocks) ** 2).sum(-1) <= ((old - blocks) ** 2).sum(-1) + 1e-18).all(), name


def test_weighted_search_follows_the_weights_and_zero_blocks_store_word_zero() -> None:
    from tools.convert.quantization.nvfp4 import quantize_nvfp4_rows_search

    row = torch.zeros(2, 32)
    row[0, :16] = torch.tensor([0.9, 0.31, -0.22, 0.05, 0.4, -0.7, 0.18, 0.61, -0.33, 0.27, 0.12, -0.5, 0.44, -0.08, 0.2, 0.36])
    divisor = nvfp4_divisor(1.0)
    plain = quantize_nvfp4_rows_search(row, divisor, device="cpu")
    weights = torch.full((32,), 1e-6)
    weights[2] = 1.0
    focus = quantize_nvfp4_rows_search(row, divisor, importance=weights, device="cpu")
    assert plain.scales[1].tolist() == [0, 0] and focus.scales[1].tolist() == [0, 0]
    assert plain.scales[0, 0].item() != 0 and plain.scales[0, 1].item() == 0

    def weighted_error(encoded):
        decoded = decode(encoded.codes, encoded.scales, divisor, 32)[0, :16]
        return float(((decoded - row[0, :16].double().numpy()) ** 2 * weights[:16].double().numpy()).sum())

    assert weighted_error(focus) <= weighted_error(plain)


def test_streamed_exhaustive_conversion_matches_the_encoder(tmp_path) -> None:
    from tools.convert.quantization.nvfp4 import quantize_nvfp4_rows_search

    name = "model.layers.0.mlp.down_proj.weight"
    source = matrices()["outliers"].to(torch.bfloat16)
    shard = "model.safetensors"
    save_file({name: source}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard}}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text("{}")
    model = Model({"text": {"config": {}}})
    shape = tuple(source.shape)
    with SafetensorsSource(tmp_path) as reader:
        model.add(
            Parameter("down", shape, tensor_source(reader, name, shape), inputs=("input",))
        )
        recipe = Recipe(model)
        recipe.assign("down", format="nvfp4", method="nvfp4_mse_all")
        path = tmp_path / "down.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=128)
    with Artifact(path) as artifact:
        object_id = artifact.directory.bindings["down"]["object"]
        codes, scales, stored = decode_nvfp4_words(artifact.read_object(object_id), shape)
    divisor = float(stored)
    assert divisor == nvfp4_divisor(float(source.float().abs().max()))
    expected = quantize_nvfp4_rows_search(source.float(), divisor, device="cpu")
    assert torch.equal(codes, expected.codes)
    assert torch.equal(scales, expected.scales)


def _precomputed_model(tmp_path, source, words_path):
    from tools.convert.quantization.nvfp4_words import set_words

    name = "model.layers.0.mlp.down_proj.weight"
    shard = "model.safetensors"
    save_file({name: source}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: shard}}), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}")
    set_words({"down": words_path})
    model = Model({"text": {"config": {}}})
    shape = tuple(source.shape)
    reader = SafetensorsSource(tmp_path)
    reader.__enter__()
    model.add(Parameter("down", shape, tensor_source(reader, name, shape), inputs=("input",)))
    recipe = Recipe(model)
    recipe.assign("down", format="nvfp4", method="nvfp4_precomputed")
    return model, recipe, reader, shape


def test_precomputed_words_are_stored_verbatim_and_a_wrong_order_is_refused(tmp_path) -> None:
    import pytest

    source = matrices()["gaussian"].to(torch.bfloat16)
    divisor = nvfp4_divisor(float(source.float().abs().max()))
    encoded = quantize_nvfp4_rows(source.float(), divisor, device="cpu")
    good = tmp_path / "good.safetensors"
    save_file({"down.codes": encoded.codes, "down.scales": encoded.scales,
               "down.divisor": torch.tensor([divisor], dtype=torch.float32)}, good)
    model, recipe, reader, shape = _precomputed_model(tmp_path, source, good)
    try:
        path = tmp_path / "down.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=128)
    finally:
        reader.__exit__(None, None, None)
    with Artifact(path) as artifact:
        object_id = artifact.directory.bindings["down"]["object"]
        codes, scales, stored = decode_nvfp4_words(artifact.read_object(object_id), shape)
    assert float(stored) == divisor
    assert torch.equal(codes, encoded.codes) and torch.equal(scales, encoded.scales)

    shuffled = tmp_path / "shuffled.safetensors"
    perm = torch.randperm(shape[0], generator=torch.Generator().manual_seed(3))
    save_file({"down.codes": encoded.codes[perm].contiguous(), "down.scales": encoded.scales[perm].contiguous(),
               "down.divisor": torch.tensor([divisor], dtype=torch.float32)}, shuffled)
    other = tmp_path / "other"
    other.mkdir()
    model, recipe, reader, shape = _precomputed_model(other, source, shuffled)
    try:
        with pytest.raises(ValueError, match="do not describe"):
            convert(model, recipe, other / "down.ninfer", device="cpu", rows_per_chunk=128)
    finally:
        reader.__exit__(None, None, None)
