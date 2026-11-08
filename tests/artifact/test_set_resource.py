from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec, encode_directory
from tools.artifact.set_resource import main
from tools.artifact.writer import ArtifactWriter, RESOURCE_EDIT_SPARE_BYTES

WEIGHT = bytes(range(256)) * 40  # 10240 bytes, spans parts when sharded


def _build(path: Path, *, max_file_bytes: int = 32_000_000_000, metadata: dict | None = None,
           architecture: str = "Gemma4ForConditionalGeneration", spare: int = 0) -> None:
    specs = [
        ResourceSpec("resource/text/chat_template.jinja", 5),
        ResourceSpec("resource/text/tokenizer.json", 2),
        TensorSpec("w", (len(WEIGHT) // 2,), "bf16", "contiguous_le_v1"),
    ]
    components = {
        "text": {
            "config": {"architectures": [architecture]},
            "resources": {
                "chat_template.jinja": "resource/text/chat_template.jinja",
                "tokenizer.json": "resource/text/tokenizer.json",
            },
        }
    }
    with ArtifactWriter(path, specs, components=components,
                        bindings={"text/w": {"object": "w"}}, metadata=metadata,
                        max_file_bytes=max_file_bytes, directory_spare_bytes=spare) as writer:
        writer.write_object("resource/text/chat_template.jinja", b"hello")
        writer.write_object("resource/text/tokenizer.json", b"{}")
        writer.write_object("w", WEIGHT)


def _files(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + "*"))


def _digest(path: Path) -> dict[str, str]:
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in _files(path) if not item.name.endswith(".json")}


def _swap(path: Path, template: Path, *extra: str) -> int:
    return main([str(path), "--resource", f"chat_template.jinja={template}", *extra])


@pytest.mark.parametrize("max_file_bytes", [32_000_000_000, 12288], ids=["single", "sharded"])
def test_swap_replaces_only_the_resource_and_restores_exactly(tmp_path, max_file_bytes):
    path = tmp_path / "gemma.ninfer"
    _build(path, max_file_bytes=max_file_bytes)
    with Artifact(path) as before:
        offset = before.payload_offset
        sharded = len(before.directory.files) > 1
    original = _digest(path)
    template = tmp_path / "fixed.jinja"
    template.write_bytes("{{- bos_token -}}\u00e9 fixed template\n".encode())

    assert _swap(path, template, "--backup", str(tmp_path / "b1.json")) == 0
    with Artifact(path) as after:
        assert after.payload_offset == offset
        assert (len(after.directory.files) > 1) == sharded
        assert after.read_object("resource/text/chat_template.jinja") == template.read_bytes()
        assert after.read_object("resource/text/tokenizer.json") == b"{}"
        assert after.read_object("w") == WEIGHT
        offsets = [obj.offset for obj in after.objects]
        assert offsets == sorted(offsets)
        record = after.directory.provenance["resource_overrides"]["text/chat_template.jinja"]
        assert record["sha256"] == hashlib.sha256(template.read_bytes()).hexdigest()
        assert record["original_sha256"] == hashlib.sha256(b"hello").hexdigest()
        grown = sum(f.payload_bytes for f in after.directory.files)

    # A second swap overwrites the tail resource instead of appending again.
    template.write_bytes(b"v2")
    assert _swap(path, template, "--backup", str(tmp_path / "b2.json")) == 0
    with Artifact(path) as again:
        assert again.read_object("resource/text/chat_template.jinja") == b"v2"
        assert again.read_object("w") == WEIGHT
        assert sum(f.payload_bytes for f in again.directory.files) == grown - len(
            "{{- bos_token -}}\u00e9 fixed template\n".encode()) + 2
        record = again.directory.provenance["resource_overrides"]["text/chat_template.jinja"]
        assert record["swaps"] == 2
        assert record["original_sha256"] == hashlib.sha256(b"hello").hexdigest()

    assert main([str(path), "--restore", str(tmp_path / "b2.json")]) == 0
    with Artifact(path) as undone:
        assert undone.read_object("resource/text/chat_template.jinja").startswith(b"{{- bos")
    assert main([str(path), "--restore", str(tmp_path / "b1.json")]) == 0
    assert _digest(path) == original


def test_dry_run_and_identical_bytes_write_nothing(tmp_path):
    path = tmp_path / "gemma.ninfer"
    _build(path)
    original = _digest(path)
    template = tmp_path / "t.jinja"
    template.write_text("new", encoding="utf-8")
    assert _swap(path, template, "--dry-run") == 0
    template.write_bytes(b"hello")
    assert _swap(path, template) == 0
    assert _digest(path) == original
    assert not list(tmp_path.glob("*.set_resource-*.json"))


def test_rejects_bad_input_without_writing(tmp_path, capsys):
    path = tmp_path / "gemma.ninfer"
    _build(path)
    original = _digest(path)
    bad = tmp_path / "bad.jinja"
    bad.write_bytes(b"\xff\xfe")
    assert _swap(path, bad) == 1
    bad.write_bytes(b"")
    assert _swap(path, bad) == 1
    good = tmp_path / "ok.jinja"
    good.write_bytes(b"x")
    assert main([str(path), "--resource", f"missing.jinja={good}"]) == 1
    assert main([str(path), "--resource", f"tokenizer.json={bad}"]) == 1
    assert "not found" in capsys.readouterr().err
    assert _digest(path) == original


def _tight(path: Path, free: int, **kwargs) -> None:
    """Build an artifact whose directory reserve has exactly ``free`` spare bytes."""
    _build(path, metadata={"name": "x"}, **kwargs)
    with Artifact(path) as artifact:
        spare = artifact.payload_offset - 32 - len(encode_directory(artifact.directory))
    for item in _files(path):
        item.unlink()
    pad = spare - free if spare >= free else spare + 4096 - free
    _build(path, metadata={"name": "x" * (1 + pad)}, **kwargs)
    with Artifact(path) as artifact:
        assert artifact.payload_offset - 32 - len(encode_directory(artifact.directory)) == free


def test_history_has_a_fixed_size_across_swaps(tmp_path):
    path = tmp_path / "gemma.ninfer"
    _build(path)
    template = tmp_path / "t.jinja"
    sizes = []
    for index in range(6):
        template.write_bytes(f"template {index}".encode())
        assert _swap(path, template, "--backup", str(tmp_path / f"b{index}.json")) == 0
        with Artifact(path) as artifact:
            sizes.append(len(encode_directory(artifact.directory)))
    assert len(set(sizes[1:])) == 1


@pytest.mark.parametrize("max_file_bytes", [32_000_000_000, 12288], ids=["single", "sharded"])
def test_full_reserve_needs_rewrite_which_restores_exactly(tmp_path, max_file_bytes, capsys):
    path = tmp_path / "gemma.ninfer"
    _tight(path, 16, max_file_bytes=max_file_bytes)
    original = _digest(path)
    template = tmp_path / "fixed.jinja"
    template.write_bytes(b"{{ fixed }}")
    assert _swap(path, template) == 1
    assert "--rewrite" in capsys.readouterr().err
    assert _digest(path) == original
    assert _swap(path, template, "--rewrite", "--backup", str(tmp_path / "b.json")) == 0
    with Artifact(path) as artifact:
        assert artifact.payload_offset > 4096
        assert artifact.payload_offset - 32 - len(encode_directory(artifact.directory)) >= RESOURCE_EDIT_SPARE_BYTES
        assert artifact.read_object("resource/text/chat_template.jinja") == b"{{ fixed }}"
        assert artifact.read_object("w") == WEIGHT
    assert not list(tmp_path.glob(".*.tmp"))
    assert main([str(path), "--restore", str(tmp_path / "b.json")]) == 0
    assert _digest(path) == original


def test_converter_reserve_leaves_room_for_swaps(tmp_path):
    path = tmp_path / "gemma.ninfer"
    _build(path, spare=RESOURCE_EDIT_SPARE_BYTES)
    with Artifact(path) as artifact:
        assert artifact.payload_offset - 32 - len(encode_directory(artifact.directory)) >= RESOURCE_EDIT_SPARE_BYTES
        assert artifact.read_object("w") == WEIGHT


def test_fixed_template_architecture_is_refused(tmp_path, capsys):
    path = tmp_path / "flash.ninfer"
    _build(path, architecture="Qwen3_8FlashNextForCausalLM")
    template = tmp_path / "t.jinja"
    template.write_bytes(b"x")
    assert _swap(path, template) == 1
    assert "official chat template" in capsys.readouterr().err


def test_qwen_dense_template_swap(tmp_path):
    path = tmp_path / "qwen.ninfer"
    _build(path, architecture="Qwen3_5ForConditionalGeneration")
    template = tmp_path / "t.jinja"
    template.write_bytes(b"{{ qwen }}")
    assert _swap(path, template, "--backup", str(tmp_path / "b.json")) == 0
    with Artifact(path) as artifact:
        assert artifact.read_object("resource/text/chat_template.jinja") == b"{{ qwen }}"


def test_tokenizer_json_resource_must_parse(tmp_path):
    path = tmp_path / "gemma.ninfer"
    _build(path)
    replacement = tmp_path / "tokenizer.json"
    replacement.write_text(json.dumps({"model": {}}), encoding="utf-8")
    assert main([str(path), "--resource", f"tokenizer.json={replacement}",
                 "--backup", str(tmp_path / "b.json")]) == 0
    with Artifact(path) as artifact:
        assert json.loads(artifact.read_object("resource/text/tokenizer.json")) == {"model": {}}
        assert artifact.read_object("resource/text/chat_template.jinja") == b"hello"
