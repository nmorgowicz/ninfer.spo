from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, ResourceSpec, TensorSpec
from tools.artifact.set_resource import main, plan
from tools.artifact.writer import ArtifactWriter

WEIGHT = bytes(range(256)) * 40  # 10240 bytes, spans parts when sharded


def _build(path: Path, *, max_file_bytes: int = 32_000_000_000) -> None:
    specs = [
        ResourceSpec("resource/text/chat_template.jinja", 5),
        ResourceSpec("resource/text/tokenizer.json", 2),
        TensorSpec("w", (len(WEIGHT) // 2,), "bf16", "contiguous_le_v1"),
    ]
    components = {
        "text": {
            "config": {"architectures": ["Gemma4ForConditionalGeneration"]},
            "resources": {
                "chat_template.jinja": "resource/text/chat_template.jinja",
                "tokenizer.json": "resource/text/tokenizer.json",
            },
        }
    }
    with ArtifactWriter(path, specs, components=components,
                        bindings={"text/w": {"object": "w"}},
                        max_file_bytes=max_file_bytes) as writer:
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
        record = after.directory.provenance["resource_overrides"][-1]
        assert record["sha256"] == hashlib.sha256(template.read_bytes()).hexdigest()
        assert record["previous_sha256"] == hashlib.sha256(b"hello").hexdigest()
        grown = sum(f.payload_bytes for f in after.directory.files)

    # A second swap overwrites the tail resource instead of appending again.
    template.write_bytes(b"v2")
    assert _swap(path, template, "--backup", str(tmp_path / "b2.json")) == 0
    with Artifact(path) as again:
        assert again.read_object("resource/text/chat_template.jinja") == b"v2"
        assert again.read_object("w") == WEIGHT
        assert sum(f.payload_bytes for f in again.directory.files) == grown - len(
            "{{- bos_token -}}\u00e9 fixed template\n".encode()) + 2
        assert len(again.directory.provenance["resource_overrides"]) == 2

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


def test_directory_that_outgrows_its_reserve_is_refused(tmp_path):
    path = tmp_path / "gemma.ninfer"
    _build(path)
    with pytest.raises(ArtifactError, match="in-place edit is impossible"):
        plan(path, "chat_template.jinja", b"x", source_name="n" * 8192)


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
