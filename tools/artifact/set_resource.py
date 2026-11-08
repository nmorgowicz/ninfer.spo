"""Replace one raw resource (e.g. chat_template.jinja) in an existing v3 artifact.

Weights never move and nothing is re-quantized. The new bytes go to the end of the last
payload file, or overwrite the old bytes when the resource already is the last object (so
repeated swaps do not grow the file). The old bytes stay behind as unreferenced space.

The directory JSON is normally rewritten inside its existing reserve (converters leave 64 KiB
spare for this; older artifacts have 0-4 KiB). If it no longer fits, ``--rewrite`` copies the
entry file once with a larger reserve - a plain file copy, not a reconversion. Continuation
files are never copied.

Before any write a backup of the directory head, the file lengths and any overwritten bytes is
saved beside the entry; ``--restore BACKUP`` puts the artifact back exactly. Stop any engine
or server using the artifact first.

    python -m tools.artifact.set_resource MODEL.ninfer --resource chat_template.jinja=FILE
    python -m tools.artifact.set_resource MODEL.ninfer --resource chat_template.jinja=FILE --rewrite
    python -m tools.artifact.set_resource MODEL.ninfer --restore MODEL.ninfer.set_resource-*.json
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .file_io import BINARY, pread, pwrite
from .framing import HEADER, MAGIC, PAYLOAD_ALIGNMENT
from .layouts import align_up
from .reader import Artifact
from .schema import ArtifactError, encode_directory, parse_directory
from .writer import RESOURCE_EDIT_SPARE_BYTES

MAX_RESOURCE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_NAME = 200
COPY_CHUNK_BYTES = 64 * 1024 * 1024
BACKUP_FORMAT = "ninfer-set-resource-backup-v1"
# Architectures whose loader accepts only fixed template digests; a swap makes them unloadable.
FIXED_TEMPLATE_ARCHITECTURES = {"Qwen3_8FlashNextForCausalLM"}
TEMPLATE_ROLES = {"chat_template.jinja", "tokenizer_config.json"}


@dataclass(frozen=True, slots=True)
class Plan:
    entry: Path
    component: str
    role: str
    object_id: str
    old_offset: int
    old_bytes: int
    new_offset: int
    new_data: bytes
    file_index: int  # payload file that receives the new bytes
    position_old: int  # byte position of new_offset in that file, current layout
    position_new: int  # same position after the edit (differs only for an entry rewrite)
    old_payload_offset: int
    new_payload_offset: int
    entry_payload_bytes: int  # entry payload bytes before the edit
    old_file_lengths: tuple[int, ...]
    new_file_lengths: tuple[int, ...]
    head: bytes  # entry [0, new_payload_offset) after the edit
    old_head: bytes
    overwritten: bytes  # existing tail bytes the new data replaces (tail-reuse case only)
    free_before: int  # spare directory bytes before the edit
    free_after: int
    notes: tuple[str, ...]
    unchanged: bool

    @property
    def rewrite(self) -> bool:
        return self.new_payload_offset != self.old_payload_offset


def _file_paths(entry: Path, directory) -> list[Path]:
    return [entry] + [entry.parent / file.path for file in directory.files[1:]]


def _read_resource_file(path: Path, role: str) -> bytes:
    data = path.read_bytes()
    if not data or len(data) > MAX_RESOURCE_BYTES:
        raise ArtifactError(f"{path}: resource must be 1 byte to 16 MiB, got {len(data)}")
    if role.endswith((".jinja", ".json", ".txt")):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactError(f"{path}: {role} must be UTF-8 ({error})") from error
        if role.endswith(".json"):
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                raise ArtifactError(f"{path}: invalid JSON for {role} ({error})") from error
    return data


def _resolve_role(directory, role: str, component: str | None) -> tuple[str, str]:
    matches = [
        (name, value["resources"][role])
        for name, value in directory.components.items()
        if role in value.get("resources", {})
        and (component is None or name == component)
    ]
    if not matches:
        where = f"component {component!r}" if component else "any component"
        raise ArtifactError(f"resource role {role!r} not found in {where}")
    if len(matches) > 1:
        names = ", ".join(name for name, _ in matches)
        raise ArtifactError(f"role {role!r} is in several components ({names}); pass --component")
    name, object_id = matches[0]
    shared = [
        f"{other}/{other_role}"
        for other, value in directory.components.items()
        for other_role, other_id in value.get("resources", {}).items()
        if other_id == object_id and (other, other_role) != (name, role)
    ]
    if shared:
        raise ArtifactError(f"object {object_id!r} is also used by {', '.join(shared)}")
    return name, object_id


def _check_architecture(directory, role: str) -> None:
    if role not in TEMPLATE_ROLES:
        return
    for name, value in directory.components.items():
        architectures = set((value.get("config") or {}).get("architectures") or ())
        fixed = architectures & FIXED_TEMPLATE_ARCHITECTURES
        if fixed:
            raise ArtifactError(
                f"{name}: {', '.join(sorted(fixed))} accepts only its official chat template "
                "digests; a swapped template would make the artifact unloadable"
            )


def _embedded_template_note(artifact: Artifact, component: str, role: str, data: bytes) -> list[str]:
    if role != "chat_template.jinja":
        return []
    object_id = artifact.directory.components[component].get("resources", {}).get(
        "tokenizer_config.json")
    if object_id is None:
        return []
    try:
        config = json.loads(artifact.read_object(object_id))
    except (ValueError, ArtifactError):
        return []
    embedded = config.get("chat_template") if isinstance(config, dict) else None
    if isinstance(embedded, str) and embedded.encode("utf-8") != data:
        return ["tokenizer_config.json also embeds a chat_template; ninfer renders "
                "chat_template.jinja, so the embedded copy is now stale but unused"]
    return []


def _override_record(provenance: dict, key: str, object_id: str, data: bytes, old_data: bytes,
                     source_name: str, now: datetime) -> dict:
    """One fixed-size record per resource: the original digest and the latest swap only."""
    overrides = provenance.get("resource_overrides")
    if isinstance(overrides, list):  # early per-swap history: keep only its first original
        first = next((item for item in overrides if f"{item.get('component')}/{item.get('role')}" == key), None)
        overrides = {} if first is None else {key: {
            "original_sha256": first.get("previous_sha256"),
            "original_bytes": first.get("previous_bytes"),
            "swaps": sum(1 for item in overrides
                         if f"{item.get('component')}/{item.get('role')}" == key),
        }}
    overrides = dict(overrides or {})
    previous = overrides.get(key) or {}
    overrides[key] = {
        "object": object_id,
        "original_sha256": previous.get("original_sha256") or hashlib.sha256(old_data).hexdigest(),
        "original_bytes": previous.get("original_bytes", len(old_data)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "source": source_name[-MAX_SOURCE_NAME:],
        "utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "swaps": int(previous.get("swaps", 0)) + 1,
        "tool": "tools.artifact.set_resource",
    }
    return overrides


def plan(entry: Path, role: str, data: bytes, *, component: str | None = None,
         source_name: str = "", now: datetime | None = None, rewrite: bool = False) -> Plan:
    """Compute every byte to be written without touching the artifact."""
    now = now or datetime.now(timezone.utc)
    with Artifact(entry) as artifact:
        directory = artifact.directory
        _check_architecture(directory, role)
        old_payload_offset = artifact.payload_offset
        old_head = pread(artifact._fds[0], old_payload_offset, 0)
        reserve_old = old_payload_offset - HEADER.size
        free_before = reserve_old - len(encode_directory(directory))
        component_name, object_id = _resolve_role(directory, role, component)
        obj = artifact.object(object_id)
        if obj.kind != "resource":
            raise ArtifactError(f"{object_id}: not a resource object")
        old_data = artifact.read_object(object_id)
        notes = _embedded_template_note(artifact, component_name, role, data)
        paths = _file_paths(entry, directory)
        old_lengths = tuple(path.stat().st_size for path in paths)

        files = [file.payload_bytes for file in directory.files]
        entry_payload_bytes = files[0]
        last = len(files) - 1
        last_start = sum(files[:last])
        payload_end = sum(files)
        if obj.offset + obj.bytes == payload_end and obj.offset >= last_start:
            new_offset = obj.offset  # already the tail object: overwrite, no growth
        else:
            new_offset = payload_end
        files[last] = new_offset + len(data) - last_start
        old_header = old_payload_offset if last == 0 else PAYLOAD_ALIGNMENT
        position_old = old_header + (new_offset - last_start)
        overwritten = b""
        if new_offset == obj.offset:
            fd = artifact._fds[0] if last == 0 else artifact._file(last)
            overwritten = pread(fd, old_lengths[last] - position_old, position_old)

        value = directory.to_json()
        objects = [item for item in value["objects"] if item["id"] != object_id]
        edited = obj.to_json()
        edited.update(offset=new_offset, bytes=len(data))
        objects.append(edited)
        objects.sort(key=lambda item: item["offset"])
        value["objects"] = objects
        value["files"] = [
            {"path": record.path, "payload_bytes": size}
            for record, size in zip(directory.files, files)
        ]
        provenance = dict(value.get("provenance") or {})
        provenance["resource_overrides"] = _override_record(
            provenance, f"{component_name}/{role}", object_id, data, old_data, source_name, now)
        value["provenance"] = provenance
        parse_directory(value, entry_name=entry.name)  # same structural checks as a reader

        encoded = encode_directory(value)
        if len(encoded) <= reserve_old:
            new_payload_offset = old_payload_offset
        elif rewrite:
            new_payload_offset = align_up(
                HEADER.size + len(encoded) + RESOURCE_EDIT_SPARE_BYTES, PAYLOAD_ALIGNMENT)
        else:
            raise ArtifactError(
                f"new directory needs {len(encoded)} bytes but the entry reserves {reserve_old} "
                f"({free_before} free); rerun with --rewrite to copy the entry file once with a "
                f"larger reserve ({old_lengths[0] / 2**30:.1f} GiB copy; no reconversion)"
            )
        reserve_new = new_payload_offset - HEADER.size
        head = (
            HEADER.pack(MAGIC, reserve_new, artifact.artifact_id)
            + encoded
            + b" " * (reserve_new - len(encoded))
        )
        assert len(head) == new_payload_offset
        new_header = new_payload_offset if last == 0 else PAYLOAD_ALIGNMENT
        new_lengths = list(old_lengths)
        new_lengths[0] = new_payload_offset + files[0]
        new_lengths[last] = new_header + files[last]
        return Plan(entry, component_name, role, object_id, obj.offset, obj.bytes,
                    new_offset, data, last, position_old,
                    new_header + (new_offset - last_start),
                    old_payload_offset, new_payload_offset, entry_payload_bytes,
                    old_lengths, tuple(new_lengths), head, old_head, overwritten,
                    free_before, reserve_new - len(encoded), tuple(notes), old_data == data)


def _write_exact(path: Path, position: int, data: bytes, length: int) -> None:
    fd = os.open(path, os.O_RDWR | BINARY)
    try:
        _pwrite_all(fd, data, position)
        os.ftruncate(fd, length)
        os.fsync(fd)
    finally:
        os.close(fd)


def _pwrite_all(fd: int, data: bytes, position: int) -> None:
    view = memoryview(data)
    while view:
        count = pwrite(fd, view, position)
        if count <= 0:
            raise OSError(f"short write at {position}")
        view = view[count:]
        position += count


def _rebuild_entry(entry: Path, head: bytes, source_offset: int, count: int,
                   tail: tuple[int, bytes] | None, length: int) -> None:
    """Write a new entry file (new head + copied payload) beside the old one, then replace it."""
    free = shutil.disk_usage(entry.parent).free
    if free < length + (1 << 30):
        raise ArtifactError(
            f"rewriting {entry.name} needs {length / 2**30:.1f} GiB free beside it "
            f"(+1 GiB margin); {free / 2**30:.1f} GiB available")
    fd_out, temporary = tempfile.mkstemp(prefix=f".{entry.name}.", suffix=".tmp", dir=entry.parent)
    try:
        fd_in = os.open(entry, os.O_RDONLY | BINARY)
        try:
            _pwrite_all(fd_out, head, 0)
            done = 0
            while done < count:
                chunk = pread(fd_in, min(COPY_CHUNK_BYTES, count - done), source_offset + done)
                if not chunk:
                    raise ArtifactError(f"{entry}: short read while copying payload")
                _pwrite_all(fd_out, chunk, len(head) + done)
                done += len(chunk)
        finally:
            os.close(fd_in)
        if tail is not None:
            _pwrite_all(fd_out, tail[1], tail[0])
        os.ftruncate(fd_out, length)
        os.fsync(fd_out)
        os.close(fd_out)
        fd_out = -1
        os.replace(temporary, entry)
    except BaseException:
        if fd_out >= 0:
            os.close(fd_out)
        Path(temporary).unlink(missing_ok=True)
        raise


def _check_lengths(paths: list[Path], lengths) -> None:
    for path, expected in zip(paths, lengths):
        actual = path.stat().st_size
        if actual != expected:
            raise ArtifactError(f"{path}: length {actual} differs from expected {expected}")


def write_backup(p: Plan, path: Path) -> None:
    record = {
        "format": BACKUP_FORMAT,
        "entry": p.entry.name,
        "object": p.object_id,
        "head_b64": base64.b64encode(p.old_head).decode("ascii"),
        "file_lengths": list(p.old_file_lengths),
        "file_index": p.file_index,
        "file_position": p.position_old,
        "overwritten_b64": base64.b64encode(p.overwritten).decode("ascii"),
    }
    with path.open("x", encoding="utf-8") as stream:  # never clobber an older backup
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())


def apply(p: Plan, backup: Path) -> None:
    with Artifact(p.entry) as artifact:  # refuse if the artifact changed since planning
        paths = _file_paths(p.entry, artifact.directory)
    _check_lengths(paths, p.old_file_lengths)
    write_backup(p, backup)
    if p.file_index != 0 or not p.rewrite:
        # Payload first: until the head is rewritten, readers see a length mismatch and refuse
        # the artifact rather than reading a half-edited directory.
        _write_exact(paths[p.file_index], p.position_new, p.new_data,
                     p.new_file_lengths[p.file_index])
    if p.rewrite:
        tail = (p.position_new, p.new_data) if p.file_index == 0 else None
        _rebuild_entry(p.entry, p.head, p.old_payload_offset, p.entry_payload_bytes, tail,
                       p.new_file_lengths[0])
    else:
        _write_exact(paths[0], 0, p.head, p.new_file_lengths[0])
    verify(p)


def verify(p: Plan) -> None:
    with Artifact(p.entry) as artifact:
        if artifact.read_object(p.object_id) != p.new_data:
            raise ArtifactError(f"{p.object_id}: read-back differs from the written resource")
        _check_lengths(_file_paths(p.entry, artifact.directory), p.new_file_lengths)


def _read_head(path: Path) -> tuple[bytes, int]:
    fd = os.open(path, os.O_RDONLY | BINARY)
    try:
        raw = pread(fd, HEADER.size, 0)
    finally:
        os.close(fd)
    if len(raw) != HEADER.size or raw[:8] != MAGIC:
        raise ArtifactError(f"{path}: not a NInfer v3 entry")
    _, json_bytes, _ = HEADER.unpack(raw)
    return raw, align_up(HEADER.size + json_bytes, PAYLOAD_ALIGNMENT)


def restore(entry: Path, backup: Path) -> None:
    record = json.loads(backup.read_text(encoding="utf-8"))
    if record.get("format") != BACKUP_FORMAT or record.get("entry") != entry.name:
        raise ArtifactError(f"{backup}: not a set_resource backup for {entry.name}")
    head = base64.b64decode(record["head_b64"])
    lengths = record["file_lengths"]
    raw, current_payload_offset = _read_head(entry)
    if raw[16:32] != head[16:32]:
        raise ArtifactError(f"{backup}: artifact_id does not match {entry}")
    old_directory = parse_directory(json.loads(head[HEADER.size:].decode("utf-8")),
                                    entry_name=entry.name)
    paths = _file_paths(entry, old_directory)
    index = record["file_index"]
    position = record["file_position"]
    overwritten = base64.b64decode(record["overwritten_b64"])
    if index != 0:
        _write_exact(paths[index], position, overwritten, lengths[index])
    if current_payload_offset != len(head):  # the edit rewrote the entry: copy it back
        tail = (position, overwritten) if index == 0 and overwritten else None
        _rebuild_entry(entry, head, current_payload_offset,
                       old_directory.files[0].payload_bytes, tail, lengths[0])
    else:
        if index == 0:
            _write_exact(entry, position, overwritten, lengths[0])
        _write_exact(entry, 0, head, lengths[0])
    _check_lengths(paths, lengths)
    Artifact(entry).close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.artifact.set_resource",
        description="Replace a raw resource (e.g. chat_template.jinja) in a v3 artifact.",
    )
    parser.add_argument("artifact", type=Path, help="entry .ninfer file")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--resource", metavar="ROLE=FILE",
                        help="resource role and replacement file, e.g. chat_template.jinja=fixed.jinja")
    action.add_argument("--restore", type=Path, metavar="BACKUP",
                        help="undo an earlier edit from its backup file")
    parser.add_argument("--component", help="component owning the role (default: the only match)")
    parser.add_argument("--rewrite", action="store_true",
                        help="if the directory no longer fits its reserve, copy the entry file "
                             "once with a larger reserve (needs free space for one entry copy)")
    parser.add_argument("--backup", type=Path,
                        help="backup path (default: <entry>.set_resource-<utc>.json beside it)")
    parser.add_argument("--dry-run", action="store_true", help="plan and report; write nothing")
    args = parser.parse_args(argv)

    try:
        if args.restore:
            if args.dry_run:
                parser.error("--dry-run does not apply to --restore")
            restore(args.artifact, args.restore)
            print(f"restored {args.artifact} from {args.restore}")
            return 0
        role, sep, file = args.resource.partition("=")
        if not sep or not role or not file:
            parser.error("--resource must be ROLE=FILE")
        data = _read_resource_file(Path(file), role)
        now = datetime.now(timezone.utc)
        p = plan(args.artifact, role, data, component=args.component,
                 source_name=Path(file).name, now=now, rewrite=args.rewrite)
        placement = "overwrite tail" if p.new_offset == p.old_offset else "append"
        print(f"{p.component}/{p.role} -> {p.object_id}: {p.old_bytes} B -> {len(data)} B "
              f"({placement} at payload offset {p.new_offset}, file {p.file_index})")
        print(f"directory reserve: {p.free_before} B free before, {p.free_after} B after")
        if p.rewrite:
            print(f"entry rewrite: copying {p.old_file_lengths[0] / 2**30:.1f} GiB to grow the "
                  f"reserve ({p.old_payload_offset} -> {p.new_payload_offset} B)")
        for note in p.notes:
            print(f"note: {note}")
        if p.unchanged:
            print("resource already has these bytes; nothing to do")
            return 0
        if args.dry_run:
            print("dry run: no changes written")
            return 0
        backup = args.backup or args.artifact.with_name(
            f"{args.artifact.name}.set_resource-{now:%Y%m%dT%H%M%SZ}.json")
        apply(p, backup)
        print(f"updated and verified {args.artifact}; backup {backup}")
        return 0
    except (ArtifactError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
