"""Replace one raw resource (e.g. chat_template.jinja) in an existing v3 artifact in place.

Weights are never moved. The new bytes go to the end of the last payload file, or overwrite
the old bytes when the resource already is the last object (so repeated swaps do not grow the
file). The directory JSON is rewritten inside its existing reserve, so the payload offset and
every other object stay where they are. The old bytes are left in place as unreferenced space.

Before any write, a backup of the entry head, every file length and the overwritten bytes is
saved beside the entry; ``--restore BACKUP`` puts the artifact back exactly. Run with the
artifact closed (no engine or server using it).

    python -m tools.artifact.set_resource MODEL.ninfer --resource chat_template.jinja=FILE
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
import sys

from .file_io import BINARY, pread, pwrite
from .framing import HEADER, MAGIC, PAYLOAD_ALIGNMENT
from .reader import Artifact
from .schema import ArtifactError, encode_directory, parse_directory

MAX_RESOURCE_BYTES = 16 * 1024 * 1024
BACKUP_FORMAT = "ninfer-set-resource-backup-v1"


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
    file_position: int  # byte position inside that file
    old_file_lengths: tuple[int, ...]
    new_file_lengths: tuple[int, ...]
    head: bytes  # entry [0, payload_offset) after the edit
    old_head: bytes
    overwritten: bytes  # existing file bytes the new data replaces (reuse case only)
    unchanged: bool


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


def plan(entry: Path, role: str, data: bytes, *, component: str | None = None,
         source_name: str = "", now: datetime | None = None) -> Plan:
    """Compute every byte to be written without touching the artifact."""
    with Artifact(entry) as artifact:
        directory = artifact.directory
        payload_offset = artifact.payload_offset
        old_head = pread(artifact._fds[0], payload_offset, 0)
        component_name, object_id = _resolve_role(directory, role, component)
        obj = artifact.object(object_id)
        if obj.kind != "resource":
            raise ArtifactError(f"{object_id}: not a resource object")
        old_data = artifact.read_object(object_id)
        paths = _file_paths(entry, directory)
        old_lengths = tuple(path.stat().st_size for path in paths)

        files = [file.payload_bytes for file in directory.files]
        last = len(files) - 1
        last_start = sum(files[:last])  # logical offset where the last file begins
        payload_end = sum(files)
        file_header = payload_offset if last == 0 else PAYLOAD_ALIGNMENT
        if obj.offset + obj.bytes == payload_end and obj.offset >= last_start:
            new_offset = obj.offset  # already the tail object: overwrite, no growth
        else:
            new_offset = payload_end
        files[last] = new_offset + len(data) - last_start
        position = file_header + (new_offset - last_start)
        overwritten = b""
        if new_offset == obj.offset:
            overwritten = pread(artifact._fds[0] if last == 0 else artifact._file(last),
                                old_lengths[last] - position, position)

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
        history = list(provenance.get("resource_overrides", []))
        history.append({
            "component": component_name,
            "role": role,
            "object": object_id,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "previous_sha256": hashlib.sha256(old_data).hexdigest(),
            "previous_bytes": len(old_data),
            "source": source_name,
            "utc": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": "tools.artifact.set_resource",
        })
        provenance["resource_overrides"] = history
        value["provenance"] = provenance
        parse_directory(value, entry_name=entry.name)  # same structural checks as a reader

        encoded = encode_directory(value)
        reserve = payload_offset - HEADER.size
        if len(encoded) > reserve:
            raise ArtifactError(
                f"new directory needs {len(encoded)} bytes but the entry reserves {reserve}; "
                "an in-place edit is impossible (re-convert, or rewrite the entry file)"
            )
        head = (
            HEADER.pack(MAGIC, reserve, artifact.artifact_id)
            + encoded
            + b" " * (reserve - len(encoded))
        )
        assert len(head) == payload_offset
        new_lengths = list(old_lengths)
        new_lengths[last] = file_header + files[last]
        return Plan(entry, component_name, role, object_id, obj.offset, obj.bytes,
                    new_offset, data, last, position, old_lengths, tuple(new_lengths),
                    head, old_head, overwritten, old_data == data)


def _write_exact(path: Path, position: int, data: bytes, length: int) -> None:
    fd = os.open(path, os.O_RDWR | BINARY)
    try:
        view = memoryview(data)
        while view:
            count = pwrite(fd, view, position)
            if count <= 0:
                raise OSError(f"{path}: short write at {position}")
            view = view[count:]
            position += count
        os.ftruncate(fd, length)
        os.fsync(fd)
    finally:
        os.close(fd)


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
        "file_position": p.file_position,
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
    # Payload first: until the head is rewritten, readers see a length mismatch and refuse
    # the artifact rather than reading a half-edited directory.
    _write_exact(paths[p.file_index], p.file_position, p.new_data, p.new_file_lengths[p.file_index])
    _write_exact(paths[0], 0, p.head, p.new_file_lengths[0])
    verify(p)


def verify(p: Plan) -> None:
    with Artifact(p.entry) as artifact:
        if artifact.read_object(p.object_id) != p.new_data:
            raise ArtifactError(f"{p.object_id}: read-back differs from the written resource")
        _check_lengths(_file_paths(p.entry, artifact.directory), p.new_file_lengths)


def restore(entry: Path, backup: Path) -> None:
    record = json.loads(backup.read_text(encoding="utf-8"))
    if record.get("format") != BACKUP_FORMAT or record.get("entry") != entry.name:
        raise ArtifactError(f"{backup}: not a set_resource backup for {entry.name}")
    head = base64.b64decode(record["head_b64"])
    lengths = record["file_lengths"]
    raw = pread_file(entry, HEADER.size)
    if raw[16:32] != head[16:32]:
        raise ArtifactError(f"{backup}: artifact_id does not match {entry}")
    old_directory = parse_directory(json.loads(head[HEADER.size:].decode("utf-8")),
                                    entry_name=entry.name)
    paths = _file_paths(entry, old_directory)
    index = record["file_index"]
    overwritten = base64.b64decode(record["overwritten_b64"])
    if overwritten:
        _write_exact(paths[index], record["file_position"], overwritten, lengths[index])
    elif index != 0:
        _write_exact(paths[index], 0, b"", lengths[index])
    _write_exact(paths[0], 0, head, lengths[0])
    _check_lengths(paths, lengths)
    Artifact(entry).close()


def pread_file(path: Path, count: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | BINARY)
    try:
        return pread(fd, count, 0)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.artifact.set_resource",
        description="Replace a raw resource (e.g. chat_template.jinja) in a v3 artifact in place.",
    )
    parser.add_argument("artifact", type=Path, help="entry .ninfer file")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--resource", metavar="ROLE=FILE",
                        help="resource role and replacement file, e.g. chat_template.jinja=fixed.jinja")
    action.add_argument("--restore", type=Path, metavar="BACKUP",
                        help="undo an earlier edit from its backup file")
    parser.add_argument("--component", help="component owning the role (default: the only match)")
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
                 source_name=Path(file).name, now=now)
        placement = "overwrite tail" if p.new_offset == p.old_offset else "append"
        print(f"{p.component}/{p.role} -> {p.object_id}: {p.old_bytes} B -> {len(data)} B "
              f"({placement} at payload offset {p.new_offset}, file {p.file_index})")
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
