#!/usr/bin/env python3
"""Verify and stage the pinned, vendored libfibre source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable


REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(RuntimeError):
    """The manifest or a file tree does not satisfy the vendor contract."""


def _safe_relpath(raw: Any) -> str:
    if not isinstance(raw, str) or not raw:
        raise ManifestError("manifest file path must be a non-empty string")
    if "\\" in raw:
        raise ManifestError(f"manifest path must use POSIX separators: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ManifestError(f"unsafe manifest path: {raw!r}")
    return path.as_posix()


def _manifest_mode(raw: Any) -> int:
    """Return a regular-file mode, accepting common JSON mode encodings."""
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ManifestError("manifest file mode must be an integer or string")
    if isinstance(raw, int):
        value = raw
        if stat.S_IFMT(value) == 0 and 100000 <= value <= 100777:
            value = int(str(value), 8)
    else:
        value = raw.strip()
        if value.startswith("0o"):
            value = int(value, 8)
        elif value.isdigit() and len(value) <= 6:
            value = int(value, 8)
        else:
            value = int(value, 10)
    if value <= 0o777:
        value |= stat.S_IFREG
    if stat.S_IFMT(value) != stat.S_IFREG:
        raise ManifestError(f"manifest mode is not a regular-file mode: {raw!r}")
    return value


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    files = data.get("files")
    if isinstance(files, dict):
        mapped_files = []
        for file_path, entry in files.items():
            if not isinstance(entry, dict):
                raise ManifestError(f"manifest entry for {file_path!r} must be an object")
            mapped_entry = dict(entry)
            mapped_entry["path"] = file_path
            mapped_files.append(mapped_entry)
        files = mapped_files
    if not isinstance(files, list) or not files:
        raise ManifestError("manifest files must be a non-empty list or object")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ManifestError("each manifest file entry must be an object")
        relpath = _safe_relpath(item.get("path"))
        if relpath in seen:
            raise ManifestError(f"duplicate manifest path: {relpath}")
        seen.add(relpath)
        digest = item.get("sha256")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise ManifestError(f"invalid SHA-256 for {relpath}")
        blob = item.get("git_blob_sha1")
        if blob is not None and (
            not isinstance(blob, str) or not REVISION_RE.fullmatch(blob)
        ):
            raise ManifestError(f"invalid Git blob SHA-1 for {relpath}")
        normalized_entry = {
            "path": relpath,
            "sha256": digest.lower(),
            "mode": _manifest_mode(item.get("mode")),
        }
        if blob is not None:
            normalized_entry["git_blob_sha1"] = blob.lower()
        normalized.append(normalized_entry)
    return data, normalized


def _revision_from_manifest(data: dict[str, Any]) -> str | None:
    versions = data.get("versions")
    candidates: list[Any] = []
    if isinstance(versions, dict):
        value = versions.get("libfibre")
        candidates.append(value)
        if isinstance(value, dict):
            candidates.append(value.get("revision"))
    upstream = data.get("upstream")
    if isinstance(upstream, dict):
        candidates.append(upstream.get("commit"))
    candidates.extend((data.get("revision"), data.get("libfibre_revision")))
    for candidate in candidates:
        if isinstance(candidate, str) and REVISION_RE.fullmatch(candidate):
            return candidate.lower()
    return None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _check_tree(root: Path, files: Iterable[dict[str, Any]]) -> None:
    if not root.is_dir():
        raise ManifestError(f"source/destination directory not found: {root}")
    for item in files:
        relpath = item["path"]
        path = root / Path(*PurePosixPath(relpath).parts)
        try:
            path_stat = path.lstat()
        except OSError as exc:
            raise ManifestError(f"missing manifest file {path}: {exc}") from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise ManifestError(f"manifest path is not a regular file: {path}")
        # Git records the executable bit, not group-write bits set by umask
        # or git archive. Staging below restores canonical file permissions.
        mode = stat.S_IFREG | (0o755 if path_stat.st_mode & stat.S_IXUSR else 0o644)
        expected_mode = item["mode"]
        if mode != expected_mode:
            raise ManifestError(
                f"mode mismatch for {path}: expected {oct(expected_mode)}, got {oct(mode)}"
            )
        actual = _digest(path)
        if actual != item["sha256"]:
            raise ManifestError(f"SHA-256 mismatch for {path}")


def _copy_tree(source: Path, destination: Path, files: Iterable[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for item in files:
            relpath = item["path"]
            source_path = source / Path(*PurePosixPath(relpath).parts)
            destination_path = staging / Path(*PurePosixPath(relpath).parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            os.chmod(destination_path, stat.S_IMODE(item["mode"]))
        if destination.exists():
            raise ManifestError(f"refusing to overwrite existing destination: {destination}")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare(
    source: Path,
    destination: Path,
    manifest: Path,
    expected_revision: str | None,
    dry_run: bool,
) -> str:
    data, files = _load_manifest(manifest)
    manifest_revision = _revision_from_manifest(data)
    if expected_revision:
        expected_revision = expected_revision.lower()
        if not REVISION_RE.fullmatch(expected_revision):
            raise ManifestError(f"invalid expected libfibre revision: {expected_revision}")
        if manifest_revision != expected_revision:
            raise ManifestError(
                "manifest libfibre revision mismatch: "
                f"expected {expected_revision}, found {manifest_revision or 'unset'}"
            )
    _check_tree(source, files)
    if destination.exists():
        _check_tree(destination, files)
        return "dry-run: destination matches manifest" if dry_run else "reused existing destination"
    if dry_run:
        return "dry-run: source matches manifest; destination would be copied"
    _copy_tree(source, destination, files)
    _check_tree(destination, files)
    return "copied vendored source to destination"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(args.source, args.dest, args.manifest, args.expected_revision, args.dry_run)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
