#!/usr/bin/env python3
"""Prepare the historical 32-way Friendster edge-list input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

PART_COUNT = 32
PART_SUFFIX_WIDTH = 2
MANIFEST_NAME = "manifest.json"
CHEAP_FINGERPRINT_BYTES = 64 * 1024

# Observed official SNAP Friendster values.  The production caller should
# pass them as expectations; this generic helper does not silently assume a
# 32 GB input for every fixture.
FRIENDSTER_EDGE_COUNT = 1_806_067_135
FRIENDSTER_HEADER_LINES = 4
FRIENDSTER_EXPECTED_LINES = FRIENDSTER_EDGE_COUNT + FRIENDSTER_HEADER_LINES
FRIENDSTER_EXPECTED_UNCOMPRESSED_BYTES = 32_364_651_776


class FriendsterPrepareError(RuntimeError):
    """The source or output is not trustworthy."""


class ExistingOutputError(FriendsterPrepareError):
    """An existing output cannot be proven reusable."""


def _source(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FriendsterPrepareError(f"source is not readable: {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise FriendsterPrepareError(f"source is not a regular file: {path}")
    return path


def _basename(source: Path) -> str:
    name = source.name
    for suffix in (".gz", ".gzip", ".tgz"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    if not name:
        raise FriendsterPrepareError(f"source basename is empty: {source}")
    return name


def _cheap_fingerprint(source: Path) -> str:
    size = source.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with source.open("rb") as stream:
        digest.update(stream.read(CHEAP_FINGERPRINT_BYTES))
        if size > CHEAP_FINGERPRINT_BYTES:
            stream.seek(size - CHEAP_FINGERPRINT_BYTES)
            digest.update(stream.read(CHEAP_FINGERPRINT_BYTES))
    return digest.hexdigest()


def _source_metadata(source: Path) -> Dict[str, Any]:
    info = source.stat()
    return {
        "path": str(source),
        "basename": _basename(source),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "cheap_fingerprint": _cheap_fingerprint(source),
    }


def _run(args: Sequence[str], stdin: Any = None, stdout: Any = subprocess.PIPE) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(args), stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, check=True
        )
    except FileNotFoundError as exc:
        raise FriendsterPrepareError(f"required command is unavailable: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise FriendsterPrepareError(
            f"command failed ({exc.returncode}): {' '.join(args)}"
            + (f": {detail}" if detail else "")
        ) from exc


def _wc_lines(path: Path) -> int:
    with path.open("rb") as stream:
        result = _run(["wc", "-l"], stdin=stream)
    fields = result.stdout.decode("ascii", "strict").split()
    if not fields or not fields[0].isdigit():
        raise FriendsterPrepareError(f"unexpected wc -l output: {path}")
    return int(fields[0])


def _first_header(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("rb") as stream:
        for _ in range(FRIENDSTER_HEADER_LINES):
            line = stream.readline()
            if not line:
                break
            lines.append(line.decode("utf-8", "replace").rstrip("\r\n"))
    return lines


def _expect(value: Optional[int], name: str) -> None:
    if value is not None and (isinstance(value, bool) or value < 0):
        raise FriendsterPrepareError(f"{name} must be a non-negative integer")


def _check_totals(
    total_lines: int,
    size: int,
    expected_lines: Optional[int],
    expected_size: Optional[int],
) -> None:
    _expect(expected_lines, "expected_lines")
    _expect(expected_size, "expected_uncompressed_bytes")
    if expected_lines is not None and total_lines != expected_lines:
        raise FriendsterPrepareError(
            f"line-count mismatch: expected {expected_lines}, got {total_lines}"
        )
    if expected_size is not None and size != expected_size:
        raise FriendsterPrepareError(
            f"uncompressed-size mismatch: expected {expected_size}, got {size}"
        )


def _names(prefix: str) -> List[str]:
    return [f"{prefix}.{index}" for index in range(PART_COUNT)]


def _verify_existing(
    output: Path,
    source: Path,
    source_meta: Dict[str, Any],
    expected_lines: Optional[int],
    expected_size: Optional[int],
) -> Dict[str, Any]:
    if not output.is_dir() or output.is_symlink():
        raise ExistingOutputError(f"refusing existing non-directory output: {output}")
    manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ExistingOutputError(f"existing output has no {MANIFEST_NAME}: {output}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExistingOutputError(f"cannot read {manifest_path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ExistingOutputError(f"malformed manifest: {manifest_path}")

    preprocessing = data.get("preprocessing")
    prefix = preprocessing.get("prefix_name") if isinstance(preprocessing, dict) else None
    if prefix != "graph" or preprocessing.get("parts") != PART_COUNT:
        raise ExistingOutputError(f"manifest is not the 32-way contract: {manifest_path}")
    if data.get("prefix") != str((output / prefix).resolve()):
        raise ExistingOutputError(f"manifest prefix mismatch: {manifest_path}")

    recorded_source = data.get("source")
    if not isinstance(recorded_source, dict):
        raise ExistingOutputError(f"manifest has no source metadata: {manifest_path}")
    for key in ("size", "mtime_ns", "cheap_fingerprint"):
        if recorded_source.get(key) != source_meta[key]:
            raise ExistingOutputError(f"source metadata changed ({key}): {output}")
    if recorded_source.get("basename") != _basename(source):
        raise ExistingOutputError(f"source basename changed: {output}")

    total_lines = data.get("total_lines")
    size = data.get("size")
    if not isinstance(total_lines, int) or not isinstance(size, int):
        raise ExistingOutputError(f"manifest totals are malformed: {manifest_path}")
    _check_totals(total_lines, size, expected_lines, expected_size)

    names = _names(prefix)
    outputs = data.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != PART_COUNT:
        raise ExistingOutputError(f"manifest does not list 32 outputs: {manifest_path}")
    if any(not isinstance(item, dict) for item in outputs):
        raise ExistingOutputError(f"manifest output entries are malformed: {manifest_path}")
    if [item.get("path") for item in outputs] != names:
        raise ExistingOutputError(f"manifest output names are invalid: {manifest_path}")
    if sorted(item.name for item in output.iterdir()) != sorted(names + [MANIFEST_NAME]):
        raise ExistingOutputError(f"existing output contains unexpected files: {output}")

    summed = 0
    for name, entry in zip(names, outputs):
        shard = output / name
        try:
            shard_stat = shard.lstat()
        except OSError as exc:
            raise ExistingOutputError(f"missing shard {shard}: {exc}") from exc
        if not stat.S_ISREG(shard_stat.st_mode):
            raise ExistingOutputError(f"shard is not a regular file: {shard}")
        if entry.get("size") != shard_stat.st_size:
            raise ExistingOutputError(f"shard size changed: {shard}")
        if entry.get("mtime_ns") != shard_stat.st_mtime_ns:
            raise ExistingOutputError(f"shard mtime changed: {shard}")
        summed += shard_stat.st_size
    if summed != size:
        raise ExistingOutputError(f"shard sizes do not sum to source size: {output}")
    return data


def _rename_parts(staging: Path, prefix: str) -> List[str]:
    part_prefix = f"{prefix}.part"
    generated = [item for item in staging.iterdir() if item.name.startswith(part_prefix)]
    wanted = [f"{part_prefix}{index:0{PART_SUFFIX_WIDTH}d}" for index in range(PART_COUNT)]
    if sorted(item.name for item in generated) != sorted(wanted):
        found = ", ".join(sorted(item.name for item in generated))
        raise FriendsterPrepareError(
            f"historical split did not create exactly 32 parts; got: {found}"
        )
    names = _names(prefix)
    for index, name in enumerate(names):
        os.rename(staging / wanted[index], staging / name)
    return names


def _make_manifest(
    output: Path,
    source_meta: Dict[str, Any],
    prefix: str,
    total_lines: int,
    size: int,
    lines_per_part: int,
    header: List[str],
    names: List[str],
    staging: Path,
) -> Dict[str, Any]:
    outputs = []
    for name in names:
        info = (staging / name).stat()
        outputs.append({"path": name, "size": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)})
    return {
        "schema_version": 1,
        "prefix": str((output / prefix).resolve()),
        "source": source_meta,
        "preprocessing": {
            "method": "gzip -dc; wc -l; split -l ceil(total_lines/32) -d -a 2; rename .partNN to .N",
            "parts": PART_COUNT,
            "prefix_name": prefix,
            "lines_per_part": lines_per_part,
            "header_first4": header,
        },
        "total_lines": total_lines,
        "size": size,
        "uncompressed_bytes": size,
        "outputs": outputs,
    }


def _publish(staging: Path, output: Path) -> None:
    try:
        os.rename(str(staging), str(output))
    except OSError as exc:
        if os.path.lexists(str(output)):
            raise ExistingOutputError(
                f"refusing to overwrite output created during preparation: {output}"
            ) from exc
        raise FriendsterPrepareError(f"cannot atomically publish {output}: {exc}") from exc


def prepare(
    source_gz: Path,
    output_dir: Path,
    expected_lines: Optional[int] = None,
    expected_uncompressed_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare and atomically publish a 32-shard graph.

    The returned manifest has an absolute prefix suitable for site.inputs.bfs.
    Existing output is reused only when source metadata and all recorded shard
    size/mtime values match; an unverified output is never overwritten.
    """

    source = _source(Path(source_gz))
    output = Path(output_dir).expanduser().resolve()
    source_meta = _source_metadata(source)
    prefix = "graph"
    if output.exists() or os.path.lexists(str(output)):
        return _verify_existing(
            output, source, source_meta, expected_lines, expected_uncompressed_bytes
        )

    _expect(expected_lines, "expected_lines")
    _expect(expected_uncompressed_bytes, "expected_uncompressed_bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        decompressed = staging / "_decompressed"
        with decompressed.open("wb") as stream:
            _run(["gzip", "-dc", str(source)], stdout=stream)
        total_lines = _wc_lines(decompressed)
        size = decompressed.stat().st_size
        _check_totals(total_lines, size, expected_lines, expected_uncompressed_bytes)
        header = _first_header(decompressed)
        lines_per_part = max(1, (total_lines + PART_COUNT - 1) // PART_COUNT)
        _run(
            [
                "split",
                "-l",
                str(lines_per_part),
                "-d",
                "-a",
                str(PART_SUFFIX_WIDTH),
                str(decompressed),
                str(staging / f"{prefix}.part"),
            ],
            stdout=subprocess.PIPE,
        )
        names = _rename_parts(staging, prefix)
        decompressed.unlink()
        manifest = _make_manifest(
            output, source_meta, prefix, total_lines, size, lines_per_part,
            header, names, staging
        )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output)
        staging = None
        return manifest
    except FriendsterPrepareError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise FriendsterPrepareError(f"Friendster preparation failed: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_gz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-lines", type=int)
    parser.add_argument("--expected-uncompressed-bytes", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare(
            args.source_gz,
            args.output_dir,
            args.expected_lines,
            args.expected_uncompressed_bytes,
        )
    except FriendsterPrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
