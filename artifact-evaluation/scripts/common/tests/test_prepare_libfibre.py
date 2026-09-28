#!/usr/bin/env python3
"""Offline tests for the vendored libfibre staging helper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import prepare_libfibre  # noqa: E402


REVISION = "0123456789abcdef0123456789abcdef01234567"


def _manifest(source: Path, files: list[tuple[str, int]]) -> dict:
    entries = {}
    for relpath, mode in files:
        path = source / relpath
        entries[relpath] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mode": format(mode, "o"),
        }
    return {
        "schema_version": 1,
        "upstream": {"commit": REVISION},
        "files": entries,
    }


class PrepareLibfibreTest(unittest.TestCase):
    def test_shared_umask_modes_use_git_executable_bit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root / "source", root / "destination"
            source.mkdir()
            (source / "file").write_text("source")
            (source / "run.sh").write_text("#!/bin/sh\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest(
                source, [("file", 0o644), ("run.sh", 0o755)])))
            (source / "file").chmod(0o664)
            (source / "run.sh").chmod(0o775)
            prepare_libfibre.prepare(source, destination, manifest, REVISION, False)
            self.assertEqual((destination / "file").stat().st_mode & 0o777, 0o644)
            self.assertEqual((destination / "run.sh").stat().st_mode & 0o777, 0o755)
            (source / "run.sh").chmod(0o664)
            with self.assertRaises(prepare_libfibre.ManifestError):
                prepare_libfibre.prepare(source, destination, manifest, REVISION, True)

    def test_copy_and_reuse_without_git_preserves_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "third_party" / "libfibre"
            destination = root / "deps" / "libfibre"
            manifest_path = root / "libfibre.vendor.json"
            (source / "src").mkdir(parents=True)
            (source / "src" / "Makefile").write_text("all:\n", encoding="utf-8")
            (source / "src" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(source / "src" / "Makefile", 0o644)
            os.chmod(source / "src" / "run.sh", 0o755)
            source_mtime = 1_700_000_000
            os.utime(source / "src" / "Makefile", (source_mtime, source_mtime))
            files = [("src/Makefile", 0o644), ("src/run.sh", 0o755)]
            manifest_path.write_text(json.dumps(_manifest(source, files)), encoding="utf-8")

            self.assertIn("copied", prepare_libfibre.prepare(
                source, destination, manifest_path, REVISION, False
            ))
            self.assertEqual((destination / "src" / "Makefile").read_text(), "all:\n")
            self.assertEqual((destination / "src" / "run.sh").stat().st_mode & 0o777, 0o755)
            self.assertEqual(
                (destination / "src" / "Makefile").stat().st_mtime_ns,
                (source / "src" / "Makefile").stat().st_mtime_ns,
            )
            (destination / "src" / "libfibre.so").write_bytes(b"generated")
            self.assertIn("reused", prepare_libfibre.prepare(
                source, destination, manifest_path, REVISION, False
            ))

    def test_mismatch_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "destination"
            manifest_path = root / "manifest.json"
            source.mkdir()
            destination.mkdir()
            (source / "file").write_bytes(b"expected")
            (destination / "file").write_bytes(b"user change")
            manifest_path.write_text(
                json.dumps(_manifest(source, [("file", 0o644)])), encoding="utf-8"
            )
            with self.assertRaises(prepare_libfibre.ManifestError):
                prepare_libfibre.prepare(source, destination, manifest_path, REVISION, False)
            self.assertEqual((destination / "file").read_bytes(), b"user change")

    def test_source_hash_mismatch_fails_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "destination"
            manifest_path = root / "manifest.json"
            source.mkdir()
            (source / "file").write_bytes(b"original")
            manifest_path.write_text(
                json.dumps(_manifest(source, [("file", 0o644)])), encoding="utf-8"
            )
            (source / "file").write_bytes(b"changed")
            with self.assertRaises(prepare_libfibre.ManifestError):
                prepare_libfibre.prepare(source, destination, manifest_path, REVISION, False)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
