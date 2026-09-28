#!/usr/bin/env python3
"""Offline tests for the historical Friendster 32-way preparation helper."""

from __future__ import annotations

import gzip
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import prepare_friendster  # noqa: E402


def _raw_graph(final_newline: bool = True) -> bytes:
    lines = [
        b"# Undirected graph: fixture\n",
        b"# Friendster\n",
        b"# Nodes: 32 Edges: 28\n",
        b"# FromNodeId\tToNodeId\n",
    ]
    lines.extend(f"{index} {index + 1}\n".encode("ascii") for index in range(28))
    raw = b"".join(lines)
    return raw if final_newline else raw[:-1]


def _write_gzip(path: Path, raw: bytes) -> None:
    with gzip.open(path, "wb") as stream:
        stream.write(raw)


class PrepareFriendsterTest(unittest.TestCase):
    def test_32_complete_lines_preserve_bytes_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "graph.gz"
            output = root / "friendster"
            raw = _raw_graph()
            _write_gzip(source, raw)
            manifest = prepare_friendster.prepare(
                source, output, expected_lines=32,
                expected_uncompressed_bytes=len(raw)
            )
            names = [f"graph.{index}" for index in range(32)]
            self.assertEqual(manifest["prefix"], str((output / "graph").resolve()))
            self.assertEqual(manifest["total_lines"], 32)
            self.assertEqual(manifest["size"], len(raw))
            self.assertEqual(
                b"".join((output / name).read_bytes() for name in names), raw
            )
            self.assertEqual([entry["path"] for entry in manifest["outputs"]], names)
            self.assertEqual(
                manifest["preprocessing"]["header_first4"],
                [
                    "# Undirected graph: fixture",
                    "# Friendster",
                    "# Nodes: 32 Edges: 28",
                    "# FromNodeId\tToNodeId",
                ],
            )
            self.assertTrue((output / "manifest.json").is_file())

    def test_no_final_newline_keeps_bytes_and_historical_wc_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "graph.gz"
            output = root / "friendster"
            raw = _raw_graph(final_newline=False)
            _write_gzip(source, raw)
            manifest = prepare_friendster.prepare(
                source, output, expected_lines=31,
                expected_uncompressed_bytes=len(raw)
            )
            self.assertEqual(manifest["total_lines"], 31)
            self.assertEqual(
                b"".join(
                    (output / f"graph.{index}").read_bytes()
                    for index in range(32)
                ),
                raw,
            )

    def test_bad_gzip_leaves_no_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "bad.gz"
            output = root / "friendster"
            source.write_bytes(b"not a gzip stream")
            with self.assertRaises(prepare_friendster.FriendsterPrepareError):
                prepare_friendster.prepare(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".friendster.*")), [])

    def test_non_32_way_input_is_rejected_without_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "graph.gz"
            output = root / "friendster"
            raw = b"".join(f"{index}\n".encode("ascii") for index in range(31))
            _write_gzip(source, raw)
            with self.assertRaises(prepare_friendster.FriendsterPrepareError):
                prepare_friendster.prepare(
                    source, output, expected_lines=31,
                    expected_uncompressed_bytes=len(raw)
                )
            self.assertFalse(output.exists())

    def test_wrong_expected_size_is_rejected_without_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "graph.gz"
            output = root / "friendster"
            raw = _raw_graph()
            _write_gzip(source, raw)
            with self.assertRaises(prepare_friendster.FriendsterPrepareError):
                prepare_friendster.prepare(
                    source, output, expected_lines=32,
                    expected_uncompressed_bytes=len(raw) + 1
                )
            self.assertFalse(output.exists())

    def test_rerun_reuses_and_never_overwrites_unverified_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "graph.gz"
            output = root / "friendster"
            raw = _raw_graph()
            _write_gzip(source, raw)
            first = prepare_friendster.prepare(
                source, output, expected_lines=32,
                expected_uncompressed_bytes=len(raw)
            )
            shard = output / "graph.0"
            shard_bytes = shard.read_bytes()
            shard_mtime = shard.stat().st_mtime_ns
            second = prepare_friendster.prepare(
                source, output, expected_lines=32,
                expected_uncompressed_bytes=len(raw)
            )
            self.assertEqual(first, second)
            self.assertEqual(shard.stat().st_mtime_ns, shard_mtime)
            time.sleep(0.001)
            shard.touch()
            with self.assertRaises(prepare_friendster.ExistingOutputError):
                prepare_friendster.prepare(
                    source, output, expected_lines=32,
                    expected_uncompressed_bytes=len(raw)
                )
            self.assertEqual(shard.read_bytes(), shard_bytes)


if __name__ == "__main__":
    unittest.main()
