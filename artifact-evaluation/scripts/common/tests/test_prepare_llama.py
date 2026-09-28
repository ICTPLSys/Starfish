"""Offline tests for LLaMA structural validation and atomic preparation."""

from pathlib import Path
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

import prepare_llama


def write_model(path: Path, header=prepare_llama.MODEL_HEADER):
    with path.open("wb") as stream:
        stream.write(struct.pack("<7i", *header))
        stream.truncate(prepare_llama.MODEL_SIZE)


def write_tokenizer(path: Path, records=prepare_llama.VOCAB_SIZE,
                    max_token_length=8, extra=b""):
    with path.open("wb") as stream:
        stream.write(struct.pack("<i", max_token_length))
        for index in range(records):
            token = b"a" if index % 2 else b"b"
            stream.write(struct.pack("<f", float(index)))
            stream.write(struct.pack("<i", len(token)))
            stream.write(token)
        stream.write(extra)


class PrepareLlama(unittest.TestCase):
    def test_model_header_and_size_validator_accepts_sparse_structural_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            write_model(path)
            info = prepare_llama.validate_model(path)
            self.assertEqual(info["header"], list(prepare_llama.MODEL_HEADER))
            self.assertEqual(info["size"], prepare_llama.MODEL_SIZE)

    def test_model_validator_rejects_bad_or_truncated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            write_model(path, (1, 2, 3, 4, 5, 6, 7))
            with self.assertRaises(prepare_llama.PreparationError):
                prepare_llama.validate_model(path)
            with path.open("r+b") as stream:
                stream.truncate(20)
            with self.assertRaises(prepare_llama.PreparationError):
                prepare_llama.validate_model(path)

    def test_tokenizer_validator_rejects_truncation_extra_and_bad_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.bin"
            write_tokenizer(path)
            self.assertEqual(
                prepare_llama.validate_tokenizer(path)["records"],
                prepare_llama.VOCAB_SIZE,
            )

            with path.open("r+b") as stream:
                stream.truncate(path.stat().st_size - 1)
            with self.assertRaises(prepare_llama.PreparationError):
                prepare_llama.validate_tokenizer(path)

            write_tokenizer(path, extra=b"x")
            with self.assertRaises(prepare_llama.PreparationError):
                prepare_llama.validate_tokenizer(path)

            with path.open("wb") as stream:
                stream.write(struct.pack("<i", 2))
                for index in range(prepare_llama.VOCAB_SIZE):
                    token = b"abc" if index == 0 else b"a"
                    stream.write(struct.pack("<f", 0.0))
                    stream.write(struct.pack("<i", len(token)))
                    stream.write(token)
            with self.assertRaises(prepare_llama.PreparationError):
                prepare_llama.validate_tokenizer(path)

    def test_prepare_stubs_tools_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "meta"
            tools = root / "tools"
            output = root / "prepared"
            meta.mkdir()
            tools.mkdir()
            (meta / "consolidated.00.pth").write_bytes(b"checkpoint")
            (meta / "params.json").write_text("{}", encoding="utf-8")
            (meta / "tokenizer.model").write_bytes(b"raw-tokenizer")

            (tools / "model.py").write_text("# offline stub model module\n")
            export_file = tools / "export.py"
            export_file.write_text(
                "import pathlib, struct, sys\n"
                "q=pathlib.Path(sys.argv[1])\n"
                "q.write_bytes(struct.pack('<7i',"
                "4096,11008,32,32,32,-32000,2048))\n"
                "with q.open('ab') as f: f.truncate(26954711068)\n",
                encoding="utf-8",
            )
            tokenizer = tools / "tokenizer.py"
            tokenizer.write_text(
                "import struct\n"
                "with open('tokenizer.bin','wb') as f:\n"
                " f.write(struct.pack('<i',8))\n"
                " for i in range(32000): f.write(struct.pack('<fi',0.0,1)); f.write(b'a')\n",
                encoding="utf-8",
            )

            first = prepare_llama.prepare(meta, tools, output, sys.executable)
            self.assertFalse(first["reused"])
            self.assertTrue((output / prepare_llama.MODEL_FILENAME).is_file())
            self.assertTrue((output / prepare_llama.TOKENIZER_FILENAME).is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertFalse(list(root.glob(".prepared.*")))

            with mock.patch.object(
                    prepare_llama.subprocess, "run",
                    side_effect=AssertionError("reuse must not invoke tools")):
                second = prepare_llama.prepare(meta, tools, output, sys.executable)
            self.assertTrue(second["reused"])
            self.assertEqual(second["source"], first["source"])
            self.assertEqual(first["model"]["path"], str(output / prepare_llama.MODEL_FILENAME))
            with (output / prepare_llama.MODEL_FILENAME).open("r+b") as stream:
                stream.seek(40)
                stream.write(b"changed")
            with self.assertRaises(prepare_llama.PreparationError):
                prepare_llama.prepare(meta, tools, output, sys.executable)



if __name__ == "__main__":
    unittest.main()
