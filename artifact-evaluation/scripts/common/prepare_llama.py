#!/usr/bin/env python3
"""Prepare and validate the fixed Starfish LLaMA input artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MODEL_FILENAME = "llama2_7b_chat.bin"
TOKENIZER_FILENAME = "tokenizer.bin"
MODEL_SIZE = 26_954_711_068
MODEL_HEADER = (4096, 11008, 32, 32, 32, -32000, 2048)
VOCAB_SIZE = 32_000
MANIFEST_VERSION = 1


class PreparationError(RuntimeError):
    """The source/tool/output contract is invalid."""


def _stat_fingerprint(path: Path) -> Dict[str, Any]:
    try:
        value = path.stat()
    except OSError as exc:
        raise PreparationError(f"cannot stat {path}: {exc}") from exc
    if not path.is_file():
        raise PreparationError(f"expected a regular file: {path}")
    return {
        "path": str(path),
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "mode": value.st_mode & 0o777,
    }


def _source_file(meta_dir: Path, name: str) -> Path:
    path = meta_dir / name
    if not path.is_file():
        raise PreparationError(f"missing Meta source file: {path}")
    return path


def _meta_sources(meta_dir: Path) -> List[Path]:
    if not meta_dir.is_dir():
        raise PreparationError(f"Meta checkpoint directory not found: {meta_dir}")
    checkpoints = sorted(meta_dir.glob("consolidated*.pth"))
    if [path.name for path in checkpoints] != ["consolidated.00.pth"]:
        raise PreparationError("Llama 2 7B requires exactly consolidated.00.pth")
    return [checkpoints[0], _source_file(meta_dir, "params.json"),
            _source_file(meta_dir, "tokenizer.model")]


def _tool_sources(tools_dir: Path) -> List[Path]:
    if not tools_dir.is_dir():
        raise PreparationError(f"tool directory not found: {tools_dir}")
    return [_source_file(tools_dir, "export.py"),
            _source_file(tools_dir, "model.py"),
            _source_file(tools_dir, "tokenizer.py")]


def validate_model(path: Path) -> Dict[str, Any]:
    """Validate the fixed LLaMA binary without hashing its large payload."""
    try:
        value = path.stat()
    except OSError as exc:
        raise PreparationError(f"cannot stat model {path}: {exc}") from exc
    if value.st_size != MODEL_SIZE:
        raise PreparationError(
            f"model size mismatch: expected {MODEL_SIZE}, got {value.st_size}"
        )
    try:
        with path.open("rb") as stream:
            raw = stream.read(28)
    except OSError as exc:
        raise PreparationError(f"cannot read model header {path}: {exc}") from exc
    if len(raw) != 28:
        raise PreparationError("model header is truncated")
    header = struct.unpack("<7i", raw)
    if header != MODEL_HEADER:
        raise PreparationError(
            f"model header mismatch: expected {MODEL_HEADER}, got {header}"
        )
    return {
        "path": str(path),
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "header": list(header),
    }


def _read_exact(stream: Any, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise PreparationError(f"tokenizer {label} is truncated")
    return data


def validate_tokenizer(path: Path, vocab_size: int = VOCAB_SIZE) -> Dict[str, Any]:
    """Validate the C tokenizer.bin record stream and EOF."""
    if vocab_size <= 0:
        raise PreparationError("vocab_size must be positive")
    try:
        value = path.stat()
    except OSError as exc:
        raise PreparationError(f"cannot stat tokenizer {path}: {exc}") from exc
    max_token_length: Optional[int] = None
    observed_max = 0
    records = 0
    try:
        with path.open("rb") as stream:
            max_token_length = struct.unpack(
                "<i", _read_exact(stream, 4, "max_token_length")
            )[0]
            if max_token_length <= 0:
                raise PreparationError("tokenizer max_token_length must be positive")
            for index in range(vocab_size):
                _read_exact(stream, 4, f"score record {index}")
                length = struct.unpack(
                    "<i", _read_exact(stream, 4, f"length record {index}")
                )[0]
                if length < 0 or length > max_token_length:
                    raise PreparationError(
                        f"tokenizer record {index} length {length} exceeds "
                        f"max_token_length {max_token_length}"
                    )
                _read_exact(stream, length, f"token record {index}")
                observed_max = max(observed_max, length)
                records += 1
            if stream.read(1):
                raise PreparationError("tokenizer contains bytes after 32000 records")
    except PreparationError:
        raise
    except (OSError, struct.error) as exc:
        raise PreparationError(f"cannot parse tokenizer {path}: {exc}") from exc
    return {
        "path": str(path),
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "records": records,
        "max_token_length": max_token_length,
        "observed_max_token_length": observed_max,
    }


def _run_tool(command: Sequence[str], cwd: Path) -> None:
    try:
        subprocess.run(list(command), cwd=str(cwd), check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError(
            "tool failed: " + " ".join(str(item) for item in command)
        ) from exc


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _load_existing(output_dir: Path, source_fingerprints: List[Dict[str, Any]],
                   tool_fingerprints: List[Dict[str, Any]], python_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    manifest_path = _manifest_path(output_dir)
    if not output_dir.is_dir() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read existing manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_VERSION:
        raise PreparationError(f"unsupported existing manifest: {manifest_path}")
    if (manifest.get("source") != source_fingerprints
            or manifest.get("tools") != tool_fingerprints
            or manifest.get("python") != python_info):
        raise PreparationError(
            f"refusing to overwrite output with different sources: {output_dir}"
        )
    model = validate_model(output_dir / MODEL_FILENAME)
    tokenizer = validate_tokenizer(output_dir / TOKENIZER_FILENAME)
    if manifest.get("model") != model or manifest.get("tokenizer") != tokenizer:
        raise PreparationError(f"prepared model/tokenizer changed: {output_dir}")
    manifest["reused"] = True
    return manifest


def prepare(meta_dir: Path, tools_dir: Path, output_dir: Path,
            python_bin: str) -> Dict[str, Any]:
    """Convert Meta LLaMA input and atomically publish validated artifacts."""
    meta_dir = Path(meta_dir).resolve()
    tools_dir = Path(tools_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not python_bin:
        raise PreparationError("python_bin must be non-empty")
    sources = _meta_sources(meta_dir)
    tools = _tool_sources(tools_dir)
    source_fingerprints = [_stat_fingerprint(path) for path in sources]
    tool_fingerprints = [_stat_fingerprint(path) for path in tools]
    python_info = _stat_fingerprint(Path(python_bin).absolute())
    existing = _load_existing(output_dir, source_fingerprints, tool_fingerprints, python_info)
    if existing is not None:
        return existing

    if output_dir.exists():
        raise PreparationError(f"refusing unverified existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", dir=str(output_dir.parent)
    ))
    tokenizer_work = staging / "tokenizer-work"
    tokenizer_work.mkdir()
    try:
        _run_tool([
            python_bin, str(tools_dir / "export.py"), str(staging / MODEL_FILENAME),
            "--version", "0", "--meta-llama", str(meta_dir)
        ], staging)

        copied_tokenizer = tokenizer_work / "tokenizer.model"
        shutil.copy2(meta_dir / "tokenizer.model", copied_tokenizer)
        _run_tool([
            python_bin, str(tools_dir / "tokenizer.py"),
            "--tokenizer-model", str(copied_tokenizer)
        ], tokenizer_work)
        os.replace(tokenizer_work / "tokenizer.bin", staging / TOKENIZER_FILENAME)
        shutil.rmtree(tokenizer_work)

        model_info = validate_model(staging / MODEL_FILENAME)
        tokenizer_info = validate_tokenizer(staging / TOKENIZER_FILENAME)
        model_info["path"] = str(output_dir / MODEL_FILENAME)
        tokenizer_info["path"] = str(output_dir / TOKENIZER_FILENAME)
        manifest = {
            "python": python_info,
            "schema_version": MANIFEST_VERSION,
            "model": model_info,
            "tokenizer": tokenizer_info,
            "source": source_fingerprints,
            "tools": tool_fingerprints,
            "python_bin": python_bin,
            "reused": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise PreparationError(f"refusing to overwrite output: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare(args.meta_dir, args.tools_dir, args.output_dir, args.python_bin)
    except (OSError, PreparationError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
