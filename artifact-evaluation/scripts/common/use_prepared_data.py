#!/usr/bin/env python3
"""Register the shared, already-prepared AE inputs without downloading data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from prepare_data import ROOT, SOURCES, register_inputs
from prepare_llama import validate_model, validate_tokenizer

TOKENIZER_SHA256 = "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
WC_BYTES = 8_053_063_696


def readable(path):
    with path.open("rb") as stream:
        if not stream.read(1):
            raise ValueError(f"empty prepared input: {path}")
    return path.stat().st_size


def inspect_prepared(directory):
    """Cheap startup validation; do not rescan tens of GB on every run."""
    directory = Path(directory).expanduser().absolute()
    model = directory / "llama/llama2_7b_chat.bin"
    tokenizer = directory / "llama/tokenizer.bin"
    validate_model(model)
    validate_tokenizer(tokenizer)
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError(f"unexpected Llama 2 tokenizer: {tokenizer}")

    prefix = directory / "bfs/graph"
    sizes = [readable(Path(f"{prefix}.{index}")) for index in range(32)]
    expected = SOURCES["bfs"]
    if sum(sizes) != expected["uncompressed_size"]:
        raise ValueError("Friendster's 32 shards do not have the expected total byte size")
    with Path(f"{prefix}.0").open("rb") as stream:
        header = b"".join(stream.readline() for _ in range(4))
    marker = f"Nodes: {expected['nodes']} Edges: {expected['edges']}".encode()
    if b"Friendster" not in header or marker not in header:
        raise ValueError("the graph header is not the complete Friendster dataset")
    inputs = {"llama": str(model), "llama_tokenizer": str(tokenizer), "bfs": str(prefix)}
    wordcount = directory / "wordcount/enwiki-small-8g.txt"
    if wordcount.exists():
        if readable(wordcount) != WC_BYTES:
            raise ValueError(f"unexpected WordCount input size: {wordcount}")
        inputs["wordcount"] = str(wordcount)
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get(
                            "AE_PREPARED_DATA_DIR", ROOT / "data/inputs")))
    parser.add_argument("--site", type=Path, default=ROOT / "data/site.json")
    args = parser.parse_args(argv)
    inputs = inspect_prepared(args.data_dir)
    register_inputs(args.site.expanduser().absolute(), inputs)
    print(json.dumps({"prepared_data": str(args.data_dir), "inputs": inputs}, indent=2))
    print(f"Registered prepared inputs in {args.site}")
    print("No downloads or dataset writes. Set the site's SSH/RDMA addresses before running.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"prepared input check failed: {error}", file=sys.stderr)
        raise SystemExit(2)
