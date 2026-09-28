#!/usr/bin/env python3
"""Generate the deterministic full-dictionary Wordcount correctness fixture."""

import argparse
import hashlib
import json
from pathlib import Path


def build(output: Path, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "input.txt", output / "expected-counts.json", output / "manifest.json"]
    if not force:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise SystemExit("refusing to overwrite: " + ", ".join(map(str, existing)))

    counts = {}
    words = []
    for index in range(65536):
        value = index
        suffix = ""
        for _ in range(6):
            suffix = chr(ord("a") + value % 26) + suffix
            value //= 26
        word = "word" + suffix
        counts[word] = index % 5 + 1
        words.extend([word] * counts[word])
    words.sort(key=lambda word: hashlib.sha256(word.encode()).digest())
    input_path = output / "input.txt"
    input_path.write_bytes(("\n".join(words) + "\n").encode())
    (output / "expected-counts.json").write_text(
        json.dumps(counts, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "unique_words": len(counts),
        "total_words": sum(counts.values()),
        "bytes": input_path.stat().st_size,
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "description": "deterministic full-dictionary functional fixture, not a performance baseline",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output, args.force)


if __name__ == "__main__":
    main()
