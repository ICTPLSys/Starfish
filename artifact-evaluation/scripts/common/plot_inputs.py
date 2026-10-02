#!/usr/bin/env python3
"""Resolve the newest measured artifact-evaluation plotting input."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN_COMMANDS = {
    "figure9": "bash scripts/figure9/run.sh",
    "figure10": "bash scripts/figure10/run.sh",
    "figure11": "bash scripts/figure9/run.sh",
    "figure12": "bash scripts/figure12/run.sh",
    "figure13": "bash scripts/figure13/run.sh",
}


def fail(figure: str, detail: str) -> "NoReturn":
    raise SystemExit(
        f"plot input error ({figure}): {detail}\n"
        f"Run this to produce the required measured result: {RUN_COMMANDS[figure]}"
    )


def latest_batch(parent: Path, figure: str) -> Path:
    if not parent.is_dir():
        fail(figure, f"result directory is missing: {parent}")
    def is_batch(path: Path) -> bool:
        if not path.is_dir():
            return False
        if figure in ("figure9", "figure11"):
            return ((path / "batch.log").is_file()
                    and (path / "runs").is_dir()
                    and (path / "logs").is_dir())
        return (path / "batch-plan.json").is_file()

    batches = sorted(
        (path for path in parent.iterdir() if is_batch(path)),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not batches:
        fail(figure, f"no batch directory exists under {parent}")
    return batches[0]


def measured_csv(path: Path, required: set[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                return False
            rows = []
            for row in reader:
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                source_type = str(row.get("source_type") or "measured").strip()
                if source_type != "measured" or not str(row.get("source") or "").strip():
                    return False
                rows.append(row)
            return bool(rows)
    except (OSError, UnicodeError, csv.Error):
        return False


def resolve(figure: str) -> Path:
    if figure == "figure9":
        batch = latest_batch(ROOT / "results/figure9", figure)
        path = batch / "figure9.csv"
        if not measured_csv(path, {"workload", "system", "ratio", "elapsed_s"}):
            fail(figure, f"newest batch has no measured CSV: {batch}")
        return path
    if figure == "figure10":
        batch = latest_batch(ROOT / "results/figure10", figure)
        candidates = [path for path in batch.glob("figure10*.csv")
                      if path.name == "figure10.csv" or
                      (path.name.startswith("figure10-r") and path.name.endswith(".csv"))]
        candidates = sorted(
            (path for path in candidates
             if measured_csv(path, {"workload", "system", "offered_load", "p99_latency"})),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        if not candidates:
            fail(figure, f"newest batch has no measured CSV: {batch}")
        return candidates[0]
    if figure == "figure11":
        batch = latest_batch(ROOT / "results/figure9", figure)
        runs = batch / "runs"
        if not runs.is_dir() or not any(runs.rglob("*.log")):
            fail(figure, f"newest Figure 9 batch has no raw run logs: {batch}")
        return runs
    if figure == "figure12":
        batch = latest_batch(ROOT / "results/figure12", figure)
        if not (batch / "batch-plan.json").is_file() or not (batch / "runs").is_dir():
            fail(figure, f"newest batch is not an indexed result: {batch}")
        return batch
    if figure == "figure13":
        batch = latest_batch(ROOT / "results/figure13", figure)
        manifest = batch / "runs.json"
        if not manifest.is_file() or manifest.stat().st_size == 0:
            fail(figure, f"newest batch has no runs.json: {batch}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            fail(figure, f"cannot read newest runs.json: {exc}")
        if payload.get("schema") != "figure13-native-runs-v1" or not payload.get("runs"):
            fail(figure, f"newest runs.json has no measured native runs: {manifest}")
        return manifest
    raise SystemExit("usage: plot_inputs.py figure9|figure10|figure11|figure12|figure13")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_inputs.py figure9|figure10|figure11|figure12|figure13")
    print(resolve(sys.argv[1]).resolve())
