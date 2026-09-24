#!/usr/bin/env python3
"""Collect verified Figure 9 runs into a measured-only, sparse CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

FIELDS = ("workload", "system", "ratio", "elapsed_s", "source_type",
          "source", "run_id", "exit_status", "correctness",
          "environment", "measurement_phase")
SYSTEMS = {"Non-FT", "Starfish", "Carbink", "Hydra"}
WORKLOADS = {"BFS", "LLM", "MG", "WC", "KV-B", "KV-A", "KV-S", "NQ"}
AE_ROOT = Path(__file__).resolve().parents[2]


def source_path(path: Path) -> str:
    """Use an artifact-root-relative source, even when the CSV is copied."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(AE_ROOT))
    except ValueError:
        return str(resolved)


def collect(runs_dir: Path) -> tuple[list[dict], int]:
    if not runs_dir.is_dir():
        raise ValueError(f"run directory not found: {runs_dir}")
    rows = []
    failed = 0
    seen_ids = set()
    contexts: dict[str, set[tuple[str, str, str, int, int]]] = {}
    for analysis_path in sorted(runs_dir.rglob("analysis.json")):
        record = json.loads(analysis_path.read_text(encoding="utf-8"))
        if record.get("status") != "passed":
            failed += 1
            continue
        manifest_path = analysis_path.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plan = manifest.get("plan", {})
        if (record.get("schema_version") != 1 or manifest.get("schema_version") != 1
                or manifest.get("status") != "passed"
                or record.get("exit_status") != 0 or record.get("correctness") != "pass"
                or record.get("workload") not in WORKLOADS
                or record.get("system") not in SYSTEMS
                or not isinstance(record.get("ratio"), int)
                or not 1 <= record["ratio"] <= 100
                or record.get("run_id") != analysis_path.parent.name
                or plan.get("app") != record.get("application")
                or plan.get("ratio") != record["ratio"]
                or plan.get("run_id") != record["run_id"]
                or not record.get("measurement_phase")):
            raise ValueError(f"inconsistent successful run: {analysis_path}")
        for name in ("client.log", "server.log", "effective.config", "server.config"):
            if not analysis_path.with_name(name).is_file():
                raise ValueError(f"successful run has no {name}: {analysis_path}")
        elapsed = float(record["elapsed_s"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid successful elapsed_s: {analysis_path}")
        run_id = record["run_id"]
        if run_id in seen_ids:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen_ids.add(run_id)
        workload = record["workload"]
        contexts.setdefault(workload, set()).add((
            str(record["environment"]), str(record["measurement_phase"]),
            str(plan.get("input")), int(manifest.get("input_bytes", -1)),
            int(manifest.get("input_mtime_ns", -1)),
        ))
        rows.append({
            "workload": workload, "system": record["system"],
            "ratio": record["ratio"], "elapsed_s": f"{elapsed:.9g}",
            "source_type": "measured",
            "source": source_path(analysis_path),
            "run_id": run_id, "exit_status": "0", "correctness": "pass",
            "environment": record["environment"],
            "measurement_phase": record["measurement_phase"],
        })
    for workload, context in contexts.items():
        if len(context) != 1:
            raise ValueError(f"{workload}: mixed input, environment, or measurement phase")
    rows.sort(key=lambda r: (r["workload"], r["system"], r["ratio"], r["run_id"]))
    return rows, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows, failed = collect(args.runs_dir)
        if args.output.exists():
            raise ValueError(f"refusing to overwrite CSV: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}; measured_rows={len(rows)}; failed_runs={failed}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
