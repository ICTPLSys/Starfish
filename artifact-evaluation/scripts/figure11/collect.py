#!/usr/bin/env python3
"""Collect Figure 11 final log records into the existing plot CSV format.

Each log record follows log_contract.py (also shown by --show-log-format).
Select one local-memory ratio and one repeat; repeats are never averaged.
The selected input may be sparse, but every included workload needs a matched
NonFT record. Normalization is component-wise, not by the total traffic bar:
fetch/fetch_nonft, eviction/eviction_nonft, memory_peak/memory_peak_nonft.
CPU remains an absolute mean core count: remote_cpu_seconds / elapsed_s.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__:
    from . import log_contract
else:
    import log_contract
CONTEXT_FIELDS = log_contract.CONTEXT_FIELDS
SYSTEMS = log_contract.SYSTEMS
WORKLOADS = log_contract.WORKLOADS
parse_line = log_contract.parse_line

AE_ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "workload", "system", "component", "value", "unit", "source_type", "source",
    "exit_status", "correctness", "run_id", "baseline_run_id", "ratio",
    "app_workers", "repeat", "environment", "workload_id", "measurement_phase",
    "raw_value", "normalizer", "raw_unit", "normalizer_unit",
)
COMPONENTS = (
    ("fetch_traffic", "fetch_bytes", "x", "bytes"),
    ("eviction_traffic", "eviction_bytes", "x", "bytes"),
    ("remote_cpu_cores", "remote_cpu_seconds", "cores", "cpu_seconds"),
    ("remote_memory", "remote_memory_peak_bytes", "x", "bytes"),
)


def collect_rows(logs_root, *, pattern="*.log", ratio=25, repeat=1):
    root = Path(logs_root).resolve()
    if not root.is_dir():
        raise ValueError(f"log directory not found: {root}")
    if not 1 <= ratio <= 100 or repeat < 1:
        raise ValueError("ratio must be 1..100 and repeat must be positive")
    records = {}
    seen_runs = set()
    files = sorted({path.resolve() for path in root.rglob(pattern) if path.is_file()})
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = parse_line(line)
                    if record is None:
                        continue
                    if record["ratio"] != ratio or record["repeat"] != repeat:
                        continue
                    if record["exit_status"] != 0 or record["correctness"] != "pass":
                        raise ValueError("failed or unverified runs cannot enter Figure 11")
                    run = record["run_id"]
                    if run in seen_runs:
                        raise ValueError(f"duplicate final record for run_id={run}")
                    seen_runs.add(run)
                    key = record["workload"], record["system"]
                    if key in records:
                        raise ValueError(
                            f"multiple selected runs for {key}; select one repeat/log directory")
                    record["_source"] = f"{path}:{number}"
                    records[key] = record
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(
            f"no figure11_result records for ratio={ratio}, repeat={repeat} under {root}")

    rows = []
    for app in WORKLOADS:
        selected = [records[app, system] for system in SYSTEMS
                    if (app, system) in records]
        if not selected:
            continue
        baseline = records.get((app, "nonft"))
        if baseline is None:
            raise ValueError(f"{app}: missing selected NonFT baseline")
        for key in ("fetch_bytes", "eviction_bytes", "remote_memory_peak_bytes"):
            if baseline[key] <= 0:
                raise ValueError(f"{app}: NonFT {key} must be positive for normalization")
        for record in selected:
            mismatched = [key for key in CONTEXT_FIELDS if record[key] != baseline[key]]
            if mismatched:
                raise ValueError(
                    f"{app}/{record['system']}: context differs from NonFT: "
                    + ", ".join(mismatched))
            for component, field, unit, raw_unit in COMPONENTS:
                cpu = component == "remote_cpu_cores"
                denominator = record["elapsed_s"] if cpu else baseline[field]
                value = record[field] / denominator
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{app}/{record['system']}: invalid {component}")
                rows.append({
                    "workload": WORKLOADS[app], "system": SYSTEMS[record["system"]],
                    "component": component, "value": format(value, ".12g"), "unit": unit,
                    "source_type": "measured",
                    "source": (record["_source"] if cpu else
                               record["_source"] + "; nonft=" + baseline["_source"]),
                    "exit_status": "0", "correctness": "pass",
                    "run_id": record["run_id"],
                    "baseline_run_id": "" if cpu else baseline["run_id"],
                    "ratio": ratio, "app_workers": record["app_workers"],
                    "repeat": repeat, "environment": record["environment"],
                    "workload_id": record["workload_id"], "measurement_phase": record["phase"],
                    "raw_value": record[field], "normalizer": denominator,
                    "raw_unit": raw_unit, "normalizer_unit": "seconds" if cpu else "bytes",
                })
    return rows


def write_csv(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=AE_ROOT / "data/figure11.csv")
    parser.add_argument("--show-log-format", action="store_true")
    args = parser.parse_args()
    if args.show_log_format:
        print(log_contract.__doc__)
        return 0
    if args.logs_root is None:
        parser.error("--logs-root is required unless --show-log-format is selected")
    try:
        rows = collect_rows(args.logs_root, pattern=args.pattern,
                            ratio=args.ratio, repeat=args.repeat)
        write_csv(rows, args.output)
        print(f"wrote {len(rows)} measured component rows to {args.output}")
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
