#!/usr/bin/env python3
"""Collect Figure 12 raw compute EC cycles and final-Work metadata from AE runs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure12 import collect_metadata as metadata
from figure12 import log_contract as contract

FIELDS = (*metadata.FIELDS, "repeat", "cycle_clock", "ec_work_intervals", "ec_scopes",
          "ec_schema_version", "ec_measurement_basis", "ec_boundary_semantics",
          "ec_initialization_cycles", "ec_work_cycles", "ec_total_cycles",
          "ec_initialization_scopes", "ec_work_scopes", "ec_total_scopes",
          "measurement_usable", "execution_status", "measurement_warning")


def read_ec_cpu(context):
    plan, log = context["plan"], context["log"]
    if plan.get("client_env", {}).get("FARLIB_RUNTIME_EC_CPU") != "1":
        raise ValueError("manifest must record FARLIB_RUNTIME_EC_CPU=1; old logs "
                         "cannot reconstruct raw EC cycles")
    records = []
    schema_version = None
    with log.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if not line.lstrip().startswith(contract.EC_PREFIX + " "):
                continue
            record = contract.parse_line(line)
            if record["system"] != plan["system"]:
                raise ValueError("EC runtime/manifest system mismatch")
            if schema_version is None:
                schema_version = record["schema_version"]
            elif record["schema_version"] != schema_version:
                raise ValueError("EC schema 1 and schema 2 records cannot be mixed")
            record["_source"] = f"{log.resolve()}:{number}"
            records.append(record)
    if not records:
        raise ValueError(f"{log}: missing runtime_ec_cpu completed-Work records")
    if schema_version == 1:
        initialization = None
        work_records = records
        basis = "work_only_legacy"
        boundary_semantics = ""
        if any(record["phase"] != "work" for record in work_records):
            raise ValueError("schema 1 may contain only Work records")
    else:
        initialization_records = [record for record in records
                                  if record["phase"] == "initialization"]
        if len(initialization_records) != 1:
            raise ValueError("schema 2 requires exactly one initialization record")
        initialization = initialization_records[0]
        if records[0] is not initialization:
            raise ValueError("schema 2 initialization must precede all Work records")
        work_records = records[1:]
        if any(record["phase"] != "work" for record in work_records):
            raise ValueError("schema 2 must contain initialization followed by Work")
        basis = "initialization_plus_work"
        boundary_semantics = contract.EC_BOUNDARY_SEMANTICS
    if not work_records:
        raise ValueError(f"{log}: missing runtime_ec_cpu completed-Work records")
    for expected, record in enumerate(work_records, 1):
        if record["boundary_sequence"] != expected:
            raise ValueError("EC Work sequences must be contiguous, starting at 1")
    work_cycles = sum(record["cycles"] for record in work_records)
    work_scopes = sum(record["scopes"] for record in work_records)
    initialization_cycles = initialization["cycles"] if initialization is not None else ""
    initialization_scopes = initialization["scopes"] if initialization is not None else ""
    total_cycles = (initialization_cycles + work_cycles
                    if initialization is not None else work_cycles)
    total_scopes = (initialization_scopes + work_scopes
                    if initialization is not None else work_scopes)
    return {
        "workload": contract.WORKLOADS[plan["app"]],
        "system": contract.SYSTEMS[plan["system"]],
        "metric": "local_ec_cpu_cycles", "unit": "cycles",
        "value": total_cycles,
        "source_type": "measured",
        "source": ";".join(record["_source"] for record in records),
        "measurement_phase": ("initialization_and_all_completed_work"
                              if initialization is not None else "all_completed_work"),
        "cycle_clock": "tsc", "ec_work_intervals": len(work_records),
        "ec_scopes": work_scopes,
        "ec_schema_version": schema_version,
        "ec_measurement_basis": basis,
        "ec_boundary_semantics": boundary_semantics,
        "ec_initialization_cycles": initialization_cycles,
        "ec_work_cycles": work_cycles,
        "ec_total_cycles": total_cycles,
        "ec_initialization_scopes": initialization_scopes,
        "ec_work_scopes": work_scopes,
        "ec_total_scopes": total_scopes,
        "boundary_sequence": len(work_records), "scope": "compute_ec",
    }


def read_run(directory, *, ratio=25, repeat=1, metrics="all", indexed=False):
    context = metadata.load_run_context(directory, ratio=ratio, allow_teardown=True)
    if context is None:
        raise ValueError(f"{directory}: selected run ratio/runtime does not match Figure 12")
    plan = context["plan"]
    selected = contract.METRIC_SELECTIONS[metrics]
    rows = []
    if "local_ec_cpu_cycles" in selected:
        rows.append(read_ec_cpu(context))
    if "metadata_space_pct" in selected:
        rows.append(metadata.read_metadata(context))
    if len(rows) == 2 and rows[0]["boundary_sequence"] != rows[1]["boundary_sequence"]:
        raise ValueError("EC and metadata disagree on the final completed Work boundary")
    for row in rows:
        row.update(
            exit_status=context["analysis"]["exit_status"], correctness="pass",
            run_id=plan["run_id"], environment=plan["site"], ratio=ratio,
            repeat=repeat if indexed else "unindexed",
            app_memory_bytes=context["footprint"],
            client_sha256=context["manifest"]["client_sha256"],
            manifest_source=str(context["manifest_path"].resolve()),
            measurement_usable=1, execution_status=context["execution_status"],
            measurement_warning=context["measurement_warning"])
    return rows


def discover_run_dirs(root, *, repeat=1):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("logs root must exist")
    index = root / "batch-plan.json"
    if index.is_file():
        plan = json.loads(index.read_text())
        if not isinstance(plan, list):
            raise ValueError("batch-plan.json must be a case list")
        directories, seen = [], set()
        for case in plan:
            if not isinstance(case, dict):
                raise ValueError("batch plan cases must be objects")
            if case.get("repeat") != repeat:
                continue
            run_id = case["run_id"]
            if (not isinstance(run_id, str) or not run_id or
                    Path(run_id).name != run_id or run_id in (".", "..") or
                    "/" in run_id or "\\" in run_id or run_id in seen):
                raise ValueError("invalid or duplicate batch run_id")
            seen.add(run_id)
            # The index is relocatable. Do not follow stale absolute run_dir
            # values into an unrelated checkout or another experiment.
            directory = root / "runs" / run_id
            if not directory.is_dir():
                raise ValueError(f"batch case missing: {directory}")
            manifest = json.loads((directory / "manifest.json").read_text())
            actual = manifest["plan"]
            if any(actual.get(key) != case.get(key)
                   for key in ("run_id", "app", "system")):
                raise ValueError(f"{directory}: batch/manifest context mismatch")
            directories.append(directory)
        if not directories:
            raise ValueError(f"batch has no cases for repeat {repeat}")
        return directories
    if repeat != 1:
        raise ValueError("--repeat requires batch-plan.json; use explicit --run-dir "
                         "for older runs, never infer repeats from directory suffixes")
    # Discover manifests, not only successful analyses: unfinished cases must
    # fail validation rather than silently disappear.
    return [path.parent for path in sorted(root.rglob("manifest.json"))]


def collect_rows(root, *, ratio=25, repeat=1, metrics="all"):
    return collect_run_dirs(discover_run_dirs(root, repeat=repeat), ratio=ratio,
                            repeat=repeat, metrics=metrics,
                            indexed=(Path(root) / "batch-plan.json").is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--logs-root", type=Path)
    selection.add_argument("--run-dir", type=Path, action="append",
                           help="Exact verified runs; use to select retries explicitly.")
    parser.add_argument("--metrics", "--metric", choices=contract.METRIC_SELECTIONS,
                        default="all")
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--show-log-format", action="store_true")
    args = parser.parse_args(argv)
    if args.show_log_format:
        print(contract.__doc__)
        return 0
    if not args.output or not (args.logs_root or args.run_dir):
        parser.error("--output and --logs-root or --run-dir are required")
    try:
        if args.output.exists() or (args.summary and args.summary.exists()):
            raise ValueError("output already exists; select a fresh path")
        if args.summary and args.summary.resolve() == args.output.resolve():
            raise ValueError("CSV output and JSON summary must be different paths")
        options = dict(ratio=args.ratio, repeat=args.repeat, metrics=args.metrics)
        rows = (collect_run_dirs(args.run_dir, **options) if args.run_dir else
                collect_rows(args.logs_root, **options))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        warnings = sorted({row["measurement_warning"] for row in rows
                           if row["measurement_warning"]})
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            summary = {"schema_version": 3, "figure": "figure12",
                       "metrics": args.metrics, "repeat": rows[0]["repeat"],
                       "cycle_clock": "tsc", "metadata_numerator": "accounted_bytes",
                       "warnings": warnings, "rows": rows}
            ec_rows = [row for row in rows if row["metric"] == "local_ec_cpu_cycles"]
            if ec_rows:
                summary.update(
                    ec_schema_versions=sorted({row["ec_schema_version"] for row in ec_rows}),
                    ec_measurement_bases=sorted({row["ec_measurement_basis"]
                                                 for row in ec_rows}),
                    ec_boundary_semantics=sorted({row["ec_boundary_semantics"]
                                                  for row in ec_rows if row["ec_boundary_semantics"]}),
                )
            with args.summary.open("x") as stream:
                json.dump(summary, stream, indent=2)
                stream.write("\n")
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        print(f"wrote {len(rows)} measured Figure 12 rows to {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


def collect_run_dirs(directories, *, ratio=25, repeat=1, metrics="all", indexed=False):
    if not 1 <= ratio <= 100 or repeat < 1:
        raise ValueError("ratio must be 1..100 and repeat must be positive")
    if metrics not in contract.METRIC_SELECTIONS:
        raise ValueError(f"unknown metric selection: {metrics}")
    if not indexed and repeat != 1:
        raise ValueError("--repeat requires batch-plan.json; explicit run directories "
                         "are recorded as unindexed, not relabelled as repetitions")
    rows, seen, footprints = [], set(), {}
    for directory in directories:
        for row in read_run(directory, ratio=ratio, repeat=repeat, metrics=metrics,
                            indexed=indexed):
            key = row["workload"], row["system"], row["metric"]
            if key in seen:
                raise ValueError(f"multiple runs for {key}; select exact --run-dir paths "
                                 "or a batch --repeat; retries are not repetitions")
            seen.add(key)
            old = footprints.setdefault(row["workload"], row["app_memory_bytes"])
            if old != row["app_memory_bytes"]:
                raise ValueError("matched runtimes disagree on application footprint")
            rows.append(row)
    if not rows:
        raise ValueError("no selected verified Figure 12 measurements found")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
