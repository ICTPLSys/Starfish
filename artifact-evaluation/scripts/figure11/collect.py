#!/usr/bin/env python3
"""Collect Figure 11 schema-v3 records into the established plot CSV.

The default requires all four components.  Use --components to explicitly
collect a measured subset.  Direct figure11_result records are preferred; if
none are present, the optional raw_runs.collect_records adapter is used.  A
run is never read through both paths.
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
    try:
        from . import raw_runs
    except ImportError:
        raw_runs = None
else:
    import log_contract
    try:
        import raw_runs
    except ImportError:
        raw_runs = None

AE_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FIELDS = log_contract.CONTEXT_FIELDS
SYSTEMS = log_contract.SYSTEMS
WORKLOADS = log_contract.WORKLOADS
COMPONENTS = log_contract.COMPONENTS
parse_line = log_contract.parse_line

FIELDS = (
    "workload", "system", "component", "value", "unit", "source_type", "source",
    "exit_status", "correctness", "run_id", "baseline_run_id", "ratio",
    "app_workers", "repeat", "environment", "baseline_environment",
    "environment_match", "warning", "workload_id", "measurement_phase",
    "aggregation", "window", "metric", "sampling", "samples",
    "raw_value", "normalizer", "raw_unit", "normalizer_unit",
    "scheduled_times_s", "sample_times_s",
    "baseline_samples", "baseline_sampling",
    "baseline_variant", "backup_enabled", "normalizer_variant",
)


def requested_components(value=None):
    """Return canonical component order for CLI or API input."""
    if value is None:
        return COMPONENTS
    return log_contract.normalize_components(value)


def _files(root, pattern):
    return sorted({path.resolve() for path in root.rglob(pattern)
                   if path.is_file()})


def _accept_record(records, seen_runs, record, source, *, ratio, repeat):
    record = log_contract.validate_record(record, source=source)
    if record["ratio"] != ratio or record["repeat"] != repeat:
        return
    if record["system"] == "nonft" and record["baseline_variant"] != "nonft-backup-off":
        return
    if record["exit_status"] != 0 or record["correctness"] != "pass":
        raise ValueError("failed or unverified runs cannot enter Figure 11")
    run = record["run_id"]
    if run in seen_runs:
        raise ValueError(f"duplicate final record for run_id={run}")
    key = record["workload"], record["system"]
    if key in records:
        raise ValueError(f"multiple selected runs for {key}")
    seen_runs.add(run)
    records[key] = record


def _collect_final_records(root, *, pattern, ratio, repeat):
    records = {}
    seen_runs = set()
    for path in _files(root, pattern):
        with path.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    parsed = parse_line(line)
                    if parsed is not None:
                        _accept_record(
                            records, seen_runs, parsed, f"{path}:{number}",
                            ratio=ratio, repeat=repeat)
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    return records


def _raw_records(root, *, pattern, ratio, repeat, components, raw_collector):
    if raw_collector is None:
        raw_collector = raw_runs.collect_records if raw_runs is not None else None
    if raw_collector is None:
        raise ValueError(
            "no schema-v3 figure11_result records and raw_runs.collect_records "
            "is unavailable; raw collection requires original logs and "
            "remote-cpu.json sidecars")
    result = raw_collector(
        root, pattern=pattern, ratio=ratio, repeat=repeat,
        components=components)
    if isinstance(result, dict):
        result = result.get("records", result.values())
    records = {}
    seen_runs = set()
    for index, record in enumerate(result):
        source = record.get("_source", f"raw record {index + 1}")
        _accept_record(
            records, seen_runs, record, source,
            ratio=ratio, repeat=repeat)
    return records


def collect_records(logs_root, *, pattern="*.log", ratio=25, repeat=1,
                    components=None, raw_collector=None):
    """Return selected schema-v3 records keyed by (workload, system)."""
    root = Path(logs_root).resolve()
    if not root.is_dir():
        raise ValueError(f"log directory not found: {root}")
    if not 1 <= ratio <= 100 or repeat < 1:
        raise ValueError("ratio must be 1..100 and repeat must be positive")
    selected = requested_components(components)
    records = _collect_final_records(
        root, pattern=pattern, ratio=ratio, repeat=repeat)
    if not records:
        records = _raw_records(
            root, pattern=pattern, ratio=ratio, repeat=repeat,
            components=selected, raw_collector=raw_collector)
    if not records:
        raise ValueError(
            f"no schema-v3 records for ratio={ratio}, repeat={repeat} under {root}")
    return records


def _context_status(record, baseline, *, allow_unmatched_environments):
    mismatched = [key for key in CONTEXT_FIELDS
                  if record[key] != baseline[key]]
    environment_mismatch = record["environment"] != baseline["environment"]
    if mismatched and (not allow_unmatched_environments or
                       any(key != "environment" for key in mismatched)):
        raise ValueError(
            f"{record['workload']}/{record['system']}: context differs from "
            f"NonFT: {', '.join(mismatched)}")
    if environment_mismatch:
        warning = (
            "unmatched environment allowed: record="
            + record["environment"] + " baseline=" + baseline["environment"])
        return "unmatched_allowed", warning
    return "matched", ""


def _positive_baseline(record, field):
    value = float(record[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{record['workload']}: NonFT {field} must be positive")
    return value


def _make_row(record, baseline, component, *, ratio, repeat,
              environment_match, warning):
    if component == "fetch_traffic":
        field, unit, raw_unit = "fetch_bytes", "x", "bytes"
        raw = float(record[field])
        denominator = _positive_baseline(baseline, field)
        aggregation, window, metric = "sum", "work", field
        normalizer_unit = "bytes"
        baseline_run_id = baseline["run_id"]
        source = record["_source"] + "; nonft=" + baseline["_source"]
        sampling, samples = "", ""
    elif component == "eviction_traffic":
        field, unit, raw_unit = "eviction_bytes", "x", "bytes"
        raw = float(record[field])
        denominator = _positive_baseline(baseline, field)
        aggregation, window, metric = "sum", "work", field
        normalizer_unit = "bytes"
        baseline_run_id = baseline["run_id"]
        source = record["_source"] + "; nonft=" + baseline["_source"]
        sampling, samples = "", ""
    elif component == "remote_cpu_cores":
        field, unit, raw_unit = "remote_cpu_seconds", "cores", "cpu_seconds"
        raw = float(record[field])
        denominator = float(record["remote_cpu_elapsed_s"])
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("remote_cpu_elapsed_s must be positive")
        aggregation = "sum"
        window = record["remote_cpu_window"]
        metric = field
        normalizer_unit = "seconds"
        baseline_run_id = ""
        source = record["_source"]
        sampling, samples = "", ""
    else:
        field, unit, raw_unit = "remote_memory_mean_bytes", "x", "bytes"
        raw = float(record[field])
        denominator = _positive_baseline(baseline, field)
        aggregation = "mean"
        window = record["remote_memory_window"]
        metric = record["remote_memory_metric"]
        normalizer_unit = "bytes"
        baseline_run_id = baseline["run_id"]
        source = record["_source"] + "; nonft=" + baseline["_source"]
        sampling = record["remote_memory_sampling"]
        samples = str(record["remote_memory_samples"])
        if record["remote_memory_samples"] != baseline["remote_memory_samples"]:
            count_warning = (
                f"memory sample counts differ: record={record['remote_memory_samples']} "
                f"baseline={baseline['remote_memory_samples']}")
            warning = "; ".join(part for part in (warning, count_warning) if part)
    value = raw / denominator
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{record['workload']}/{record['system']}: invalid {component}")
    return {
        "workload": WORKLOADS[record["workload"]],
        "system": SYSTEMS[record["system"]],
        "component": component,
        "value": format(value, ".12g"),
        "unit": unit,
        "source_type": "measured",
        "source": source,
        "exit_status": "0",
        "correctness": "pass",
        "run_id": record["run_id"],
        "baseline_run_id": baseline_run_id,
        "ratio": record["ratio"],
        "app_workers": record["app_workers"],
        "repeat": record["repeat"],
        "environment": record["environment"],
        "baseline_environment": baseline["environment"],
        "environment_match": environment_match,
        "warning": warning,
        "workload_id": record["workload_id"],
        "measurement_phase": window,
        "aggregation": aggregation,
        "window": window,
        "metric": metric,
        "sampling": sampling,
        "samples": samples,
        "raw_value": raw,
        "normalizer": denominator,
        "raw_unit": raw_unit,
        "normalizer_unit": normalizer_unit,
        "scheduled_times_s": ",".join(map(str, record["remote_memory_scheduled_times_s"]))
            if component == "remote_memory" else "",
        "sample_times_s": ",".join(map(str, record["remote_memory_sample_times_s"]))
            if component == "remote_memory" else "",
        "baseline_samples": str(baseline["remote_memory_samples"])
            if component == "remote_memory" else "",
        "baseline_sampling": baseline["remote_memory_sampling"]
            if component == "remote_memory" else "",
        "baseline_variant": record.get("baseline_variant", record["system"]),
        "backup_enabled": str(record["backup_enabled"]).lower()
            if "backup_enabled" in record else "",
        "normalizer_variant": baseline.get("baseline_variant", baseline["system"])
            if component != "remote_cpu_cores" else "",
    }


def collect_rows(logs_root, *, pattern="*.log", ratio=25, repeat=1,
                 components=None, allow_unmatched_environments=False,
                 raw_collector=None):
    selected = requested_components(components)
    records = collect_records(
        logs_root, pattern=pattern, ratio=ratio, repeat=repeat,
        components=selected, raw_collector=raw_collector)
    rows = []
    for app in WORKLOADS:
        selected_records = [
            records[app, system] for system in SYSTEMS
            if (app, system) in records
        ]
        if not selected_records:
            continue
        baseline = records.get((app, "nonft"))
        if baseline is None:
            raise ValueError(f"{app}: missing selected NonFT baseline")
        for record in selected_records:
            available = set(log_contract.component_names(record))
            missing = sorted(set(selected) - available)
            if missing:
                raise ValueError(
                    f"{app}/{record['system']}: missing selected component(s): "
                    + ", ".join(missing))
            baseline_missing = sorted(
                set(selected) - set(log_contract.component_names(baseline)))
            if baseline_missing:
                raise ValueError(
                    f"{app}: NonFT missing selected component(s): "
                    + ", ".join(baseline_missing))
            environment_match, warning = _context_status(
                record, baseline,
                allow_unmatched_environments=allow_unmatched_environments)
            for component in selected:
                rows.append(_make_row(
                    record, baseline, component, ratio=ratio, repeat=repeat,
                    environment_match=environment_match, warning=warning))
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
    parser.add_argument("--components", default=None,
                        help="comma-separated explicit subset; default is all four")
    parser.add_argument("--allow-unmatched-environments", action="store_true",
                        help="allow but annotate environment mismatches")
    parser.add_argument("--output", type=Path, default=AE_ROOT / "data/figure11.csv")
    parser.add_argument("--show-log-format", action="store_true")
    args = parser.parse_args()
    if args.show_log_format:
        print(log_contract.__doc__)
        return 0
    if args.logs_root is None:
        parser.error("--logs-root is required unless --show-log-format is selected")
    try:
        rows = collect_rows(
            args.logs_root, pattern=args.pattern, ratio=args.ratio,
            repeat=args.repeat, components=args.components,
            allow_unmatched_environments=args.allow_unmatched_environments)
        write_csv(rows, args.output)
        print(f"wrote {len(rows)} measured component rows to {args.output}")
        return 0
    except (OSError, ValueError, OverflowError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
