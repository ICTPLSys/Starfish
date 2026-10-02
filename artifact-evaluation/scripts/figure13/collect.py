#!/usr/bin/env python3
"""Collect Figure 13 recovery durations and throughput windows from logs.

No benchmark or fault injection is performed. See --show-log-format for the
producer contract. Select one ratio/repeat/context; never average repetitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__:
    from . import log_contract
else:
    import log_contract

AE_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FIELDS = log_contract.CONTEXT_FIELDS
COMMON_FIELDS = (
    "source_type", "source", "exit_status", "correctness", "run_id", "workload",
    "environment", "workload_id", "ratio", "app_workers", "repeat", "phase", "time_origin",
    "comparison_kind", "failure_scope", "source_validation",
)
RECOVERY_FIELDS = (
    "scenario", "system", "recovery_s", "failure_elapsed_s", "recovered_elapsed_s",
    "rebuild_start_elapsed_s", "recovery_definition",
) + COMMON_FIELDS
TRACE_FIELDS = (
    "system", "elapsed_s", "throughput_ops_per_s", "event", "normalization",
    "reference_ops_per_s", "scenario", "time_alignment", "display_offset_s",
    "original_window_start_s", "original_window_end_s", "completed_ops",
    "original_failure_elapsed_s", "original_recovered_elapsed_s",
    "reference_run_id", "reference_source", "sample_kind", "window_scope", "original_time_origin",
    "recovery_definition", "original_rebuild_start_elapsed_s",
) + COMMON_FIELDS


def number(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite derived metric/time")
    return format(value, ".17g")


def same_time(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)


def read_logs(root, pattern):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"log directory not found: {root}")
    results, samples = {}, {}
    paths = sorted({path.resolve() for path in root.rglob(pattern) if path.is_file()})
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = log_contract.parse_line(line)
                    if record is None:
                        continue
                    record["_source"] = f"{path}:{line_number}"
                    run_id = record["run_id"]
                    if record["_kind"] == "result":
                        if run_id in results:
                            raise ValueError(f"duplicate final result: {run_id}")
                        results[run_id] = record
                    else:
                        samples.setdefault(run_id, []).append(record)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    orphaned = set(samples) - results.keys()
    if orphaned:
        raise ValueError("samples lack final result: " + ", ".join(sorted(orphaned)))
    return results, samples


def validate_windows(run, samples):
    if not samples:
        raise ValueError(f"{run['run_id']}: trace has no samples")
    windows = sorted(samples, key=lambda item: (item["window_start_s"], item["window_end_s"]))
    previous_end = None
    seen = set()
    for item in windows:
        start, end = item["window_start_s"], item["window_end_s"]
        if (start, end) in seen:
            raise ValueError(f"{run['run_id']}: duplicate sample window")
        seen.add((start, end))
        if end > run["run_end_s"]:
            raise ValueError(f"{run['run_id']}: sample ends after run_end_s")
        if previous_end is not None and start < previous_end and not same_time(start, previous_end):
            raise ValueError(f"{run['run_id']}: overlapping/duplicate sample windows")
        previous_end = end
    if not any(item["window_end_s"] <= run["failure_elapsed_s"] for item in windows):
        raise ValueError(f"{run['run_id']}: no complete pre-failure sample")
    if not any(item["window_start_s"] >= run["recovered_elapsed_s"] for item in windows):
        raise ValueError(f"{run['run_id']}: no complete post-recovery sample")
    return windows


def pre_failure_reference(run, windows):
    # Boundary-crossing windows may be excluded from the weighted mean, but
    # missing pre-failure observations must not turn an old isolated sample
    # into the reference for an otherwise unobserved steady-state interval.
    covered = [item for item in windows
               if item["window_end_s"] > run["steady_start_s"]
               and item["window_start_s"] < run["failure_elapsed_s"]]
    if (not covered
            or (covered[0]["window_start_s"] > run["steady_start_s"]
                and not same_time(covered[0]["window_start_s"], run["steady_start_s"]))
            or (covered[-1]["window_end_s"] < run["failure_elapsed_s"]
                and not same_time(covered[-1]["window_end_s"], run["failure_elapsed_s"]))):
        raise ValueError("Starfish samples do not cover the steady pre-failure interval")
    for left, right in zip(covered, covered[1:]):
        if not same_time(left["window_end_s"], right["window_start_s"]):
            raise ValueError("Starfish pre-failure reference contains a sampling gap")
    chosen = [item for item in windows
              if item["window_start_s"] >= run["steady_start_s"]
              and item["window_end_s"] <= run["failure_elapsed_s"]]
    if not chosen:
        raise ValueError("Starfish has no complete steady pre-failure window")
    count = sum(item["completed_ops"] for item in chosen)
    duration = sum(item["window_end_s"] - item["window_start_s"] for item in chosen)
    rate = count / duration
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Starfish pre-failure reference must be finite and positive")
    return {
        "run_id": run["run_id"], "source": run["_source"],
        "window_start_s": chosen[0]["window_start_s"],
        "window_end_s": chosen[-1]["window_end_s"],
        "completed_ops": count, "duration_s": duration, "ops_per_s": rate,
        "sample_sources": [item["_source"] for item in chosen],
        "method": "sum_completed_ops_divided_by_sum_complete_window_seconds",
    }


def base_row(run):
    row = {key: run[key] for key in CONTEXT_FIELDS}
    row.update(source_type="measured", source=run["_source"], exit_status="0",
               correctness="pass", run_id=run["run_id"],
               system=log_contract.SYSTEMS[run["system"]], scenario=run["scenario"],
               comparison_kind=run.get("comparison_kind", "unspecified"),
               failure_scope=run.get("failure_scope", "unspecified"),
               source_validation=run.get("source_validation", "verified_explicit_protocol"))
    return row


def collect_rows(logs_root, *, pattern="*.log", ratio=25, repeat=1,
                 trace_scenario="1-node", failure_at_s=20.0, panels="both"):
    results, samples = read_logs(logs_root, pattern)
    return collect_records(results, samples, ratio=ratio, repeat=repeat,
                           trace_scenario=trace_scenario, failure_at_s=failure_at_s,
                           panels=panels)


def collect_native(result_paths, *, ratio=25, repeat=1, trace_scenario="1-node",
                   failure_at_s=20.0, steady_before_s=10.0, panels="both"):
    if __package__:
        from .raw_runs import read_run
    else:
        from raw_runs import read_run
    results, samples, details = {}, {}, {}
    site_indexed = False
    source_kinds = set()
    for path in result_paths:
        run, windows, evidence = read_run(Path(path), steady_before_s=steady_before_s)
        site_input = (
            evidence.get("adapter") == "figure13.site_runs"
            or evidence.get("source_type") == "site_result_json"
        )
        source_kinds.add(site_input)
        site_indexed = site_indexed or site_input
        run.update(
            comparison_kind="diagnostic",
            failure_scope="logical_service_process",
            source_validation=(
                "verified_site_evidence" if site_input
                else "verified_native_evidence"
            ),
        )
        # Shared workload identity does not imply identical runtime resources.
        # Carbink has additional compaction workers. Preserve the full native
        # identity/placement, but compare only identical workload + host fields.
        payload = evidence["workload_id_payload"]
        common_payload = {key: value for key, value in payload.items()
                          if key not in ("app_cpus", "background_cpus")}
        evidence["native_workload_id"] = run["workload_id"]
        run["workload_id"] = hashlib.sha256(json.dumps(
            common_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        evidence["comparison_workload_payload"] = common_payload
        if run["run_id"] in results:
            raise ValueError(f"duplicate selected native run_id: {run['run_id']}")
        results[run["run_id"]] = run
        samples[run["run_id"]] = windows
        details[run["run_id"]] = evidence
    if len(source_kinds) > 1:
        raise ValueError("do not mix historical native results and new site results")
    recovery, trace, info = collect_records(
        results, samples, ratio=ratio, repeat=repeat, trace_scenario=trace_scenario,
        failure_at_s=failure_at_s, native=True, indexed=site_indexed, panels=panels)
    info.update(input_format="native_recovery_result", native_runs=details,
                comparison_kind="diagnostic",
                recovery_definition="background_start_to_done",
                workload_id_scope="same workload and hosts; CPU placement retained per run, not equated",
                placements={name: {key: record["workload_id_payload"][key]
                                   for key in ("app_cpus", "background_cpus")
                                   if key in record["workload_id_payload"]}
                            for name, record in details.items()},
                note="Owned logical memory-service failures; not physical-node failure measurements.")
    if site_indexed:
        info["repeat_source"] = "figure13-site-result-v1"
    return recovery, trace, info


def collect_records(results, samples, *, ratio=25, repeat=1,
                    trace_scenario="1-node", failure_at_s=20.0, native=False,
                    indexed=False,
                    panels="both"):
    if not 1 <= ratio <= 100 or repeat < 1:
        raise ValueError("ratio must be 1..100 and repeat must be positive")
    if trace_scenario not in log_contract.SCENARIOS:
        raise ValueError("unknown trace scenario")
    if not math.isfinite(failure_at_s) or failure_at_s < 0:
        raise ValueError("display failure anchor must be finite and nonnegative")
    if panels not in ("both", "recovery", "throughput"):
        raise ValueError("unknown panel selection")
    if native and not indexed and repeat != 1:
        raise ValueError("native result files are unindexed; --repeat cannot relabel them")
    selected = [run for run in results.values()
                if run["ratio"] == ratio and (
                    (indexed and run["repeat"] == repeat)
                    or (native and not indexed and run["repeat"] == "unindexed")
                    or (not native and run["repeat"] == repeat)
                )]
    if native and not indexed and len(selected) != len(results):
        raise ValueError("explicit native result ratio does not match the selected ratio")
    if not selected:
        raise ValueError("no final Figure 13 records for selected ratio/repeat")
    context = {key: selected[0][key] for key in CONTEXT_FIELDS}
    definitions = {run.get("recovery_definition", "failure_to_reconstruction") for run in selected}
    if len(definitions) != 1:
        raise ValueError("do not mix background duration and failure-to-completion definitions")
    recovery_runs, trace_runs = {}, {}
    for run in selected:
        if (run["exit_status"] != 0 or run["correctness"] != "pass"
                or run["failure_confirmed"] != 1 or run["recovery_verified"] != 1):
            raise ValueError(f"{run['run_id']}: failed/unverified workload, fault or recovery")
        mismatched = [key for key in CONTEXT_FIELDS if run[key] != context[key]]
        if mismatched:
            raise ValueError(f"{run['run_id']}: mixed context: " + ", ".join(mismatched))
        key = run["scenario"], run["system"]
        if run["panel"] in ("both", "recovery") and panels != "throughput":
            if key in recovery_runs:
                raise ValueError(f"duplicate recovery condition: {key}")
            recovery_runs[key] = run
        if (run["panel"] in ("both", "trace") and run["scenario"] == trace_scenario
                and panels != "recovery"):
            if run["system"] in trace_runs:
                raise ValueError(f"duplicate throughput condition: {key}")
            trace_runs[run["system"]] = run

    recovery_rows = []
    for key in sorted(recovery_runs):
        run = recovery_runs[key]
        row = base_row(run)
        definition = run.get("recovery_definition", "failure_to_reconstruction")
        start = (run["rebuild_start_elapsed_s"] if definition == "background_start_to_done"
                 else run["failure_elapsed_s"])
        row.update(
            recovery_s=number(run["recovered_elapsed_s"] - start),
            failure_elapsed_s=number(run["failure_elapsed_s"]),
            recovered_elapsed_s=number(run["recovered_elapsed_s"]),
            rebuild_start_elapsed_s=number(run["rebuild_start_elapsed_s"])
                if "rebuild_start_elapsed_s" in run else "",
            recovery_definition=definition)
        recovery_rows.append(row)
    if not recovery_rows and not trace_runs:
        raise ValueError("no records belong to the selected panels/scenario")
    trace_rows, reference = [], None
    if trace_runs:
        if "starfish" not in trace_runs:
            raise ValueError("throughput panel needs the selected Starfish reference run")
        windows = {system: validate_windows(run, samples.get(run["run_id"], []))
                   for system, run in trace_runs.items()}
        reference = pre_failure_reference(trace_runs["starfish"], windows["starfish"])
        trace_rows = build_trace_rows(trace_runs, windows, reference, failure_at_s)
    return recovery_rows, trace_rows, {
        "context": context, "trace_scenario": trace_scenario,
        "reference": reference, "time_alignment": "failure_aligned",
        "display_failure_s": failure_at_s, "ratio": ratio,
        "repeat": repeat if indexed else ("unindexed" if native else repeat),
        "panels": panels,
    }


def build_trace_rows(runs, windows, reference, failure_at_s):
    rows = []
    for system in log_contract.SYSTEMS:
        if system not in runs:
            continue
        run = runs[system]
        offset = failure_at_s - run["failure_elapsed_s"]
        base = base_row(run)
        base.update(
            normalization="starfish_pre_failure",
            reference_ops_per_s=number(reference["ops_per_s"]),
            reference_run_id=reference["run_id"], reference_source=reference["source"],
            time_alignment="failure_aligned", display_offset_s=number(offset),
            time_origin="failure_aligned_display", original_time_origin=run["time_origin"],
            original_failure_elapsed_s=number(run["failure_elapsed_s"]),
            original_recovered_elapsed_s=number(run["recovered_elapsed_s"]),
            recovery_definition=run.get("recovery_definition", "failure_to_reconstruction"),
            original_rebuild_start_elapsed_s=number(run["rebuild_start_elapsed_s"])
                if "rebuild_start_elapsed_s" in run else "")
        previous_end = None
        for item in windows[system]:
            start, end = item["window_start_s"], item["window_end_s"]
            if previous_end is not None and start > previous_end and not same_time(start, previous_end):
                gap = dict(base)
                gap.update(
                    elapsed_s=number((previous_end + start) / 2 + offset),
                    throughput_ops_per_s="", event="none", sample_kind="missing",
                    window_scope="unobserved",
                    original_window_start_s=number(previous_end),
                    original_window_end_s=number(start), completed_ops="",
                    source=run["_source"] + "; missing interval before " + item["_source"])
                rows.append(gap)
            if any(start < event_time < end for event_time in
                   (run["failure_elapsed_s"], run["recovered_elapsed_s"])):
                window_scope = "crosses_event"
            elif end <= run["failure_elapsed_s"]:
                window_scope = "pre_failure"
            elif start >= run["recovered_elapsed_s"]:
                window_scope = "post_recovery"
            else:
                window_scope = "recovery"
            row = dict(base)
            row.update(
                elapsed_s=number(end + offset),
                throughput_ops_per_s=number(item["completed_ops"] / (end - start)),
                event="none", sample_kind="observed", window_scope=window_scope,
                original_window_start_s=number(start), original_window_end_s=number(end),
                completed_ops=item["completed_ops"],
                source=item["_source"] + "; result=" + run["_source"])
            rows.append(row)
            previous_end = end
        # Annotation only. It must not become a blank data point that breaks
        # the throughput line if recovery falls between sampling boundaries.
        event = dict(base)
        if "rebuild_start_elapsed_s" in run:
            event.update(elapsed_s=number(run["rebuild_start_elapsed_s"] + offset),
                         throughput_ops_per_s="", event="rebuild_start", sample_kind="event")
            rows.append(event)
            event = dict(base)
        event.update(elapsed_s=number(run["recovered_elapsed_s"] + offset),
                     throughput_ops_per_s="", event="recovered", sample_kind="event")
        rows.append(event)
    failure = base_row(runs["starfish"])
    failure.update(
        system="all", run_id="", elapsed_s=number(failure_at_s),
        throughput_ops_per_s="", event="failure", sample_kind="event",
        normalization="starfish_pre_failure",
        reference_ops_per_s=number(reference["ops_per_s"]),
        reference_run_id=reference["run_id"], reference_source=reference["source"],
        time_alignment="failure_aligned", time_origin="failure_aligned_display",
        original_time_origin="work_start",
        source="; ".join(runs[system]["_source"] for system in log_contract.SYSTEMS if system in runs))
    rows.append(failure)
    return rows


def write_csv(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--manifest", type=Path, help="explicit six-condition native manifest or indexed batch runs.json")
    inputs.add_argument("--logs-root", type=Path, help="explicit result/sample protocol logs")
    inputs.add_argument("--run-result", type=Path, action="append",
                        help="Select exact native recovery/revalidation JSON files; repeat per run.")
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--trace-scenario", choices=log_contract.SCENARIOS, default="1-node")
    parser.add_argument("--failure-at-s", type=float, default=20,
                        help="display coordinate for aligned observed failures; not injection time")
    parser.add_argument("--steady-before-s", type=float, default=10,
                        help="Native input: measured pre-failure reference interval in seconds.")
    parser.add_argument("--panels", choices=("both", "recovery", "throughput"), default="both")
    parser.add_argument("--output-dir", type=Path, default=AE_ROOT / "results/figure13" /
                        ("collected-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--show-log-format", action="store_true")
    args = parser.parse_args()
    if args.show_log_format:
        print(log_contract.__doc__)
        return 0
    if args.logs_root is None and not args.run_result and args.manifest is None:
        parser.error("--manifest, --logs-root or --run-result is required")
    try:
        manifest_path = args.manifest
        if args.logs_root and (args.logs_root / "runs.json").is_file():
            manifest_path = args.logs_root / "runs.json"
        indexed = False
        if manifest_path:
            from inputs import load_manifest
            manifest, entries, indexed = load_manifest(manifest_path, args.repeat)
            args.run_result = [Path(e["result"]) for e in entries]
        native_repeat = (
            args.repeat
            if indexed and manifest.get("_batch_schema") == "figure13-site-batch-v1"
            else (1 if indexed else args.repeat)
        )
        options = dict(ratio=args.ratio, repeat=native_repeat, trace_scenario=args.trace_scenario,
                       failure_at_s=args.failure_at_s, panels=args.panels)
        if args.run_result:
            recovery, trace, info = collect_native(args.run_result,
                steady_before_s=args.steady_before_s, **options)
        else:
            recovery, trace, info = collect_rows(args.logs_root, pattern=args.pattern, **options)
        if indexed:
            for row in recovery + trace:
                row["repeat"] = str(args.repeat)
        if manifest_path:
            info.update(manifest=str(manifest_path.resolve()), selected_repeat=args.repeat if indexed else None,
                        repeat_source="batch-plan.json" if indexed else "historical_unindexed")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_csv(args.output_dir / "figure13-recovery.csv", RECOVERY_FIELDS, recovery)
        write_csv(args.output_dir / "figure13-throughput.csv", TRACE_FIELDS, trace)
        info.update(logs_root=str(args.logs_root.resolve()) if args.logs_root else None,
                    run_results=[str(path.resolve()) for path in args.run_result or []],
                    log_pattern=args.pattern,
                    recovery_rows=recovery, throughput_rows=trace)
        (args.output_dir / "collection.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {len(recovery)} recovery rows and {len(trace)} trace/event rows to {args.output_dir}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
