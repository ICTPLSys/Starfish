#!/usr/bin/env python3
"""Collect metadata capacities saved at final Work end from successful AE runs.

Not peak/mean/RSS/live-data size. Excludes application object handles, payload
and parity buffers; includes identified measurement auxiliary state. Uses the manifest's full
application footprint, never the local cache, as the percentage denominator.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

PREFIX = "runtime_metadata"
COMPONENTS = ("region_bytes", "group_bytes", "stripe_bytes", "mapping_bytes", "span_bytes")
COUNTS = ("local_regions", "remote_regions", "stripes", "group_slots", "spans")
INTEGERS = ("schema_version", "boundary_sequence", "snapshot_us", "metadata_bytes",
            "measurement_aux_bytes", "accounted_bytes", *COMPONENTS, *COUNTS)
CONVENTIONS = {
    "phase": "work_end", "snapshot": "last_completed_work",
    "scope": "region_group_stripe_span", "accounting": "sizeof_plus_capacity",
    "consistency": "component_locked",
}
SYSTEMS = {"starfish": "Starfish", "carbink": "Carbink"}
APPS = {"bfs": "BFS", "llama": "LLM", "mg": "MG", "wordcount": "WC",
        "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S", "nq": "NQ"}
FIELDS = ("workload", "system", "metric", "value", "unit", "source_type", "source",
          "exit_status", "correctness", "run_id", "environment", "ratio",
          "measurement_phase", "app_memory_bytes", "metadata_bytes",
          "core_metadata_bytes", "metadata_numerator",
          "measurement_aux_bytes", "accounted_bytes", *COMPONENTS, *COUNTS,
          "boundary_sequence", "snapshot_us", "accounting", "scope",
          "client_sha256", "manifest_source")


def parse_line(line):
    tokens = shlex.split(line.strip())
    if not tokens or tokens[0] != PREFIX:
        return None
    record = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator or not key or not value or key in record:
            raise ValueError(f"invalid or duplicate metadata field: {token!r}")
        record[key] = value
    for key in (*INTEGERS, "system", *CONVENTIONS):
        if key not in record:
            raise ValueError(f"missing metadata field: {key}")
    for key in INTEGERS:
        value = record[key]
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"{key} must be a nonnegative integer")
        record[key] = int(value)
    if record["schema_version"] != 1 or record["boundary_sequence"] < 1:
        raise ValueError("unsupported metadata schema or no completed Work")
    if record["system"] not in SYSTEMS:
        raise ValueError("unsupported metadata runtime")
    for key, value in CONVENTIONS.items():
        if record[key] != value:
            raise ValueError(f"unexpected metadata convention: {key}")
    if record["metadata_bytes"] != sum(record[key] for key in COMPONENTS):
        raise ValueError("metadata component sum does not close")
    if record["accounted_bytes"] != (
            record["metadata_bytes"] + record["measurement_aux_bytes"]):
        raise ValueError("metadata plus measurement auxiliary sum does not close")
    return record


def load_run_context(directory, *, ratio=25, allow_teardown=False):
    """Validate the common AE run envelope without assuming a metric exists."""
    directory = Path(directory)
    analysis = json.loads((directory / "analysis.json").read_text())
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(analysis, dict) or not isinstance(manifest, dict):
        raise ValueError(f"{directory}: analysis and manifest must be objects")
    plan = manifest["plan"]
    if not isinstance(plan, dict):
        raise ValueError(f"{directory}: manifest plan must be an object")
    if plan["ratio"] != ratio or plan["system"] not in SYSTEMS:
        return None
    app = plan["app"]
    if app not in APPS:
        raise ValueError(f"unknown application: {app}")
    successful = not (analysis.get("exit_status") != 0 or analysis.get("correctness") != "pass"
            or analysis.get("status") != "passed" or manifest.get("status") != "passed"
            or manifest.get("exit_status") != 0)
    if (analysis.get("application") != app or analysis.get("ratio") != ratio
            or analysis.get("system") != SYSTEMS[plan["system"]]
            or analysis.get("run_id") != plan["run_id"]
            or analysis.get("environment") != plan["site"]):
        raise ValueError(f"{directory}: analysis/manifest context mismatch")
    warning = ""
    if not successful:
        recovered = None
        if (allow_teardown and manifest.get("status") != "passed"
                and manifest.get("exit_status", analysis.get("exit_status")) == analysis.get("exit_status")
                and analysis.get("exit_status") in (124, -15, -9)):
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
            from measurement_acceptance import recover_measurement
            recovered = recover_measurement(directory, analysis, manifest)
        if recovered is None:
            raise ValueError(f"{directory}: only successful verified runs are accepted "
                             "(or verified Starfish KV postcheck teardown failures)")
        warning = recovered["measurement_warning"]
    footprint = plan["workload_footprint_bytes"]
    if isinstance(footprint, bool) or not isinstance(footprint, int) or footprint <= 0:
        raise ValueError("recorded application footprint must be positive bytes")
    return {"directory": directory, "analysis": analysis, "manifest": manifest,
            "manifest_path": manifest_path, "plan": plan, "footprint": footprint,
            "log": directory / "client.log", "measurement_warning": warning,
            "execution_status": "teardown_failed" if warning else "passed"}


def read_metadata(context):
    directory = context["directory"]
    plan = context["plan"]
    manifest = context["manifest"]
    manifest_path = context["manifest_path"]
    footprint = context["footprint"]
    app = plan["app"]
    ratio = plan["ratio"]
    if plan.get("client_env", {}).get("FARLIB_RUNTIME_METADATA") != "1":
        raise ValueError("manifest must record FARLIB_RUNTIME_METADATA=1")
    if plan.get("client_env", {}).get("FARLIB_FIXED_SIX_GROUPS", "0") != "0":
        raise ValueError("metadata does not yet cover fixed-six policy tables; "
                         "FARLIB_FIXED_SIX_GROUPS must be 0")
    log = context["log"]
    records = []
    with log.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if not line.lstrip().startswith(PREFIX + " "):
                continue
            record = parse_line(line)
            record["_source"] = f"{log.resolve()}:{number}"
            records.append(record)
    if len(records) != 1:
        raise ValueError(f"{directory}: expected exactly one metadata record, got {len(records)}")
    record = records[0]
    if record["system"] != plan["system"]:
        raise ValueError("runtime/manifest system mismatch")
    return {
        "workload": APPS[app], "system": SYSTEMS[record["system"]],
        "metric": "metadata_space_pct",
        "value": format(100 * record["accounted_bytes"] / footprint, ".12g"),
        "unit": "% app memory", "source_type": "measured", "source": record["_source"],
        "exit_status": 0, "correctness": "pass", "run_id": plan["run_id"],
        "environment": plan["site"], "ratio": ratio,
        "measurement_phase": "final_work_end", "app_memory_bytes": footprint,
        # Runtime schema v1 named its core-only field metadata_bytes.
        # The current Figure 12 numerator includes the separately logged
        # auxiliary storage, so expose that raw core field unambiguously.
        "metadata_bytes": record["accounted_bytes"],
        "core_metadata_bytes": record["metadata_bytes"],
        "metadata_numerator": "accounted_bytes",
        **{key: record[key] for key in
           ("measurement_aux_bytes", "accounted_bytes",
            *COMPONENTS, *COUNTS, "boundary_sequence", "snapshot_us", "accounting", "scope")},
        "client_sha256": manifest["client_sha256"],
        "manifest_source": str(manifest_path.resolve()),
    }


def read_run(directory, *, ratio=25):
    context = load_run_context(directory, ratio=ratio)
    return read_metadata(context) if context is not None else None


def collect_run_dirs(directories, *, ratio=25):
    if not 1 <= ratio <= 100:
        raise ValueError("ratio must be 1..100")
    rows, seen = [], set()
    for directory in directories:
        row = read_run(directory, ratio=ratio)
        if row is None:
            continue
        key = row["workload"], row["system"]
        if key in seen:
            raise ValueError(f"multiple runs for {key}; select one repetition directory")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("no successful runtime metadata measurements found")
    footprints = {}
    for row in rows:
        previous = footprints.setdefault(row["workload"], row["app_memory_bytes"])
        if previous != row["app_memory_bytes"]:
            raise ValueError("matched runtimes disagree on application footprint")
    return rows


def collect_rows(root, *, ratio=25):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("logs root must exist")
    directories = [analysis.parent
                   for analysis in sorted(root.rglob("analysis.json"))
                   if (analysis.parent / "manifest.json").is_file()]
    return collect_run_dirs(directories, ratio=ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--logs-root", type=Path)
    selection.add_argument("--run-dir", type=Path, action="append",
                           help="Select exact run directories; repeat for matched conditions.")
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    try:
        rows = (collect_run_dirs(args.run_dir, ratio=args.ratio) if args.run_dir
                else collect_rows(args.logs_root, ratio=args.ratio))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            with args.summary.open("x") as stream:
                json.dump({"schema_version": 2, "metric": "metadata_space_pct",
                           "numerator": "core_metadata_bytes + measurement_aux_bytes",
                           "phase": "final_work_end", "runs": rows}, stream, indent=2)
                stream.write("\n")
        print(f"wrote {len(rows)} measured metadata rows to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
