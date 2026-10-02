#!/usr/bin/env python3
"""Collect verified Figure 10 KV-B/NQ latency points without averaging runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

AE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AE_ROOT / "scripts/common"))
import carbink_support
import kv_latency
from measurement_acceptance import recover_measurement
import nq_latency
from figure9_support import SYSTEM_LABEL
from run_case import config_value
from workloads import validate_design2, validate_hydra, validate_nonft_kv

COLUMNS = ("workload", "system", "offered_load", "p99_latency", "load_unit",
           "latency_unit", "source_type", "source", "exit_status", "correctness",
           "offered_load_ops", "scheduled", "completed_in_window_ops_s",
           "p99_service_us", "p99_dispatch_us", "repeat", "profile_id",
           "actual_offered_ops_s", "completed", "deadline_dropped", "drop_fraction",
           "max_queue_delay_us", "completed_after_deadline", "measurement_usable",
           "execution_status", "warning", "latency_metric", "p99_total_us")


def point(directory):
    analysis = json.loads((directory / "analysis.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    plan = manifest["plan"]
    app = plan.get("app")
    if app not in ("kv-b", "nq") or analysis.get("application") != app:
        raise ValueError("latency result has a mismatched application")
    latency_key = "nq_latency" if app == "nq" else "kv_latency"
    adapter = nq_latency if app == "nq" else kv_latency
    spec = plan.get(latency_key)
    if not spec:
        return None
    normal = (analysis.get("status") == "passed" and analysis.get("exit_status") == 0
              and analysis.get("correctness") == "pass"
              and manifest.get("status") == "passed")
    recovered = False
    measurement_warning = ""
    endpoints = analysis.get("endpoints", manifest.get("endpoints", []))
    if not isinstance(endpoints, list):
        raise ValueError("latency point endpoint records must be a list")
    if normal:
        if not endpoints or any(endpoint.get("status") not in ("stopped", "already_exited")
                                for endpoint in endpoints):
            raise ValueError("latency point has incomplete memory-service cleanup")
    else:
        accepted = recover_measurement(directory, analysis, manifest)
        if accepted is None:
            return None
        if not isinstance(accepted, dict):
            raise ValueError("measurement acceptance returned an invalid record")
        measurement_warning = str(accepted.get("measurement_warning") or "")
        result = accepted.get(latency_key, accepted)
        if not isinstance(result, dict) or "p99_latency_ns" not in result:
            raise ValueError("measurement acceptance returned no latency result")
        recovered = True
    log = (directory / "client.log").read_text(errors="replace")
    workers = plan["worker_profile"]
    if not recovered:
        result = adapter.parse_result(log, spec, expected_workers=workers["app_workers"],
                                      histogram_dir=directory / "histograms")[latency_key]
        saved = analysis.get(latency_key) or {}
        if saved.get("p99_latency_ns") != result["p99_latency_ns"]:
            raise ValueError("saved P99 differs from independently reparsed raw log")
    system = plan["system"]
    resident = int(config_value(plan["effective_config"], "local_resident_budget_bytes") or "0")
    if system == "nonft" and app == "kv-b":
        validate_nonft_kv(log, expected_workers=workers["app_workers"])
    elif system == "nonft":
        if not re.search(r"^exact used bytes: 0\s*$", log, re.M):
            raise ValueError("NQ NonFT remote allocations were not released")
    elif system == "starfish":
        validate_design2(log)
    elif system == "hydra":
        validate_hydra(log, endpoint_count=len(endpoints), expected_workers=workers,
                       expected_resident_bytes=resident)
    elif system == "carbink":
        carbink_support.validate_log(log, workers, expected_resident_bytes=resident)
    else:
        raise ValueError("unknown latency system")
    repeat_match = re.search(r"-r([1-9][0-9]*)$", directory.name)
    if repeat_match is None:
        raise ValueError("latency run directory must identify its repeat")
    measured = result["phases"]["measurement"]
    profile = {
        "client_sha256": manifest.get("client_sha256"),
        "application": app,
        "placement": {key: plan.get(key) for key in (
            "compute_ip", "numa_node", "ib_device", "ib_port", "ratio",
            "worker_profile", "configured_cpu_profile", "effective_config")},
        "environment": {key: value for key, value in plan.get("client_env", {}).items()
                        if key not in ("FARLIB_KVS_OFFERED_LOAD_OPS",
                                       "FARLIB_KVS_LATENCY_OUTPUT_DIR",
                                       "NHOP_OFFERED_LOAD_OPS", "NHOP_LATENCY_OUTPUT_DIR")},
        "measurement": {key: value for key, value in spec.items()
                        if key not in ("offered_load_ops", "histogram_dir")},
    }
    profile_id = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    p99_scale = 1e6 if app == "nq" else 1e3
    return {
        "workload": "NQ" if app == "nq" else "KV-B", "system": SYSTEM_LABEL[system],
        "offered_load": spec["offered_load_ops"] / (1e3 if app == "nq" else 1e6),
        "p99_latency": (int(measured["p99_service_ns"]) if app == "kv-b"
                        else result["p99_latency_ns"]) / p99_scale,
        "latency_metric": "service" if app == "kv-b" else "total",
        "p99_total_us": result["p99_latency_ns"] / 1000,
        "load_unit": "Kops" if app == "nq" else "Mops",
        "latency_unit": "ms" if app == "nq" else "us",
        "source_type": "measured", "source": str((directory / "analysis.json").resolve()),
        "exit_status": analysis.get("exit_status", 0), "correctness": "pass",
        "offered_load_ops": spec["offered_load_ops"], "scheduled": measured["scheduled"],
        "completed_in_window_ops_s": measured["completed_in_window_ops_s"],
        # Component fields remain microseconds for both workloads; only the
        # primary NQ curve is converted to the plot's milliseconds contract.
        "p99_service_us": int(measured["p99_service_ns"]) / 1000,
        "p99_dispatch_us": int(measured["p99_dispatch_ns"]) / 1000,
        "repeat": int(repeat_match[1]),
        "profile_id": profile_id,
        "actual_offered_ops_s": measured["realized_offered_ops_s"],
        "completed": measured["completed"],
        "deadline_dropped": measured["deadline_dropped"],
        "drop_fraction": measured["drop_fraction"],
        "max_queue_delay_us": spec.get("max_queue_delay_us", 0),
        "completed_after_deadline": measured["completed_after_deadline"],
        "measurement_usable": 1,
        "execution_status": "teardown_failed" if recovered else "passed",
        "warning": measurement_warning,
    }


def _write_collection_status(path, *, repeat, rows, errors, status="error"):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "repeat": repeat,
                                "rows": len(rows), "errors": errors},
                               indent=2) + "\n", encoding="utf-8")


def collect(root, output, repeat=1, status_output=None):
    root = Path(root)
    output = Path(output)
    if not root.is_dir():
        raise ValueError(f"run directory not found: {root}")
    rows, keys, profiles, errors = [], set(), {}, []
    for path in sorted(root.rglob("analysis.json")):
        repeat_match = re.search(r"-r([1-9][0-9]*)$", path.parent.name)
        if repeat_match is None:
            errors.append({"run_dir": str(path.parent), "status": "error",
                           "reason": "latency run directory must identify its repeat"})
            continue
        if int(repeat_match[1]) != repeat:
            continue
        try:
            row = point(path.parent)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append({"run_dir": str(path.parent), "status": "error",
                           "reason": str(error)})
            continue
        if row is None:
            errors.append({"run_dir": str(path.parent), "status": "error",
                           "reason": "case did not produce a verified measurement"})
            continue
        profile_key = (row["workload"], row["system"])
        if (profile_key in profiles and profiles[profile_key] != row["profile_id"]):
            raise ValueError("mixed runtime/placement/measurement profiles in one curve")
        profiles[profile_key] = row["profile_id"]
        key = (row["workload"], row["system"], row["offered_load_ops"])
        if key in keys:
            raise ValueError("duplicate measured load point; select one batch/repeat")
        keys.add(key)
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_collection_status(status_output, repeat=repeat, rows=rows, errors=errors,
                             status="passed" if not errors else "error")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--status-output", type=Path)
    args = parser.parse_args()
    try:
        if args.repeat < 1:
            raise ValueError("repeat must be positive")
        output = args.output or args.logs_root / "figure10.csv"
        status_output = args.status_output or args.logs_root / "collection-status.json"
        rows = collect(args.logs_root, output, args.repeat, status_output=status_output)
        print(f"collected {len(rows)} measured points: {output}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
