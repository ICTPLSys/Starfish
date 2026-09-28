#!/usr/bin/env python3
"""Run Figure 10 KV-B and NQ offered-load points in a verified batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
AE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AE_ROOT / "scripts/common"))

from batch_status import classify_case
from collect import collect
from nq_latency import DEFAULT_MAX_QUEUE_DELAY_US as NQ_QUEUE_DELAY_US
from paper_loads import OFFERED_LOAD_OPS

SYSTEMS = ("nonft", "starfish", "hydra", "carbink")
APPS = ("kv-b", "nq")
SYSTEM_DISPLAY = {"nonft": "NONFT", "starfish": "STARFISH",
                  "hydra": "HYDRA", "carbink": "CARBINK"}
HEADER_DISPLAY = {"nonft": "nonFT", "starfish": "Starfish",
                  "hydra": "Hydra", "carbink": "Carbink"}
APP_DISPLAY = {"kv-b": "KV-B", "nq": "NQ"}


def selection(value, choices):
    items = value.split(",")
    if not items or any(item not in choices for item in items):
        raise ValueError("choose a comma-separated list from " + ",".join(choices))
    if len(items) != len(set(items)):
        raise ValueError("duplicate application/system selection")
    return items


def selected_apps(value):
    chosen = selection(value, APPS)
    # A multi-application batch always keeps the paper-facing KV-B, then NQ order.
    return [app for app in APPS if app in chosen]


def plans(args):
    apps = selected_apps(args.apps)
    systems = selection(args.systems, SYSTEMS)
    if not apps:
        raise ValueError("at least one application must be selected")
    if args.recipe is not None and (len(apps) != 1 or len(systems) != 1):
        raise ValueError(
            "an explicit recipe requires one selected application and one selected system")

    explicit = None
    if args.loads_ops is not None:
        if len(apps) != 1:
            raise ValueError("explicit --loads-ops requires exactly one selected application")
        try:
            explicit = [int(value) for value in args.loads_ops.split(",")]
        except ValueError as error:
            raise ValueError("offered loads must be positive, distinct integers") from error
        if (not explicit or any(value <= 0 for value in explicit)
                or len(explicit) != len(set(explicit))):
            raise ValueError("offered loads must be positive, distinct integers")

    # System-major order is intentional: each system completes KV-B, then NQ,
    # before the next system is started.
    for system in systems:
        for app in apps:
            queue_us = args.max_queue_delay_us
            if queue_us is None:
                queue_us = NQ_QUEUE_DELAY_US if app == "nq" else 0
            recipe = args.recipe or args.config_root / app / f"{system}.config"
            load_source = ("explicit" if explicit is not None else
                           "ae_selected_load_grid" if app == "nq"
                           else "paper_figure10_x_axis")
            loads = explicit if explicit is not None else OFFERED_LOAD_OPS[app][system]
            for load in loads:
                for repeat in range(1, args.repeats + 1):
                    run_id = f"{app}-{system}-{load}ops-r{repeat}"
                    directory = args.out / "runs" / run_id
                    command = [
                        sys.executable, str(AE_ROOT / "scripts/common/run_case.py"),
                        "--app", app, "--system", system, "--ratio", str(args.ratio),
                        "--site", str(args.site), "--recipe", str(recipe),
                        "--build-root", str(args.build_root), "--out", str(directory),
                        "--timeout", str(args.timeout), "--offered-load-ops", str(load),
                        "--latency-warmup-ms", str(args.warmup_ms),
                        "--latency-measure-ms", str(args.measure_ms),
                        "--latency-drain-ms", str(args.drain_ms),
                        "--max-queue-delay-us", str(queue_us),
                    ]
                    yield {"app": app, "system": system, "run_id": run_id,
                           "repeat": repeat, "offered_load_ops": load,
                           "load_source": load_source,
                           "command": command, "run_dir": str(directory)}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _header(systems):
    if tuple(systems) == SYSTEMS:
        return "This script will run nonFT, Starfish, Hydra, Carbink sequentially."
    return "This script will run " + ", ".join(HEADER_DISPLAY[s] for s in systems) \
        + " sequentially."


def _status_word(status):
    return "finish" if status == "passed" else "WARNING" if status in ("timeout", "warning") else "ERROR"


def _new_counter():
    return {"passed": 0, "timeout": 0, "warning": 0, "error": 0}


def _merge_counter(target, status):
    target[status] = target.get(status, 0) + 1


def _finish_line(prefix, counter):
    return (f"{prefix} FINISH (passed={counter['passed']} "
            f"warnings={counter['timeout'] + counter['warning']} errors={counter['error']})")


def execute(parser, args):
    try:
        if (not 1 <= args.ratio <= 100 or min(args.repeats, args.timeout,
                args.warmup_ms, args.measure_ms, args.drain_ms) <= 0):
            raise ValueError("ratio must be 1..100; durations and repeats must be positive")
        if (args.max_queue_delay_us is not None
                and not 0 <= args.max_queue_delay_us <= 60_000_000):
            raise ValueError("queue deadline must be 0..60000000 microseconds")
        for key in ("site", "build_root", "config_root", "out"):
            setattr(args, key, getattr(args, key).resolve())
        if args.recipe:
            args.recipe = args.recipe.resolve()
        if args.dry_run and not args.site.is_file():
            args.site = AE_ROOT / "scripts/common/site.example.json"
        if not args.site.is_file():
            raise ValueError(f"machine configuration missing: {args.site}")

        apps = selected_apps(args.apps)
        systems = selection(args.systems, SYSTEMS)
        matrix = list(plans(args))
        if args.dry_run:
            # run_case's dry-run path is pure planning: it does not create the
            # output directory, probe a server, or write an experiment record.
            for plan in matrix:
                result = subprocess.run(plan["command"] + ["--dry-run"], cwd=AE_ROOT,
                                        stdin=subprocess.DEVNULL, capture_output=True,
                                        text=True, check=False)
                if result.returncode:
                    raise ValueError(result.stderr.strip() or "case planning failed")
                print(json.dumps({**plan, "effective_plan": json.loads(result.stdout)},
                                 sort_keys=True))
            return 0

        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "runs").mkdir()
        print(_header(systems), flush=True)
        status_path = args.out / "batch-status.json"
        _write_json(args.out / "batch-plan.json", matrix)
        completed = []
        failures = False
        stop_batch = False
        matrix_by_curve = {(system, app): [plan for plan in matrix
                                           if plan["system"] == system and plan["app"] == app]
                           for system in systems for app in apps}

        for system in systems:
            system_counter = _new_counter()
            title = "NonFT" if system == "nonft" else HEADER_DISPLAY[system]
            print(f"{title} Start...", flush=True)
            for app in apps:
                app_counter = _new_counter()
                print(f"{SYSTEM_DISPLAY[system]} App {APP_DISPLAY[app]} START", flush=True)
                for plan in matrix_by_curve[system, app]:
                    print(f"START {plan['run_id']}", flush=True)
                    log_path = args.out / (plan["run_id"] + ".runner.log")
                    returncode = 2
                    invocation_error = ""
                    try:
                        with log_path.open("w", encoding="utf-8") as output:
                            result = subprocess.run(
                                plan["command"], cwd=AE_ROOT, stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, check=False)
                        returncode = result.returncode
                    except (OSError, subprocess.SubprocessError) as error:
                        invocation_error = str(error)

                    classified = classify_case(Path(plan["run_dir"]), returncode)
                    if invocation_error:
                        classified.update(status="error", reason=
                                          "runner invocation failed: " + invocation_error)
                    record = {
                        "run_id": plan["run_id"], "app": plan["app"],
                        "system": plan["system"], "repeat": plan["repeat"],
                        "offered_load_ops": plan["offered_load_ops"],
                        "run_dir": plan["run_dir"], **classified,
                    }
                    repeat = plan["repeat"]
                    csv_name = "figure10.csv" if args.repeats == 1 \
                        else f"figure10-r{repeat}.csv"
                    collection_error = None
                    try:
                        collect(args.out / "runs", args.out / csv_name, repeat,
                                status_output=args.out / "collection-status.json")
                    except (OSError, KeyError, TypeError, ValueError) as error:
                        # A collector integrity error (mixed profile or duplicate
                        # point) is unsafe to continue with; ordinary case errors
                        # above remain warning/error records and are continued.
                        failures = True
                        collection_error = str(error)
                        record["collection"] = {
                            "status": "error", "safe_to_continue": False,
                            "reason": collection_error,
                        }
                        stop_batch = True
                    else:
                        status_path_for_collection = args.out / "collection-status.json"
                        if status_path_for_collection.is_file():
                            collection_record = json.loads(
                                status_path_for_collection.read_text(encoding="utf-8"))
                            record["collection"] = collection_record
                            current_errors = [error for error in collection_record.get("errors", [])
                                              if error.get("run_dir") == plan["run_dir"]]
                            if current_errors:
                                # A failed point is retained in collection-status,
                                # while verified rows remain in the CSV.
                                if classified["status"] == "passed":
                                    classified.update(status="error", reason=(
                                        "collector rejected this point: " +
                                        "; ".join(error["reason"] for error in current_errors)))
                                    record.update(classified)
                                    failures = True
                    status = classified["status"]
                    if collection_error:
                        status = "error"
                        failures = True
                        classified.update(status="error", safe_to_continue=False,
                                          reason="collector integrity error: " + collection_error)
                        record.update(classified)
                        print(f"ERROR {SYSTEM_DISPLAY[system]} {APP_DISPLAY[app]} "
                              f"{plan['offered_load_ops']} QPS collection: {collection_error}",
                              file=sys.stderr, flush=True)
                    _merge_counter(app_counter, status)
                    _merge_counter(system_counter, status)
                    if status != "passed":
                        failures = True
                        level = _status_word(status)
                        reason = (record.get("reason") or
                                  "case did not pass verification")
                        print(f"{SYSTEM_DISPLAY[system]} {APP_DISPLAY[app]} "
                              f"{plan['offered_load_ops']} QPS {level}: {reason}",
                              file=sys.stderr, flush=True)
                    else:
                        print(f"{SYSTEM_DISPLAY[system]} {APP_DISPLAY[app]} "
                              f"{plan['offered_load_ops']} QPS finish", flush=True)
                    completed.append(record)
                    _write_json(status_path, completed)
                    if stop_batch or not classified.get("safe_to_continue", False):
                        stop_batch = True
                        break
                app_prefix = f"{SYSTEM_DISPLAY[system]} App {APP_DISPLAY[app]}"
                if stop_batch:
                    print(f"{app_prefix} ABORTED", flush=True)
                else:
                    print(_finish_line(app_prefix, app_counter), flush=True)
                if stop_batch:
                    break
            system_prefix = HEADER_DISPLAY[system]
            if stop_batch:
                print(f"{system_prefix} ABORTED", flush=True)
            else:
                print(_finish_line(system_prefix, system_counter), flush=True)
            if stop_batch:
                break

        _write_json(status_path, completed)
        total_counter = _new_counter()
        for record in completed:
            _merge_counter(total_counter, record["status"])
        if stop_batch:
            print(f"BATCH ABORTED (attempted={len(completed)} planned={len(matrix)} "
                  f"passed={total_counter['passed']} warnings={total_counter['timeout'] + total_counter['warning']} "
                  f"errors={total_counter['error']})", flush=True)
        else:
            print(_finish_line("BATCH", total_counter), flush=True)
        if failures:
            print(f"completed {len(completed)} cases with failures; results: {args.out}",
                  file=sys.stderr)
            return 1
        print(f"completed {len(completed)} cases; measured CSV and logs: {args.out}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"error: {error}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", default="kv-b,nq")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--site", type=Path, default=AE_ROOT / "data/site.json")
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--build-root", type=Path, default=AE_ROOT / "build")
    parser.add_argument("--config-root", type=Path, default=AE_ROOT / "configs")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--loads-ops", help="explicit aggregate request rates, comma separated")
    parser.add_argument("--warmup-ms", type=int, default=10000)
    parser.add_argument("--measure-ms", type=int, default=10000)
    parser.add_argument("--drain-ms", type=int, default=60000)
    parser.add_argument("--max-queue-delay-us", type=int, default=None,
                        help="pre-execution queue deadline in us; NQ default 1500000, KV default 0; 0 disables")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=AE_ROOT / "results/figure10" /
                        ("batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
