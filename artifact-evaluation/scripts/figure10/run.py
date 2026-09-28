#!/usr/bin/env python3
"""Pass Figure 10 paper load points to KV-B/NQ applications.

Application argument contract:
  KV-B: kvs_throughput CONFIG --workload kv-b --offered-load-ops INTEGER
  NQ:   nhop_graph CONFIG GRAPH_PREFIX --offered-load-ops INTEGER

--offered-load-ops is the aggregate application request rate, not per-worker.
Rate control and latency measurement belong to the application. This launcher
expects ready memory services and endpoint/CPU settings supplied separately;
it does not build binaries, render runtime configs or start/stop servers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
from paper_loads import OFFERED_LOAD_OPS, PLOT_UNITS

AE_ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ("hydra", "carbink", "starfish", "nonft")
BINARY = {
    "kv-b": Path("benchmark/kvs/kvs_throughput"),
    "nq": Path("benchmark/microbenchmarks/nhop_graph"),
}


def selection(value, choices):
    items = value.split(",")
    if not items or any(item not in choices for item in items):
        raise ValueError("choose a comma-separated list from " + ",".join(choices))
    if len(items) != len(set(items)):
        raise ValueError("duplicate application/system selection")
    return items


def plans(args):
    for app in selection(args.apps, OFFERED_LOAD_OPS):
        for system in selection(args.systems, SYSTEMS):
            binary = args.build_root / system / BINARY[app]
            config = args.config_root / app / f"{system}.config"
            unit, scale = PLOT_UNITS[app]
            for load in OFFERED_LOAD_OPS[app][system]:
                for repeat in range(1, args.repeats + 1):
                    command = [str(binary), str(config)]
                    if app == "kv-b":
                        command += ["--workload", "kv-b"]
                    else:
                        command.append(str(args.graph))
                    command += ["--offered-load-ops", str(load)]
                    yield {
                        "app": app, "system": system,
                        "run_id": f"{app}-{system}-{load}ops-r{repeat}",
                        "offered_load_ops": load,
                        "offered_load": load / scale, "load_unit": unit,
                        "load_source": "paper_figure10_x_axis",
                        "repeat": repeat, "command": command,
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apps", default="kv-b,nq")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--build-root", type=Path, default=AE_ROOT / "build")
    parser.add_argument("--config-root", type=Path, default=AE_ROOT / "configs")
    parser.add_argument("--graph", type=Path, default=Path("/data/starfish-ae/bfs/graph"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=AE_ROOT / "results/figure10" /
                        ("batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--dry-run", action="store_true",
                        help="print application commands only; no files or processes")
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        args.build_root = args.build_root.resolve()
        args.config_root = args.config_root.resolve()
        args.graph = args.graph.resolve()
        matrix = list(plans(args))
        if args.dry_run:
            for plan in matrix:
                print(json.dumps(plan, sort_keys=True))
            return 0

        args.out.mkdir(parents=True, exist_ok=False)
        failures = 0
        for plan in matrix:
            directory = args.out / plan["run_id"]
            directory.mkdir()
            (directory / "plan.json").write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            print(shlex.join(plan["command"]), flush=True)
            # No application-readiness gate: execute the declared interface.
            # A missing binary or unsupported argument is recorded as a failure.
            result = {"run_id": plan["run_id"], "status": "failed",
                      "exit_status": None, "error": None}
            try:
                with (directory / "client.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(plan["command"], stdout=log,
                                               stderr=subprocess.STDOUT, check=False)
                result["exit_status"] = completed.returncode
                result["status"] = "completed" if completed.returncode == 0 else "failed"
            except OSError as exc:
                result["error"] = str(exc)
            (directory / "status.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")
            failures += result["status"] == "failed"
        # Exit success is not a correctness/P99 measurement assertion. No
        # paper latency values or fabricated measurement CSVs are emitted.
        print(f"completed matrix: {len(matrix)} cases, {failures} failed; logs: {args.out}")
        return 1 if failures else 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
