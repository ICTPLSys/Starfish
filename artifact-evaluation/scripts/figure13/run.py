#!/usr/bin/env python3
"""Run Figure 13 through the normal AE site/deployment/correctness workflow.

Each case calls common/run_case.py with an explicit KV recovery overlay.
No frozen binaries, account-specific paths or shared filesystem are required.
--dry-run renders plans only: no files, SSH, services or fault injection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ("hydra", "carbink", "starfish")
SCENARIOS = ("1-node", "2-node")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select(value, allowed):
    values = value.split(",")
    if not values or len(set(values)) != len(values) or any(x not in allowed for x in values):
        raise ValueError("select distinct values from " + ",".join(allowed))
    return values


def make_plans(args):
    if args.apps != "kv-b" or args.ratio != 25:
        raise ValueError("Figure 13 uses the complete KV-B workload at ratio 25")
    if args.repeats < 1 or args.timeout < 1:
        raise ValueError("repeats and timeout must be positive")
    if not args.site.is_file():
        raise ValueError("AE site missing; prepare data/site.json first: " + str(args.site))
    if args.observer_cpu is not None and args.observer_cpu < 0:
        raise ValueError("observer-cpu must be nonnegative")
    plans = []
    for system in select(args.systems, SYSTEMS):
        for scenario in select(args.scenarios, SCENARIOS):
            for repeat in range(1, args.repeats + 1):
                run_id = f"kv-b-{system}-{scenario}-r{repeat}"
                directory = args.out / "runs" / run_id
                command = [sys.executable, "-B", str(ROOT / "scripts/common/run_case.py"),
                    "--app", "kv-b", "--system", system, "--ratio", "25",
                    "--site", str(args.site), "--build-root", str(args.build_root),
                    "--out", str(directory), "--timeout", str(args.timeout),
                    "--figure13-scenario", scenario, "--figure13-repeat", str(repeat)]
                if args.observer_cpu is not None:
                    command += ["--figure13-observer-cpu", str(args.observer_cpu)]
                # Render every selected case before starting the first service.
                # Common dry-run performs no local writes or remote operations.
                rendered = subprocess.run(command + ["--dry-run"], text=True,
                                          capture_output=True, check=False)
                if rendered.returncode:
                    raise ValueError(f"{run_id}: " + rendered.stderr.strip())
                effective_plan = json.loads(rendered.stdout)
                plans.append(dict(app="kv-b", system=system, scenario=scenario,
                    ratio=25, repeat=repeat, run_id=run_id, run_dir=str(directory),
                    result=str(directory / "figure13-result.json"), command=command,
                    effective_plan=effective_plan,
                    failure_scope="logical_service_process",
                    metric="background rebuild start to final completion, seconds; excludes detection",
                    expected_completed_requests=1_000_000_000))
    return plans


def cleanup_verified(directory):
    path = directory / "manifest.json"
    if not path.is_file():
        # Preflight failures happen before any memory service is started.
        return all(read(item).get("pid") is None
                   for item in directory.glob("endpoints/*/endpoint-manifest.json"))
    manifest = read(path)
    endpoints = manifest.get("endpoints", [])
    return bool(endpoints) and all(
        state.get("status") in ("stopped", "already_exited", "not_started", "preflight_pass")
        or state.get("cleanup_verified") is True for state in endpoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "data/site.json")
    parser.add_argument("--build-root", type=Path, default=ROOT / "build")
    parser.add_argument("--apps", default="kv-b")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--observer-cpu", type=int)
    parser.add_argument("--out", type=Path, default=ROOT / "results/figure13" /
        ("batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        for name in ("site", "build_root", "out"):
            setattr(args, name, getattr(args, name).resolve())
        plans = make_plans(args)
        batch = dict(schema="figure13-site-batch-v1", site=str(args.site),
                     build_root=str(args.build_root), cases=plans,
                     failure_scope="logical_service_process")
        if args.dry_run:
            print(json.dumps(dict(batch, dry_run=True), indent=2))
            return 0
        for case in plans:
            plan = case["effective_plan"]
            for key in ("client_bin", "local_server_source"):
                if not Path(plan[key]).is_file():
                    raise ValueError("build Figure 13 sources first; missing " + plan[key])
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "runs").mkdir()
        save(args.out / "batch-plan.json", dict(batch, started=now()))
        states, runs = [], []
        for case in plans:
            directory = Path(case["run_dir"])
            state = dict(run_id=case["run_id"], started=now(), status="RUNNING", passed=False)
            save(args.out / "batch-status.json", dict(updated=now(), cases=states + [state]))
            print(f"START {case['run_id']}", flush=True)
            # Do not create the case directory here: common/run_case owns it.
            try:
                completed = subprocess.run(case["command"], check=False)
                state["exit_status"] = completed.returncode
                if completed.returncode:
                    raise RuntimeError("AE case failed; inspect its manifest/client.log")
                from raw_runs import read_run
                run, _, details = read_run(case["result"])
                if run["system"] != case["system"] or run["scenario"] != case["scenario"]:
                    raise RuntimeError("observed condition differs from selected case")
                save(directory / "figure13-analysis.json", dict(run=run, evidence=details))
                state.update(status="FINISHED", passed=True)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                state.update(status="ERROR", error=str(exc))
            state["cleanup_verified"] = cleanup_verified(directory)
            if not state["cleanup_verified"]:
                state.update(status="ABORTED", passed=False,
                             error="owned service cleanup unverified; batch stopped")
            if state["passed"]:
                runs.append({key: case[key] for key in (
                    "system", "scenario", "repeat", "run_id", "result")})
            state["finished"] = now()
            states.append(state)
            save(args.out / "batch-status.json", dict(updated=now(), cases=states))
            save(args.out / "runs.json", {
                "schema": "figure13-native-runs-v1", "batch_plan": "batch-plan.json",
                "runs": runs, "pending": [item["run_id"] for item in states if not item["passed"]]})
            if not state["cleanup_verified"]:
                break
        return 0 if len(states) == len(plans) and all(x["passed"] for x in states) else 1
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
