#!/usr/bin/env python3
"""Run or plan the Figure 12 metadata/EC-CPU batch.

The default is the complete system-major batch.  ``--dry-run`` is the
explicit read-only mode; ``--execute`` remains accepted as a compatibility
spelling. Figure 12 owns matrix orchestration and provenance; deployment,
cleanup and correctness remain the responsibility of ``run_case.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True

AE_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = AE_ROOT / "scripts" / "common"
SCRIPTS_DIR = AE_ROOT / "scripts"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from batch_status import classify_case  # noqa: E402
from figure12 import collect as figure12_collect  # noqa: E402


APPS = ("bfs", "llama", "mg", "wordcount", "kv-b", "kv-a", "kv-s", "nq")
SYSTEMS = ("starfish", "carbink")
METRICS = ("all", "metadata", "ec-cpu")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection(value: str, allowed: Iterable[str], label: str) -> List[str]:
    choices = tuple(allowed)
    values = value.split(",")
    if (not values or any(not item or item not in choices for item in values)
            or len(values) != len(set(values))):
        raise ValueError(
            f"{label} must be a comma-separated selection from {','.join(choices)} "
            "without duplicates")
    return values


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_site(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"site file must contain a JSON object: {path}")
    return value


def _mapping_entry(mapping: Dict[str, Any], app: str,
                   system: str) -> Any:
    root = mapping.get("sites", mapping)
    if not isinstance(root, dict):
        raise ValueError("site map must contain an object or a 'sites' object")
    for key in (f"{app}/{system}", f"{system}/{app}"):
        if key in root:
            return root[key]
    for outer, inner in ((app, system), (system, app)):
        value = root.get(outer)
        if isinstance(value, dict) and inner in value:
            return value[inner]
    return None


def _site_for(base_path: Path, site_map: Optional[Path], app: str,
              system: str, metric: str) -> Tuple[Path, Dict[str, Any], str]:
    base = _load_site(base_path)
    source = str(base_path.resolve())
    if site_map is not None:
        mapping = _load_site(site_map)
        entry = _mapping_entry(mapping, app, system)
        if entry is not None:
            if isinstance(entry, str):
                selected = Path(entry)
                if not selected.is_absolute():
                    selected = site_map.parent / selected
                selected = selected.resolve()
                base = _load_site(selected)
                source = str(selected)
            elif isinstance(entry, dict):
                if "path" in entry:
                    selected = Path(entry["path"])
                    if not selected.is_absolute():
                        selected = site_map.parent / selected
                    selected = selected.resolve()
                    base = _load_site(selected)
                    source = str(selected)
                overlay = entry.get("overlay", {
                    key: value for key, value in entry.items() if key != "path"
                })
                if not isinstance(overlay, dict):
                    raise ValueError("site map overlay must be an object")
                base = _merge(base, overlay)
            else:
                raise ValueError("site map entries must be paths or objects")

    # Make metric selection authoritative: a stale opt-in in the base site
    # must not silently enable an unselected producer.
    flags: Dict[str, bool] = {
        "runtime_metadata": metric in ("all", "metadata"),
        "runtime_ec_cpu": metric in ("all", "ec-cpu"),
    }
    base = _merge(base, flags)
    return base_path, base, source


def _materialize_site(payload: Dict[str, Any], out: Path, app: str,
                      system: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{app}-{system}.json"
    _write_json(destination, payload)
    return destination


def _case_command(args: argparse.Namespace, app: str, system: str, ratio: int,
                  run_dir: Path, site_path: Path, *, dry_run: bool) -> List[str]:
    command = [
        sys.executable, str(AE_ROOT / "scripts" / "common" / "run_case.py"),
        "--app", app, "--system", system, "--ratio", str(ratio),
        "--site", str(site_path), "--out", str(run_dir),
        "--timeout", str(args.timeout), "--build-root", str(args.build_root),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _build_command(args: argparse.Namespace, system: str, *, dry_run: bool) -> List[str]:
    command = [
        "bash", str(AE_ROOT / "scripts" / "common" / "build.sh"),
        "--system", system, "--build-dir", str(args.build_root / system),
    ]
    if args.build_jobs is not None:
        command.extend(("--jobs", str(args.build_jobs)))
    if dry_run:
        command.append("--dry-run")
    return command


def plans(args: argparse.Namespace, *, site_paths: Optional[Dict[Tuple[str, str], Path]] = None,
          dry_run: bool = False) -> List[Dict[str, Any]]:
    """Return the Figure10-shaped ordered plan list used by the collector."""
    apps = _selection(args.apps, APPS, "applications")
    systems = _selection(args.systems, SYSTEMS, "systems")
    if not 1 <= args.ratio <= 100:
        raise ValueError("ratio must be 1..100")
    _positive(args.repeats, "repeats")
    _positive(args.timeout, "timeout")
    if args.build_jobs is not None:
        _positive(args.build_jobs, "build-jobs")
    result = []
    for system in systems:
        for app in apps:
            for repeat in range(1, args.repeats + 1):
                run_id = f"{app}-{system}-{args.ratio}-r{repeat}"
                run_dir = args.out / "runs" / run_id
                site_path = ((site_paths or {}).get((app, system))
                             or args.site)
                overlay_flags = {}
                if args.metric in ("all", "metadata"):
                    overlay_flags["runtime_metadata"] = True
                if args.metric in ("all", "ec-cpu"):
                    overlay_flags["runtime_ec_cpu"] = True
                result.append({
                    "app": app, "system": system, "ratio": args.ratio,
                    "repeat": repeat, "run_id": run_id,
                    "run_dir": str(run_dir), "site": str(site_path),
                    "site_overlay": overlay_flags, "metric": args.metric,
                    "command": _case_command(
                        args, app, system, args.ratio, run_dir, site_path,
                        dry_run=dry_run),
                })
    return result


def _effective_sites(args: argparse.Namespace, *, output: Optional[Path]
                     ) -> Tuple[Dict[Tuple[str, str], Path], Dict[str, Any]]:
    base = args.site.resolve()
    if not base.is_file():
        if args.execute:
            raise ValueError(f"site config missing: {base}")
        base = (AE_ROOT / "scripts" / "common" / "site.example.json").resolve()
    mapping_digest = None
    if args.site_map is not None:
        mapping_path = args.site_map.resolve()
        if not mapping_path.is_file():
            raise ValueError(f"site map missing: {mapping_path}")
        mapping_digest = _sha256(mapping_path)
    paths: Dict[Tuple[str, str], Path] = {}
    provenance: Dict[str, Any] = {
        "base": str(base), "base_sha256": _sha256(base),
        "map": str(args.site_map.resolve()) if args.site_map else None,
        "map_sha256": mapping_digest, "cases": {},
    }
    temp_output = output / "site-overlays" if output is not None else None
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if temp_output is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="figure12-site-")
        temp_output = Path(temp_dir.name)
    for system in _selection(args.systems, SYSTEMS, "systems"):
        for app in _selection(args.apps, APPS, "applications"):
            _, payload, source = _site_for(base, args.site_map, app, system,
                                           args.metric)
            path = _materialize_site(payload, temp_output, app, system)
            paths[(app, system)] = path
            provenance["cases"][f"{app}/{system}"] = {
                "source": source, "source_sha256": _sha256(Path(source)),
                "overlay": {key: value for key, value in payload.items()
                            if key in ("runtime_metadata", "runtime_ec_cpu")},
                "materialized": str(path),
            }
    if temp_dir is not None:
        provenance["_temporary_directory"] = temp_dir
    return paths, provenance


def _cleanup_site_provenance(provenance: Dict[str, Any]) -> None:
    temporary = provenance.pop("_temporary_directory", None)
    if temporary is not None:
        temporary.cleanup()


def _parse_dry_run_output(stdout: str) -> Dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"raw": lines[-1]}
    return value if isinstance(value, dict) else {"value": value}


def _new_batch(args: argparse.Namespace, matrix: List[Dict[str, Any]],
               site_provenance: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1, "kind": "figure12_batch", "status": "planned",
        "created_at": _now(), "updated_at": _now(),
        "selection": {
            "apps": _selection(args.apps, APPS, "applications"),
            "systems": _selection(args.systems, SYSTEMS, "systems"),
            "ratio": args.ratio, "repeats": args.repeats, "metric": args.metric,
        },
        "flags": {
            "runtime_metadata": args.metric in ("all", "metadata"),
            "runtime_ec_cpu": args.metric in ("all", "ec-cpu"),
        },
        "build": {"enabled": bool(args.build), "jobs": args.build_jobs,
                  "commands": {system: _build_command(args, system, dry_run=False)
                               for system in _selection(args.systems, SYSTEMS, "systems")}},
        "site": site_provenance,
        "case_count": len(matrix), "plan_file": "batch-plan.json",
        "status_file": "batch-status.json", "completed": 0,
        "passed": 0, "failed": 0, "aborted": False,
    }


def _save_batch(path: Path, batch: Dict[str, Any]) -> None:
    durable = {key: value for key, value in batch.items()
               if not key.startswith("_")}
    _write_json(path, durable)


def _validate_figure12_metrics(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate this completed case using only the local Figure 12 reader."""
    selected = plan["metric"]
    rows = figure12_collect.read_run(
        Path(plan["run_dir"]), ratio=plan["ratio"], repeat=plan["repeat"],
        metrics=selected, indexed=True)
    expected = tuple(figure12_collect.contract.METRIC_SELECTIONS[selected])
    actual = tuple(row.get("metric") for row in rows)
    if len(rows) != len(expected) or set(actual) != set(expected):
        raise ValueError(
            f"expected Figure12 metrics {expected}, got {actual or 'none'}")
    for row in rows:
        if row["metric"] == "local_ec_cpu_cycles" and (
                row.get("ec_schema_version") != 2 or
                row.get("ec_measurement_basis") != "initialization_plus_work"):
            raise ValueError("new Figure12 runs require initialization + Work EC "
                             "instrumentation; rebuild the selected client")
    return rows


def _invoke_case(plan: Dict[str, Any], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    node_log = log_path.with_suffix(".nodes-before.json")
    try:
        probe = subprocess.run(
            ["bash", str(AE_ROOT / "scripts/check_nodes.sh"), "--site",
             str(plan["site"]), "--require-idle", "--json"],
            cwd=AE_ROOT, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, check=False)
        nodes = json.loads(probe.stdout)
        _write_json(node_log, {"exit_status": probe.returncode,
                               "stderr": probe.stderr, "result": nodes})
        if (probe.returncode != 0 or not nodes.get("nodes") or
                not all(node.get("availability", {}).get("idle") is True
                        for node in nodes["nodes"])):
            raise ValueError("strict node-idle check did not pass")
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        return {**plan, "status": "error", "safe_to_continue": False,
                "measurement_usable": False, "exit_status": 2,
                "runner_exit_status": None, "runner_log": str(log_path),
                "reason": f"case not started: node-idle check failed: {error}"}
    invocation_error = ""
    returncode = 2
    try:
        workdir = Path(plan["run_dir"]).parent.parent / "workdirs" / plan["run_id"]
        workdir.mkdir(parents=True, exist_ok=False)
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(plan["command"], cwd=workdir,
                                    stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, check=False)
        returncode = result.returncode
    except (OSError, subprocess.SubprocessError) as error:
        invocation_error = str(error)
    classified = classify_case(Path(plan["run_dir"]), returncode)
    if invocation_error:
        classified.update(status="error", reason="runner invocation failed: " + invocation_error)
    elif (classified.get("status") == "passed"
          or classified.get("measurement_usable") is True):
        try:
            metrics = _validate_figure12_metrics(plan)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            metric_reason = f"Figure12 metric validation failed: {error}"
            classified["measurement_usable"] = False
            classified["figure12_metrics_valid"] = False
            if classified.get("status") == "passed":
                classified.update(status="error", reason=metric_reason)
            else:
                original = classified.get("reason", "")
                classified["reason"] = f"{original}; {metric_reason}" if original else metric_reason
        else:
            classified["figure12_metrics_valid"] = True
            classified["figure12_metric_count"] = len(metrics)
    return {**plan, **classified, "runner_exit_status": returncode,
            "runner_log": str(log_path)}


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        args.site = Path(args.site).resolve()
        args.build_root = Path(args.build_root).resolve()
        args.out = Path(args.out).resolve()
        if args.site_map is not None:
            args.site_map = Path(args.site_map).resolve()
        if args.execute and args.dry_run:
            raise ValueError("--execute and --dry-run are mutually exclusive")
        # The historical Figure 9/10 scripts execute by default.  Keep
        # --execute as an explicit compatibility flag while making --dry-run
        # the only way to suppress all runner invocations.
        args.execute = not args.dry_run
        if args.execute and args.out.exists():
            raise ValueError(f"refusing to overwrite existing batch: {args.out}")
        # Resolve overlays in a temporary directory before creating the
        # durable batch root.  This preserves the overwrite guard and keeps
        # planning read-only; execute mode rematerializes them below.
        paths, planning_site_provenance = _effective_sites(args, output=None)
        matrix = plans(args, site_paths=paths, dry_run=args.dry_run)
        if args.dry_run:
            try:
                errors = 0
                for plan in matrix:
                    result = subprocess.run(plan["command"], cwd=AE_ROOT,
                                            stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, check=False)
                    record = {**plan, "effective_plan": _parse_dry_run_output(result.stdout)}
                    if result.stderr.strip():
                        record["stderr"] = result.stderr
                        print(f"{plan['run_id']} dry-run stderr: "
                              f"{result.stderr.rstrip()}",
                              file=sys.stderr)
                    print(json.dumps(record, sort_keys=True))
                    if result.returncode != 0:
                        errors += 1
                if errors:
                    print(f"dry-run: {errors} plan(s) failed", file=sys.stderr)
                    return 1
                return 0
            finally:
                _cleanup_site_provenance(planning_site_provenance)
        _cleanup_site_provenance(planning_site_provenance)

        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "runs").mkdir()
        (args.out / "logs").mkdir()
        durable_paths, site_provenance = _effective_sites(args, output=args.out)
        matrix = plans(args, site_paths=durable_paths, dry_run=False)
        _write_json(args.out / "batch-plan.json", matrix)
        batch = _new_batch(args, matrix, site_provenance)
        batch["status"] = "running"
        _save_batch(args.out / "batch.json", batch)
        completed: List[Dict[str, Any]] = []
        stop_batch = False
        build_results: Dict[str, Dict[str, Any]] = {}
        systems = _selection(args.systems, SYSTEMS, "systems")
        apps = _selection(args.apps, APPS, "applications")
        print("This script will run " + ", ".join(system.title() for system in systems)
              + " sequentially.", flush=True)
        for system in systems:
            system_failures = 0
            print(f"{system.upper()} Start...", flush=True)
            build_failed = False
            if args.build:
                build_log = args.out / "logs" / f"build-{system}.log"
                command = _build_command(args, system, dry_run=False)
                try:
                    with build_log.open("w", encoding="utf-8") as output:
                        result = subprocess.run(command, cwd=AE_ROOT,
                                                stdin=subprocess.DEVNULL,
                                                stdout=output, stderr=subprocess.STDOUT,
                                                check=False)
                    build_results[system] = {
                        "status": "passed" if result.returncode == 0 else "error",
                        "exit_status": result.returncode, "command": command,
                        "log": str(build_log),
                    }
                except (OSError, subprocess.SubprocessError) as error:
                    build_results[system] = {
                        "status": "error", "exit_status": 2,
                        "command": command, "log": str(build_log),
                        "reason": str(error),
                    }
                build_failed = build_results[system]["status"] != "passed"
                batch["build_results"] = build_results
                _save_batch(args.out / "batch.json", batch)
                if build_failed:
                    print(f"{system.upper()} build ERROR: "
                          f"{build_results[system].get('reason', 'build failed')}",
                          file=sys.stderr, flush=True)
            for app in apps:
                app_failures = 0
                print(f"{system.upper()} App {app.upper()} START", flush=True)
                app_plans = [item for item in matrix
                             if item["system"] == system and item["app"] == app]
                for plan in app_plans:
                    print(f"START {plan['run_id']}", flush=True)
                    if build_failed:
                        record = {
                            **plan, "status": "error", "safe_to_continue": True,
                            "reason": "build failed; case not started",
                            "measurement_usable": False,
                            "exit_status": build_results[system]["exit_status"],
                            "runner_exit_status": None,
                        }
                    else:
                        record = _invoke_case(
                            plan, args.out / "logs" / f"{plan['run_id']}.runner.log")
                    completed.append(record)
                    _write_json(args.out / "batch-status.json", completed)
                    batch["completed"] = len(completed)
                    batch["passed"] = sum(row.get("status") == "passed"
                                          for row in completed)
                    batch["failed"] = len(completed) - batch["passed"]
                    if record.get("safe_to_continue") is not True:
                        stop_batch = True
                        batch["aborted"] = True
                    status = record.get("status", "error")
                    if status != "passed":
                        app_failures += 1
                        system_failures += 1
                        reason = (record.get("reason")
                                  or "case did not pass verification")
                        level = ("WARNING" if status in ("timeout", "warning")
                                 else "ERROR")
                        print(f"{system.upper()} {app.upper()} {level}: {reason}",
                              file=sys.stderr, flush=True)
                    else:
                        print(f"{system.upper()} {app.upper()} finish", flush=True)
                    _save_batch(args.out / "batch.json", batch)
                    if stop_batch:
                        break
                if stop_batch:
                    print(f"{system.upper()} App {app.upper()} ABORTED! "
                          f"(failures={app_failures})", flush=True)
                    break
                print(f"{system.upper()} App {app.upper()} FINISH! "
                      f"(failures={app_failures})", flush=True)
            if stop_batch:
                print(f"{system.upper()} ABORTED! (failures={system_failures})",
                      flush=True)
                break
            print(f"{system.upper()} FINISH! (failures={system_failures})", flush=True)
        if stop_batch:
            print("BATCH ABORTED", file=sys.stderr, flush=True)
        else:
            print(f"BATCH FINISH! (passed={sum(row.get('status') == 'passed' for row in completed)} "
                  f"failures={sum(row.get('status') != 'passed' for row in completed)})",
                  flush=True)
        batch["build_results"] = build_results
        batch["status"] = "aborted" if stop_batch else (
            "passed" if len(completed) == len(matrix)
            and all(row.get("status") == "passed" for row in completed)
            else "failed")
        batch["updated_at"] = _now()
        batch["completed"] = len(completed)
        batch["passed"] = sum(row.get("status") == "passed" for row in completed)
        batch["failed"] = len(completed) - batch["passed"]
        _save_batch(args.out / "batch.json", batch)
        return 0 if batch["status"] == "passed" else 1
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", default=",".join(APPS))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--ratio", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--metric", "--metrics", dest="metric",
                        choices=METRICS, default="all")
    parser.add_argument("--site", type=Path, default=AE_ROOT / "data" / "site.json")
    parser.add_argument("--site-map", type=Path,
                        help="optional JSON map of app/system site paths or overlays")
    parser.add_argument("--build-root", type=Path, default=AE_ROOT / "build")
    parser.add_argument("--build-jobs", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=AE_ROOT / "results" / "figure12" /
                        ("batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="explicit compatibility spelling for the default execute mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate plans without starting applications or servers")
    args = parser.parse_args()
    return execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
