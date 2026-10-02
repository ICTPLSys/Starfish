"""Optional Figure13 KV recovery helpers for the common runner.

The helpers are inert unless run_case explicitly receives --figure13-scenario.
They keep endpoint topology, configuration overlays, owned fault injection,
completion-window evidence, and the final result contract out of the normal
Figures 9-12 path.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCHEMA = "figure13-site-result-v1"
SYSTEMS = ("starfish", "hydra", "carbink")
SCENARIOS = ("1-node", "2-node")
FAULT_DELAY_S = 20.0
CARBINK_PEER_PORT_OFFSET = 10_000
MAX_CARBINK_SERVER_PORT = 65_535 - CARBINK_PEER_PORT_OFFSET
AE_ROOT = Path(__file__).resolve().parents[2]
_RAW_RUNS = None


def _raw_runs():
    global _RAW_RUNS
    if _RAW_RUNS is None:
        path = AE_ROOT / "scripts" / "figure13" / "raw_runs.py"
        spec = importlib.util.spec_from_file_location(
            "figure13_raw_runs_common", path)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load Figure13 raw-run adapter")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RAW_RUNS = module
    return _RAW_RUNS


def prepare(args, *, physical_count: int) -> Optional[Dict[str, Any]]:
    scenario = getattr(args, "figure13_scenario", None)
    if scenario is None:
        return None
    if scenario not in SCENARIOS:
        raise ValueError("figure13_scenario must be 1-node or 2-node")
    if args.app != "kv-b" or args.ratio != 25:
        raise ValueError("Figure13 common runs require --app kv-b --ratio 25")
    if (getattr(args, "offered_load_ops", None) is not None
            or getattr(args, "latency_warmup_ms", None) is not None
            or getattr(args, "latency_measure_ms", None) is not None
            or getattr(args, "latency_drain_ms", None) is not None
            or getattr(args, "max_queue_delay_us", None) is not None):
        raise ValueError("Figure13 recovery does not combine with latency/load options")
    if args.system not in SYSTEMS:
        raise ValueError("Figure13 common runs require Starfish, Hydra or Carbink")
    repeat = getattr(args, "figure13_repeat", None)
    if repeat is None:
        repeat = 1
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("figure13_repeat must be a positive integer")
    if physical_count not in (7, 8):
        raise ValueError("Figure13 requires seven physical memory endpoints, or eight")
    dual = scenario == "2-node"
    if args.system == "starfish":
        active = 7 if dual else 6
        standby = [7 if dual else 6]
    else:
        active = 6
        standby = [6] if not dual else [6, 7]
    logical = active + len(standby)
    if logical not in (7, 8):
        raise ValueError("Figure13 logical endpoint count must be seven or eight")
    if physical_count > logical:
        raise ValueError(
            "Figure13 must use all site endpoints; selected logical count is smaller "
            "than the physical inventory")
    if physical_count < logical and not (physical_count == 7 and logical == 8):
        raise ValueError("Figure13 lacks enough physical endpoints for its topology")
    observer = getattr(args, "figure13_observer_cpu", None)
    if observer is not None:
        if isinstance(observer, bool):
            raise ValueError("figure13_observer_cpu must be a nonnegative integer")
        try:
            observer = int(observer)
        except (TypeError, ValueError) as exc:
            raise ValueError("figure13_observer_cpu must be a nonnegative integer") from exc
        if observer < 0:
            raise ValueError("figure13_observer_cpu must be nonnegative")
    return {
        "enabled": True,
        "scenario": scenario,
        "repeat": repeat,
        "system": args.system,
        "app": args.app,
        "ratio": args.ratio,
        "physical_endpoints": physical_count,
        "physical_host_count": physical_count,
        "logical_endpoints": logical,
        "observer_cpu_requested": observer,
        "active_endpoints": active,
        "standby_endpoints": standby,
        "failed_endpoints": [0] if not dual else [0, 1],
        "observer_cpu": observer,
        "fault_delay_s": FAULT_DELAY_S,
        "trigger_event": "request_start" if args.system == "starfish"
                         else "runtime_packing_ready",
        "requests": 1_000_000_000,
        "resident_enabled": args.system == "starfish",
        "backup_policy": "credits1" if args.system == "starfish" else "disabled",
    }


def materialize_specs(specs: Sequence[Dict[str, Any]],
                      profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected = int(profile["logical_endpoints"])
    physical = [copy.deepcopy(spec) for spec in specs]
    if len(physical) > expected:
        raise ValueError("Figure13 must use all physical endpoints")
    if len(physical) < expected:
        if not (len(physical) == 7 and expected == 8):
            raise ValueError("Figure13 physical/logical endpoint mismatch")
        duplicate = copy.deepcopy(physical[-1])
        duplicate["requested_server_port"] = int(duplicate["server_port"])
        duplicate["server_port"] = int(duplicate["server_port"]) + 1
        if duplicate["server_port"] > 65535:
            raise ValueError("Figure13 duplicate endpoint port exceeds 65535")
        duplicate["physical_duplicate"] = True
        duplicate["physical_source_index"] = duplicate["inventory_index"]
        duplicate["port_reason"] = "physical_duplicate_plus_one"
        physical.append(duplicate)
    if len(physical) != expected:
        raise ValueError("Figure13 endpoint materialization produced wrong count")
    for spec in physical:
        spec.setdefault("requested_server_port", spec["server_port"])
        spec.setdefault("port_reason", "site_requested")
    port_policy = "site_distinct"
    ports = [int(spec["server_port"]) for spec in physical]
    if profile["system"] == "carbink" and len(set(ports)) != len(ports):
        base = ports[0]
        if base + expected - 1 > MAX_CARBINK_SERVER_PORT:
            raise ValueError("Figure13 derived Carbink service port exceeds 65535")
        for logical, spec in enumerate(physical):
            spec["server_port"] = base + logical
            spec["port_reason"] = "logical_index_derived"
        port_policy = "logical_index_derived"
    if profile["system"] == "carbink":
        # The legacy server derives each peer listener as SERVER_PORT+10000.
        # Check both listener families in the actual host collision domain;
        # this is especially important when logical endpoint 7 is colocated
        # with the duplicated seventh physical endpoint.
        occupied_by_host: Dict[str, set] = {}
        for spec in physical:
            host = str(spec["memory_addr"])
            main_port = int(spec["server_port"])
            if not 1 <= main_port <= MAX_CARBINK_SERVER_PORT:
                raise ValueError(
                    "Figure13 Carbink server port must leave room for its +10000 peer port")
            peer_port = main_port + CARBINK_PEER_PORT_OFFSET
            occupied = occupied_by_host.setdefault(host, set())
            if main_port in occupied or peer_port in occupied:
                raise ValueError(
                    f"Figure13 Carbink main/peer port collision on memory host {host}")
            occupied.update((main_port, peer_port))
            spec["peer_server_port"] = peer_port
    else:
        occupied_by_host: Dict[str, set] = {}
        for spec in physical:
            host = str(spec["memory_addr"])
            port = int(spec["server_port"])
            occupied = occupied_by_host.setdefault(host, set())
            if port in occupied:
                raise ValueError(
                    f"Figure13 server port collision on memory host {host}")
            occupied.add(port)
    seen = set()
    mapping = []
    for logical, spec in enumerate(physical):
        pair = (spec["memory_addr"], int(spec["server_port"]))
        if pair in seen:
            raise ValueError("Figure13 logical endpoint address/port collision")
        seen.add(pair)
        spec["index"] = logical
        spec.setdefault("physical_duplicate", False)
        spec.setdefault("physical_source_index", spec["inventory_index"])
        spec["figure13"] = True
        spec["system"] = profile["system"]
        spec["stage_memory_server"] = True
        mapping.append({
            "logical_index": logical,
            "port_policy": port_policy,
            "physical_inventory_index": spec["physical_source_index"],
            "memory_host": spec["memory_host"],
            "memory_addr": spec["memory_addr"],
            "requested_server_port": int(spec["requested_server_port"]),
            "server_port": spec["server_port"],
            "peer_server_port": spec.get("peer_server_port"),
            "port_reason": spec["port_reason"],
            "duplicated_physical_service": bool(spec["physical_duplicate"]),
        })
    profile["port_policy"] = port_policy
    if profile["system"] == "carbink":
        profile["peer_port_offset"] = CARBINK_PEER_PORT_OFFSET
    return physical, mapping


def _replace(lines: List[str], key: str, value: str) -> None:
    matches = []
    for index, line in enumerate(lines):
        fields = line.split("#", 1)[0].split()
        if fields and fields[0] == key:
            matches.append(index)
    if len(matches) > 1:
        raise ValueError("duplicate configuration key: " + key)
    line = f"{key} {value}\n"
    if matches:
        lines[matches[0]] = line
    else:
        lines.append(line)


def overlay_config(effective: str, profile: Dict[str, Any]) -> str:
    lines = effective.splitlines(keepends=True)
    failures = 2 if len(profile["failed_endpoints"]) == 2 else 1
    standby = profile["standby_endpoints"]
    for key, value in (
        ("ft_background_rebuild", "1"),
        ("ft_background_rebuild_failures", str(failures)),
        ("ft_rebuild_bandwidth_mbps", "2500"),
        ("qp_timeout", "4"),
        ("qp_retry_cnt", "0"),
        ("ft_standby_endpoint", str(standby[0])),
    ):
        _replace(lines, key, value)
    if profile["system"] in ("hydra", "carbink"):
        _replace(lines, "ft_standby_endpoint2",
                 str(standby[1] if len(standby) > 1 else -1))
    if profile["system"] == "starfish":
        _replace(lines, "ft_rmw_read_failure_fallback", "1")
        _replace(lines, "local_resident_budget_bytes", "3435973836")
    if profile["system"] == "carbink":
        _replace(lines, "server_buffer_size", "10118758400")
    return "".join(lines)


def apply_environment(environment: Dict[str, str],
                       profile: Dict[str, Any]) -> Dict[str, str]:
    result = dict(environment)
    result["FARLIB_KVS_COMPLETION_SERIES"] = "1"
    result["FARLIB_KVS_COMPLETION_SERIES_CPU"] = str(profile["observer_cpu"])
    if profile["system"] == "starfish":
        for key in (
            "FARLIB_SIMPLE_HOTCOLD_RELINK",
            "FARLIB_SIMPLE_LOCAL_ROUTING",
            "FARLIB_SIMPLE_REMOTE_HOTCOLD",
            "FARLIB_SIMPLE_REGION_BUDGET_FAST",
            "FARLIB_SIMPLE_REGION_BUDGET_INTERVAL_MS",
            "FARLIB_SIMPLE_REGION_HEAT_INTERVAL_MS",
            "FARLIB_PLANNER_BUDGET_EARLY_EXIT",
            "FARLIB_RESIDENT_PROFILE_REQUIRE_WORK_PHASE",
        ):
            result.pop(key, None)
        result.update({
            "FARLIB_BACKUP_CREDITS": "1",
            "FARLIB_SIMPLE_HOTCOLD": "0",
            "FARLIB_SIMPLE_LOCAL_RESIDENT": "0",
            "FARLIB_SIMPLE_SIX_GROUPS": "0",
            "FARLIB_SIMPLE_REGION_BUDGET": "0",
            "FARLIB_SIMPLE_REGION_HEAT": "0",
            "FARLIB_SIMPLE_DIRTY_OBSERVE": "0",
            "FARLIB_LIST_ONLY_SIX": "0",
            "FARLIB_LIST_ONLY_SIX_GROUPS": "0",
            "FARLIB_ALL_NONRESIDENT_BACKUP": "1",
            "FARLIB_PROFILED_BACKUP_NONBLOCKING_ON_FULL": "0",
            "FARLIB_EC_BEHAVIOR_ROUTING": "0",
        })
    if profile["system"] == "carbink":
        result.update({
            "FARLIB_CARBINK_RECOVERY": "1",
            "FARLIB_HYDRA_RUNTIME_HOT_PACKING": "1",
            "FARLIB_HYDRA_RECLAIM_BATCH": "64",
            "FARLIB_HYDRA_RECLAIM_NOTIFY": "0",
            "FARLIB_HYDRA_ZERO_COPY_WRITE": "1",
            "FARLIB_HYDRA_FAST_WRITE_PREPARE": "1",
            "FARLIB_HYDRA_WRITE_BATCH": "64",
        })
    return result


def resolve_observer_cpu(site: Dict[str, Any], effective: str,
                         requested: Optional[int]) -> int:
    import topology
    if requested is not None:
        return int(requested)
    profile = topology.check_cpu_profile(site, effective)
    used = list(profile["app_cpus"]) + list(profile["background_cpus"])
    if not used:
        raise ValueError("cannot derive observer CPU from an empty worker profile")
    return max(used) + 1

def observer_cpu_profile(site: Dict[str, Any], effective: str,
                         observer_cpu: int) -> Dict[str, Any]:
    import topology
    profile = topology.check_cpu_profile(site, effective)
    records = topology._normalise_cpu_records(topology._read_cpu_records())
    allowed = set(os.sched_getaffinity(0))
    if observer_cpu not in records:
        raise ValueError(f"observer CPU {observer_cpu} is not online")
    if observer_cpu not in allowed:
        raise ValueError(f"observer CPU {observer_cpu} is outside sched_getaffinity")
    selected = set(profile["app_cpus"]) | set(profile["background_cpus"])
    if observer_cpu in selected:
        raise ValueError("observer CPU overlaps application/background CPU set")
    observer_physical = (
        records[observer_cpu]["socket"], records[observer_cpu]["core"])
    for cpu in selected:
        physical = (records[cpu]["socket"], records[cpu]["core"])
        if physical == observer_physical:
            raise ValueError(
                f"observer CPU {observer_cpu} shares physical core with CPU {cpu}")
    return {
        "cpu": observer_cpu,
        "socket": records[observer_cpu]["socket"],
        "core": records[observer_cpu]["core"],
        "node": records[observer_cpu]["node"],
        "allowed": True,
        "disjoint_from_runtime_workers": True,
    }


def _failure_script(state: Dict[str, Any], identity_script: Callable[..., str]) -> str:
    identity = identity_script(
        int(state["pid"]), state["server_bin"],
        state["remote_dir"] + "/server.config", state.get("starttime"))
    return identity + "; kill -KILL $pid; for n in $(seq 1 50); do " + (
        "if [ ! -d /proc/$pid ]; then exit 0; fi; "
        "current=$(awk '{print $22}' /proc/$pid/stat); "
        "state=$(awk '{print $3}' /proc/$pid/stat); "
        'if [ "$current" != ' + str(state["starttime"]) +
        ' ] || [ "$state" = Z ]; then exit 0; fi; sleep 0.1; done; exit 4')


def inject_owned_failures(states: Sequence[Dict[str, Any]],
                          identity_script: Callable[..., str],
                          ssh: Callable[..., subprocess.CompletedProcess],
                          *, trigger_event: str, trigger_line: str,
                          trigger_observed_monotonic_ns: int,
                          scheduled_delay_s: float) -> Dict[str, Any]:
    command_begin = time.monotonic_ns()

    def inject(state):
        begin = time.monotonic_ns()
        response = ssh(state["memory_host"],
                       _failure_script(state, identity_script),
                       check=False, timeout=60)
        end = time.monotonic_ns()
        return state, response, begin, end

    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        responses = list(pool.map(inject, states))
    receipts = []
    for state, response, begin, end in responses:
        if response.returncode:
            raise RuntimeError(
                f"owned Figure13 injection failed for endpoint {state['index']}: "
                f"exit {response.returncode}")
        receipts.append({
            "endpoint": state["index"],
            "memory_host": state["memory_host"],
            "memory_addr": state["memory_addr"],
            "server_port": state["server_port"],
            "pid": state["pid"],
            "starttime": state["starttime"],
            "signal": "SIGKILL",
            "owned_identity_verified": True,
            "process_stopped": True,
            "command_begin_monotonic_ns": begin,
            "command_end_monotonic_ns": end,
        })
    return {
        "schema_version": 1,
        "injected": True,
        "endpoints": [item["endpoint"] for item in receipts],
        "trigger_event": trigger_event,
        "trigger_line": trigger_line.rstrip("\n"),
        "trigger_observed_monotonic_ns": trigger_observed_monotonic_ns,
        "scheduled_delay_s": scheduled_delay_s,
        "command_begin_monotonic_ns": command_begin,
        "command_end_monotonic_ns": time.monotonic_ns(),
        "owned_identity_verified": True,
        "process_stopped": True,
        "receipts": receipts,
        "signal": "SIGKILL",
    }


def run_client(command: Sequence[str], *, stdin, stdout, env: Dict[str, str],
               timeout: int, cwd: Path, failure_states: Sequence[Dict[str, Any]],
               identity_script: Callable[..., str],
               ssh: Callable[..., subprocess.CompletedProcess],
               trigger_event: str,
               trigger_delay_s: float,
               execution_state: Optional[Dict[str, Any]] = None
               ) -> Tuple[int, Dict[str, Any]]:
    deadline = time.monotonic() + timeout
    child_env = dict(env)
    # Keep client logging independent of synchronous SSH fault injection.
    # A PIPE drained by this control loop can fill while injection waits,
    # blocking the benchmark's foreground threads on diagnostic output.
    stdout.flush()
    log_reader = Path(stdout.name).open("rb")
    try:
        process = subprocess.Popen(
            list(command), stdin=stdin, stdout=stdout,
            stderr=subprocess.STDOUT, cwd=str(cwd), env=child_env,
            bufsize=0)
    except BaseException:
        log_reader.close()
        raise
    if execution_state is not None:
        execution_state.update(pid=process.pid, reaped=False)
    trigger_line = ""
    trigger_seen_ns = None
    request_start_seen_ns = None
    packing_ready_seen_ns = None
    packing_ready_line = ""
    trigger_due = None
    injection = None
    pending = b""
    def consume(line: str) -> None:
        nonlocal trigger_line, trigger_seen_ns, request_start_seen_ns
        nonlocal packing_ready_seen_ns, packing_ready_line, trigger_due
        if "kvs_phase" in line and "event=request_start" in line:
            request_start_seen_ns = time.monotonic_ns()
        if (trigger_event == "runtime_packing_ready"
                and "runtime_packing_ready" in line
                and packing_ready_seen_ns is None):
            packing_ready_line = line
            packing_ready_seen_ns = time.monotonic_ns()
        if trigger_seen_ns is not None:
            return
        if request_start_seen_ns is None:
            return
        if trigger_event == "request_start":
            trigger_line = line
            trigger_seen_ns = request_start_seen_ns
        elif packing_ready_seen_ns is not None:
            # A packing-ready marker can be emitted during initialization.
            # Keep it, but require request_start before arming the fault and
            # anchor the receipt at the later observed event.
            trigger_line = packing_ready_line
            trigger_seen_ns = max(request_start_seen_ns, packing_ready_seen_ns)
        else:
            return
        trigger_due = max(time.monotonic(), trigger_seen_ns / 1_000_000_000) + trigger_delay_s
    def drain_ready(block: bool) -> None:
        nonlocal pending
        chunk = log_reader.read(65536)
        if not chunk:
            if block and process.poll() is None:
                time.sleep(0.05)
            return
        while True:
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                consume(raw.decode("utf-8", errors="replace") + "\n")
            chunk = log_reader.read(65536)
            if not chunk:
                break
    try:
        while True:
            drain_ready(True)
            if pending and process.poll() is not None:
                consume(pending.decode("utf-8", errors="replace"))
                pending = b""
            if process.poll() is not None:
                drain_ready(False)
                if pending:
                    consume(pending.decode("utf-8", errors="replace"))
                    pending = b""
                break
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            if trigger_due is not None and time.monotonic() >= trigger_due:
                injection = inject_owned_failures(
                    failure_states, identity_script, ssh,
                    trigger_event=trigger_event, trigger_line=trigger_line,
                    trigger_observed_monotonic_ns=trigger_seen_ns,
                    scheduled_delay_s=trigger_delay_s)
                trigger_due = None
        if injection is None:
            raise RuntimeError(
                "client exited before Figure13 trigger/injection event")
        status = process.returncode
        if execution_state is not None:
            execution_state.update(exit_status=status, reaped=True)
        return status, injection
    finally:
        log_reader.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if execution_state is not None and process.poll() is not None:
            execution_state.update(exit_status=process.returncode, reaped=True)

def validate_run(run_dir: Path, plan: Dict[str, Any],
                 injection: Dict[str, Any]) -> Dict[str, Any]:
    raw = _raw_runs()
    client_text, evidence = raw._read_client_log(run_dir / "client.log")
    window_json = raw._read_json(run_dir / "kvs.completed-100ms.json")
    request_start_ns, request_end_ns = raw._native_request_times(
        evidence, window_json)
    expected_requests = int(plan["requests"])
    windows, window_details = raw._read_windows(
        run_dir, request_start_ns, request_end_ns, expected_requests)
    failed = [int(value) for value in plan["failed_endpoints"]]
    grouped = raw._group_rebuild_events(evidence, failed)
    aggregate, event_details = raw._validate_rebuild_events(
        grouped, request_start_ns, request_end_ns)
    if not injection.get("owned_identity_verified") or not injection.get(
            "process_stopped"):
        raise ValueError("Figure13 injection ownership evidence is incomplete")
    receipt_records = evidence["receipts"]
    if len(receipt_records) != 1:
        raise ValueError("Figure13 requires exactly one kvs_receipt")
    receipt = receipt_records[0]["tokens"]
    completed = sum(int(receipt.get(key, "0")) for key in (
        "completed_get", "completed_put", "completed_remove"))
    if completed != expected_requests:
        raise ValueError("Figure13 kvs completed count differs from plan")
    post = evidence["post_verify"]
    if len(post) != 1 or post[0]["tokens"].get("failures") not in ("0", 0):
        raise ValueError("Figure13 post-verification failed")
    starts = {endpoint: events["start"]["tokens"]
              for endpoint, events in grouped.items()}
    dones = {endpoint: events["done"]["tokens"]
             for endpoint, events in grouped.items()}
    return {
        "passed": True,
        "schema_version": 1,
        "request_start_ns": request_start_ns,
        "request_end_ns": request_end_ns,
        "failure_endpoints": failed,
        "completed_windows": {
            "csv": str(run_dir / "kvs.completed-100ms.csv"),
            "json": str(run_dir / "kvs.completed-100ms.json"),
            "rows": len(windows),
            "completed": window_details["completed"],
        },
        "background_rebuild": {
            "aggregate": aggregate,
            "start": starts,
            "done": dones,
            "events": event_details,
        },
        "receipt": receipt,
        "raw_client_log": str(run_dir / "client.log"),
        "client_log_sha256": evidence["sha256"],
    }


def write_result(run_dir: Path, *, passed: bool, plan: Dict[str, Any],
                 analysis_path: str = "analysis.json",
                 manifest_path: str = "manifest.json") -> Path:
    if not passed:
        raise ValueError("Figure13 result is emitted only after common success")
    result = {
        "schema": SCHEMA,
        "passed": True,
        "status": "passed",
        "system": plan["system"],
        "scenario": plan["scenario"],
        "repeat": plan["repeat"],
        "run_id": plan["run_id"],
        "manifest": manifest_path,
        "analysis": analysis_path,
    }
    path = run_dir / "figure13-result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path
