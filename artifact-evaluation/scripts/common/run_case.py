#!/usr/bin/env python3
"""Run one AE case against one or more memory-service endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from endpoint import (HOST, REMOTE_PATH, endpoint_name,
                      validate_site_endpoints)
from render_config import render, replace
from workloads import BINARY_RELATIVE, FOOTPRINT_BYTES, WORKLOAD_LABEL
from workloads import client_command, client_environment, client_process_environment
from workloads import parse_result, validate_design2
from workloads import validate_hydra
from figure9_support import APPLICATIONS, SYSTEM_LABEL, require_supported
import recovery_check
import site_defaults
import topology
import server_staging


AE_ROOT = Path(__file__).resolve().parents[2]
SERVER_KEYS = ("server_count", "server_addr", "server_port",
               "server_buffer_size", "ib_device_name", "ib_port")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(AE_ROOT), *args],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def config_value(rendered: str, key: str) -> Optional[str]:
    for line in rendered.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == key:
            return fields[1]
    return None


def server_config(rendered: str) -> str:
    """Reduce a client recipe to the memory service's per-instance keys."""
    values: Dict[str, str] = {}
    for line in rendered.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in SERVER_KEYS:
            values[fields[0]] = fields[1]
    if len(values) != len(SERVER_KEYS):
        raise ValueError("rendered config lacks a required memory-server setting")
    return "".join(f"{key} {values[key]}\n" for key in SERVER_KEYS)


def check_hugepages(local_bytes: int, numa_node: Optional[int] = None) -> None:
    """The RDMA client mmaps its entire local buffer with 2 MiB HugePages."""
    fields: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(":")
        if key in ("HugePages_Free", "Hugepagesize"):
            fields[key] = int(value.split()[0])
    page_bytes = fields["Hugepagesize"] * 1024
    if numa_node is not None:
        free_path = (Path(f"/sys/devices/system/node/node{numa_node}/hugepages") /
                     f"hugepages-{page_bytes // 1024}kB/free_hugepages")
        fields["HugePages_Free"] = int(free_path.read_text().strip())
    required = (local_bytes + page_bytes - 1) // page_bytes
    if fields["HugePages_Free"] < required:
        raise ValueError(
            f"not enough HugePages for client_buffer_size={local_bytes}: "
            f"need {required} free pages of {page_bytes} bytes, "
            f"have {fields['HugePages_Free']}"
        )


def site_config(path: Path, *, dry_run: bool,
                system: Optional[str] = None, app: Optional[str] = None) -> Dict[str, Any]:
    site = site_defaults.resolve(json.loads(path.read_text(encoding="utf-8")), AE_ROOT,
                                 system=system, app=app)
    required = ("name", "remote_run_root", "ib_device", "inputs")
    absent = [key for key in required if not site.get(key)]
    if absent:
        raise ValueError(f"site configuration is missing: {', '.join(absent)}")
    if not isinstance(site["name"], str) or not isinstance(site["inputs"], dict):
        raise ValueError("site name and inputs have invalid types")
    if (not REMOTE_PATH.fullmatch(site["remote_run_root"])
            or ".." in Path(site["remote_run_root"]).parts):
        raise ValueError("remote_run_root must be an absolute path without spaces or '..'")

    if "memory_endpoints" in site:
        site["memory_endpoints"] = validate_site_endpoints(
            site["memory_endpoints"], system=system, dry_run=dry_run)
        site["legacy_memory_endpoint"] = False
    else:
        legacy = ("memory_host", "memory_addr", "memory_server_bins", "server_port")
        absent = [key for key in legacy if not site.get(key)]
        if absent:
            raise ValueError(f"site configuration is missing: {', '.join(absent)}")
        if not isinstance(site["memory_server_bins"], dict):
            raise ValueError("memory_server_bins must be an object")
        if (site["memory_host"].startswith("-")
                or not HOST.fullmatch(site["memory_host"])):
            raise ValueError("memory_host must be a simple SSH host or user@host")
        if not dry_run:
            try:
                ipaddress.ip_address(site["memory_addr"])
            except ValueError as exc:
                raise ValueError("memory_addr must be an IP address") from exc
            if site["memory_host"].startswith("CHANGE_"):
                raise ValueError("replace the example SSH host before executing")
        site["memory_endpoints"] = validate_site_endpoints([{
            "memory_host": site["memory_host"],
            "memory_addr": site["memory_addr"],
            "server_port": site["server_port"],
            "memory_server_bins": site["memory_server_bins"],
            "memory_ib_device": site.get("memory_ib_device"),
            "memory_ld_library_path": site.get("memory_ld_library_path"),
            "memory_numa_node": site["memory_numa_node"],
            "memory_ib_port": site["memory_ib_port"],
            "stage_memory_server": site["stage_memory_server"],
        }], system=system, dry_run=dry_run)
        site["legacy_memory_endpoint"] = True

    for key in ("memory_ib_device",):
        if key in site and (not isinstance(site[key], str) or not site[key].strip()):
            raise ValueError(f"{key} must be a non-empty string")
    if "memory_ld_library_path" in site and not isinstance(
            site["memory_ld_library_path"], str):
        raise ValueError("memory_ld_library_path must be a string")
    try:
        wait_s = float(site.get("server_start_wait_s", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError("server_start_wait_s must be finite and in [0, 300]") from exc
    if not math.isfinite(wait_s) or not 0 <= wait_s <= 300:
        raise ValueError("server_start_wait_s must be finite and in [0, 300]")
    site["server_start_wait_s"] = wait_s
    return site


def endpoint_specs(site: Dict[str, Any], system: str,
                   *, dry_run: bool) -> List[Dict[str, Any]]:
    result = []
    for endpoint in site["memory_endpoints"]:
        server_bin = endpoint["memory_server_bins"].get(system)
        if not server_bin and not dry_run:
            raise ValueError(
                f"endpoint {endpoint['index']} has no memory server binary for {system}"
            )
        if server_bin and (not REMOTE_PATH.fullmatch(server_bin)
                           or ".." in Path(server_bin).parts):
            raise ValueError(
                f"endpoint {endpoint['index']} server binary must be an absolute "
                "path without spaces or '..'"
            )
        result.append({
            **endpoint,
            "server_bin": server_bin,
            "memory_ib_device": endpoint.get("memory_ib_device")
                or site.get("memory_ib_device", site["ib_device"]),
            "memory_ld_library_path": endpoint.get("memory_ld_library_path")
                or site.get("memory_ld_library_path"),
        })
    return result


def render_endpoint_records(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"server_addr": spec["memory_addr"],
             "server_port": spec["server_port"]} for spec in specs]


def server_start_command(state):
    environment = {"SERVER_PORT": str(state["server_port"]),
                   "FARLIB_RDMA_DEVICE": state["memory_ib_device"]}
    if state.get("memory_ld_library_path"):
        environment["LD_LIBRARY_PATH"] = state["memory_ld_library_path"]
    argv = ["env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin"]
    argv += [key + "=" + value for key, value in environment.items()]
    argv += topology.memory_binding(state["memory_numa_node"], server=True)
    argv += [state["server_bin"], "server.config"]
    return ("cd " + shlex.quote(state["remote_dir"]) + "; nohup "
            + shlex.join(argv) + " > server.log 2>&1 < /dev/null & echo $!")


def check_memory_capacity(states, bytes_per_service):
    totals = {}
    for state in states:
        key = (state["memory_host"], state["memory_numa_node"])
        totals[key] = totals.get(key, 0) + bytes_per_service
    for (host, node), required in totals.items():
        response = ssh(host, f"cat /sys/devices/system/node/node{node}/meminfo").stdout
        found = re.search(r"MemFree:\s+(\d+) kB", response)
        if found is None:
            raise ValueError(f"cannot read free memory on {host} NUMA {node}")
        available = int(found.group(1)) * 1024
        if required > available:
            raise ValueError(f"memory services on {host} NUMA {node} reserve "
                             f"{required} bytes together, but only {available} bytes are free")


def ssh(host: str, script: str, *, check: bool = True
        ) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         "env -u BASH_ENV -u ENV bash --noprofile --norc -c "
         + shlex.quote("set -e; " + script)],
        text=True, capture_output=True, check=False, timeout=30,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"memory host {host}: {detail}")
    return result


def scp(source: str, destination: str) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", source, destination],
                   check=True, timeout=60)



def process_identity_script(pid: int, server_bin: str,
                            remote_config: str,
                            starttime: Optional[str] = None) -> str:
    expected_bin = shlex.quote(server_bin)
    expected_config = shlex.quote(Path(remote_config).name)
    expected_cwd = shlex.quote(str(Path(remote_config).parent))
    expected_start = shlex.quote(str(starttime)) if starttime is not None else None
    start_check = ""
    if expected_start is not None:
        start_check = 'if [ "$start" != ' + expected_start + ' ]; then exit 11; fi; '
    return (
        "set -eu; "
        + "pid=" + str(pid) + "; "
        + "if [ ! -d /proc/$pid ]; then exit 10; fi; "
        + "uid=$(stat -c %u /proc/$pid); "
        + "if [ \"$uid\" != \"$(id -u)\" ]; then exit 11; fi; "
        + "start=$(awk '{print $22}' /proc/$pid/stat); "
        + start_check
        + "state=$(awk '{print $3}' /proc/$pid/stat); "
        + "if [ \"$state\" = Z ]; then exit 10; fi; "
        + "expected=$(readlink -f " + expected_bin + "); "
        + "exe=$(readlink -f /proc/$pid/exe); "
        + "if [ \"$exe\" != \"$expected\" ]; then exit 11; fi; "
        + "cwd=$(readlink -f /proc/$pid/cwd); "
        + "if [ \"$cwd\" != " + expected_cwd + " ]; then exit 11; fi; "
        + "cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline); cmd=" + "$" + "{cmd% }; "
        + "argv0=" + "$" + "{cmd%% *}; argv1=" + "$" + "{cmd#* }; "
        + "argv_real=$(readlink -f \"$argv0\"); "
        + "if [ \"$argv_real\" != \"$expected\" ] || "
        + "[ \"$argv1\" != " + expected_config + " ]; then exit 11; fi"
    )


def stop_server_process(host: str, state: Dict[str, Any]) -> str:
    pid = state.get("pid")
    if pid is None:
        return "not_started"
    identity = process_identity_script(
        int(pid), state["server_bin"], state["remote_dir"] + "/server.config",
        state.get("starttime"))
    script = (
        "set -eu; pid=" + str(int(pid)) + "; "
        + "if ( " + identity + " ); then :; else rc=$?; "
        + "if [ \"$rc\" = 10 ]; then exit 10; fi; exit 11; fi; "
        + "kill -TERM $pid; "
        + "for n in $(seq 1 30); do "
        + "if ! kill -0 $pid 2>/dev/null; then exit 0; fi; "
        + "if [ \"$(awk '{print $3}' /proc/$pid/stat 2>/dev/null)\" = Z ]; then exit 0; fi; "
        + "sleep 0.1; done; "
        + "if ( " + identity + " ); then :; else rc=$?; "
        + "if [ \"$rc\" = 10 ]; then exit 0; fi; exit 11; fi; "
        + "kill -KILL $pid; "
        + "for n in $(seq 1 10); do "
        + "if ! kill -0 $pid 2>/dev/null; then exit 0; fi; "
        + "if [ \"$(awk '{print $3}' /proc/$pid/stat 2>/dev/null)\" = Z ]; then exit 0; fi; "
        + "sleep 0.1; done; exit 4"
    )
    result = ssh(host, script, check=False)
    if result.returncode == 0:
        return "stopped"
    if result.returncode == 10:
        return "already_exited"
    if result.returncode == 11:
        raise RuntimeError("cleanup ownership mismatch; process was not signaled")
    raise RuntimeError("server did not exit after bounded TERM/KILL cleanup "
                       f"(exit={result.returncode}; {result.stderr.strip()})")

def write_endpoint_manifest(state: Dict[str, Any]) -> None:
    payload = {key: value for key, value in state.items() if key != "local_dir"}
    write_json(Path(state["local_dir"]) / "endpoint-manifest.json", payload)


def inject_owned_failure(state: Dict[str, Any]) -> Dict[str, Any]:
    identity = process_identity_script(
        state["pid"], state["server_bin"], state["remote_dir"] + "/server.config",
        state["starttime"])
    started = time.monotonic_ns()
    # Verification and signaling are in the same remote shell. No name-based
    # kills, system-wide service stops, or machine restarts are used.
    script = identity + "; kill -KILL $pid; for n in $(seq 1 50); do " + (
        "if [ ! -d /proc/$pid ]; then exit 0; fi; "
        "current=$(awk '{print $22}' /proc/$pid/stat); "
        "state=$(awk '{print $3}' /proc/$pid/stat); "
        'if [ "$current" != ' + shlex.quote(str(state["starttime"])) +
        ' ] || [ "$state" = Z ]; then exit 0; fi; sleep 0.1; done; exit 4')
    response = ssh(state["memory_host"], script, check=False)
    if response.returncode:
        raise RuntimeError("owned recovery injection failed; exit status "
                           + str(response.returncode))
    proof = {"endpoint": state["index"], "pid": state["pid"],
             "starttime": state["starttime"], "memory_host": state["memory_host"],
             "owned_identity_verified": True, "process_stopped": True,
             "signal": "SIGKILL", "trigger": "first_captured_chat_output",
             "started_monotonic_ns": started, "completed_monotonic_ns": time.monotonic_ns(),
             "time": now()}
    state["failure_injection"] = proof
    write_endpoint_manifest(state)
    return proof


def run(args: argparse.Namespace) -> int:
    require_supported(args.app, args.system, recipe=getattr(args, "recipe", None))
    if not 1 <= args.ratio <= 100 or args.timeout <= 0:
        raise ValueError("ratio must be 1..100 and timeout must be positive")
    site = site_config(args.site, dry_run=args.dry_run, system=args.system, app=args.app)
    specs = endpoint_specs(site, args.system, dry_run=args.dry_run)
    if not specs:
        raise ValueError("at least one memory endpoint is required")
    config = (getattr(args, "recipe", None)
              or AE_ROOT / "configs" / args.app / f"{args.system}.config")
    config = Path(config).resolve()
    config_text = config.read_text(encoding="utf-8")
    recover_endpoint = getattr(args, "recover_endpoint", None)
    reference_chat = getattr(args, "reference_chat", None)
    capture_chat = (getattr(args, "capture_chat", False) or recover_endpoint is not None
                    or reference_chat is not None)
    if capture_chat and args.app != "llama":
        raise ValueError("chat capture is only supported for LLaMA")
    recovery_check.validate_request(
        args.app, args.system, recover_endpoint, len(specs),
        config_value(config_text, "ft_method"), config_value(config_text, "ft_standby_endpoint"))
    capture_path = (args.out / "chat-output.txt").resolve() if capture_chat else None
    ec = config_value(config_text, "ft_method") == "ec_batch"
    hydra = args.system == "hydra"
    if args.system == "nonft" and config_value(config_text, "ft_method") not in (None, "none"):
        raise ValueError("NonFT recipe must not request a protected FT method")
    if args.system == "starfish" and not ec:
        raise ValueError("Starfish Figure 9 requires EC protection (ft_method ec_batch)")
    if hydra and config_value(config_text, "ft_method") not in ("hydra", "ec_split"):
        raise ValueError("Hydra Figure 9 requires page EC (ft_method hydra or ec_split)")
    if (ec or hydra) and len(specs) < 6:
        raise ValueError("4+2 EC requires at least six active endpoints; "
                         "set memory_server_count to 6 or 7 for a single memory host")
    if ec or hydra:
        if (int(config_value(config_text, "ft_ec_data_shards") or "4") != 4
                or int(config_value(config_text, "ft_ec_parity_shards") or "2") != 2):
            raise ValueError("Figure 9 protected recipes require the implemented RS(4,2) layout")
    build_dir = AE_ROOT / "build" / args.system
    client_bin = build_dir / BINARY_RELATIVE[args.app]
    input_path = Path(site["inputs"].get(args.app, ""))
    tokenizer = Path(site["inputs"].get("llama_tokenizer", "")) if args.app == "llama" else None
    command, stdin_path = client_command(
        args.app, client_bin, args.out / "effective.config", input_path, AE_ROOT, tokenizer)
    command = topology.memory_binding(site["numa_node"]) + command
    batch_id = args.out.parent.parent.name if args.out.parent.name == "runs" else "single"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id + args.out.name):
        raise ValueError("batch and run directory names must be simple identifiers")
    remote_dir = site["remote_run_root"].rstrip("/") + "/" + batch_id + "/" + args.out.name
    if not REMOTE_PATH.fullmatch(remote_dir) or ".." in Path(remote_dir).parts:
        raise ValueError("invalid remote run directory")
    for index, spec in enumerate(specs):
        spec["remote_dir"] = remote_dir + "/" + endpoint_name(index)
        if spec["stage_memory_server"]:
            spec["server_bin"] = remote_dir + "/" + endpoint_name(index) + "/server"
            spec["memory_ld_library_path"] = remote_dir + "/" + endpoint_name(index)
    endpoint_records = render_endpoint_records(specs)
    effective = render(config, system=args.system, ratio=args.ratio,
                       footprint_bytes=FOOTPRINT_BYTES[args.app],
                       ib_device=site["ib_device"], ib_port=site["ib_port"],
                       server_endpoints=endpoint_records)
    if ec or hydra:
        lines = effective.splitlines(keepends=True)
        standby = int(config_value(effective, "ft_standby_endpoint") or "-1")
        if standby == -1 and len(specs) == 7:
            standby = 6
        if standby < -1 or standby >= len(specs) or (standby >= 0 and len(specs) < 7):
            raise ValueError("EC standby requires six other active memory endpoints")
        replace(lines, "ft_standby_endpoint", str(standby))
        effective = "".join(lines)
    plan = {
        "app": args.app, "system": args.system, "ratio": args.ratio,
        "run_id": args.out.name, "site": site["name"],
        "compute_ip": site.get("compute_ip"), "numa_node": site["numa_node"],
        "ib_device": site["ib_device"], "ib_port": site["ib_port"],
        "recipe": str(config), "client_bin": str(client_bin),
        "effective_config": effective,
        "local_server_source": str(build_dir / "server"),
        "memory_host": specs[0]["memory_host"] if len(specs) == 1 else None,
        "memory_addr": specs[0]["memory_addr"] if len(specs) == 1 else None,
        "memory_server_bin": specs[0]["server_bin"] if len(specs) == 1 else None,
        "remote_run_dir": remote_dir if len(specs) == 1 else None,
        "memory_endpoints": [
            {"index": index, "memory_host": spec["memory_host"],
             "memory_addr": spec["memory_addr"], "server_port": spec["server_port"],
             "server_bin": spec["server_bin"],
             "stage_memory_server": spec["stage_memory_server"],
             "start_command": server_start_command(spec),
             "memory_ib_device": spec["memory_ib_device"],
             "memory_ib_port": spec["memory_ib_port"],
             "memory_numa_node": spec["memory_numa_node"],
             "remote_run_dir": remote_dir + "/" + endpoint_name(index)}
            for index, spec in enumerate(specs)
        ],
        "input": str(input_path),
        "client_command": command,
        "client_env": recovery_check.client_env(
            client_environment(args.app, site, args.system), capture_path, ec),
        "failure_injection_endpoint": recover_endpoint,
        "reference_chat": str(reference_chat) if reference_chat else None,
        "feature_profile": {"starfish": "resident_local_six", "nonft": "nonft",
                            "hydra": "hydra_page_ec_fixed_resident"}[args.system],
        "workload_footprint_bytes": FOOTPRINT_BYTES[args.app],
    }
    # Pure profile checks belong to planning too. Physical-core/NUMA topology
    # and observed affinity are still verified only by the real preflight.
    plan["worker_profile"] = topology.runtime_workers(site, effective)
    plan["configured_cpu_profile"] = (
        topology.check_cpu_profile(site, effective)
        if site.get("fibre_cpu_set") and site.get("background_cpu_base") is not None
        else None)
    if args.dry_run:
        plan["available"] = {
            "recipe": config.is_file(),
            "runtime": (AE_ROOT / "runtime" / args.system / "CMakeLists.txt").is_file(),
            "client_binary": client_bin.is_file(),
            "memory_server_path": all(spec["server_bin"] for spec in specs),
            "memory_endpoint_count": len(specs),
            "cpu_placement_configured": (
                bool(site.get("fibre_cpu_set")) and site.get("background_cpu_base") is not None),
        }
        print(json.dumps(plan, sort_keys=True))
        return 0

    if args.out.exists():
        raise ValueError(f"refusing to overwrite existing run directory: {args.out}")
    if not config.is_file() or not (AE_ROOT / "runtime" / args.system / "CMakeLists.txt").is_file():
        raise ValueError(f"runtime or recipe missing for {args.app}/{args.system}")
    if not os.access(client_bin, os.X_OK):
        raise ValueError(f"client binary missing or not executable: {client_bin}")
    local_server = build_dir / "server"
    private_libraries = []
    if any(spec["stage_memory_server"] for spec in specs):
        if not os.access(local_server, os.X_OK):
            raise ValueError(f"local memory-server binary missing: {local_server}; run build.sh")
        private_libraries = server_staging.dependencies(local_server)
    input_files = ([Path(f"{input_path}.{i}") for i in range(32)]
                   if args.app == "bfs" else [input_path])
    missing = [path for path in input_files if not path.is_file()]
    if missing:
        raise ValueError(f"workload input missing: {missing[0]}")
    input_stats = [path.stat() for path in input_files]
    if any(stat.st_size == 0 for stat in input_stats):
        raise ValueError("workload input contains an empty file")
    if stdin_path is not None and not stdin_path.is_file():
        raise ValueError(f"prompt file not found: {stdin_path}")
    if tokenizer is not None and not tokenizer.is_file():
        raise ValueError(f"LLaMA tokenizer not found: {tokenizer}")

    rendered_count = config_value(effective, "server_count")
    if rendered_count is None or int(rendered_count) != len(specs):
        raise ValueError("rendered server_count does not match endpoint count")
    plan["verified_cpu_placement"] = topology.check_compute(site, runtime_config=effective)
    check_hugepages(FOOTPRINT_BYTES[args.app] * args.ratio // 100, site["numa_node"])
    if args.check_local:
        print(f"local preflight PASS: {args.app}/{args.system}")
        return 0

    server_configs = []
    for spec in specs:
        rendered = render(
            config, system=args.system, ratio=args.ratio,
            footprint_bytes=FOOTPRINT_BYTES[args.app],
            server_endpoints=[{"server_addr": spec["memory_addr"],
                               "server_port": spec["server_port"]}],
            ib_device=spec["memory_ib_device"], ib_port=spec["memory_ib_port"])
        server_configs.append(server_config(rendered))

    args.out.mkdir(parents=True)
    source_archive = args.out / "server-source.tar.gz"
    if any(spec["stage_memory_server"] for spec in specs):
        server_staging.pack_source(AE_ROOT / "runtime" / args.system, source_archive)
    endpoint_root = args.out / "endpoints"
    endpoint_root.mkdir()
    (args.out / "effective.config").write_text(effective, encoding="utf-8")
    (args.out / "server.config").write_text(server_configs[0], encoding="utf-8")
    (args.out / "client.command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8")
    endpoint_states: List[Dict[str, Any]] = []
    for index, (spec, server_effective) in enumerate(zip(specs, server_configs)):
        name = endpoint_name(index)
        endpoint_remote_dir = remote_dir + "/" + name
        if (not REMOTE_PATH.fullmatch(endpoint_remote_dir)
                or ".." in Path(endpoint_remote_dir).parts):
            raise ValueError("invalid endpoint remote run directory")
        local_dir = endpoint_root / name
        local_dir.mkdir()
        (local_dir / "server.config").write_text(server_effective, encoding="utf-8")
        state = {
            "index": index, "name": name,
            "memory_host": spec["memory_host"], "memory_addr": spec["memory_addr"],
            "server_port": spec["server_port"], "memory_ib_device": spec["memory_ib_device"],
            "memory_numa_node": spec["memory_numa_node"],
            "memory_ib_port": spec["memory_ib_port"],
            "memory_ld_library_path": spec.get("memory_ld_library_path"),
            "server_bin": spec["server_bin"], "remote_dir": endpoint_remote_dir,
            "stage_memory_server": spec["stage_memory_server"],
            "local_dir": str(local_dir), "status": "planned", "pid": None,
            "remote_dir_created": False, "server_sha256": None, "error": "",
        }
        endpoint_states.append(state)
        write_endpoint_manifest(state)

    check_memory_capacity(endpoint_states, int(config_value(effective, "server_buffer_size")))
    local_server_sha256 = sha256(local_server) if any(
        state["stage_memory_server"] for state in endpoint_states) else None
    for state in endpoint_states:
        if not state["server_bin"]:
            raise ValueError(f"endpoint {state['index']} has no server binary for {args.system}")
        host = state["memory_host"]
        ssh(host, "test ! -e " + shlex.quote(state["remote_dir"]))
        if not state["stage_memory_server"]:
            ssh(host, server_staging.check_script(state["server_bin"],
                                                  state["memory_ld_library_path"]))
        ssh(host, topology.preflight_script(state["memory_ib_device"], state["memory_numa_node"],
                                           state["memory_ib_port"]))
        ssh(host, "command -v ss >/dev/null; "
            + "listeners=$(ss -H -ltn " + shlex.quote("sport = :" + str(state["server_port"]))
            + '); test -z "$listeners"')
        state["server_sha256"] = (local_server_sha256 if state["stage_memory_server"]
            else ssh(host, "sha256sum " + shlex.quote(state["server_bin"])).stdout.split()[0])
        state["status"] = "preflight_pass"
        write_endpoint_manifest(state)

    manifest = {
        "schema_version": 1, "start_time": now(), "status": "running",
        "server_source_sha256": sha256(source_archive) if source_archive.exists() else None,
        "plan": plan, "client_sha256": sha256(client_bin),
        "server_sha256": endpoint_states[0]["server_sha256"]
            if len(endpoint_states) == 1 else None,
        "server_sha256_by_endpoint": [s["server_sha256"] for s in endpoint_states],
        "effective_config_sha256": sha256(args.out / "effective.config"),
        "server_config_sha256": sha256(args.out / "server.config"),
        "server_config_sha256_by_endpoint": [
            sha256(Path(s["local_dir"]) / "server.config") for s in endpoint_states
        ],
        "input_bytes": sum(stat.st_size for stat in input_stats),
        "input_mtime_ns": max(stat.st_mtime_ns for stat in input_stats),
        "input_files": [str(path) for path in input_files],
        "declared_input_sha256": site.get("input_sha256", {}).get(args.app),
        "prompt_sha256": sha256(stdin_path) if stdin_path else None,
        "tokenizer_sha256": sha256(tokenizer) if tokenizer else None,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
    }
    write_json(args.out / "manifest.json", manifest)

    exit_status = 1
    check: Dict[str, Any] = {}
    error = ""
    injection: Dict[str, Any] = {}
    try:
        for state in endpoint_states:
            host = state["memory_host"]
            remote_dir_q = shlex.quote(state["remote_dir"])
            bin_q = shlex.quote(state["server_bin"])
            ssh(host, "mkdir -m 700 -p " + shlex.quote(str(Path(state["remote_dir"]).parent))
                + "; mkdir -m 700 " + remote_dir_q)
            state["remote_dir_created"] = True
            if state["stage_memory_server"]:
                deployment = server_staging.stage(
                    host, state["remote_dir"], local_server, private_libraries,
                    ssh=ssh, scp=scp, source_archive=source_archive)
                copied_hash = ssh(host, "sha256sum " + bin_q).stdout.split()[0]
                if not deployment["rebuilt_on_memory_host"] and copied_hash != state["server_sha256"]:
                    raise ValueError("staged memory server checksum mismatch")
                state["rebuilt_on_memory_host"] = deployment["rebuilt_on_memory_host"]
                state["server_sha256"] = copied_hash
            scp(str(Path(state["local_dir"]) / "server.config"),
                f"{host}:{state['remote_dir']}/server.config")
            start = server_start_command(state)
            response = ssh(host, start).stdout.strip()
            if not response.isdigit():
                state["status"] = "start_failed"
                state["error"] = f"cannot identify owned server PID: {response!r}"
                raise RuntimeError(state["error"])
            state["pid"] = int(response)
            state["starttime"] = ssh(
                host, "awk '{print $22}' /proc/%d/stat" % state["pid"]).stdout.strip()
            if not state["starttime"].isdigit():
                raise RuntimeError("invalid process starttime")
            state["status"] = "starting"
            write_endpoint_manifest(state)
            time.sleep(site["server_start_wait_s"])
            ssh(host, process_identity_script(
                state["pid"], state["server_bin"],
                state["remote_dir"] + "/server.config", state["starttime"]))
            state["status"] = "running"
            write_endpoint_manifest(state)

        environment = client_process_environment(args.app, site, args.system, os.environ)
        environment = recovery_check.client_env(environment, capture_path, ec)
        if site.get("compute_ld_library_path"):
            environment["LD_LIBRARY_PATH"] = site["compute_ld_library_path"]
        with (args.out / "client.log").open("w", encoding="utf-8") as output:
            with (stdin_path.open("r", encoding="utf-8") if stdin_path
                  else open(os.devnull)) as input_stream:
                try:
                    if recover_endpoint is None:
                        exit_status = recovery_check.run_client(
                            command, stdin=input_stream, stdout=output, env=environment,
                            timeout=args.timeout, capture=capture_path, inject=None,
                            evidence=injection)
                    else:
                        def inject():
                            proof = inject_owned_failure(endpoint_states[recover_endpoint])
                            write_json(args.out / "failure-injection.json", proof)
                            return proof
                        exit_status = recovery_check.run_client(
                            command, stdin=input_stream, stdout=output, env=environment,
                            timeout=args.timeout, capture=capture_path, inject=inject,
                            evidence=injection)
                except subprocess.TimeoutExpired:
                    exit_status = 124
        if exit_status != 0:
            raise RuntimeError(f"client exit status {exit_status}")
        check = parse_result(args.app, args.out.joinpath("client.log").read_text(
            encoding="utf-8", errors="replace"))
        if reference_chat is not None:
            check["chat_reference"] = recovery_check.compare_chat(capture_path, reference_chat)
        if args.system == "starfish":
            check["design2"] = validate_design2(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"))
        if hydra:
            check["hydra"] = validate_hydra(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                endpoint_count=len(specs), expected_workers=plan["worker_profile"],
                expected_resident_bytes=int(config_value(
                    effective, "local_resident_budget_bytes") or "0"))
        if ec:
            check["recovery"] = recovery_check.validate_log(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                injection if recover_endpoint is not None else None,
                endpoint_count=len(specs),
                answer_matches_reference=check.get("chat_reference", {}).get("matched", False))
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        error = str(exc)
    finally:
        for state in reversed(endpoint_states):
            if state.get("pid") is not None:
                try:
                    state["status"] = stop_server_process(state["memory_host"], state)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    state["status"] = "cleanup_failed"
                    state["error"] = str(exc)
                    error += f"; {state['name']} cleanup check: {exc}"
            if state.get("remote_dir_created"):
                try:
                    scp(f"{state['memory_host']}:{state['remote_dir']}/server.log",
                        str(Path(state["local_dir"]) / "server.log"))
                except (OSError, subprocess.SubprocessError) as exc:
                    state["status"] = "log_unavailable"
                    state["error"] = str(exc)
                    error += f"; {state['name']} server log unavailable: {exc}"
            try:
                write_endpoint_manifest(state)
            except OSError as exc:
                error += f"; {state['name']} manifest write: {exc}"
        first_log = Path(endpoint_states[0]["local_dir"]) / "server.log"
        if first_log.is_file():
            (args.out / "server.log").write_bytes(first_log.read_bytes())
        passed = exit_status == 0 and bool(check) and not error
        analysis = {
            "schema_version": 1, "application": args.app,
            "workload": WORKLOAD_LABEL[args.app],
            "system": SYSTEM_LABEL[args.system],
            "ratio": args.ratio, "run_id": args.out.name,
            "environment": site["name"], "exit_status": exit_status,
            "correctness": "pass" if passed else "fail",
            "status": "passed" if passed else "failed",
            "elapsed_s": check.get("elapsed_s") if passed else None,
            "measurement_phase": check.get("measurement_phase"),
            "measurement_method": check.get("measurement_method"),
            "correctness_scope": check.get("correctness_scope"),
            "correctness_evidence": check.get("correctness_evidence"),
            "design2": check.get("design2"),
            "hydra": check.get("hydra"),
            "recovery": check.get("recovery"),
            "chat_reference": check.get("chat_reference"),
            "failure_injection": injection or None,
            "ft_method": config_value(effective, "ft_method") or "none",
            "endpoint_count": len(endpoint_states), "endpoints": endpoint_states,
            "error": error,
        }
        write_json(args.out / "analysis.json", analysis)
        manifest.update({
            "end_time": now(), "status": analysis["status"],
            "exit_status": exit_status, "error": error,
            "memory_server_pid": endpoint_states[0].get("pid"),
            "memory_server_pids": [s.get("pid") for s in endpoint_states],
            "endpoints": endpoint_states,
            "server_sha256_by_endpoint": [s["server_sha256"] for s in endpoint_states],
            "server_sha256": endpoint_states[0]["server_sha256"]
                if len(endpoint_states) == 1 else None,
            "failure_injection": injection or None,
        })
        write_json(args.out / "manifest.json", manifest)
    if analysis["status"] == "passed":
        print(f"PASS {analysis['run_id']}: work {analysis['elapsed_s']:.6f} s "
              f"(details: {args.out / 'analysis.json'})")
    else:
        print(f"FAIL {analysis['run_id']}: {error or 'incomplete run'} "
              f"(details: {args.out / 'analysis.json'})", file=sys.stderr)
    return 0 if analysis["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=APPLICATIONS, required=True)
    parser.add_argument("--system", choices=tuple(SYSTEM_LABEL), required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--capture-chat", action="store_true")
    parser.add_argument("--reference-chat", type=Path)
    parser.add_argument("--recover-endpoint", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-local", action="store_true",
                        help="validate local inputs and recipe without writing or using SSH")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
