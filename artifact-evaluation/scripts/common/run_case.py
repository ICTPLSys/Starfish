#!/usr/bin/env python3
"""Run one AE case against one or more memory-service endpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from render_config import render, replace, load_recipe
from workloads import BINARY_RELATIVE, FOOTPRINT_BYTES, WORKLOAD_LABEL, GENERATED_INPUTS
from workloads import client_command, client_environment, client_process_environment
from workloads import parse_result, validate_design2
from workloads import validate_hydra, validate_nonft_kv
from figure9_support import APPLICATIONS, SYSTEM_LABEL, require_supported
import recovery_check
import site_defaults
import topology
import server_staging
import carbink_support
import kv_recovery
import kv_latency
import nq_latency
import remote_cpu
from measurement_acceptance import recover_measurement


AE_ROOT = Path(__file__).resolve().parents[2]
SERVER_KEYS = ("server_count", "server_addr", "server_port",
               "server_buffer_size", "ib_device_name", "ib_port")

NONFT_RESIDENT_REQUIRED_KEYS = (
    "enable_region_resident_placement",
)

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
                   *, dry_run: bool, figure13: bool = False) -> List[Dict[str, Any]]:
    result = []
    inventory = site["memory_endpoints"]
    selections = site.get("memory_endpoint_indices_by_system", {})
    if not isinstance(selections, dict) or set(selections) - set(site_defaults.SYSTEMS):
        raise ValueError("memory_endpoint_indices_by_system requires known runtime keys")
    for runtime, indices in selections.items():
        if (not isinstance(indices, list) or not indices
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                       for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"memory endpoint selection for {runtime} must contain unique nonnegative integer indices")
    selected = (list(range(len(inventory))) if figure13
               else selections.get(system, list(range(len(inventory)))))
    if any(i >= len(inventory) for i in selected):
        raise ValueError(f"memory endpoint selection for {system} exceeds the inventory")
    # Keep the site inventory intact. In particular, a seven-memory-node site
    # can explicitly reserve its seventh node for Carbink without pretending
    # the current six-endpoint compaction protocol supports a seventh member.
    # Bounds apply to the selected runtime: fast-check derives a one-endpoint
    # NonFT site from the full inventory but does not select Carbink there.
    for index, inventory_index in enumerate(selected):
        endpoint = inventory[inventory_index]
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
            "index": index,
            "inventory_index": inventory_index,
            "server_bin": server_bin,
            "memory_ib_device": endpoint.get("memory_ib_device")
                or site.get("memory_ib_device", site["ib_device"]),
            "memory_ld_library_path": endpoint.get("memory_ld_library_path")
                or site.get("memory_ld_library_path"),
        })
    if system == "carbink" and not figure13:
        result = carbink_support.endpoint_ports(result)
    return result


def render_endpoint_records(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"server_addr": spec["memory_addr"],
             "server_port": spec["server_port"]} for spec in specs]


def server_start_command(state):
    environment = {"SERVER_PORT": str(state["server_port"]),
                   "FARLIB_RDMA_DEVICE": state["memory_ib_device"]}
    if state.get("figure13") and state.get("system") == "carbink":
        environment["FARLIB_CARBINK_RECOVERY"] = "1"
    if state.get("memory_ld_library_path"):
        environment["LD_LIBRARY_PATH"] = state["memory_ld_library_path"]
    if state.get("disable_server_pin"):
        environment["FARLIB_DISABLE_SERVER_PIN"] = "1"
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
        free_kb = int(found.group(1))
        def node_kb(field):
            value = re.search(re.escape(field) + r":\s+(\d+) kB", response)
            return int(value.group(1)) if value else 0
        # Clean inactive file pages are reclaimable on this NUMA node.
        # Exclude active pages, slab, and pages awaiting writeback.
        reclaimable_kb = max(0, node_kb("Inactive(file)")
                            - node_kb("Dirty") - node_kb("Writeback"))
        available = (free_kb + reclaimable_kb) * 1024
        if required > available:
            raise ValueError(f"memory services on {host} NUMA {node} reserve "
                             f"{required} bytes together, but only {available} bytes "
                             "are free or clean inactive file cache")


def ssh(host: str, script: str, *, check: bool = True, timeout: int = 30
        ) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         "env -u BASH_ENV -u ENV bash --noprofile --norc -c "
         + shlex.quote("set -e; " + script)],
        text=True, capture_output=True, check=False, timeout=timeout,
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
        start_check = 'if [ "$start" != ' + expected_start + ' ]; then return 11; fi; '
    return (
        "set -eu; "
        + "pid=" + str(pid) + "; "
        + "check_owned_identity() { "
        + "if [ ! -d /proc/$pid ]; then return 10; fi; "
        + "uid=$(stat -c %u /proc/$pid 2>/dev/null) || return 11; "
        + "if [ \"$uid\" != \"$(id -u)\" ]; then return 11; fi; "
        + "snapshot=$(cat /proc/$pid/stat 2>/dev/null) || return 11; "
        + "fields=${snapshot##*) }; set -- $fields; "
        + "[ $# -ge 20 ] || return 11; state=$1; start=${20}; "
        + start_check
        + "case $state in Z|X|x) return 10;; esac; "
        + "expected=$(readlink -f " + expected_bin + "); "
        + "exe=$(readlink -f /proc/$pid/exe); "
        + "if [ \"$exe\" != \"$expected\" ]; then return 11; fi; "
        + "cwd=$(readlink -f /proc/$pid/cwd); "
        + "if [ \"$cwd\" != " + expected_cwd + " ]; then return 11; fi; "
        + "cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline); cmd=" + "$" + "{cmd% }; "
        + "argv0=" + "$" + "{cmd%% *}; argv1=" + "$" + "{cmd#* }; "
        + "argv_real=$(readlink -f \"$argv0\"); "
        + "if [ \"$argv_real\" != \"$expected\" ] || "
        + "[ \"$argv1\" != " + expected_config + " ]; then return 11; fi; "
        + "return 0; }; "
        # Exiting tasks can lose proc links before becoming zombies or disappearing.
        + "identity_rc=11; for identity_attempt in $(seq 1 40); do "
        + "if check_owned_identity; then identity_rc=0; break; else identity_rc=$?; fi; "
        + "if [ $identity_rc = 10 ]; then exit 10; fi; sleep 0.05; done; "
        + "if [ $identity_rc != 0 ]; then exit 11; fi"
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
        + "if ! kill -TERM $pid 2>/dev/null; then "
        + "if ( " + identity + " ); then exit 4; else rc=$?; "
        + "if [ $rc = 10 ]; then exit 0; fi; exit 11; fi; fi; "
        + "for n in $(seq 1 100); do "
        + "if ! kill -0 $pid 2>/dev/null; then exit 0; fi; "
        + "if [ \"$(awk '{print $3}' /proc/$pid/stat 2>/dev/null)\" = Z ]; then exit 0; fi; "
        + "sleep 0.1; done; "
        + "if ( " + identity + " ); then :; else rc=$?; "
        + "if [ \"$rc\" = 10 ]; then exit 0; fi; exit 11; fi; "
        + "if ! kill -KILL $pid 2>/dev/null; then "
        + "if ( " + identity + " ); then exit 4; else rc=$?; "
        + "if [ $rc = 10 ]; then exit 0; fi; exit 11; fi; fi; "
        + "for n in $(seq 1 200); do "
        + "if ! kill -0 $pid 2>/dev/null; then exit 0; fi; "
        + "if [ \"$(awk '{print $3}' /proc/$pid/stat 2>/dev/null)\" = Z ]; then exit 0; fi; "
        + "sleep 0.1; done; exit 4"
    )
    # Large registered pools may outlive signal delivery while the kernel
    # releases their mappings. This bounded wait is outside application timing.
    result = ssh(host, script, check=False, timeout=60)
    if result.returncode == 0:
        return "stopped"
    if result.returncode == 10:
        return "already_exited"
    if result.returncode == 11:
        # Normal server shutdown can remove /proc links between identity reads.
        # Recheck without signaling; a live identity mismatch still fails closed.
        recheck = ssh(host, identity, check=False)
        if recheck.returncode == 10:
            return "already_exited"
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
    # Client commands run with cwd=args.out; resolve both paths before command
    # construction so a relative invocation cannot accidentally double-prefix
    # effective.config or select a different build after chdir.
    args.out = args.out.resolve()
    if getattr(args, "build_root", None) is not None:
        args.build_root = args.build_root.resolve()
    repeat_suffix = re.search(r"-r([1-9][0-9]*)$", args.out.name)
    repeat = getattr(args, "repeat", None)
    if repeat is None:
        repeat = (int(repeat_suffix[1]) if repeat_suffix
                  else getattr(args, "figure13_repeat", None) or 1)
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    if repeat_suffix and int(repeat_suffix[1]) != repeat:
        raise ValueError("repeat disagrees with the run directory suffix")
    if getattr(args, "figure13_scenario", None) is not None:
        if getattr(args, "figure13_repeat", None) is None:
            args.figure13_repeat = repeat
        if args.figure13_repeat != repeat:
            raise ValueError("Figure13 repeat disagrees with the batch repeat")
    baseline_variant = getattr(args, "baseline_variant", None) or args.system
    if baseline_variant == "nonft-backup-off":
        if args.system != "nonft":
            raise ValueError("nonft-backup-off requires --system nonft")
        support_system = baseline_variant
    elif baseline_variant == args.system:
        support_system = args.system
    else:
        raise ValueError(
            "baseline_variant must equal --system or be nonft-backup-off "
            "with --system nonft")
    require_supported(args.app, support_system, recipe=getattr(args, "recipe", None))
    latency_adapter = nq_latency if args.app == "nq" else kv_latency
    latency_key = "nq_latency" if args.app == "nq" else "kv_latency"
    latency = latency_adapter.specification(args)
    if not 1 <= args.ratio <= 100 or args.timeout <= 0:
        raise ValueError("ratio must be 1..100 and timeout must be positive")
    site = site_config(args.site, dry_run=args.dry_run, system=args.system, app=args.app)
    physical_inventory_count = len(site["memory_endpoints"])
    physical_host_count = len({str(item["memory_addr"])
                               for item in site["memory_endpoints"]})
    figure13 = kv_recovery.prepare(args, physical_count=physical_host_count)
    if figure13 is not None:
        # Figure13 needs seven distinct physical memory hosts.  Keep the
        # inventory count separately because aliases must not be counted as
        # additional hosts in the topology contract.
        figure13["physical_inventory_endpoints"] = physical_inventory_count
        if not site.get("compute_ip"):
            raise ValueError("Figure13 site must identify compute_ip")
        if any(item["memory_addr"] == site["compute_ip"]
               for item in site["memory_endpoints"]):
            raise ValueError("Figure13 compute and memory hosts must be distinct")
    specs = endpoint_specs(
        site, args.system, dry_run=args.dry_run, figure13=figure13 is not None)
    endpoint_mapping: List[Dict[str, Any]] = []
    if figure13 is not None:
        specs, endpoint_mapping = kv_recovery.materialize_specs(specs, figure13)
        figure13["run_id"] = args.out.name
        figure13["physical_host_count"] = len({
            str(spec["memory_addr"]) for spec in specs
        })
        if figure13["physical_host_count"] not in (7, 8):
            raise ValueError("Figure13 requires seven or eight physical memory hosts")
    client_site = dict(site)
    if figure13 is not None:
        client_site["runtime_metadata"] = False
        client_site["runtime_ec_cpu"] = False
        client_site["remote_memory_samples"] = False
        client_site.pop("remote_memory_observer_cpu", None)
    if not specs:
        raise ValueError("at least one memory endpoint is required")
    config = (getattr(args, "recipe", None)
              or AE_ROOT / "configs" / args.app / f"{args.system}.config")
    config = Path(config).resolve()
    config_text = load_recipe(config)
    recover_endpoint = getattr(args, "recover_endpoint", None)
    reference_chat = getattr(args, "reference_chat", None)
    capture_chat = (getattr(args, "capture_chat", False) or recover_endpoint is not None
                    or reference_chat is not None)
    if capture_chat and args.app != "llama":
        raise ValueError("chat capture is only supported for LLaMA")
    capture_path = (args.out / "chat-output.txt").resolve() if capture_chat else None
    ec = config_value(config_text, "ft_method") == "ec_batch"
    hydra = args.system == "hydra"
    carbink = args.system == "carbink"
    if carbink and config_value(config_text, "ft_method") != "carbink":
        raise ValueError("Carbink Figure 9 requires ft_method carbink")
    if carbink and ((figure13 is None and len(specs) != 6)
                     or (figure13 is not None and len(specs) not in (7, 8))):
        raise ValueError("Carbink endpoint count is incompatible with this run")
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
    build_dir = (getattr(args, "build_root", None) or AE_ROOT / "build") / args.system
    server_relative = carbink_support.SERVER_BINARY if carbink else Path("server")
    client_bin = build_dir / BINARY_RELATIVE[args.app]
    input_path = None if args.app in GENERATED_INPUTS else Path(site["inputs"].get(args.app, ""))
    tokenizer = Path(site["inputs"].get("llama_tokenizer", "")) if args.app == "llama" else None
    command, stdin_path = client_command(
        args.app, client_bin, args.out / "effective.config", input_path, AE_ROOT, tokenizer,
        local_bytes=FOOTPRINT_BYTES[args.app] * args.ratio // 100)
    # Graph loaders need ordinary pages beyond one NUMA node's free capacity.
    compute_numa_node = None if args.app in ("bfs", "nq") else site["numa_node"]
    if compute_numa_node is not None:
        command = topology.memory_binding(compute_numa_node) + command
    batch_id = args.out.parent.parent.name if args.out.parent.name == "runs" else "single"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id + args.out.name):
        raise ValueError("batch and run directory names must be simple identifiers")
    local_batch_root = (args.out.parent.parent if args.out.parent.name == "runs"
                        else args.out.parent).resolve()
    remote_namespace = batch_id + "-" + hashlib.sha256(
        str(local_batch_root).encode("utf-8")).hexdigest()[:12]
    remote_parts = [site["remote_run_root"].rstrip("/"), remote_namespace,
                    args.out.name]
    remote_dir = "/".join(remote_parts)
    if not REMOTE_PATH.fullmatch(remote_dir) or ".." in Path(remote_dir).parts:
        raise ValueError("invalid remote run directory")
    for index, spec in enumerate(specs):
        spec["remote_dir"] = remote_dir + "/" + endpoint_name(index)
        if carbink:
            spec["disable_server_pin"] = not bool(site.get("carbink_server_pin_cores"))
        if spec["stage_memory_server"]:
            spec["server_bin"] = remote_dir + "/" + endpoint_name(index) + "/server"
            spec["memory_ld_library_path"] = remote_dir + "/" + endpoint_name(index)
    endpoint_records = render_endpoint_records(specs)
    raw_feature_overrides = site.get("feature_overrides", {})
    if (not isinstance(raw_feature_overrides, dict)
            or set(raw_feature_overrides) - {"backup", "resident"}):
        raise ValueError("feature_overrides supports only backup and resident")
    feature_overrides = dict(raw_feature_overrides)
    if figure13 is not None:
        # Figure13 uses the frozen policy explicitly.  Do not inherit a
        # possibly stale site feature override from the normal Figure9-12
        # profiles, and do not mutate the site object itself.
        feature_overrides.update({
            "backup": args.system == "starfish",
            "resident": args.system == "starfish",
        })
    if baseline_variant == "nonft-backup-off":
        feature_overrides.update(backup=False, resident=True)
    effective = render(config, system=args.system, ratio=args.ratio,
                       footprint_bytes=FOOTPRINT_BYTES[args.app],
                       ib_device=site["ib_device"], ib_port=site["ib_port"],
                       server_endpoints=endpoint_records,
                       backup_enabled=feature_overrides.get("backup"),
                       resident_enabled=feature_overrides.get("resident"))
    if (baseline_variant == "nonft-backup-off"
            and config_value(effective, "remote_backup_budget_pct") is None):
        rendered_lines = effective.splitlines(keepends=True)
        replace(rendered_lines, "remote_backup_budget_pct", "0")
        effective = "".join(rendered_lines)
    if figure13 is not None:
        effective = kv_recovery.overlay_config(effective, figure13)
    rendered_backup = config_value(effective, "enable_selective_backup")
    if rendered_backup not in ("0", "1"):
        raise ValueError("rendered config lacks enable_selective_backup=0/1")
    backup_enabled = rendered_backup == "1"
    rendered_backup_bytes = config_value(effective, "remote_backup_budget_bytes")
    rendered_backup_pct = config_value(effective, "remote_backup_budget_pct")
    if baseline_variant == "nonft-backup-off" and (
            backup_enabled or rendered_backup_bytes != "0"
            or rendered_backup_pct != "0"):
        raise ValueError(
            "nonft-backup-off did not render enable_selective_backup=0 "
            "and zero remote backup budgets")
    if baseline_variant == "nonft-backup-off":
        resident_budget = config_value(effective, "local_resident_budget_bytes")
        try:
            resident_budget_value = int(resident_budget) if resident_budget is not None else 0
        except ValueError as exc:
            raise ValueError("nonft-backup-off did not render a numeric Resident budget") from exc
        if resident_budget_value <= 0:
            raise ValueError("nonft-backup-off requires a positive Resident budget")
        disabled = [key for key in NONFT_RESIDENT_REQUIRED_KEYS
                    if config_value(effective, key) != "1"]
        if disabled:
            raise ValueError("nonft-backup-off disabled Resident placement/policy: "
                             + ", ".join(disabled))
    if ec or hydra:
        lines = effective.splitlines(keepends=True)
        standby = int(config_value(effective, "ft_standby_endpoint") or "-1")
        if standby == -1 and len(specs) == 7:
            standby = 6
        if standby < -1 or standby >= len(specs) or (standby >= 0 and len(specs) < 7):
            raise ValueError("EC standby requires six other active memory endpoints")
        replace(lines, "ft_standby_endpoint", str(standby))
        effective = "".join(lines)
    recovery_check.validate_request(
        args.app, args.system, recover_endpoint, len(specs),
        config_value(effective, "ft_method"), config_value(effective, "ft_standby_endpoint"))
    memory_sampling = None
    requested_memory = getattr(args, "collect_remote_memory", False)
    if figure13 is not None and requested_memory:
        raise ValueError("Figure13 recovery does not combine with steady-state memory sampling")
    if figure13 is None:
        site_memory = client_site.get("remote_memory_samples", False)
        if not isinstance(site_memory, bool):
            raise ValueError("remote_memory_samples must be a boolean")
        if requested_memory or site_memory:
            requested_cpu = getattr(args, "remote_memory_observer_cpu", None)
            if requested_cpu is None:
                requested_cpu = client_site.get("remote_memory_observer_cpu")
            memory_cpu = topology.observer_cpu(site, effective, requested_cpu)
            memory_origin = ("kvs_request_start"
                             if args.app in ("kv-b", "kv-a", "kv-s") and latency is None
                             else "profile_start_work")
            client_site.update(remote_memory_samples=True,
                               remote_memory_observer_cpu=memory_cpu,
                               remote_memory_work_origin=memory_origin)
            memory_sampling = {"enabled": True, "observer_cpu": memory_cpu,
                               "schema_version": 3,
                               "planned_points_s": [3] if args.app == "bfs"
                                   else [10, 20, 30, 40, 50],
                               "time_origin": memory_origin,
                               "window": "work_start_3s" if args.app == "bfs"
                                   else "work_start_10_20_30_40_50s",
                               "stop_at": "kvs_request_drain_end"
                                   if memory_origin == "kvs_request_start"
                                   else "profile_end_work",
                               "short_work_policy": "missing_no_scheduled_sample"}
    plan = {
        "app": args.app, "system": args.system, "ratio": args.ratio,
        "repeat": repeat,
        "remote_namespace": remote_namespace,
        "remote_namespace_source": str(local_batch_root),
        "baseline_variant": baseline_variant,
        "backup_enabled": backup_enabled,
        "run_id": args.out.name, "site": site["name"],
        "compute_ip": site.get("compute_ip"), "numa_node": compute_numa_node,
        "compute_memory_policy": "inherited" if compute_numa_node is None else "bind",
        "nic_numa_node": site["numa_node"],
        "ib_device": site["ib_device"], "ib_port": site["ib_port"],
        "recipe": str(config), "client_bin": str(client_bin),
        "effective_config": effective,
        "feature_overrides": feature_overrides,
        "figure13": figure13,
        "logical_physical_endpoint_map": endpoint_mapping,
        "memory_endpoint_inventory": [
            {"inventory_index": i, "memory_host": item["memory_host"],
             "memory_addr": item["memory_addr"], "server_port": item["server_port"],
             "selected": any(spec["inventory_index"] == i for spec in specs)}
            for i, item in enumerate(site["memory_endpoints"])
        ],
        "local_server_source": str(build_dir / server_relative),
        "memory_host": specs[0]["memory_host"] if len(specs) == 1 else None,
        "memory_addr": specs[0]["memory_addr"] if len(specs) == 1 else None,
        "memory_server_bin": specs[0]["server_bin"] if len(specs) == 1 else None,
        "remote_run_dir": remote_dir if len(specs) == 1 else None,
        "memory_endpoints": [
            {"index": index, "memory_host": spec["memory_host"],
             "inventory_index": spec["inventory_index"],
             "memory_addr": spec["memory_addr"], "server_port": spec["server_port"],
             **({"requested_server_port": spec["requested_server_port"],
                 "peer_server_port": spec.get("peer_server_port"),
                 "port_reason": spec.get("port_reason")}
                if figure13 is not None or carbink else {}),
             "server_bin": spec["server_bin"],
             "stage_memory_server": spec["stage_memory_server"],
             "start_command": server_start_command(spec),
             "memory_ib_device": spec["memory_ib_device"],
             "memory_ib_port": spec["memory_ib_port"],
             "memory_numa_node": spec["memory_numa_node"],
             "remote_run_dir": remote_dir + "/" + endpoint_name(index)}
            for index, spec in enumerate(specs)
        ],
        "input": str(input_path) if input_path is not None else None,
        "client_command": command,
        "client_env": recovery_check.client_env(
            client_environment(args.app, client_site, args.system), capture_path, ec),
        "failure_injection_endpoint": recover_endpoint,
        "reference_chat": str(reference_chat) if reference_chat else None,
        "feature_profile": {"starfish": "resident_local_six", "nonft": "nonft",
                            "hydra": "hydra_page_ec_fixed_resident",
                            "carbink": "carbink_full_stripe_background_compaction"}[args.system],
        "workload_footprint_bytes": FOOTPRINT_BYTES[args.app],
    }
    if memory_sampling is not None:
        plan["remote_memory_sampling"] = memory_sampling
    if figure13 is not None:
        plan["feature_profile"] = {
            "starfish": "figure13_profiled_backup_recovery",
            "hydra": "figure13_page_recovery",
            "carbink": "figure13_shadow_compaction_recovery",
        }[args.system]
        plan["figure13"]["observer_cpu"] = kv_recovery.resolve_observer_cpu(
            site, effective, figure13.get("observer_cpu_requested"))
        plan["client_env"] = kv_recovery.apply_environment(
            plan["client_env"], figure13)
    if latency is not None:
        plan[latency_key] = latency
        plan["client_env"] = latency_adapter.environment(plan["client_env"], latency)
    # Pure profile checks belong to planning too. Physical-core/NUMA topology
    # and observed affinity are still verified only by the real preflight.
    plan["worker_profile"] = topology.runtime_workers(site, effective)
    plan["configured_cpu_profile"] = (
        topology.check_cpu_profile(site, effective)
        if site.get("fibre_cpu_set") and site.get("background_cpu_base") is not None
        else None)
    if carbink:
        plan["server_configs"] = [
            carbink_support.server_config(
                effective, spec, site, figure13=figure13 is not None)
            for spec in specs]
        plan["carbink_server_pin_cores"] = site.get("carbink_server_pin_cores")
        plan["memory_service_reservation"] = carbink_support.memory_lower_bound(
            effective, plan["worker_profile"],
            endpoint_count=len(specs) if figure13 is not None else None,
            figure13=figure13 is not None)
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
    if latency is not None:
        probe_env = client_process_environment(args.app, site, args.system, os.environ)
        probe_env = latency_adapter.environment(probe_env, latency)
        if site.get("compute_ld_library_path"):
            probe_env["LD_LIBRARY_PATH"] = site["compute_ld_library_path"]
        # This entry point exits before runtime/data initialization. Reject an
        # old binary before creating output or starting any memory service.
        plan[latency_key + "_capability"] = latency_adapter.probe(client_bin, probe_env)
    local_server = build_dir / server_relative
    private_libraries = []
    if any(spec["stage_memory_server"] for spec in specs):
        if not os.access(local_server, os.X_OK):
            raise ValueError(f"local memory-server binary missing: {local_server}; run build.sh")
        private_libraries = server_staging.dependencies(local_server)
    # MG generates its fixed workload; a cwd override would silently change it.
    if args.app == "mg" and Path("mg.input").exists():
        raise ValueError("Figure 9 MG requires compiled defaults; remove the cwd mg.input override")
    input_files = ([] if input_path is None else
                   [Path(f"{input_path}.{i}") for i in range(32)]
                   if args.app in ("bfs", "nq") else [input_path])
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
    plan["verified_cpu_placement"] = topology.check_compute(
        site, runtime_config=effective)
    if memory_sampling is not None:
        memory_sampling["observer_cpu_validation"] = topology.check_observer_cpu(
            site, effective, memory_sampling["observer_cpu"])
    if figure13 is not None:
        plan["figure13"]["observer_cpu"] = kv_recovery.resolve_observer_cpu(
            site, effective, figure13.get("observer_cpu_requested"))
        plan["figure13"]["observer_cpu_validation"] = (
            kv_recovery.observer_cpu_profile(
                site, effective, plan["figure13"]["observer_cpu"]))
    check_hugepages(FOOTPRINT_BYTES[args.app] * args.ratio // 100, compute_numa_node)
    if args.check_local:
        print(f"local preflight PASS: {args.app}/{args.system}")
        return 0

    server_configs = []
    for spec in specs:
        if carbink:
            server_configs.append(carbink_support.server_config(
                effective, spec, site, figure13=figure13 is not None))
            continue
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
        server_staging.pack_source(AE_ROOT / "runtime" / args.system, source_archive,
                                   system=args.system)
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
            "server_port": spec["server_port"],
            "requested_server_port": spec.get("requested_server_port"),
            "peer_server_port": spec.get("peer_server_port"),
            "port_reason": spec.get("port_reason"),
            "memory_ib_device": spec["memory_ib_device"],
            "memory_numa_node": spec["memory_numa_node"],
            "memory_ib_port": spec["memory_ib_port"],
            "memory_ld_library_path": spec.get("memory_ld_library_path"),
            "server_bin": spec["server_bin"], "remote_dir": endpoint_remote_dir,
            "stage_memory_server": spec["stage_memory_server"],
            "disable_server_pin": spec.get("disable_server_pin", False),
            "figure13": figure13 is not None,
            "system": args.system,
            "physical_duplicate": bool(spec.get("physical_duplicate", False)),
            "physical_source_index": spec.get("physical_source_index"),
            "local_dir": str(local_dir), "status": "planned", "pid": None,
            "remote_dir_created": False, "launch_attempted": False,
            "server_sha256": None, "error": "",
        }
        endpoint_states.append(state)
        write_endpoint_manifest(state)

    required_service_bytes = (
        plan["memory_service_reservation"]["minimum_bytes_per_service"] if carbink
        else int(config_value(effective, "server_buffer_size")))
    check_memory_capacity(endpoint_states, required_service_bytes)
    local_server_sha256 = sha256(local_server) if any(
        state["stage_memory_server"] for state in endpoint_states) else None
    def preflight_endpoint(state: Dict[str, Any]) -> None:
        if not state["server_bin"]:
            raise ValueError(f"endpoint {state['index']} has no server binary for {args.system}")
        host = state["memory_host"]
        ssh(host, "test ! -e " + shlex.quote(state["remote_dir"]))
        if not state["stage_memory_server"]:
            ssh(host, server_staging.check_script(state["server_bin"],
                                                  state["memory_ld_library_path"]))
        ssh(host, topology.preflight_script(state["memory_ib_device"], state["memory_numa_node"],
                                           state["memory_ib_port"]))
        listener_ports = [state["server_port"]]
        if state.get("system") == "carbink":
            listener_ports.append(state["peer_server_port"])
        for listener_port in listener_ports:
            ssh(host, "command -v ss >/dev/null; "
                + "listeners=$(ss -H -ltn "
                + shlex.quote("sport = :" + str(listener_port))
                + '); test -z "$listeners"')
        state["server_sha256"] = (local_server_sha256 if state["stage_memory_server"]
            else ssh(host, "sha256sum " + shlex.quote(state["server_bin"])).stdout.split()[0])
        state["status"] = "preflight_pass"
        write_endpoint_manifest(state)

    def run_parallel_endpoint_steps(worker) -> None:
        """Run independent endpoint setup steps, joining every worker first."""
        if not endpoint_states:
            return
        failures = []
        with ThreadPoolExecutor(max_workers=len(endpoint_states)) as executor:
            futures = [executor.submit(worker, state) for state in endpoint_states]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise failures[0]

    run_parallel_endpoint_steps(preflight_endpoint)
    manifest = {
        "schema_version": 1, "start_time": now(), "status": "running",
        "repeat": repeat,
        "remote_namespace": remote_namespace,
        "remote_namespace_source": str(local_batch_root),
        "baseline_variant": baseline_variant,
        "backup_enabled": backup_enabled,
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
        "input_mtime_ns": max((stat.st_mtime_ns for stat in input_stats), default=None),
        "input_files": [str(path) for path in input_files],
        "declared_input_sha256": site.get("input_sha256", {}).get(args.app),
        "prompt_sha256": sha256(stdin_path) if stdin_path else None,
        "tokenizer_sha256": sha256(tokenizer) if tokenizer else None,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
    }
    collect_remote_cpu = bool(getattr(args, "collect_remote_cpu", False))
    remote_cpu_collector = None
    remote_cpu_result: Dict[str, Any] = {}
    remote_cpu_finalized = False
    client_launch_ns: Optional[int] = None
    client_end_ns: Optional[int] = None
    if collect_remote_cpu:
        plan["remote_cpu"] = {
            "schema_version": remote_cpu.SCHEMA_VERSION,
            "window": remote_cpu.REMOTE_CPU_WINDOW,
            "sampling_interval_s": remote_cpu.DEFAULT_INTERVAL_SECONDS,
            "method": "owned memory-server processes",
        }
        manifest["remote_cpu_enabled"] = True
    write_json(args.out / "manifest.json", manifest)

    exit_status = 1
    timed_out = False
    client_execution: Dict[str, Any] = {}
    recovered_measurement: Dict[str, Any] = {}
    measurement_recovery_error = ""
    check: Dict[str, Any] = {}
    error = ""
    injection: Dict[str, Any] = {}
    try:
        def launch_endpoint(state: Dict[str, Any]) -> None:
            host = state["memory_host"]
            remote_dir_q = shlex.quote(state["remote_dir"])
            bin_q = shlex.quote(state["server_bin"])
            try:
                ssh(host, "mkdir -m 700 -p " + shlex.quote(str(Path(state["remote_dir"]).parent))
                    + "; mkdir -m 700 " + remote_dir_q)
                state["remote_dir_created"] = True
                if state["stage_memory_server"]:
                    deployment = server_staging.stage(
                        host, state["remote_dir"], local_server, private_libraries,
                        ssh=ssh, scp=scp, source_archive=source_archive, system=args.system)
                    copied_hash = ssh(host, "sha256sum " + bin_q).stdout.split()[0]
                    if (not deployment["rebuilt_on_memory_host"]
                            and copied_hash != state["server_sha256"]):
                        raise ValueError("staged memory server checksum mismatch")
                    state["rebuilt_on_memory_host"] = deployment["rebuilt_on_memory_host"]
                    state["server_sha256"] = copied_hash
                scp(str(Path(state["local_dir"]) / "server.config"),
                    f"{host}:{state['remote_dir']}/server.config")
                start = server_start_command(state)
            except Exception:
                if (state.get("pid") is None
                        and state.get("launch_attempted") is False):
                    state["status"] = "not_started"
                    write_endpoint_manifest(state)
                raise
            # Persist intent before the SSH start call. If the call's outcome
            # is unknown, cleanup must remain fail-closed even with pid=None.
            state["launch_attempted"] = True
            write_endpoint_manifest(state)
            birth_lower_ns = time.monotonic_ns()
            response = ssh(host, start).stdout.strip()
            birth_upper_ns = time.monotonic_ns()
            if not response.isdigit():
                state["status"] = "start_failed"
                state["error"] = f"cannot identify owned server PID: {response!r}"
                raise RuntimeError(state["error"])
            state["pid"] = int(response)
            state["birth_local_lower_bound_ns"] = birth_lower_ns
            state["birth_local_upper_bound_ns"] = birth_upper_ns
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

        run_parallel_endpoint_steps(launch_endpoint)
        if collect_remote_cpu:
            try:
                remote_cpu_collector = remote_cpu.RemoteCpuCollector(
                    endpoint_states, app=args.app,
                    log_path=args.out / "client.log")
                # Require one owned snapshot before the client starts. The
                # initial cumulative schedstat is retained as initialization.
                remote_cpu_collector.start()
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                remote_cpu_result = {
                    "schema_version": remote_cpu.SCHEMA_VERSION,
                    "status": "failed",
                    "remote_cpu_window": remote_cpu.REMOTE_CPU_WINDOW,
                    "reason": "collector_start: " + str(exc),
                    "endpoints": [],
                    "provenance": {
                        "sampling_interval_s": remote_cpu.DEFAULT_INTERVAL_SECONDS,
                        "first_snapshot_included": False,
                    },
                }
                write_json(args.out / "remote-cpu.json", remote_cpu_result)
                remote_cpu_finalized = True
                raise
        environment = client_process_environment(
            args.app, client_site, args.system, os.environ)
        environment = recovery_check.client_env(environment, capture_path, ec)
        environment = latency_adapter.environment(environment, latency)
        if figure13 is not None:
            environment = kv_recovery.apply_environment(environment, figure13)
        if site.get("compute_ld_library_path"):
            environment["LD_LIBRARY_PATH"] = site["compute_ld_library_path"]
        client_launch_ns = time.monotonic_ns()
        with (args.out / "client.log").open("w", encoding="utf-8") as output:
            with (stdin_path.open("r", encoding="utf-8") if stdin_path
                  else open(os.devnull)) as input_stream:
                try:
                    if figure13 is not None:
                        exit_status, injection = kv_recovery.run_client(
                            command, stdin=input_stream, stdout=output, env=environment,
                            timeout=args.timeout, cwd=args.out,
                            failure_states=[
                                endpoint_states[index]
                                for index in figure13["failed_endpoints"]
                            ],
                            identity_script=process_identity_script, ssh=ssh,
                            trigger_event=figure13["trigger_event"],
                            trigger_delay_s=figure13["fault_delay_s"],
                            execution_state=client_execution)
                        write_json(args.out / "failure-injection.json", injection)
                    elif recover_endpoint is None:
                        exit_status = recovery_check.run_client(
                            command, stdin=input_stream, stdout=output, env=environment,
                            timeout=args.timeout, capture=capture_path, inject=None,
                            evidence=injection, execution_state=client_execution,
                            record_launch_time=collect_remote_cpu)
                    else:
                        def inject():
                            proof = inject_owned_failure(endpoint_states[recover_endpoint])
                            write_json(args.out / "failure-injection.json", proof)
                            return proof
                        exit_status = recovery_check.run_client(
                            command, stdin=input_stream, stdout=output, env=environment,
                            timeout=args.timeout, capture=capture_path, inject=inject,
                            evidence=injection, execution_state=client_execution,
                            record_launch_time=collect_remote_cpu)
                except subprocess.TimeoutExpired:
                    exit_status = 124
                    timed_out = True
        client_end_ns = time.monotonic_ns()
        if remote_cpu_collector is not None and not remote_cpu_finalized:
            client_launch_ns = int(client_execution.get(
                "client_launch_monotonic_ns") or client_launch_ns)
            remote_cpu_result = remote_cpu_collector.finish(
                client_end_ns=client_end_ns, client_launch_ns=client_launch_ns)
            write_json(args.out / "remote-cpu.json", remote_cpu_result)
            remote_cpu_finalized = True
        if exit_status != 0:
            if timed_out:
                raise RuntimeError(f"client timed out after {args.timeout} s")
            raise RuntimeError(f"client exit status {exit_status}")
        log_text = args.out.joinpath("client.log").read_text(
            encoding="utf-8", errors="replace")
        if latency is not None:
            check = latency_adapter.parse_result(
                log_text, latency, expected_workers=plan["worker_profile"]["app_workers"],
                histogram_dir=args.out / "histograms")
        else:
            check = parse_result(
                args.app, log_text, expected_bfs_repetitions=site.get("bfs_repetitions", 1),
                expected_post_samples=4096 if args.app in ("kv-b", "kv-a", "kv-s") else None)
        if args.system == "nonft" and args.app in ("kv-b", "kv-a", "kv-s"):
            check["nonft_kv"] = validate_nonft_kv(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                expected_workers=plan["worker_profile"]["app_workers"])
        if reference_chat is not None:
            check["chat_reference"] = recovery_check.compare_chat(capture_path, reference_chat)
        if args.system == "starfish" and figure13 is None:
            check["design2"] = validate_design2(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                expected_bfs_repetitions=site.get("bfs_repetitions", 1)
                if args.app == "bfs" else None)
        if hydra:
            check["hydra"] = validate_hydra(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                endpoint_count=len(specs), expected_workers=plan["worker_profile"],
                expected_resident_bytes=int(config_value(
                    effective, "local_resident_budget_bytes") or "0"))
        if carbink:
            check["carbink"] = carbink_support.validate_log(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                plan["worker_profile"],
                expected_resident_bytes=int(config_value(
                    effective, "local_resident_budget_bytes") or "0"))
        if figure13 is not None:
            check["figure13"] = kv_recovery.validate_run(
                args.out, plan["figure13"], injection)
        if ec and figure13 is None:
            check["recovery"] = recovery_check.validate_log(
                args.out.joinpath("client.log").read_text(encoding="utf-8", errors="replace"),
                injection if recover_endpoint is not None else None,
                endpoint_count=len(specs),
                answer_matches_reference=check.get("chat_reference", {}).get("matched", False))
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        error = str(exc)
        if remote_cpu_collector is not None and not remote_cpu_finalized:
            try:
                launch = client_execution.get(
                    "client_launch_monotonic_ns") or client_launch_ns
                remote_cpu_result = remote_cpu_collector.finish(
                    client_end_ns=None, client_launch_ns=launch)
                write_json(args.out / "remote-cpu.json", remote_cpu_result)
                remote_cpu_finalized = True
            except (OSError, RuntimeError, ValueError) as cpu_error:
                error += "; remote CPU finalization: " + str(cpu_error)
        try:
            recovered_measurement = recover_measurement(args.out, {
                "application": args.app, "exit_status": exit_status,
                "timed_out": timed_out, "failure_injection": injection or None,
            }, manifest) or {}
        except (OSError, KeyError, TypeError, ValueError) as recovery_error:
            measurement_recovery_error = str(recovery_error)
    finally:
        if remote_cpu_collector is not None and not remote_cpu_finalized:
            try:
                launch = client_execution.get(
                    "client_launch_monotonic_ns") or client_launch_ns
                remote_cpu_result = remote_cpu_collector.finish(
                    client_end_ns=None, client_launch_ns=launch)
                write_json(args.out / "remote-cpu.json", remote_cpu_result)
                remote_cpu_finalized = True
            except (OSError, RuntimeError, ValueError) as cpu_error:
                error += "; remote CPU finalization: " + str(cpu_error)
        for state in reversed(endpoint_states):
            if state.get("pid") is not None:
                try:
                    state["status"] = stop_server_process(state["memory_host"], state)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    state["status"] = "cleanup_failed"
                    state["error"] = str(exc)
                    error += f"; {state['name']} cleanup check: {exc}"
            if (state.get("remote_dir_created")
                    and not (state.get("launch_attempted") is False
                             and state.get("pid") is None)):
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
        if not passed and recovered_measurement:
            check = recovered_measurement
        analysis = {
            "schema_version": 1, "application": args.app,
            "repeat": repeat,
            "baseline_variant": baseline_variant,
            "backup_enabled": backup_enabled,
            "workload": WORKLOAD_LABEL[args.app],
            "system": SYSTEM_LABEL[args.system],
            "ratio": args.ratio, "run_id": args.out.name,
            "environment": site["name"], "exit_status": exit_status,
            "timed_out": timed_out, "timeout_s": args.timeout,
            "client_cleanup": client_execution or None,
            "measurement_usable": passed or bool(recovered_measurement),
            "measurement_correctness": "pass" if passed or recovered_measurement else "fail",
            "measurement_warning": recovered_measurement.get("measurement_warning"),
            "measurement_recovery_error": measurement_recovery_error,
            "correctness": "pass" if passed else "fail",
            "status": "passed" if passed else "failed",
            "elapsed_s": check.get("elapsed_s") if passed or recovered_measurement else None,
            "measurement_phase": check.get("measurement_phase"),
            "measurement_method": check.get("measurement_method"),
            "iteration_elapsed_s": check.get("iteration_elapsed_s"),
            "measured_repetitions": check.get("measured_repetitions"),
            "correctness_scope": check.get("correctness_scope"),
            "correctness_evidence": check.get("correctness_evidence"),
            "design2": check.get("design2"),
            "hydra": check.get("hydra"),
            "carbink": check.get("carbink"),
            "nonft_kv": check.get("nonft_kv"),
            "recovery": check.get("recovery"),
            "figure13": check.get("figure13"),
            "chat_reference": check.get("chat_reference"),
            "failure_injection": injection or None,
            "ft_method": config_value(effective, "ft_method") or "none",
            "endpoint_count": len(endpoint_states), "endpoints": endpoint_states,
            "error": error,
        }
        if latency is not None:
            analysis[latency_key] = check.get(latency_key)
        if collect_remote_cpu:
            analysis["remote_cpu"] = remote_cpu_result
        write_json(args.out / "analysis.json", analysis)
        manifest.update({
            "end_time": now(), "status": analysis["status"],
            "baseline_variant": baseline_variant,
            "backup_enabled": backup_enabled,
            "exit_status": exit_status, "error": error,
            "memory_server_pid": endpoint_states[0].get("pid"),
            "memory_server_pids": [s.get("pid") for s in endpoint_states],
            "endpoints": endpoint_states,
            "server_sha256_by_endpoint": [s["server_sha256"] for s in endpoint_states],
            "server_sha256": endpoint_states[0]["server_sha256"]
                if len(endpoint_states) == 1 else None,
            "failure_injection": injection or None,
        })
        if collect_remote_cpu:
            manifest["remote_cpu"] = remote_cpu_result
            manifest["remote_cpu_sidecar"] = str(args.out / "remote-cpu.json")
        write_json(args.out / "manifest.json", manifest)
        if figure13 is not None and analysis["status"] == "passed":
            kv_recovery.write_result(
                args.out, passed=True, plan=plan["figure13"])
    if analysis["status"] == "passed":
        print(f"PASS {analysis['run_id']}: work {analysis['elapsed_s']:.6f} s "
              f"(details: {args.out / 'analysis.json'})")
    else:
        label = "WARNING" if analysis["measurement_usable"] else "FAIL"
        detail = analysis.get("measurement_warning") or error or "incomplete run"
        print(f"{label} {analysis['run_id']}: {detail} "
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
    parser.add_argument("--repeat", type=int,
                        help="batch repetition index; defaults to the run directory suffix or 1")
    parser.add_argument("--baseline-variant",
                        choices=("nonft-backup-off",))
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--offered-load-ops", type=int)
    parser.add_argument("--latency-warmup-ms", type=int)
    parser.add_argument("--latency-measure-ms", type=int)
    parser.add_argument("--latency-drain-ms", type=int)
    parser.add_argument("--max-queue-delay-us", type=int)
    parser.add_argument("--capture-chat", action="store_true")
    parser.add_argument("--reference-chat", type=Path)
    parser.add_argument("--recover-endpoint", type=int)
    parser.add_argument("--figure13-scenario", choices=("1-node", "2-node"))
    parser.add_argument("--figure13-repeat", type=int)
    parser.add_argument("--figure13-observer-cpu", type=int)
    parser.add_argument("--collect-remote-cpu", "--remote-cpu",
                        dest="collect_remote_cpu", action="store_true",
                        help="opt in to owned memory-service CPU sidecar sampling")
    parser.add_argument("--collect-remote-memory", action="store_true",
                        help="enable timed remote-memory samples for Figure 11")
    parser.add_argument("--remote-memory-observer-cpu", type=int,
                        help="override the automatically derived memory observer CPU")
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
