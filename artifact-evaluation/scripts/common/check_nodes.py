#!/usr/bin/env python3
"""Read-only activity snapshot of every node in configs/machines.json."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from node_probe import idle_exempt_process

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS = 15
LOAD_POLICY = {
    "process_cpu_percent": 100.0,  # One logical CPU = 100%.
    "process_rss_bytes": 8 * 1024**3,
    "host_cpu_percent": 10.0,
    "minimum_available_memory_fraction": 0.20,
    "idle_exempt_services": ["clickhouse/clickhouse-server"],
}


def ssh_target(value):
    if not isinstance(value, str) or value.count("@") > 1:
        raise ValueError("invalid SSH host")
    parts = value.split("@")
    if len(parts) == 2 and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", parts[0]):
        raise ValueError("invalid SSH user")
    host = parts[-1]
    try:
        ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
            raise ValueError("SSH host must be an IP or simple configured alias")
    return value


def load_targets(inventory, site, user=None):
    data = json.loads(Path(inventory).read_text())
    servers = data.get("servers", [])
    if not isinstance(servers, list) or not servers:
        raise ValueError("machine inventory must contain a nonempty servers list")
    raw_site = json.loads(Path(site).read_text()) if site and Path(site).is_file() else {}
    records = [raw_site] + raw_site.get("memory_endpoints", [])
    overrides, usernames = {}, set()
    for record in records:
        host = record.get("memory_host")
        address = record.get("memory_ip") or record.get("memory_addr") or record.get("ip")
        if host:
            host = ssh_target(host)
            if "@" in host:
                usernames.add(host.split("@")[0])
            if address:
                overrides[str(ip_address(address))] = host
    # A single explicitly configured memory account is reusable for the node
    # inventory. Heterogeneous accounts retain per-endpoint overrides.
    default_user = next(iter(usernames)) if len(usernames) == 1 else None
    if user is not None:
        ssh_target(user + "@example")
    targets, seen = [], set()
    for record in servers:
        address = str(ip_address(record["ip"]))
        if address in seen:
            raise ValueError("duplicate node IP in inventory: " + address)
        seen.add(address)
        host = record.get("ssh_host") or overrides.get(address)
        host = host or (default_user + "@" + address if default_user else address)
        if user:
            host = user + "@" + host.rsplit("@", 1)[-1]
        targets.append({"ip": address, "hostname": record.get("hostname", address),
                        "ssh_host": ssh_target(host)})
    return targets


def local_addresses():
    try:
        result = subprocess.run(["ip", "-j", "address", "show"], text=True,
                                capture_output=True, timeout=3, check=True)
        return {item["local"] for link in json.loads(result.stdout)
                for item in link.get("addr_info", []) if item.get("local")}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return set()


def inspect_node(target, source, local_ips):
    local = target["ip"] in local_ips
    command = ([sys.executable, "-B", "-"] if local else [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=5",
        "-o", "ForwardAgent=no", "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=1",
        target["ssh_host"], "python3 -B -"])
    row = {**target, "transport": "local" if local else "ssh"}
    try:
        result = subprocess.run(command, input=source, text=True, capture_output=True,
                                timeout=TIMEOUT_SECONDS, check=False)
        if result.returncode:
            row.update(status="UNREACHABLE" if not local and result.returncode == 255 else "UNKNOWN",
                       error=(result.stderr or result.stdout or
                              f"probe exited {result.returncode}").strip()[:500])
            return row
        snapshot = json.loads(result.stdout)
        if (not isinstance(snapshot, dict) or not isinstance(snapshot.get("processes"), list)
                or any(not isinstance(p, dict) for p in snapshot["processes"])):
            raise ValueError("probe returned an invalid snapshot")
        processes = snapshot["processes"]
        if any(p.get("kind") != "candidate_server" for p in processes):
            status = "EXPERIMENT"
        elif not snapshot.get("visibility_complete", False):
            status = "PARTIAL"
        elif processes:
            status = "REVIEW"
        else:
            status = "NO_MATCH"
        row.update(status=status, snapshot=snapshot)
    except subprocess.TimeoutExpired:
        row.update(status="UNKNOWN", error=f"node check exceeded {TIMEOUT_SECONDS}s")
    except (OSError, ValueError, TypeError, KeyError) as error:
        row.update(status="UNKNOWN", error=str(error))
    return row


def clean(value):
    return "".join(char if char.isprintable() else " " for char in str(value))


def number(value, scale=1, digits=1):
    return "-" if value is None else f"{value / scale:.{digits}f}"


def availability(row):
    """An idle snapshot requires both low load and no known experiment."""
    snap = row.get("snapshot", {})
    processes = snap.get("processes", [])
    known = [p for p in processes if p.get("kind") != "candidate_server"]
    candidates = [p for p in processes if p.get("kind") == "candidate_server"]
    resource = {p.get("pid"): p for p in (
        snap.get("top_processes", []) + snap.get("top_memory_processes", []) + processes)}
    exempt = [p for p in snap.get("idle_exempt_processes", []) if idle_exempt_process(p)]
    foreground = [p for p in resource.values() if not idle_exempt_process(p)]
    high_cpu = [p for p in foreground
                if (p.get("cpu_percent") or 0) >= LOAD_POLICY["process_cpu_percent"]]
    high_memory = [p for p in foreground
                   if (p.get("rss_bytes") or 0) >= LOAD_POLICY["process_rss_bytes"]]
    cpu = snap.get("cpu_percent")
    exempt_complete = (all(p.get("host_cpu_percent") is not None for p in exempt)
                       and all(p.get("pid") in {q.get("pid") for q in exempt}
                               for p in resource.values() if idle_exempt_process(p)))
    effective_cpu = (max(0.0, cpu - sum(p["host_cpu_percent"] for p in exempt))
                     if cpu is not None and exempt_complete else None)
    memory = snap.get("memory", {})
    total, available = memory.get("total_bytes"), memory.get("available_bytes")
    pressure = (total is not None and total > 0 and available is not None
                and available / total < LOAD_POLICY["minimum_available_memory_fraction"])
    cpu_busy = bool(high_cpu) or (effective_cpu is not None
                                 and effective_cpu >= LOAD_POLICY["host_cpu_percent"])
    memory_busy = bool(high_memory) or pressure
    complete = (not row.get("error") and snap.get("visibility_complete", False)
                and cpu is not None and total is not None and total > 0 and available is not None
                and "top_memory_processes" in snap and exempt_complete)
    if cpu_busy or memory_busy:
        load = "CPU + memory" if cpu_busy and memory_busy else "high CPU" if cpu_busy else "high memory"
    else:
        load = "clear" if complete else "unknown"
    idle = False if known or cpu_busy or memory_busy else None if not complete or candidates else True
    labels = {"run_chat_far": "LLaMA", "gapbs_bfs_chunked": "BFS", "mg": "MG",
              "wordcount_far": "WC", "nhop_graph": "NQ", "kvs_throughput": "KV"}
    groups = {}
    for process in known:
        kind = process.get("kind")
        name = process.get("name", "")
        if kind == "memory_service":
            label = "memory"
        elif kind == "launcher":
            label = "launcher"
        elif kind == "network_benchmark":
            label = "IB test"
        else:
            label = next((value for key, value in labels.items()
                          if name == key or any(name.startswith(key + sep) for sep in ("-", "_", "."))),
                         "benchmark")
        groups[label] = groups.get(label, 0) + 1
    runtime = ", ".join(label + (f" x{count}" if count > 1 else "")
                        for label, count in groups.items())
    if not runtime:
        runtime = "possible server" if candidates else "none" if complete else "unknown"
    return {"idle": idle, "load": load, "runtime": runtime,
            "effective_host_cpu_percent": effective_cpu,
            "idle_exempt_processes": exempt,
            "high_cpu_processes": high_cpu, "high_memory_processes": high_memory}


def colorize(text, color, *, bold=False):
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return text
    return f"\x1b[{'1;' if bold else ''}{color}m{text}\x1b[0m"


def display(report):
    nodes = report["nodes"]
    states = [(row, row.get("availability") or availability(row)) for row in nodes]
    idle = sum(state["idle"] is True for _, state in states)
    busy = sum(state["idle"] is False for _, state in states)
    unknown = len(nodes) - idle - busy
    heading = f"  {'✓' if idle else '?' if unknown else '✗'}  {idle}/{len(nodes)} NODES IDLE"
    print()
    print(colorize(heading, 32 if idle else 33 if unknown else 31, bold=True) +
          f"    {busy} busy" + (f"    {unknown} unknown" if unknown else ""))
    print("  " + "─" * 66)
    print("  NODE              IDLE       LOAD            RUNTIME / TEST")
    for row, state in states:
        value = state["idle"]
        mark, shade = ("✓ YES", 32) if value is True else ("✗ NO", 31) if value is False else ("? UNKNOWN", 33)
        load = "unreachable" if row.get("status") == "UNREACHABLE" else state["load"]
        print(f"  {row['ip']:<17} {colorize(mark.ljust(10), shade)} "
              f"{load:<15} {clean(state['runtime'])}")
    print("  " + "─" * 66)
    print("  Snapshot only · details and thresholds: --verbose")


def display_verbose(report):
    print("Read-only node activity snapshot — " + report["timestamp_utc"])
    print("NODE             HOST             STATUS       CPU%   AVAILABLE/TOTAL GiB  TCP LISTEN")
    for row in report["nodes"]:
        snap = row.get("snapshot", {})
        memory = snap.get("memory", {})
        available = number(memory.get("available_bytes"), 1024**3)
        total = number(memory.get("total_bytes"), 1024**3)
        ports = ",".join(str(p) for p in snap.get("tcp_listen_ports", [])) or "-"
        print(f"{row['ip']:<16} {clean(snap.get('hostname', row['hostname'])):<16} "
              f"{row['status']:<12} {number(snap.get('cpu_percent')):>5} "
              f"{available + '/' + total:>21}  {ports}")
    for row in report["nodes"]:
        print("\n[" + row["ip"] + "] " + row["status"])
        if "error" in row:
            print("  " + clean(row["error"]))
            continue
        snap = row["snapshot"]
        for warning in snap.get("warnings", []):
            print("  WARNING: " + clean(warning))
        processes = snap.get("processes", [])
        if not processes:
            print("  No known workload, launcher, or memory-service process found.")
        for p in processes:
            print(f"  {clean(p.get('kind', '?')):<18} user={clean(p.get('user', '?'))} "
                  f"pid={p['pid']} name={clean(p.get('name', '?'))} "
                  f"cpu={number(p.get('cpu_percent'))}% "
                  f"rss={number(p.get('rss_bytes'), 1024**3)}GiB "
                  f"age={number(p.get('elapsed_s'), digits=0)}s")
        others = [p for p in snap.get("top_processes", [])
                  if p["pid"] not in {q["pid"] for q in processes}]
        if others:
            print("  Other top CPU: " + "; ".join(
                f"{clean(p.get('user', '?'))}/{p['pid']} {clean(p.get('name', '?'))} "
                f"{number(p.get('cpu_percent'))}%" for p in others))
        state = row.get("availability") or availability(row)
        if state["idle_exempt_processes"]:
            print("  Excluded from idle load: " + "; ".join(
                f"{clean(p['user'])}/{p['pid']} {clean(p['name'])}"
                for p in state["idle_exempt_processes"]))
            print("  Host CPU after exclusion: " +
                  number(state["effective_host_cpu_percent"]) + "%")
        memory = snap.get("memory", {})
        print("  HugePages free/total: " + str(memory.get("hugepages_free", "-")) +
              "/" + str(memory.get("hugepages_total", "-")))
    print("\nCPU is sampled for about 1s: host CPU is 0–100%; process CPU uses one core=100%.")
    print("EXPERIMENT means recognized processes exist, including idle memory services.")
    print("NO_MATCH is not a reservation or an availability guarantee; REVIEW means a candidate server.")
    print("PARTIAL/UNKNOWN/UNREACHABLE cannot establish whether the node is free.")
    print("Exit 0 means a complete snapshot, not that all nodes are idle.")
    print("Idle thresholds: process CPU >=100% (one core) or RSS >=8 GiB; "
          "host CPU >=10% or available RAM <20%. Known experiment processes also block idle.")
    print("The clickhouse-owned clickhouse-server daemon is excluded from process load "
          "and aggregate CPU checks. Actual low available RAM still blocks idle.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Exit codes: 0 = complete snapshot (nodes may be busy), 1 = incomplete "
        "snapshot, 2 = invalid configuration. Requires existing SSH access; no sudo."))
    parser.add_argument("--inventory", type=Path, default=ROOT / "configs/machines.json")
    parser.add_argument("--site", type=Path, default=ROOT / "data/site.json",
                        help="reuse SSH accounts/endpoint aliases from this site")
    parser.add_argument("--user", help="override the SSH username; otherwise reuse site/SSH settings")
    parser.add_argument("--json", action="store_true", help="print machine-readable snapshots")
    parser.add_argument("--verbose", action="store_true", help="show process/resource details and idle thresholds")
    args = parser.parse_args()
    try:
        targets = load_targets(args.inventory, args.site, args.user)
        source = Path(__file__).with_name("node_probe.py").read_text()
        local_ips = local_addresses()
        if args.verbose and not args.json:
            print(f"Checking {len(targets)} nodes in parallel (read-only)...", flush=True)
        with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
            nodes = list(pool.map(lambda node: inspect_node(node, source, local_ips), targets))
        for row in nodes:
            row["availability"] = availability(row)
        report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "load_policy": LOAD_POLICY, "nodes": nodes}
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            display(report)
            if args.verbose:
                print()
                display_verbose(report)
        return int(any("error" in row or not row.get("snapshot", {}).get("visibility_complete", False)
                       for row in nodes))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(2, "ERROR: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
