#!/usr/bin/env python3
"""Read-only, best-effort activity probe for one Linux experiment node.

The module intentionally uses only the Linux proc filesystem and the standard
library.  It does not reserve resources, start workloads, or write temporary
files.  ``collect_snapshot`` is also usable when this file is fed to
``python3 -B -`` by the multi-node checker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import math
import os
from pathlib import Path
import pwd
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, MutableMapping, Optional


_PROC_ROOT = Path("/proc")
_PROC_STAT = _PROC_ROOT / "stat"
_PROC_MEMINFO = _PROC_ROOT / "meminfo"
_PROC_MOUNTS = _PROC_ROOT / "mounts"
_PROC_UPTIME = _PROC_ROOT / "uptime"
_SAMPLE_MAX_SECONDS = 2.0

_KNOWN_WORKLOADS = frozenset(
    {
        "run_chat_far",
        "gapbs_bfs_chunked",
        "mg",
        "wordcount_far",
        "nhop_graph",
        "kvs_throughput",
        "bandwidth_microbenchmark",
        "object_size",
    }
)
_NETWORK_BENCHMARKS = frozenset(
    {
        # Common perftest spellings; matching remains exact by basename.
        "ib_read_bw",
        "ib_write_bw",
        "ib_send_bw",
        "ib_read_lat",
        "ib_write_lat",
        "ib_send_lat",
    }
)
_LAUNCHER_SCRIPTS = (
    "scripts/common/run_case.py",
    "scripts/figure9/run.sh",
    "scripts/figure10/run.py",
    "scripts/figure11/run.py",
    "scripts/figure12/run.py",
    "scripts/figure13/run.py",
    "scripts/run_fast_check.sh",
)
_CHECKER_SCRIPTS = (
    "scripts/check_nodes.sh",
    "scripts/common/check_nodes.py",
)
_EXPERIMENT_MARKERS = ("starfish", "hydra", "carbink", "nonft", "artifact-evaluation")
_INTERPRETERS = frozenset(
    {
        "python",
        "python2",
        "python3",
        "python3.8",
        "python3.9",
        "python3.10",
        "python3.11",
        "python3.12",
        "bash",
        "sh",
        "dash",
        "zsh",
    }
)
_SERVER_NAMES = frozenset(
    {
        "server",
        "memory_server",
        "memory-service",
        "memory_service",
        "memserver",
        "mem_server",
        "starfish_server",
        "hydra_server",
        "carbink_server",
        "nonft_server",
    }
)


@dataclass(frozen=True)
class _ProcRecord:
    pid: int
    ppid: int
    comm: str
    argv: tuple[str, ...]
    exe: str
    user: str
    rss_bytes: Optional[int]
    utime_ticks: int
    stime_ticks: int
    starttime_ticks: int
    state: str = "S"


def _new_state() -> dict[str, Any]:
    return {"warnings": [], "partial": False, "counts": {}}


def _count(state: MutableMapping[str, Any], key: str, amount: int = 1) -> None:
    counts = state.setdefault("counts", {})
    counts[key] = int(counts.get(key, 0)) + amount


def _warn(
    state: MutableMapping[str, Any],
    message: str,
    *,
    partial: bool = False,
    key: Optional[str] = None,
) -> None:
    if key is not None:
        _count(state, key)
    elif message not in state["warnings"]:
        state["warnings"].append(message)
    if partial:
        state["partial"] = True


def _is_permission_error(exc: BaseException) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "errno", None) in {
        errno.EPERM,
        errno.EACCES,
    }


def _flush_aggregate_warnings(state: MutableMapping[str, Any]) -> None:
    counts = state.get("counts", {})
    aggregate = (
        ("stat_denied", "proc stat unavailable for {n} process(es); process visibility is partial"),
        ("stat_malformed", "proc stat malformed for {n} process(es); process visibility is partial"),
        ("cmdline_denied", "proc cmdline unavailable for {n} process(es); process visibility is partial"),
        ("cmdline_unavailable", "proc cmdline unavailable for {n} process(es); process visibility is partial"),
        ("status_denied", "proc status unavailable for {n} process(es)"),
        ("exited", "{n} process(es) exited while being sampled"),
    )
    for key, template in aggregate:
        value = int(counts.get(key, 0))
        if value:
            message = template.format(n=value)
            if message not in state["warnings"]:
                state["warnings"].append(message)
    for key, template in (
        ("proc_scan_denied", "proc directory scan was denied; process visibility is partial"),
        ("proc_stat_unavailable", "host CPU counters unavailable; CPU percentage is incomplete"),
        ("meminfo_unavailable", "memory counters unavailable; memory snapshot is incomplete"),
        ("ss_unavailable", "listening-port scan unavailable; port visibility is partial"),
    ):
        if counts.get(key):
            if template not in state["warnings"]:
                state["warnings"].append(template)


def _parse_proc_stat(raw: str) -> dict[str, Any]:
    """Parse ``/proc/<pid>/stat`` without splitting the parenthesized comm."""
    first_space = raw.find(" ")
    open_paren = raw.find("(", first_space + 1)
    close_paren = raw.rfind(")")
    if first_space <= 0 or open_paren <= first_space or close_paren <= open_paren:
        raise ValueError("malformed proc stat")
    pid = int(raw[:first_space])
    comm = raw[open_paren + 1 : close_paren]
    tail = raw[close_paren + 1 :].strip().split()
    # tail[0] is state (field 3); ppid is field 4, utime/stime are 14/15,
    # and starttime is field 22.
    if len(tail) <= 19:
        raise ValueError("short proc stat")
    return {
        "pid": pid,
        "comm": comm,
        "state": tail[0],
        "ppid": int(tail[1]),
        "utime_ticks": int(tail[11]),
        "stime_ticks": int(tail[12]),
        "starttime_ticks": int(tail[19]),
    }


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read()


def _read_proc_status(pid: int, state: MutableMapping[str, Any]) -> tuple[str, int]:
    user = "?"
    rss_bytes = None
    try:
        text = _read_bytes(_PROC_ROOT / str(pid) / "status").decode("utf-8", "replace")
    except FileNotFoundError:
        _count(state, "exited")
        return user, rss_bytes
    except OSError as exc:
        if _is_permission_error(exc):
            _warn(state, "", partial=True, key="status_denied")
        return user, rss_bytes
    uid: Optional[int] = None
    for line in text.splitlines():
        if line.startswith("Uid:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    uid = int(fields[1])
                except ValueError:
                    pass
        elif line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    rss_bytes = int(fields[1]) * 1024
                except ValueError:
                    pass
    if uid is not None:
        try:
            user = pwd.getpwuid(uid).pw_name
        except (KeyError, OSError):
            user = str(uid)
    return user, rss_bytes


def _read_proc_record(pid: int, state: MutableMapping[str, Any]) -> Optional[_ProcRecord]:
    proc_dir = _PROC_ROOT / str(pid)
    try:
        raw_stat = _read_bytes(proc_dir / "stat").decode("utf-8", "replace")
        stat = _parse_proc_stat(raw_stat)
    except FileNotFoundError:
        _count(state, "exited")
        return None
    except OSError as exc:
        if _is_permission_error(exc):
            _warn(state, "", partial=True, key="stat_denied")
        return None
    except (TypeError, ValueError, UnicodeError):
        _warn(state, "", partial=True, key="stat_malformed")
        return None

    try:
        raw_cmdline = _read_bytes(proc_dir / "cmdline")
        argv = tuple(
            part.decode("utf-8", "replace")
            for part in raw_cmdline.split(b"\0")
            if part
        )
    except FileNotFoundError:
        _count(state, "exited")
        return None
    except OSError as exc:
        argv = ()
        if _is_permission_error(exc):
            _warn(state, "", partial=True, key="cmdline_denied")
        else:
            _warn(state, "", partial=True, key="cmdline_unavailable")

    try:
        exe = os.readlink(proc_dir / "exe")
    except FileNotFoundError:
        exe = argv[0] if argv else ""
    except OSError as exc:
        # An unreadable exe is not itself a visibility failure: argv0 and comm
        # remain useful classification sources.
        exe = argv[0] if argv else ""
        if _is_permission_error(exc):
            _count(state, "exe_denied")

    user, rss_bytes = _read_proc_status(pid, state)
    return _ProcRecord(
        pid=pid,
        ppid=int(stat["ppid"]),
        comm=str(stat["comm"]),
        argv=argv,
        exe=exe,
        user=user,
        rss_bytes=rss_bytes,
        utime_ticks=int(stat["utime_ticks"]),
        stime_ticks=int(stat["stime_ticks"]),
        starttime_ticks=int(stat["starttime_ticks"]),
        state=str(stat["state"]),
    )


def _scan_processes(state: MutableMapping[str, Any]) -> list[_ProcRecord]:
    records: list[_ProcRecord] = []
    try:
        entries = list(os.scandir(_PROC_ROOT))
    except OSError as exc:
        if _is_permission_error(exc):
            _warn(state, "", partial=True, key="proc_scan_denied")
        else:
            _warn(state, "", partial=True, key="proc_scan_denied")
        return records
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        record = _read_proc_record(pid, state)
        if record is not None:
            records.append(record)
    return records


def _read_parent_pid(pid: int) -> Optional[int]:
    try:
        parsed = _parse_proc_stat(
            _read_bytes(_PROC_ROOT / str(pid) / "stat").decode("utf-8", "replace")
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, UnicodeError):
        return None
    return int(parsed["ppid"])


def _ancestor_pids() -> set[int]:
    """Return this probe and its launcher chain, so neither is reported."""
    result: set[int] = set()
    current = os.getpid()
    fallback_parent = os.getppid()
    while current > 1 and current not in result:
        result.add(current)
        parent = _read_parent_pid(current)
        if parent is None:
            parent = fallback_parent if current == os.getpid() else 1
        current = parent
    if fallback_parent > 1:
        result.add(fallback_parent)
    return result


def _clock_ticks() -> int:
    try:
        value = int(os.sysconf("SC_CLK_TCK"))
        return value if value > 0 else 100
    except (AttributeError, OSError, TypeError, ValueError):
        return 100


def _read_cpu_counters(state: MutableMapping[str, Any]) -> Optional[tuple[int, int]]:
    try:
        text = _read_bytes(_PROC_STAT).decode("ascii", "replace")
    except OSError as exc:
        _warn(state, "", partial=True, key="proc_stat_unavailable")
        return None
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        fields = line.split()[1:]
        try:
            values = [int(value) for value in fields]
        except ValueError:
            break
        if len(values) < 5:
            break
        # Linux reports guest/guest_nice in addition to user/nice.  The first
        # eight counters are the non-duplicated aggregate used by procps.
        total = sum(values[:8])
        idle = values[3] + values[4]  # idle + iowait
        return total, idle
    _warn(state, "", partial=True, key="proc_stat_unavailable")
    return None


def _cpu_percent(
    before: Optional[tuple[int, int]],
    after: Optional[tuple[int, int]],
) -> Optional[float]:
    if before is None or after is None:
        return None
    total_delta = after[0] - before[0]
    idle_delta = after[1] - before[1]
    if total_delta <= 0:
        return None
    value = 100.0 * (total_delta - idle_delta) / total_delta
    return max(0.0, min(100.0, value))


def _read_loadavg(state: MutableMapping[str, Any]) -> list[float]:
    try:
        values = list(os.getloadavg())[:3]
    except (AttributeError, OSError):
        _warn(state, "load average unavailable")
        return []
    result = [float(value) for value in values]
    return (result + [0.0, 0.0, 0.0])[:3]


def _read_meminfo(state: MutableMapping[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        text = _read_bytes(_PROC_MEMINFO).decode("ascii", "replace")
    except OSError:
        _warn(state, "", partial=True, key="meminfo_unavailable")
        return {
            "total_bytes": None,
            "available_bytes": None,
            "hugepages_total": None,
            "hugepages_free": None,
        }
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[0].endswith(":"):
            continue
        try:
            number = int(fields[1])
        except ValueError:
            continue
        unit = fields[2].lower() if len(fields) >= 3 else ""
        multiplier = 1024 if unit == "kb" else 1024 * 1024 if unit == "mb" else 1
        values[fields[0][:-1]] = number * multiplier

    total = int(values.get("MemTotal", 0))
    available = values.get("MemAvailable")
    if available is None:
        # MemFree is the only value that is unambiguously available on old
        # kernels; do not present a cached-page estimate as a measured value.
        available = int(values.get("MemFree", 0))
        _warn(state, "MemAvailable missing; falling back to MemFree")
    return {
        "total_bytes": total,
        "available_bytes": int(available),
        "hugepages_total": int(values.get("HugePages_Total", 0)),
        "hugepages_free": int(values.get("HugePages_Free", 0)),
    }


def _read_uptime(state: MutableMapping[str, Any]) -> Optional[float]:
    try:
        text = _read_bytes(_PROC_UPTIME).decode("ascii", "replace")
        return float(text.split()[0])
    except (OSError, IndexError, TypeError, ValueError):
        _warn(state, "process uptime unavailable; elapsed times may be zero")
        return None


def _normalise_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def _basename(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1].lower()


def _known_executable(name: str, known: Iterable[str]) -> bool:
    # Benchmarks are often retained with version/build suffixes, e.g.
    # kvs_throughput-v6 or mg.D.x. Match only the executable stem, not argv text.
    return any(name == base or any(name.startswith(base + separator)
                                  for separator in ("-", "_", "."))
               for base in known)


def _matches_launcher_script(arg: str) -> bool:
    normalized = _normalise_path(arg)
    return any(normalized == script or normalized.endswith("/" + script) for script in _LAUNCHER_SCRIPTS)


def _matches_checker_script(arg: str) -> bool:
    normalized = _normalise_path(arg)
    return any(normalized == script or normalized.endswith("/" + script) for script in _CHECKER_SCRIPTS)


def _looks_like_server(record: _ProcRecord) -> bool:
    names = {_basename(record.exe), _basename(record.argv[0]) if record.argv else ""}
    if any(_known_executable(name, _SERVER_NAMES) for name in names):
        return True
    return False


def _server_is_experiment(record: _ProcRecord) -> bool:
    candidates = [_normalise_path(record.exe)]
    if record.argv:
        candidates.append(_normalise_path(record.argv[0]))
        # These are individual arguments, not a substring search over a
        # reconstructed command line.  Paths are accepted because configs and
        # launch directories carry the experiment identity.
        candidates.extend(_normalise_path(arg) for arg in record.argv[1:] if "/" in arg or "\\" in arg)
    return any(marker in candidate for candidate in candidates for marker in _EXPERIMENT_MARKERS)


def _classify(record: _ProcRecord) -> Optional[str]:
    names = {_basename(record.exe), _basename(record.argv[0]) if record.argv else "", _basename(record.comm)}
    if any(_known_executable(name, _KNOWN_WORKLOADS) for name in names):
        return "workload"
    if any(_known_executable(name, _NETWORK_BENCHMARKS) for name in names):
        return "network_benchmark"
    interpreter = any(name in _INTERPRETERS for name in names)
    if interpreter and any(_matches_checker_script(arg) for arg in record.argv[1:]):
        return None
    if interpreter and any(_matches_launcher_script(arg) for arg in record.argv[1:]):
        return "launcher"
    if _looks_like_server(record):
        return "memory_service" if _server_is_experiment(record) else "candidate_server"
    return None


def _process_view(record: _ProcRecord, cpu_percent: Optional[float], uptime: Optional[float], hz: int) -> dict[str, Any]:
    elapsed = None
    if uptime is not None:
        elapsed = max(0.0, uptime - (record.starttime_ticks / float(hz)))
    executable = (record.exe or (record.argv[0] if record.argv else "")).split(" ", 1)[0]
    view = {
        "pid": record.pid,
        "user": record.user,
        "name": _basename(executable) or record.comm,
        "kind": "other",
        # Some daemons rewrite argv0 to a whole process title. Never export
        # those arguments through the executable fallback.
        "exe": executable,
        "cpu_percent": None if cpu_percent is None else max(0.0, float(cpu_percent)),
        "rss_bytes": None if record.rss_bytes is None else max(0, int(record.rss_bytes)),
        "elapsed_s": elapsed,
    }
    # /proc/exe resolves the daemon symlink to the shared "clickhouse" binary
    # when readable. Retain the exact server invocation, never the client.
    if (record.user == "clickhouse" and record.argv
            and _basename(record.argv[0]) == "clickhouse-server"
            and _basename(executable) in {"clickhouse", "clickhouse-server"}):
        view["idle_exempt_service"] = "clickhouse-server"
    return view


def _read_listen_ports(state: MutableMapping[str, Any]) -> list[int]:
    try:
        result = subprocess.run(
            ["ss", "-H", "-lnt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        _warn(state, "", partial=True, key="ss_unavailable")
        return []
    if result.returncode != 0:
        _warn(state, "", partial=True, key="ss_unavailable")
    ports: set[int] = set()
    for line in (result.stdout or "").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        address = fields[3]
        port_text = ""
        if "]:" in address:
            port_text = address.rsplit("]:", 1)[1]
        elif ":" in address:
            port_text = address.rsplit(":", 1)[1]
        elif address.isdigit():
            port_text = address
        try:
            port = int(port_text)
        except ValueError:
            continue
        if 0 < port <= 65535:
            ports.add(port)
    return sorted(ports)


def _sample_seconds(value: Any) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = 1.0
    if not math.isfinite(seconds):
        seconds = 1.0
    return min(_SAMPLE_MAX_SECONDS, max(0.0, seconds))


def idle_exempt_process(process: Mapping[str, Any]) -> bool:
    """The installed ClickHouse daemon is an allowed background service."""
    return (process.get("user") == "clickhouse"
            and (process.get("idle_exempt_service") == "clickhouse-server"
                 or (process.get("name") == "clickhouse-server"
                     and _basename(process.get("exe", "")) == "clickhouse-server"))
            and process.get("kind", "other") == "other")


def _top_with_nonexempt(views: list) -> list:
    # Keep raw leaders, plus leaders after exemptions, so background services
    # cannot hide another busy process below the raw top three.
    leaders = views[:3] + [p for p in views if not idle_exempt_process(p)][:3]
    return list({p["pid"]: p for p in leaders}.values())


def _collect_process_views(
    first: Iterable[_ProcRecord],
    second: Iterable[_ProcRecord],
    *,
    wall_seconds: float,
    uptime: Optional[float],
    hz: int,
    excluded: set[int],
    memory_views: Optional[list] = None,
    idle_exempt_views: Optional[list] = None,
    host_cpu_ticks: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = {(record.pid, record.starttime_ticks): record for record in first}
    matched: list[dict[str, Any]] = []
    top: list[dict[str, Any]] = []
    denominator = max(float(wall_seconds), 1e-9)
    for record in second:
        if record.pid in excluded or record.state in {"Z", "X"}:
            continue
        previous = before.get((record.pid, record.starttime_ticks))
        cpu = None
        ticks = None
        if previous is not None:
            ticks = (
                record.utime_ticks
                + record.stime_ticks
                - previous.utime_ticks
                - previous.stime_ticks
            )
            cpu = max(0.0, 100.0 * (ticks / float(hz)) / denominator)
        view = _process_view(record, cpu, uptime, hz)
        if idle_exempt_views is not None and idle_exempt_process(view):
            idle_exempt_views.append({
                **view,
                "host_cpu_percent": (100.0 * max(0, ticks) / host_cpu_ticks
                                     if ticks is not None and host_cpu_ticks
                                     and host_cpu_ticks > 0 else None),
            })
        if memory_views is not None and view["rss_bytes"] is not None:
            memory_views.append(view)
        if cpu is not None and cpu > 0:
            top.append(view)
        kind = _classify(record)
        if kind is not None:
            selected = dict(view)
            selected["kind"] = kind
            matched.append(selected)
    matched.sort(key=lambda item: (int(item["pid"]), str(item["name"])))
    top.sort(key=lambda item: (-float(item["cpu_percent"]), int(item["pid"])))
    if memory_views is not None:
        memory_views.sort(key=lambda item: (-item["rss_bytes"], item["pid"]))
        memory_views[:] = _top_with_nonexempt(memory_views)
    return matched, _top_with_nonexempt(top)


def collect_snapshot(sample_seconds: float = 1.0) -> dict[str, Any]:
    """Collect a bounded, read-only snapshot of this Linux node."""
    state = _new_state()
    duration = _sample_seconds(sample_seconds)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = "unknown"
        _warn(state, "hostname unavailable")

    started = time.monotonic()
    cpu_before = _read_cpu_counters(state)
    first = _scan_processes(state)
    excluded = _ancestor_pids()
    time.sleep(duration)
    wall_seconds = max(0.0, time.monotonic() - started)
    second = _scan_processes(state)
    cpu_after = _read_cpu_counters(state)
    uptime = _read_uptime(state)
    hz = _clock_ticks()
    top_memory_processes = []
    idle_exempt_processes = []
    processes, top_processes = _collect_process_views(
        first,
        second,
        wall_seconds=wall_seconds if wall_seconds > 0 else duration,
        uptime=uptime,
        hz=hz,
        excluded=excluded,
        memory_views=top_memory_processes,
        idle_exempt_views=idle_exempt_processes,
        host_cpu_ticks=(cpu_after[0] - cpu_before[0]
                        if cpu_after is not None and cpu_before is not None else None),
    )

    # hidepid is checked after process collection so a restricted mount is
    # explicitly represented even when the current process remains visible.
    try:
        mounts = _read_bytes(_PROC_MOUNTS).decode("utf-8", "replace")
        for line in mounts.splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "/proc":
                options = fields[3].split(",")
                hidepid = next((item for item in options if item.startswith("hidepid=")), "")
                if hidepid and hidepid != "hidepid=0":
                    _warn(state, "proc hidepid restricts process visibility", partial=True)
                break
    except OSError:
        _warn(state, "proc mount options unavailable", partial=True)

    loadavg = _read_loadavg(state)
    memory = _read_meminfo(state)
    ports = _read_listen_ports(state)
    _flush_aggregate_warnings(state)
    return {
        "hostname": hostname,
        "timestamp_utc": timestamp,
        "cpu_percent": _cpu_percent(cpu_before, cpu_after),
        "loadavg": loadavg,
        "memory": memory,
        "processes": processes,
        "top_processes": top_processes,
        "top_memory_processes": top_memory_processes,
        "idle_exempt_processes": idle_exempt_processes,
        "tcp_listen_ports": ports,
        "warnings": list(state["warnings"]),
        "visibility_complete": not bool(state["partial"]),
    }


def main() -> int:
    print(__import__("json").dumps(collect_snapshot(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
