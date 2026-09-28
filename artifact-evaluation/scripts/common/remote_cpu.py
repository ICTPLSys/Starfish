"""Opt-in CPU accounting for owned memory-service processes.

The collector is intentionally independent from the normal runner path. It
samples each owned process through a persistent SSH probe, and it only emits a
CPU value when both an owned-process identity and a causal runtime work-end
marker are available. A missing marker or a process reuse is never represented
as zero CPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = 1
REMOTE_CPU_WINDOW = "initialization_and_requests"
DEFAULT_INTERVAL_SECONDS = 0.250
# All five systems use the same runtime traffic end marker.  The marker is
# emitted by the existing runtime work window, not by the runner's final
# verification or service cleanup.
RUNTIME_WORK_END = re.compile(
    r"^runtime_traffic_work\b[^\n]*\bend_monotonic_ns=(\d+)\b[^\n]*$",
    re.MULTILINE)

# This program runs on the memory host as the SSH login user. Every request
# carries the exact owned PID and its expected starttime/exe/cwd.
PROBE_SCRIPT = r'''
import json
import os
from pathlib import Path
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def stat_fields(pid):
    text = Path("/proc/%d/stat" % pid).read_text()
    close = text.rfind(")")
    if close < 0:
        raise ValueError("malformed /proc stat")
    return text[close + 2:].split()


def process_row(request):
    pid = int(request["pid"])
    proc = Path("/proc/%d" % pid)
    if not proc.is_dir():
        raise RuntimeError("missing_process")
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError("not_owned")
    fields = stat_fields(pid)
    if len(fields) < 20:
        raise ValueError("short /proc stat")
    state = fields[0]
    starttime = int(fields[19])
    if str(starttime) != str(request["starttime"]):
        raise RuntimeError("process_reused")
    if state in ("Z", "X", "x"):
        raise RuntimeError("process_exited")
    exe = os.path.realpath("/proc/%d/exe" % pid)
    cwd = os.path.realpath("/proc/%d/cwd" % pid)
    expected_exe = os.path.realpath(request["server_bin"])
    expected_cwd = os.path.realpath(request["remote_dir"])
    if exe != expected_exe or cwd != expected_cwd:
        raise RuntimeError("identity_mismatch")
    threads = []
    task_root = proc / "task"
    for task in sorted(task_root.iterdir(), key=lambda item: item.name):
        if not task.name.isdigit():
            continue
        parts = (task / "schedstat").read_text().split()
        if not parts:
            raise ValueError("malformed schedstat")
        threads.append({"tid": int(task.name), "runtime_ns": int(parts[0])})
    if not threads:
        raise RuntimeError("missing_threads")
    uptime_s = float(Path("/proc/uptime").read_text().split()[0])
    sample_ns = time.monotonic_ns()
    birth_ns = sample_ns - int(uptime_s * 1_000_000_000)
    birth_ns += int(starttime * 1_000_000_000 / HZ)
    return {
        "pid": pid,
        "uid": proc.stat().st_uid,
        "starttime": starttime,
        "state": state,
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "threads": threads,
        "birth_monotonic_ns": birth_ns,
        "exe": exe,
        "cwd": cwd,
    }


for line in sys.stdin:
    try:
        request = json.loads(line)
        rows = []
        for item in request["processes"]:
            try:
                rows.append(process_row(item))
            except Exception as error:
                rows.append({
                    "pid": int(item["pid"]),
                    "error": type(error).__name__ + ": " + str(error),
                })
        print(json.dumps({
            "clock_ticks_per_second": HZ,
            "sample_monotonic_ns": time.monotonic_ns(),
            "rows": rows,
        }), flush=True)
    except Exception as error:
        print(json.dumps({
            "clock_ticks_per_second": HZ,
            "sample_monotonic_ns": time.monotonic_ns(),
            "error": type(error).__name__ + ": " + str(error),
            "rows": [],
        }), flush=True)
'''


def _app_name(app: str) -> str:
    return str(app).strip().lower().replace("_", "-")


def request_end_spec(app: str) -> Optional[str]:
    """Return the common runtime work-end marker used by every workload."""
    del app
    return "runtime_traffic_work.end_monotonic_ns"


def find_request_end(app: str, text: str) -> Optional[Dict[str, Any]]:
    """Return the last generic runtime work-end marker in client output."""
    del app
    matches = list(RUNTIME_WORK_END.finditer(text))
    if not matches:
        return None
    lines = [match.group(0) for match in matches]
    marker = lines[-1]
    fields = dict(token.split("=", 1) for token in marker.split()[1:]
                  if "=" in token)
    if (fields.get("schema_version") != "1"
            or fields.get("scope") != "client_rdma"
            or fields.get("phase") != "work"):
        raise ValueError("runtime work-end marker has invalid provenance")
    return {
        "kind": "runtime_traffic_work.end_monotonic_ns",
        "line": marker,
        "end_monotonic_ns": int(matches[-1].group(1)),
        "marker_count": len(matches),
    }


def unavailable(reason: str, *, app: Optional[str] = None,
                endpoints: Optional[Sequence[Dict[str, Any]]] = None,
                provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an explicit unavailable sidecar without manufacturing a metric."""
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "unavailable",
        "remote_cpu_window": REMOTE_CPU_WINDOW,
        "reason": str(reason),
        "endpoints": list(endpoints or []),
        "provenance": dict(provenance or {}),
    }
    if app is not None:
        result["application"] = _app_name(app)
    return result


class PersistentProbe:
    """One SSH request/response stream for one memory host."""

    def __init__(self, host: str, *, timeout_s: float = 10.0,
                 popen: Callable[..., Any] = subprocess.Popen):
        command = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host,
            "python3 -u -c " + shlex.quote(PROBE_SCRIPT),
        ]
        self.process = popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self.timeout_s = timeout_s
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.last_remote: Optional[Dict[str, Any]] = None

    def sample(self, processes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        request = json.dumps({"processes": list(processes)}) + "\n"
        sent_ns = time.monotonic_ns()
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("CPU probe has no stdio")
        self.process.stdin.write(request)
        self.process.stdin.flush()
        if not self.selector.select(self.timeout_s):
            raise TimeoutError("CPU probe response timeout")
        line = self.process.stdout.readline()
        received_ns = time.monotonic_ns()
        if not line:
            raise RuntimeError("CPU probe exited")
        try:
            remote = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("CPU probe returned malformed JSON") from error
        self.last_remote = remote
        return {
            "send_ns": sent_ns,
            "receive_ns": received_ns,
            "rtt_ns": received_ns - sent_ns,
            "remote": remote,
        }

    def close(self) -> None:
        try:
            self.selector.close()
        finally:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)


class RemoteCpuCollector:
    """Sample owned memory services until the live request-end marker arrives."""

    def __init__(
        self,
        states: Sequence[Dict[str, Any]],
        *,
        app: str,
        log_path: Optional[Path] = None,
        interval_s: float = DEFAULT_INTERVAL_SECONDS,
        probe_factory: Optional[Callable[[str, Sequence[Dict[str, Any]]],
                                           Any]] = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ):
        self.states = [dict(state) for state in states]
        self.app = _app_name(app)
        self.log_path = Path(log_path) if log_path is not None else None
        self.interval_s = float(interval_s)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("CPU sampling interval must be positive and finite")
        self.probe_factory = probe_factory or (
            lambda host, _requests: PersistentProbe(host))
        self.clock = clock
        if not self.states:
            raise ValueError("CPU collector requires at least one endpoint")
        self._groups: Dict[str, List[Dict[str, Any]]] = {}
        seen = set()
        for state in self.states:
            for field in ("memory_host", "pid", "starttime", "server_bin",
                          "remote_dir"):
                if field not in state:
                    raise ValueError("CPU state lacks " + field)
            host = str(state["memory_host"])
            key = (host, int(state["pid"]))
            if key in seen:
                raise ValueError("duplicate owned CPU PID")
            seen.add(key)
            if "birth_local_lower_bound_ns" not in state:
                raise ValueError("CPU state lacks birth_local_lower_bound_ns")
            self._groups.setdefault(host, []).append(state)
        self._probes: Dict[str, Any] = {}
        self._stop = threading.Event()
        self._first_sample = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._marker: Optional[Dict[str, Any]] = None
        self._marker_seen_ns: Optional[int] = None
        self._error: Optional[str] = None
        self._last_threads: Dict[tuple[str, int], Dict[int, int]] = {}
        self._sched_total: Dict[tuple[str, int], int] = {}
        self._last_proc_ticks: Dict[tuple[str, int], int] = {}
        self._client_launch_ns: Optional[int] = None
        self._client_end_ns: Optional[int] = None

    def _requests(self, host: str) -> List[Dict[str, Any]]:
        return [{
            "pid": int(state["pid"]),
            "starttime": str(state["starttime"]),
            "server_bin": str(state["server_bin"]),
            "remote_dir": str(state["remote_dir"]),
        } for state in self._groups[host]]

    def _scan_marker(self) -> None:
        # Kept as a no-op compatibility seam. The last generic marker is read
        # at finish so repeated Work windows can be selected atomically.
        return

    def _accumulate(self, host: str, row: Dict[str, Any],
                    hz: int) -> Dict[str, Any]:
        if "error" in row:
            raise RuntimeError("owned CPU probe: " + str(row["error"]))
        pid = int(row["pid"])
        key = (host, pid)
        expected = next(
            (state for state in self._groups[host]
             if int(state["pid"]) == pid), None)
        if expected is None:
            raise RuntimeError("owned CPU probe: unexpected endpoint row")
        if str(row.get("starttime")) != str(expected["starttime"]):
            raise RuntimeError("owned CPU probe: process_reused")
        if int(row.get("uid", -1)) < 0:
            raise RuntimeError("owned CPU probe: missing_uid")
        threads = {}
        delta_ns = 0
        for thread in row.get("threads", []):
            tid = int(thread["tid"])
            runtime_ns = int(thread["runtime_ns"])
            if runtime_ns < 0:
                raise RuntimeError("owned CPU probe: negative_schedstat")
            previous = self._last_threads.setdefault(key, {}).get(tid)
            if previous is None:
                # The first snapshot is intentionally included in full. It
                # covers CPU consumed during service initialization before the
                # collector's first causal request.
                delta_ns += runtime_ns
            elif runtime_ns < previous:
                raise RuntimeError("owned CPU probe: schedstat_decreased")
            else:
                delta_ns += runtime_ns - previous
            threads[tid] = runtime_ns
        if not threads:
            raise RuntimeError("owned CPU probe: missing_threads")
        self._last_threads[key] = threads
        self._sched_total[key] = self._sched_total.get(key, 0) + delta_ns
        proc_ticks = int(row["user_ticks"]) + int(row["system_ticks"])
        previous_ticks = self._last_proc_ticks.get(key)
        if previous_ticks is not None and proc_ticks < previous_ticks:
            raise RuntimeError("owned CPU probe: proc_ticks_decreased")
        self._last_proc_ticks[key] = proc_ticks
        return {
            "pid": pid,
            "starttime": int(row["starttime"]),
            "birth_monotonic_ns": int(row["birth_monotonic_ns"]),
            "thread_count": len(threads),
            "schedstat_total_ns": self._sched_total[key],
            "proc_cpu_ticks": proc_ticks,
            "proc_cpu_seconds": proc_ticks / float(hz),
            "clock_ticks_per_second": hz,
        }

    def _take_sample(self) -> None:
        host_results = {}
        send_ns = None
        receive_ns = None
        for host in self._groups:
            response = self._probes[host].sample(self._requests(host))
            host_results[host] = response
            send_ns = response["send_ns"] if send_ns is None else min(
                send_ns, response["send_ns"])
            receive_ns = response["receive_ns"] if receive_ns is None else max(
                receive_ns, response["receive_ns"])
        rows = []
        seen_keys = set()
        for host, response in host_results.items():
            remote = response["remote"]
            if remote.get("error"):
                raise RuntimeError("owned CPU probe: " + str(remote["error"]))
            hz = int(remote["clock_ticks_per_second"])
            if hz <= 0:
                raise RuntimeError("owned CPU probe: invalid clock frequency")
            for row in remote.get("rows", []):
                key = (host, int(row.get("pid", -1)))
                if key in seen_keys:
                    raise RuntimeError("owned CPU probe: duplicate endpoint row")
                seen_keys.add(key)
                rows.append((host, self._accumulate(host, row, hz)))
        if len(rows) != len(self.states):
            raise RuntimeError("owned CPU probe: missing endpoint row")
        self._samples.append({
            "sequence": len(self._samples),
            "send_ns": int(send_ns),
            "receive_ns": int(receive_ns),
            "rtt_ns": int(receive_ns - send_ns),
            "endpoints": {(host, int(row["pid"])): row for host, row in rows},
        })
        self._first_sample.set()

    def _loop(self) -> None:
        next_ns = self.clock()
        while not self._stop.is_set():
            try:
                self._take_sample()
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                self._error = str(error)
                self._stop.set()
                return
            next_ns += int(self.interval_s * 1_000_000_000)
            remaining = (next_ns - self.clock()) / 1_000_000_000
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                next_ns = self.clock()

    def start(self, *, wait_s: float = 15.0) -> None:
        """Start sampling and require a first owned snapshot before requests."""
        try:
            for host in self._groups:
                self._probes[host] = self.probe_factory(
                    host, self._requests(host))
            self._thread = threading.Thread(target=self._loop,
                                            name="remote-cpu", daemon=True)
            self._thread.start()
            if not self._first_sample.wait(wait_s):
                raise TimeoutError("owned CPU probe first snapshot timeout")
            if self._error:
                raise RuntimeError(self._error)
        except Exception:
            self.close()
            raise

    def _crosscheck(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        sched_ns = int(endpoint["schedstat_total_ns"])
        proc_ns = float(endpoint["proc_cpu_seconds"]) * 1_000_000_000
        hz = int(endpoint["clock_ticks_per_second"])
        threads = max(1, int(endpoint["thread_count"]))
        # user+system ticks are quantized per thread. Two ticks per observed
        # thread is a conservative allowance for a boundary read race.
        tolerance_ns = max(1_000_000.0,
                           2.0 * threads * 1_000_000_000 / hz)
        difference_ns = abs(sched_ns - proc_ns)
        return {
            "schedstat_cpu_seconds": sched_ns / 1_000_000_000,
            "proc_cpu_seconds": proc_ns / 1_000_000_000,
            "difference_seconds": difference_ns / 1_000_000_000,
            "tolerance_seconds": tolerance_ns / 1_000_000_000,
            "status": "passed" if difference_ns <= tolerance_ns else "failed",
        }

    def result(self) -> Dict[str, Any]:
        if self._marker is None and self.log_path is not None:
            if self.log_path.is_file():
                text = self.log_path.read_text(
                    encoding="utf-8", errors="replace")
                self._marker = find_request_end(self.app, text)
        if self._error:
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "remote_cpu_window": REMOTE_CPU_WINDOW,
                "reason": self._error,
                "endpoints": [],
                "provenance": {"sampling_interval_s": self.interval_s,
                                "sample_count": len(self._samples)},
            }
        if self._marker is None:
            return unavailable(
                "missing_runtime_work_end_marker", app=self.app,
                provenance={"sampling_interval_s": self.interval_s,
                            "sample_count": len(self._samples)})
        measurement_end_ns = int(self._marker["end_monotonic_ns"])
        before = [sample for sample in self._samples
                  if sample["receive_ns"] <= measurement_end_ns]
        if not before:
            return unavailable(
                "no_sample_before_runtime_work_end", app=self.app,
                provenance={"sampling_interval_s": self.interval_s,
                            "sample_count": len(self._samples)})
        boundary = before[-1]
        birth_lower = min(
            int(state["birth_local_lower_bound_ns"]) for state in self.states)
        elapsed_ns = measurement_end_ns - birth_lower
        if elapsed_ns <= 0:
            return unavailable(
                "nonpositive_initialization_and_request_wall", app=self.app,
                provenance={"sampling_interval_s": self.interval_s,
                            "sample_count": len(self._samples)})
        endpoint_rows = []
        crosschecks = []
        for state in self.states:
            key = (str(state["memory_host"]), int(state["pid"]))
            endpoint = boundary["endpoints"].get(key)
            if endpoint is None:
                return unavailable(
                    "missing_boundary_endpoint", app=self.app,
                    provenance={"sample_count": len(self._samples)})
            checked = self._crosscheck(endpoint)
            crosschecks.append(checked)
            endpoint_rows.append({
                "endpoint": int(state.get("index", len(endpoint_rows))),
                "host": str(state["memory_host"]),
                "pid": int(state["pid"]),
                "starttime": str(state["starttime"]),
                "schedstat_cpu_seconds": checked["schedstat_cpu_seconds"],
                "proc_cpu_seconds": checked["proc_cpu_seconds"],
                "threads": int(endpoint["thread_count"]),
                "birth_monotonic_ns_remote": int(
                    endpoint["birth_monotonic_ns"]),
                "birth_local_lower_bound_ns": int(
                    state["birth_local_lower_bound_ns"]),
                "birth_local_upper_bound_ns": int(
                    state.get("birth_local_upper_bound_ns",
                              state["birth_local_lower_bound_ns"])),
                "crosscheck": checked,
            })
        if any(item["status"] != "passed" for item in crosschecks):
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "remote_cpu_window": REMOTE_CPU_WINDOW,
                "reason": "proc_stat_schedstat_crosscheck_failed",
                "endpoints": endpoint_rows,
                "provenance": {"sampling_interval_s": self.interval_s,
                                "sample_count": len(self._samples),
                                "work_end_marker": self._marker},
            }
        cpu_seconds = sum(
            item["schedstat_cpu_seconds"] for item in endpoint_rows)
        first_post = next((
            sample for sample in self._samples
            if sample["send_ns"] >= measurement_end_ns), None)
        post_provenance = None
        if first_post is not None:
            post_cpu_seconds = sum(
                row["schedstat_total_ns"] / 1_000_000_000
                for row in first_post["endpoints"].values())
            post_provenance = {
                "sequence": first_post["sequence"],
                "send_monotonic_ns": first_post["send_ns"],
                "receive_monotonic_ns": first_post["receive_ns"],
                "cpu_seconds_upper_bound": post_cpu_seconds,
                "marker_to_send_s": (
                    first_post["send_ns"] - measurement_end_ns)
                / 1_000_000_000,
            }
        # This is an uncertainty interval, not a claim that the before-end
        # sample happened exactly at client exit. The sample's SSH RTT is also
        # recorded because the remote observation has no shared clock with the
        # local client-end observer.
        end_error_ns = max(
            0, measurement_end_ns - boundary["receive_ns"]) + boundary["rtt_ns"]
        start_error_ns = max(
            int(state.get("birth_local_upper_bound_ns",
                          state["birth_local_lower_bound_ns"]))
            - int(state["birth_local_lower_bound_ns"])
            for state in self.states)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "remote_cpu_seconds": cpu_seconds,
            "remote_cpu_elapsed_s": elapsed_ns / 1_000_000_000,
            "remote_cpu_window": REMOTE_CPU_WINDOW,
            "endpoints": endpoint_rows,
            "provenance": {
                "method": "owned /proc/<pid>/task/<tid>/schedstat runtime_ns "
                           "cross-checked with /proc/<pid>/stat user+system ticks",
                "sampling_interval_s": self.interval_s,
                "sample_count": len(self._samples),
                "first_snapshot_included": True,
                "estimator": "last_pre_end_lower_bound",
                "denominator": "local monotonic earliest owned-service birth "
                               "lower bound to last runtime work-end marker",
                "client_launch_monotonic_ns": self._client_launch_ns,
                "client_end_monotonic_ns": self._client_end_ns,
                "client_end_hook": "client_process_exit",
                "measurement_end_monotonic_ns": measurement_end_ns,
                "measurement_end_source": "runtime_traffic_work.end_monotonic_ns",
                "start_boundary_error_s": start_error_ns / 1_000_000_000,
                "end_boundary_error_s": end_error_ns / 1_000_000_000,
                "end_sample_relation": "before_work_end",
                "first_post_end_sample": post_provenance,
                "end_sample_send_monotonic_ns": boundary["send_ns"],
                "end_sample_receive_monotonic_ns": boundary["receive_ns"],
                "end_marker_monotonic_ns": measurement_end_ns,
                "end_probe_rtt_s": boundary["rtt_ns"] / 1_000_000_000,
                "end_marker_to_sample_receive_s": max(
                    0, measurement_end_ns - boundary["receive_ns"])
                / 1_000_000_000,
                "work_end_marker": self._marker,
                "boundary_sample": boundary["sequence"],
                "crosscheck": crosschecks,
            },
        }

    def finish(self, *, client_end_ns: Optional[int] = None,
               client_launch_ns: Optional[int] = None) -> Dict[str, Any]:
        """Stop sampling, close probes, and return the fail-closed sidecar."""
        self._client_launch_ns = (None if client_launch_ns is None
                                  else int(client_launch_ns))
        self._client_end_ns = (None if client_end_ns is None
                               else int(client_end_ns))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 4))
        for probe in self._probes.values():
            try:
                probe.close()
            except (OSError, RuntimeError):
                pass
        return self.result()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 4))
        for probe in self._probes.values():
            try:
                probe.close()
            except (OSError, RuntimeError):
                pass
