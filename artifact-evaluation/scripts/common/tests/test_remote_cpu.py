"""Synthetic CPU-sidecar checks; no workload or memory service is launched."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_cpu


def state(pid=101, host="mem-a"):
    lower = time.monotonic_ns() - 2_000_000_000
    return {
        "index": 0,
        "memory_host": host,
        "pid": pid,
        "starttime": "77",
        "server_bin": "/run/case/endpoint/server",
        "remote_dir": "/run/case/endpoint",
        "birth_local_lower_bound_ns": lower,
        "birth_local_upper_bound_ns": lower + 2_000_000,
    }


def response(pid=101, runtime_ns=100_000_000, ticks=10, starttime="77"):
    return {
        "clock_ticks_per_second": 100,
        "rows": [{
            "pid": pid,
            "starttime": int(starttime),
            "uid": 1000,
            "birth_monotonic_ns": 10,
            "user_ticks": ticks,
            "system_ticks": 0,
            "threads": [{"tid": pid, "runtime_ns": runtime_ns}],
        }],
    }


class FakeProbe:
    """Mock persistent SSH/proc probe returning deterministic snapshots."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.closed = False

    def sample(self, _requests):
        remote = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        sent = time.monotonic_ns()
        received = sent + 1_000_000
        return {
            "send_ns": sent,
            "receive_ns": received,
            "rtt_ns": received - sent,
            "remote": remote,
        }

    def close(self):
        self.closed = True


def work_line(end_ns, window_id=1):
    return (
        "runtime_traffic_work schema_version=1 scope=client_rdma phase=work "
        f"window_id={window_id} start_monotonic_ns={end_ns - 1000000000} "
        f"end_monotonic_ns={end_ns} elapsed_ns=1000000000 "
        "read_bytes=1 write_bytes=2 rmw_read_bytes=0 fetch_bytes=1 "
        "eviction_bytes=2 snapshot=rolling_per_worker "
        "time_bounds=snapshot_bracket accounting=existing_counter_sites\n")


class RemoteCpuTests(unittest.TestCase):
    def make_collector(self, probe, app="bfs", log_path=None):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(log_path or root.name) / "client.log"
        collector = remote_cpu.RemoteCpuCollector(
            [state()], app=app, log_path=path, interval_s=0.250,
            probe_factory=lambda _host, _requests: probe)
        collector._probes["mem-a"] = probe
        return collector, path

    def test_generic_work_end_and_all_apps(self):
        for app in ("bfs", "llama", "mg", "wc", "kv-b", "kv-a", "kv-s", "nq"):
            with self.subTest(app=app):
                self.assertEqual(
                    remote_cpu.request_end_spec(app),
                    "runtime_traffic_work.end_monotonic_ns")
        end = time.monotonic_ns()
        marker = remote_cpu.find_request_end("bfs", work_line(end))
        self.assertEqual(marker["end_monotonic_ns"], end)
        self.assertEqual(marker["marker_count"], 1)

    def test_first_snapshot_is_included_and_boundary_is_not_exact(self):
        probe = FakeProbe([response(runtime_ns=100_000_000, ticks=10)])
        collector, path = self.make_collector(probe)
        collector._take_sample()
        end = time.monotonic_ns() + 2_000_000
        path.write_text(work_line(end), encoding="utf-8")
        result = collector.finish(client_end_ns=end + 1_000_000)
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["remote_cpu_seconds"], 0.1)
        self.assertEqual(result["remote_cpu_window"],
                         "initialization_and_requests")
        provenance = result["provenance"]
        self.assertTrue(provenance["first_snapshot_included"])
        self.assertEqual(provenance["estimator"], "last_pre_end_lower_bound")
        self.assertEqual(provenance["end_sample_relation"], "before_work_end")
        self.assertGreaterEqual(provenance["end_boundary_error_s"], 0)
        self.assertFalse(provenance.get("end_sample_exact", False))
        self.assertTrue(probe.closed)

    def test_last_work_end_marker_wins_for_repeated_work(self):
        probe = FakeProbe([response()])
        collector, path = self.make_collector(probe)
        collector._take_sample()
        first = time.monotonic_ns()
        second = first + 1_000_000
        path.write_text(work_line(first, 1) + work_line(second, 2),
                        encoding="utf-8")
        result = collector.finish(client_end_ns=second + 1_000_000)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["provenance"]["measurement_end_monotonic_ns"], second)
        self.assertEqual(result["provenance"]["work_end_marker"]["marker_count"], 2)

    def test_missing_end_marker_is_unavailable(self):
        probe = FakeProbe([response()])
        collector, path = self.make_collector(probe)
        collector._take_sample()
        path.write_text("benchmark output without runtime boundary\n",
                        encoding="utf-8")
        result = collector.finish(client_end_ns=time.monotonic_ns())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "missing_runtime_work_end_marker")
        self.assertNotIn("remote_cpu_seconds", result)

    def test_process_reuse_fails_closed(self):
        probe = FakeProbe([
            response(runtime_ns=100_000_000, ticks=10),
            response(runtime_ns=110_000_000, ticks=11, starttime="78"),
        ])
        collector, _path = self.make_collector(probe)
        collector._take_sample()
        with self.assertRaisesRegex(RuntimeError, "process_reused"):
            collector._take_sample()
        collector._error = "owned CPU probe: process_reused"
        result = collector.finish(client_end_ns=time.monotonic_ns())
        self.assertEqual(result["status"], "failed")
        self.assertIn("process_reused", result["reason"])

    def test_duplicate_probe_pid_fails_closed(self):
        duplicate = response()
        duplicate["rows"].append(dict(duplicate["rows"][0]))
        probe = FakeProbe([duplicate])
        collector, _path = self.make_collector(probe)
        with self.assertRaisesRegex(RuntimeError, "duplicate endpoint row"):
            collector._take_sample()


if __name__ == "__main__":
    unittest.main()
