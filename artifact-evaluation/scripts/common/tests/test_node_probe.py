"""Focused offline tests for the read-only node activity probe."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import node_probe


class NodeProbe(unittest.TestCase):
    def test_proc_stat_parser_handles_parentheses_in_comm(self):
        # Fields after comm: state, ppid, ..., utime(14), stime(15), ..., starttime(22).
        tail = ["S", "7"] + ["0"] * 9 + ["11", "13"] + ["0"] * 6 + ["997"]
        parsed = node_probe._parse_proc_stat("42 (worker)with)paren) " + " ".join(tail))
        self.assertEqual(parsed["pid"], 42)
        self.assertEqual(parsed["comm"], "worker)with)paren")
        self.assertEqual(parsed["ppid"], 7)
        self.assertEqual(parsed["utime_ticks"], 11)
        self.assertEqual(parsed["stime_ticks"], 13)
        self.assertEqual(parsed["starttime_ticks"], 997)

    def test_classifier_uses_exact_names_and_script_paths(self):
        def record(exe, argv, comm="python3"):
            return node_probe._ProcRecord(
                pid=10, ppid=1, comm=comm, argv=tuple(argv), exe=exe,
                user="tester", rss_bytes=0, utime_ticks=0, stime_ticks=0,
                starttime_ticks=1,
            )

        self.assertEqual(
            node_probe._classify(record("/opt/bin/mg", ["mg"])), "workload"
        )
        self.assertEqual(
            node_probe._classify(record("/usr/bin/python3", ["python3", "/x/scripts/figure10/run.py"])),
            "launcher",
        )
        self.assertIsNone(
            node_probe._classify(record("/tmp/not-mg-wrapper", ["not-mg-wrapper", "mg"])),
        )
        self.assertEqual(
            node_probe._classify(record("/opt/hydra_server", ["hydra_server"])),
            "memory_service",
        )
        self.assertEqual(
            node_probe._classify(record("/opt/server", ["server", "/tmp/unrelated.conf"])),
            "candidate_server",
        )
        for name in ("ib_read_bw", "ib_write_bw", "ib_send_bw", "ib_read_lat",
                     "ib_write_lat", "ib_send_lat"):
            self.assertEqual(node_probe._classify(record("/bin/" + name, [name])),
                             "network_benchmark")
        for name in ("lat", "code-server", "clickhouse-server", "language-server"):
            self.assertIsNone(node_probe._classify(record("/bin/" + name, [name])))
        self.assertEqual(node_probe._classify(record(
            "/opt/starfish/binaries/kvs_throughput-v6", ["kvs_throughput-v6"])), "workload")
        self.assertEqual(node_probe._classify(record("/opt/mg.D.x", ["mg.D.x"])), "workload")

    def test_missing_counters_are_unknown_and_titles_do_not_leak_arguments(self):
        self.assertIsNone(node_probe._cpu_percent(None, (10, 2)))
        record = node_probe._ProcRecord(
            pid=2, ppid=1, comm="daemon", argv=("daemon --password=secret",),
            exe="daemon --password=secret", user="tester", rss_bytes=None,
            utime_ticks=0, stime_ticks=0, starttime_ticks=1)
        view = node_probe._process_view(record, None, None, 100)
        self.assertEqual(view["exe"], "daemon")
        self.assertNotIn("secret", json.dumps(view))
        self.assertIsNone(view["cpu_percent"])
        self.assertIsNone(view["rss_bytes"])

    def test_zombies_and_ancestors_are_not_active_workloads(self):
        base = dict(ppid=1, comm="mg", argv=("mg",), exe="/bin/mg",
                    user="tester", rss_bytes=0, utime_ticks=0, stime_ticks=0,
                    starttime_ticks=1)
        records = [node_probe._ProcRecord(pid=2, state="Z", **base),
                   node_probe._ProcRecord(pid=3, state="S", **base)]
        matched, top = node_probe._collect_process_views(
            records, records, wall_seconds=1, uptime=10, hz=100, excluded={3})
        self.assertEqual((matched, top), ([], []))

    def test_idle_cpu_does_not_hide_a_high_memory_process(self):
        record = node_probe._ProcRecord(
            pid=10, ppid=1, comm="unknown-work", argv=("unknown-work",), exe="/bin/unknown-work",
            user="tester", rss_bytes=20 * 1024**3, utime_ticks=0, stime_ticks=0, starttime_ticks=1)
        memory = []
        matched, cpu = node_probe._collect_process_views(
            [record], [record], wall_seconds=1, uptime=10, hz=100, excluded=set(),
            memory_views=memory)
        self.assertEqual(matched, [])
        self.assertEqual(cpu, [])
        self.assertEqual(memory[0]["rss_bytes"], 20 * 1024**3)

    def test_resolved_clickhouse_symlink_does_not_exempt_client(self):
        for invocation, expected in (("clickhouse-server", True), ("clickhouse-client", False)):
            record = node_probe._ProcRecord(
                pid=10, ppid=1, comm="clickhouse", argv=(invocation,),
                exe="/usr/bin/clickhouse", user="clickhouse", rss_bytes=0,
                utime_ticks=0, stime_ticks=0, starttime_ticks=1)
            view = node_probe._process_view(record, 400, 10, 100)
            self.assertEqual(node_probe.idle_exempt_process(view), expected)

    def test_exempt_cpu_uses_host_ticks_and_preserves_nonexempt_leaders(self):
        def record(pid, name, user, ticks):
            return node_probe._ProcRecord(
                pid=pid, ppid=1, comm=name, argv=(name,), exe="/usr/bin/" + name,
                user=user, rss_bytes=(20 - pid) * 1024**3, utime_ticks=ticks,
                stime_ticks=0, starttime_ticks=1)
        # Several allowed daemons must not hide a fourth high-load process.
        before = [record(p, "clickhouse-server", "clickhouse", 0) for p in range(1, 5)]
        before.append(record(5, "training", "tester", 0))
        after = [record(p, "clickhouse-server", "clickhouse", 500) for p in range(1, 5)]
        after.append(record(5, "training", "tester", 200))
        memory, exempt = [], []
        matched, cpu = node_probe._collect_process_views(
            before, after, wall_seconds=1, uptime=10, hz=100, excluded=set(),
            memory_views=memory, idle_exempt_views=exempt, host_cpu_ticks=10000)
        self.assertEqual(matched, [])
        self.assertEqual(len(exempt), 4)
        self.assertEqual(exempt[0]["host_cpu_percent"], 5)
        self.assertEqual(exempt[0]["cpu_percent"], 500)
        self.assertIn(5, [p["pid"] for p in cpu])
        self.assertIn(5, [p["pid"] for p in memory])

    def test_meminfo_falls_back_when_memavailable_is_missing(self):
        state = node_probe._new_state()
        text = (
            "MemTotal:       100 kB\n"
            "MemFree:         20 kB\n"
            "Buffers:         10 kB\n"
            "Cached:          30 kB\n"
            "SReclaimable:     5 kB\n"
            "Shmem:            2 kB\n"
            "HugePages_Total:  8\n"
            "HugePages_Free:   3\n"
        )
        with patch.object(node_probe, "_read_bytes", return_value=text.encode()):
            memory = node_probe._read_meminfo(state)
        self.assertEqual(memory["total_bytes"], 100 * 1024)
        self.assertEqual(memory["available_bytes"], 20 * 1024)
        self.assertEqual(memory["hugepages_total"], 8)
        self.assertEqual(memory["hugepages_free"], 3)
        self.assertTrue(any("MemAvailable missing" in warning for warning in state["warnings"]))

    def test_ss_parser_handles_ipv4_and_ipv6_without_owner_claims(self):
        result = subprocess.CompletedProcess(
            ["ss"], 0,
            stdout=(
                "LISTEN 0 128 0.0.0.0:1893 0.0.0.0:*\n"
                "LISTEN 0 128 [::]:1900 [::]:*\n"
                "LISTEN 0 128 127.0.0.1:1893 0.0.0.0:*\n"
            ),
            stderr="",
        )
        state = node_probe._new_state()
        with patch.object(node_probe.subprocess, "run", return_value=result):
            self.assertEqual(node_probe._read_listen_ports(state), [1893, 1900])
        self.assertEqual(state["warnings"], [])

    def test_snapshot_shape_is_json_safe_and_excludes_probe_ancestors(self):
        empty = []
        with patch.object(node_probe, "_read_cpu_counters", side_effect=[(100, 50), (200, 100)]), \
             patch.object(node_probe, "_scan_processes", side_effect=[empty, empty]), \
             patch.object(node_probe, "_ancestor_pids", return_value={1, 2}), \
             patch.object(node_probe.time, "sleep"), \
             patch.object(node_probe.time, "monotonic", side_effect=[0.0, 1.0]), \
             patch.object(node_probe, "_read_uptime", return_value=100.0), \
             patch.object(node_probe, "_read_bytes", return_value=b"MemTotal: 1 kB\nMemAvailable: 1 kB\n"), \
             patch.object(node_probe.subprocess, "run", return_value=subprocess.CompletedProcess(["ss"], 0, stdout="", stderr="")):
            snapshot = node_probe.collect_snapshot(sample_seconds=1)
        json.dumps(snapshot)
        self.assertEqual(snapshot["cpu_percent"], 50.0)
        self.assertEqual(snapshot["processes"], [])
        self.assertEqual(snapshot["top_processes"], [])
        self.assertIn("visibility_complete", snapshot)


if __name__ == "__main__":
    unittest.main()
