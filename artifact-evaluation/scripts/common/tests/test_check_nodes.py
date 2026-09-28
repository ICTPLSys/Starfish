"""Read-only controller tests; SSH is mocked and no hosts are contacted."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_nodes


def snapshot(processes=None, complete=True):
    return {"hostname": "test-node", "processes": processes or [],
            "visibility_complete": complete, "cpu_percent": 2.5,
            "memory": {"total_bytes": 8 * 1024**3, "available_bytes": 4 * 1024**3},
            "tcp_listen_ports": [22, 1893], "warnings": [], "top_processes": [],
            "top_memory_processes": []}


class CheckNodes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = self.root / "machines.json"
        self.inventory.write_text(json.dumps({"servers": [
            {"ip": "192.0.2.1", "hostname": "one"},
            {"ip": "192.0.2.2", "hostname": "two"}]}))
        self.site = self.root / "site.json"
        self.site.write_text(json.dumps({"memory_ip": "192.0.2.2",
                                         "memory_host": "reviewer@192.0.2.2"}))

    def test_existing_eight_node_inventory_and_site_accounts(self):
        actual = check_nodes.load_targets(check_nodes.ROOT / "configs/machines.json", None)
        self.assertEqual([r["ip"].split(".")[-1] for r in actual],
                         ["22", "54", "56", "58", "74", "76", "82", "84"])
        targets = check_nodes.load_targets(self.inventory, self.site)
        self.assertEqual([r["ssh_host"] for r in targets],
                         ["reviewer@192.0.2.1", "reviewer@192.0.2.2"])
        override = check_nodes.load_targets(self.inventory, self.site, user="another")
        self.assertEqual(override[0]["ssh_host"], "another@192.0.2.1")

    def test_heterogeneous_accounts_and_inventory_override(self):
        self.site.write_text(json.dumps({"memory_endpoints": [
            {"ip": "192.0.2.1", "memory_host": "alice@node-one"},
            {"ip": "192.0.2.2", "memory_host": "bob@node-two"}]}))
        self.assertEqual([r["ssh_host"] for r in check_nodes.load_targets(self.inventory, self.site)],
                         ["alice@node-one", "bob@node-two"])
        self.inventory.write_text(json.dumps({"servers": [
            {"ip": "192.0.2.1", "ssh_host": "custom@alias"}]}))
        self.assertEqual(check_nodes.load_targets(self.inventory, self.site)[0]["ssh_host"],
                         "custom@alias")

    def test_bad_destinations_and_duplicate_nodes_are_rejected_before_ssh(self):
        for target in ("-oProxyCommand=bad", "alice@host;cmd", "alice@host name", "a@b@c"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                check_nodes.ssh_target(target)
        self.inventory.write_text('{"servers":[{"ip":"192.0.2.1"},{"ip":"192.0.2.1"}]}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_nodes.load_targets(self.inventory, None)

    def probe(self, result, local=False):
        target = {"ip": "192.0.2.1", "ssh_host": "reviewer@node-one", "hostname": "one"}
        completed = mock.Mock(returncode=0, stdout=json.dumps(result), stderr="")
        with mock.patch.object(check_nodes.subprocess, "run", return_value=completed) as call:
            row = check_nodes.inspect_node(target, "READ_ONLY_PROBE_SOURCE", {"192.0.2.1"} if local else set())
        return row, call.call_args

    def test_status_and_noninteractive_readonly_ssh_contract(self):
        for processes, complete, status in (
            ([{"pid": 1, "kind": "workload"}], True, "EXPERIMENT"),
            ([{"pid": 2, "kind": "candidate_server"}], True, "REVIEW"),
            ([], True, "NO_MATCH"), ([], False, "PARTIAL")):
            row, call = self.probe(snapshot(processes, complete))
            self.assertEqual(row["status"], status)
            command = call.args[0]
            self.assertEqual(command[0], "ssh")
            for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "UpdateHostKeys=no"):
                self.assertIn(option, command)
            self.assertNotIn("sudo", command)
            self.assertEqual(command[-1], "python3 -B -")
            self.assertEqual(call.kwargs["input"], "READ_ONLY_PROBE_SOURCE")
            self.assertEqual(call.kwargs["timeout"], 15)
        row, call = self.probe(snapshot(), local=True)
        self.assertNotIn("ssh", call.args[0])
        self.assertEqual(row["transport"], "local")

    def test_failures_are_not_reported_as_idle(self):
        target = {"ip": "192.0.2.1", "ssh_host": "reviewer@node-one", "hostname": "one"}
        failed = mock.Mock(returncode=255, stdout="", stderr="Permission denied")
        with mock.patch.object(check_nodes.subprocess, "run", return_value=failed):
            self.assertEqual(check_nodes.inspect_node(target, "", set())["status"], "UNREACHABLE")
        with mock.patch.object(check_nodes.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ssh", 15)):
            self.assertEqual(check_nodes.inspect_node(target, "", set())["status"], "UNKNOWN")

    def test_terminal_output_has_no_control_codes_or_free_claim(self):
        row, _ = self.probe(snapshot())
        row["snapshot"]["hostname"] = "host\x1b[31m\nspoof"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            check_nodes.display({"timestamp_utc": "test", "nodes": [row]})
        self.assertNotIn("\x1b", out.getvalue())
        self.assertIn("1/1 NODES IDLE", out.getvalue())
        self.assertIn("✓ YES", out.getvalue())
        self.assertIn("Snapshot only", out.getvalue())
        self.assertNotIn("TCP LISTEN", out.getvalue())
        self.assertEqual(out.getvalue().count("192.0.2.1"), 1)
        details = io.StringIO()
        with contextlib.redirect_stdout(details):
            check_nodes.display_verbose({"timestamp_utc": "test", "nodes": [row]})
        self.assertIn("TCP LISTEN", details.getvalue())

    def test_high_load_and_runtime_are_independent_idle_checks(self):
        row = {"snapshot": snapshot(), "status": "NO_MATCH"}
        self.assertTrue(check_nodes.availability(row)["idle"])
        for field, proc in (
                ("top_processes", {"pid": 1, "name": "training", "cpu_percent": 250}),
                ("top_memory_processes", {"pid": 2, "name": "data", "rss_bytes": 12 * 1024**3})):
            row = {"snapshot": snapshot(), "status": "NO_MATCH"}
            row["snapshot"][field] = [proc]
            self.assertFalse(check_nodes.availability(row)["idle"])
        row = {"snapshot": snapshot([{"pid": 3, "kind": "memory_service", "name": "server"}])}
        self.assertFalse(check_nodes.availability(row)["idle"])
        self.assertEqual(check_nodes.availability(row)["runtime"], "memory")
        row = {"snapshot": snapshot([{"pid": 4, "kind": "workload", "name": "kvs_throughput-v6"}])}
        self.assertEqual(check_nodes.availability(row)["runtime"], "KV")

    def test_permission_failures_and_candidates_are_never_yes(self):
        for row in (
                {"status": "UNREACHABLE", "error": "Permission denied"},
                {"snapshot": snapshot(complete=False)},
                {"snapshot": snapshot([{"pid": 1, "kind": "candidate_server"}])}):
            self.assertIsNone(check_nodes.availability(row)["idle"])

    def test_host_pressure_prevents_idle_even_without_known_process(self):
        row = {"snapshot": snapshot()}
        row["snapshot"]["cpu_percent"] = 15
        self.assertFalse(check_nodes.availability(row)["idle"])
        row["snapshot"]["cpu_percent"] = 1
        row["snapshot"]["memory"]["available_bytes"] = 1024**3
        self.assertFalse(check_nodes.availability(row)["idle"])

    def test_clickhouse_exemption_keeps_raw_metrics_and_other_load_checks(self):
        ch = {"pid": 10, "name": "clickhouse-server", "exe": "/usr/bin/clickhouse-server",
              "user": "clickhouse", "kind": "other", "cpu_percent": 1200,
              "host_cpu_percent": 12.5, "rss_bytes": 12 * 1024**3}
        row = {"snapshot": snapshot(), "hostname": "test", "ip": "192.0.2.1",
               "status": "NO_MATCH"}
        snap = row["snapshot"]
        snap.update(cpu_percent=14, top_processes=[ch], top_memory_processes=[ch],
                    idle_exempt_processes=[ch])
        state = check_nodes.availability(row)
        self.assertIs(state["idle"], True)
        self.assertEqual(state["effective_host_cpu_percent"], 1.5)
        self.assertEqual(snap["cpu_percent"], 14)
        self.assertEqual(snap["top_processes"][0]["cpu_percent"], 1200)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            check_nodes.display_verbose({"timestamp_utc": "test", "nodes": [row]})
        self.assertIn("Excluded from idle load", out.getvalue())
        for field, extra in (
                ("top_processes", {"pid": 11, "cpu_percent": 200}),
                ("top_memory_processes", {"pid": 11, "rss_bytes": 9 * 1024**3}),
                ("processes", {"pid": 11, "kind": "workload", "name": "mg"})):
            old = snap[field]
            snap[field] = old + [extra]
            self.assertIs(check_nodes.availability(row)["idle"], False)
            snap[field] = old
        snap["cpu_percent"] = 25
        self.assertIs(check_nodes.availability(row)["idle"], False)
        snap["cpu_percent"] = 14
        snap["visibility_complete"] = False
        self.assertIsNone(check_nodes.availability(row)["idle"])
        snap["visibility_complete"] = True
        ch["host_cpu_percent"] = None
        self.assertIsNone(check_nodes.availability(row)["idle"])
        ch["host_cpu_percent"] = 12.5
        snap["memory"]["available_bytes"] = 1024**3
        self.assertIs(check_nodes.availability(row)["idle"], False)

    def test_exemption_does_not_match_other_clickhouse_processes(self):
        for name, user in (("clickhouse-client", "clickhouse"),
                           ("clickhouse-server", "tester"), ("training", "clickhouse")):
            p = {"pid": 12, "name": name, "user": user, "exe": "/bin/" + name,
                 "cpu_percent": 400}
            snap = snapshot()
            snap["top_processes"] = [p]
            self.assertIs(check_nodes.availability({"snapshot": snap})["idle"], False)

    def test_terminal_highlight_respects_no_color(self):
        with mock.patch.object(check_nodes.sys.stdout, "isatty", return_value=True):
            with mock.patch.dict(check_nodes.os.environ, {}, clear=True):
                self.assertEqual(check_nodes.colorize("YES", 32, bold=True),
                                 "\x1b[1;32mYES\x1b[0m")
            with mock.patch.dict(check_nodes.os.environ, {"NO_COLOR": "1"}, clear=True):
                self.assertEqual(check_nodes.colorize("YES", 32, bold=True), "YES")


if __name__ == "__main__":
    unittest.main()
