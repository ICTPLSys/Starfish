"""Focused offline coverage of the three-case fast-check contract."""

import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fast_check
from render_config import render
from workloads import FOOTPRINT_BYTES


class FastCheck(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / "site.json"
        self.original = json.loads(
            (fast_check.ROOT / "scripts/common/site.example.json").read_text())
        self.original["memory_host"] = "test-memory-host"
        self.site.write_text(json.dumps(self.original))

    def test_one_host_expands_to_seven_unique_service_ports(self):
        sites = fast_check.case_sites(self.site)
        self.assertEqual(len(sites["nonft"]["memory_endpoints"]), 1)
        ec = sites["starfish"]["memory_endpoints"]
        self.assertEqual([x["server_port"] for x in ec], list(range(1893, 1900)))
        self.assertEqual({x["memory_host"] for x in ec}, {"test-memory-host"})
        self.assertEqual(json.loads(self.site.read_text()), self.original)

    def test_port_overflow_and_incomplete_topology_rejected(self):
        self.original["server_port"] = 65532
        self.site.write_text(json.dumps(self.original))
        with self.assertRaises(ValueError):
            fast_check.case_sites(self.site)
        self.original["server_port"] = 1893
        self.site.write_text(json.dumps(self.original))
        normalized = fast_check.case_sites(self.site)["starfish"]
        normalized["memory_endpoints"] = normalized["memory_endpoints"][:6]
        self.site.write_text(json.dumps(normalized))
        with self.assertRaises(ValueError):
            fast_check.case_sites(self.site)

    def test_explicit_endpoints_preserved(self):
        normalized = fast_check.case_sites(self.site)["starfish"]
        for i, endpoint in enumerate(normalized["memory_endpoints"]):
            endpoint["memory_addr"] = f"192.0.2.{20 + i}"
        self.site.write_text(json.dumps(normalized))
        sites = fast_check.case_sites(self.site)
        self.assertEqual(sites["starfish"]["memory_endpoints"],
                         normalized["memory_endpoints"])

    def test_same_starfish_recipe_and_full_workload(self):
        args = [fast_check.case_args(name, system, recovery, self.site,
                                    self.root / "out", 1800)
                for name, system, recovery in fast_check.CASES]
        self.assertEqual(len(args), 3)
        self.assertEqual([a.ratio for a in args], [25, 25, 25])
        self.assertEqual([a.recover_endpoint for a in args], [None, None, 0])
        self.assertEqual(args[1].recipe, args[2].recipe)
        self.assertTrue(all(a.capture_chat for a in args))
        rendered = render(
            args[1].recipe, system="starfish", ratio=25,
            footprint_bytes=FOOTPRINT_BYTES["llama"], ib_device="mlx5_1",
            server_endpoints=[{"server_addr": "192.0.2.20", "server_port": 1893 + i}
                              for i in range(7)])
        self.assertIn("ft_method ec_batch", rendered)
        self.assertIn("ft_standby_endpoint 6", rendered)
        self.assertIn("server_count 7", rendered)
        self.assertIn("client_buffer_size 6738677767", rendered)

    def test_answers_require_complete_three_way_match(self):
        for name, _, _ in fast_check.CASES:
            directory = self.root / "runs" / name
            directory.mkdir(parents=True)
            (directory / "chat-output.txt").write_bytes(b"Hello!")
        self.assertTrue(fast_check.verify_answers(self.root)["matching_chat_outputs"])
        (self.root / "runs" / fast_check.CASES[2][0] /
         "chat-output.txt").write_bytes(b"Hello")
        with self.assertRaises(ValueError):
            fast_check.verify_answers(self.root)

    def test_sequence_no_repeats_and_no_plot_publication(self):
        out = self.root / "out"
        calls = []
        def run(args):
            calls.append(copy.deepcopy(args))
            if not args.check_local:
                args.out.mkdir(parents=True)
                (args.out / "analysis.json").write_text(json.dumps(
                    {"status": "passed", "elapsed_s": 1,
                     "recovery": {"status": "passed",
                                  "injection": {"process_stopped": True}}}))
                (args.out / "chat-output.txt").write_bytes(b"Hello!")
            return 0
        with patch.object(fast_check.run_case, "run", side_effect=run), \
                patch.object(fast_check.subprocess, "run") as build:
            self.assertEqual(fast_check.execute(argparse.Namespace(
                site=self.site, out=out, timeout=1800, dry_run=False)), 0)
        self.assertEqual(build.call_count, 2)
        actual = [a for a in calls if not a.check_local]
        self.assertEqual([a.out.name for a in actual],
                         [name for name, _, _ in fast_check.CASES])
        self.assertEqual(json.loads((out / "summary.json").read_text())["status"],
                         "passed")

    def test_real_run_case_plan_uses_recipe_capture_and_injection(self):
        sites = fast_check.case_sites(self.site)
        for name, system, recovery in fast_check.CASES:
            site = self.root / f"{system}.json"
            site.write_text(json.dumps(sites[system]))
            args = fast_check.case_args(name, system, recovery, site,
                                        self.root / "out", 1800, dry_run=True)
            with patch.object(fast_check.run_case, "ssh") as ssh, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(fast_check.run_case.run(args), 0)
            ssh.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["ratio"], 25)
            self.assertEqual(plan["failure_injection_endpoint"], recovery)
            self.assertTrue(plan["client_env"]["FARLIB_CAPTURE_CHAT_OUTPUT"].endswith(
                name + "/chat-output.txt"))
            self.assertNotIn("-n", plan["client_command"])
            if system == "starfish":
                self.assertTrue(plan["recipe"].endswith("starfish_ec.config"))
                self.assertEqual(len(plan["memory_endpoints"]), 7)
                self.assertEqual(plan["client_env"]["FARLIB_EC_RECOVERY_VERIFY"], "1")
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
