"""Focused regressions for EC, colocated endpoints and binary deployment."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_case
import server_staging
import site_defaults

AE = run_case.AE_ROOT


class Figure9Launch(unittest.TestCase):
    def plan(self, root, raw, app="llama", system="starfish", out=None,
             build_root=None):
        site = root / "site.json"
        site.write_text(json.dumps(raw))
        args = argparse.Namespace(app=app, system=system, ratio=25, site=site,
                                  out=root / "case" if out is None else out,
                                  timeout=1800, dry_run=True,
                                  build_root=build_root)
        with patch.object(run_case, "ssh") as ssh, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run_case.run(args), 0)
        ssh.assert_not_called()
        self.assertFalse(args.out.exists())
        return json.loads(output.getvalue())

    def test_different_clone_roots_have_different_remote_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            plans = []
            for name in ("clone-a", "clone-b"):
                root = Path(tmp) / name
                root.mkdir()
                plans.append(self.plan(root, {"memory_ip": "10.208.130.76",
                                              "memory_server_count": 6}))
            namespaces = {
                plan["remote_namespace"] for plan in plans
            }
            self.assertEqual(len(namespaces), 2)
            self.assertTrue(all(plan["remote_namespace_source"].endswith(
                ("clone-a", "clone-b")[index])
                for index, plan in enumerate(plans)))

    def test_relative_out_and_build_root_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = Path.cwd()
            try:
                os.chdir(root)
                plan = self.plan(root, {"memory_ip": "10.208.130.76",
                                        "memory_server_count": 6},
                                 out=Path("case"), build_root=Path("build"))
            finally:
                os.chdir(old)
            self.assertTrue(Path(plan["client_bin"]).is_absolute())

    def test_memory_sampling_plan_uses_work_hooks_and_bfs_single_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            for system in ("nonft", "starfish", "hydra", "carbink"):
                for app in ("llama", "bfs", "mg", "wordcount", "kv-b", "kv-a", "kv-s", "nq"):
                    with self.subTest(system=system, app=app):
                        plan = self.plan(Path(tmp), {
                            "memory_ip": "10.208.130.76",
                            "memory_server_count": 6,
                            "compute_ip": "10.208.130.56",
                            "remote_memory_samples": True,
                            "remote_memory_observer_cpu": 47,
                        }, app=app, system=system)
                        sampling = plan["remote_memory_sampling"]
                        self.assertEqual(
                            sampling["time_origin"],
                            "kvs_request_start" if app in ("kv-b", "kv-a", "kv-s")
                            else "profile_start_work")
                        self.assertEqual(
                            sampling["stop_at"],
                            "kvs_request_drain_end" if app in ("kv-b", "kv-a", "kv-s")
                            else "profile_end_work")
                        self.assertEqual(sampling["schema_version"], 3)
                        self.assertEqual(sampling["planned_points_s"],
                                         [3] if app == "bfs" else [10, 20, 30, 40, 50])
                        self.assertNotIn("FARLIB_BENCHMARK_START_MONOTONIC_NS", plan["client_env"])

    def test_colocated_six_and_seven_endpoint_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            for count in (6, 7):
                for app in ("llama", "bfs"):
                    plan = self.plan(Path(tmp), {"memory_ip": "10.208.130.76",
                        "memory_server_count": count, "server_port": 1893}, app)
                    endpoints = plan["memory_endpoints"]
                    self.assertEqual(len(endpoints), count)
                    self.assertEqual({x["memory_host"] for x in endpoints}, {"10.208.130.76"})
                    self.assertEqual([x["server_port"] for x in endpoints], list(range(1893, 1893+count)))
                    self.assertEqual(len({x["remote_run_dir"] for x in endpoints}), count)
                    self.assertIn("ft_method ec_batch", plan["effective_config"])
                    standby = "6" if count == 7 else "-1"
                    self.assertIn("ft_standby_endpoint " + standby, plan["effective_config"])
                    for item in endpoints:
                        self.assertTrue(item["stage_memory_server"])
                        self.assertTrue(item["server_bin"].endswith("/server"))
                        self.assertIn("env -i", item["start_command"])
                        subprocess.run(["bash", "-n"], input=item["start_command"],
                                       text=True, check=True)

    def test_bad_ec_count_and_ports_rejected_without_ssh(self):
        with tempfile.TemporaryDirectory() as tmp:
            for raw in ({"memory_ip": "10.208.130.76"},
                        {"memory_ip": "10.208.130.76", "memory_server_count": 7,
                         "server_port": 65534}):
                with self.assertRaises(ValueError):
                    self.plan(Path(tmp), raw)

    def test_explicit_binary_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp), {"memory_ip": "10.208.130.76",
                "memory_server_count": 6,
                "memory_server_bins": {"starfish": "/opt/ae/server"}})
            for item in plan["memory_endpoints"]:
                self.assertFalse(item["stage_memory_server"])
                self.assertEqual(item["server_bin"], "/opt/ae/server")

    def test_normalized_site_retains_staging_choice(self):
        raw = {"memory_ip": "10.208.130.76"}
        once = site_defaults.resolve(raw, AE)
        twice = site_defaults.resolve(once, AE)
        self.assertTrue(twice["stage_memory_server"])

    def test_real_local_stage_and_refuse_overwrite(self):
        # Exercises file copy, ELF dependency check and executable permission;
        # no SSH, RDMA, or benchmark processes.
        with tempfile.TemporaryDirectory() as tmp:
            def ssh(host, script):
                return subprocess.run(["bash", "-ec", script], text=True,
                                      capture_output=True, check=True)
            def scp(source, target):
                shutil.copy2(source, target.split(":", 1)[1])
            source = Path("/usr/bin/true")
            private = server_staging.dependencies(source)
            target = server_staging.stage("local", tmp, source, private, ssh=ssh, scp=scp)
            subprocess.run([target["binary"]], check=True)
            with self.assertRaises(subprocess.CalledProcessError):
                server_staging.stage("local", tmp, source, private, ssh=ssh, scp=scp)

    def test_missing_library_is_a_preflight_error(self):
        result = subprocess.CompletedProcess([], 0, "libfibre.so => not found", "")
        with patch.object(server_staging.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "unresolved"):
                server_staging.dependencies(Path("/missing/server"))

    def test_colocated_capacity_is_counted_together(self):
        states = [{"memory_host": "memory", "memory_numa_node": 0} for _ in range(7)]
        memory = subprocess.CompletedProcess([], 0, "Node 0 MemFree: 81920 kB", "")
        with patch.object(run_case, "ssh", return_value=memory) as ssh:
            with self.assertRaisesRegex(ValueError, "together"):
                run_case.check_memory_capacity(states, 12 * 1024 * 1024)
            ssh.assert_called_once()


if __name__ == "__main__":
    unittest.main()
