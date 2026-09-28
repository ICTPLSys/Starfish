"""Offline checks for path defaults, machine mappings and NUMA launch plans."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fast_check
import run_case
import site_defaults
import topology
from render_config import render

AE = run_case.AE_ROOT


class SiteDefaults(unittest.TestCase):
    def test_compute56_approved_placement_defaults_and_override(self):
        site = site_defaults.resolve({"compute_ip": "10.208.130.56",
                                      "memory_ip": "10.208.130.76"}, AE)
        self.assertEqual(site["fibre_workers"], 24)
        self.assertEqual(site["fibre_cpu_set"], "0-17,24-29")
        self.assertEqual(site["background_cpu_base"], 18)
        self.assertIs(site["allow_cross_numa_app"], True)
        custom = site_defaults.resolve({"compute_ip": "10.208.130.56",
                                        "memory_ip": "10.208.130.76",
                                        "fibre_cpu_set": "0-15,24-31",
                                        "background_cpu_base": 16}, AE)
        self.assertEqual(custom["fibre_cpu_set"], "0-15,24-31")
        self.assertEqual(custom["background_cpu_base"], 16)
        other = site_defaults.resolve({"compute_ip": "10.208.130.58",
                                       "memory_ip": "10.208.130.74"}, AE)
        self.assertNotIn("allow_cross_numa_app", other)

    def test_plain_ip_suffices_and_paths_are_automatic(self):
        with patch.dict(os.environ, {}, clear=True):
            site = site_defaults.resolve({"memory_ip": "10.208.130.76"}, AE)
        self.assertEqual(site["memory_host"], "10.208.130.76")
        self.assertEqual(site["memory_addr"], "10.208.130.76")
        self.assertEqual(site["server_port"], 1893)
        self.assertEqual(site["inputs"]["llama"],
                         "/data/starfish-ae/llama/llama2_7b_chat.bin")
        self.assertEqual(site["memory_server_bins"]["starfish"],
                         str(AE / "build/starfish/server"))

    def test_known_devices_and_numa_are_not_uniform(self):
        site = run_case.site_config(
            AE / "scripts/common/site.eight-server.example.json", dry_run=True)
        self.assertEqual(site["compute_ip"], "10.208.130.56")
        expected = {"22": ("mlx5_1", 1), "54": ("mlx5_0", 0),
                    "58": ("mlx5_1", 0), "74": ("mlx5_1", 0),
                    "76": ("mlx5_1", 0), "82": ("mlx5_0", 1), "84": ("mlx5_0", 0)}
        for item in site["memory_endpoints"]:
            self.assertEqual((item["memory_ib_device"], item["memory_numa_node"]),
                             expected[item["memory_addr"].split(".")[-1]])
            self.assertEqual(item["server_port"], 1893)
        compute22 = site_defaults.resolve({"compute_ip": "10.208.130.22",
                                           "memory_ip": "10.208.130.54"}, AE)
        self.assertEqual((compute22["ib_device"], compute22["numa_node"]), ("mlx5_1", 1))

    def test_custom_paths_devices_and_ports_take_precedence(self):
        raw = {"memory_ip": "10.208.130.76", "memory_project_root": "/opt/Starfish",
               "data_dir": "/datasets", "inputs": {"llama": "/models/custom.bin"},
               "memory_ib_device": "mlx5_7", "memory_numa_node": 2, "server_port": 2001,
               "memory_server_bins": {"nonft": "/custom/nonft-server"}}
        site = site_defaults.resolve(raw, AE)
        self.assertEqual(site["inputs"]["llama"], "/models/custom.bin")
        self.assertEqual(site["inputs"]["bfs"], "/datasets/bfs/graph")
        self.assertEqual(site["memory_server_bins"]["starfish"],
                         "/opt/Starfish/artifact-evaluation/build/starfish/server")
        self.assertEqual(site["memory_server_bins"]["nonft"], "/custom/nonft-server")
        self.assertEqual(site["memory_ib_device"], "mlx5_7")
        self.assertEqual(site["memory_numa_node"], 2)
        self.assertEqual(site["server_port"], 2001)
        self.assertNotIn("name", raw)

    def test_endpoint_overrides_and_user_at_ip(self):
        site = site_defaults.resolve({
            "memory_endpoints": [{"ip": "10.208.130.54", "memory_host": "reviewer@10.208.130.54",
                                  "memory_project_root": "/srv/project", "server_port": 2345,
                                  "memory_numa_node": 1}]}, AE)
        item = site["memory_endpoints"][0]
        self.assertEqual(item["memory_host"], "reviewer@10.208.130.54")
        self.assertEqual(item["memory_addr"], "10.208.130.54")
        self.assertEqual(item["memory_ib_device"], "mlx5_0")
        self.assertEqual(item["memory_numa_node"], 1)
        self.assertEqual(item["memory_server_bins"]["starfish"],
                         "/srv/project/artifact-evaluation/build/starfish/server")
        with self.assertRaises(ValueError):
            site_defaults.resolve({"memory_host": "unresolved-ssh-alias"}, AE)

    def test_partial_input_override_and_environment_default(self):
        with patch.dict(os.environ, {"AE_PREPARED_DATA_DIR": "/shared/data"}):
            site = site_defaults.resolve({"memory_ip": "192.0.2.1",
                                           "inputs": {"bfs": "/my/graph"}}, AE)
        self.assertEqual(site["inputs"]["bfs"], "/my/graph")
        self.assertEqual(site["inputs"]["wordcount"], "/shared/data/wordcount/enwiki-small-8g.txt")

    def test_binding_and_read_only_device_checks(self):
        self.assertEqual(topology.memory_binding(1), ["numactl", "--membind=1"])
        self.assertEqual(topology.memory_binding(0, server=True),
                         ["numactl", "--membind=0", "--cpunodebind=0"])
        script = topology.preflight_script("mlx5_0", 1)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        self.assertIn("link_layer", script)
        self.assertIn("InfiniBand", script)
        self.assertIn("device/numa_node", script)
        self.assertIn("$1 >= 100", script)
        self.assertIn("ports/2/state", topology.preflight_script("mlx5_0", 1, 2))
        with self.assertRaises(ValueError):
            topology.preflight_script("mlx5_0; touch /tmp/bad", 0)
        with self.assertRaises(ValueError):
            topology.memory_binding(-1)
        with patch.object(topology.subprocess, "check_output", return_value="[]"):
            with self.assertRaisesRegex(ValueError, "not local"):
                topology.check_compute({"compute_ip": "10.208.130.56"})

    def test_fast_check_keeps_seven_distinct_hosts_and_binding(self):
        path = AE / "scripts/common/site.eight-server.example.json"
        sites = fast_check.case_sites(path, dry_run=True)
        with tempfile.TemporaryDirectory() as tmp:
            site_path = Path(tmp) / "site.json"
            site_path.write_text(json.dumps(sites["starfish"]))
            args = fast_check.case_args("llama-starfish-recovery-25", "starfish", 0,
                                        site_path, Path(tmp) / "out", 1800, dry_run=True)
            with patch.object(run_case, "ssh") as ssh, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(run_case.run(args), 0)
            ssh.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertEqual(len({item["memory_host"] for item in plan["memory_endpoints"]}), 7)
            self.assertEqual(plan["client_command"][:2], ["numactl", "--membind=0"])
            self.assertEqual(plan["memory_endpoints"][0]["memory_numa_node"], 1)
            self.assertEqual(plan["memory_endpoints"][5]["memory_ib_device"], "mlx5_0")
            self.assertFalse(args.out.exists())

    def test_ib_port_override_reaches_the_generated_runtime_config(self):
        site = site_defaults.resolve({"memory_ip": "10.208.130.76",
                                      "ib_port": 2, "memory_ib_port": 2}, AE)
        self.assertEqual(site["ib_port"], 2)
        self.assertEqual(site["memory_ib_port"], 2)
        config = render(AE / "configs/llama/nonft.config", system="nonft",
                        ratio=25, footprint_bytes=1000000,
                        server_addr="192.0.2.20", server_port=1893,
                        ib_device="mlx5_1", ib_port=2)
        self.assertIn("ib_port 2", config)
        self.assertIn("ib_port 2", run_case.server_config(config))
        with self.assertRaises(ValueError):
            site_defaults.resolve({"memory_ip": "192.0.2.20", "ib_port": 0}, AE)


if __name__ == "__main__":
    unittest.main()
