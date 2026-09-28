"""Offline checks for multi-endpoint rendering and runner ownership boundaries."""

from pathlib import Path
import sys
import unittest
from unittest import mock
import subprocess
import tempfile

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))

import endpoint
import run_case
from render_config import render

AE = COMMON.parents[1]


class MultiEndpoint(unittest.TestCase):
    def test_endpoint_validation_allows_same_address_different_ports(self):
        records = endpoint.validate_render_endpoints([
            {"server_addr": "192.0.2.20", "server_port": 1400},
            {"server_addr": "192.0.2.20", "server_port": 1401},
        ])
        self.assertEqual(records[1]["server_port"], 1401)
        with self.assertRaisesRegex(ValueError, "duplicate address/port"):
            endpoint.validate_render_endpoints([
                {"server_addr": "192.0.2.20", "server_port": 1400},
                {"server_addr": "192.0.2.20", "server_port": 1400},
            ])

    def test_ipv6_canonicalization_and_ssh_option_guard(self):
        records = endpoint.validate_render_endpoints([
            {"server_addr": "2001:0db8::1", "server_port": 1400},
        ])
        self.assertEqual(records[0]["server_addr"], "2001:db8::1")
        with self.assertRaises(ValueError):
            endpoint.validate_site_endpoints([{
                "memory_host": "-oProxyCommand=bad",
                "memory_addr": "192.0.2.20",
                "server_port": 1400,
                "memory_server_bins": {"starfish": "/opt/server"},
            }], system="starfish", dry_run=True)

    def test_legacy_site_is_one_endpoint(self):
        site = run_case.site_config(
            AE / "scripts" / "common" / "site.example.json",
            dry_run=True, system="nonft")
        self.assertTrue(site["legacy_memory_endpoint"])
        self.assertEqual(len(site["memory_endpoints"]), 1)

    def test_multi_site_is_normalized_without_remote_access(self):
        site = run_case.site_config(
            AE / "scripts" / "common" / "site.multi-endpoint.example.json",
            dry_run=True, system="starfish")
        self.assertFalse(site["legacy_memory_endpoint"])
        self.assertEqual(len(site["memory_endpoints"]), 6)
        specs = run_case.endpoint_specs(site, "starfish", dry_run=True)
        self.assertEqual([x["server_port"] for x in specs], list(range(1400, 1406)))

    def test_client_and_per_server_config_shapes(self):
        template = AE / "configs" / "wordcount" / "recomputable_ec.config"
        endpoints = [{"server_addr": "192.0.2.20", "server_port": 1400 + i}
                     for i in range(6)]
        client = render(template, system="starfish", ratio=25,
                        footprint_bytes=10_000_000, ib_device="mlx5_0",
                        server_endpoints=endpoints)
        self.assertIn("server_count 6", client)
        self.assertIn("server_ports 1400,1401,1402,1403,1404,1405", client)
        one = render(template, system="starfish", ratio=25,
                     footprint_bytes=10_000_000, ib_device="mlx5_0",
                     server_endpoints=[endpoints[2]])
        self.assertIn("server_count 1", one)
        self.assertIn("server_port 1402", one)
        self.assertIn("ft_method ec_batch", one)

    def test_hydra_keeps_ec_method_and_does_not_add_starfish_backup(self):
        template = AE / "configs" / "wordcount" / "hydra.config"
        endpoints = [{"server_addr": "192.0.2.20", "server_port": 1400 + i}
                     for i in range(6)]
        config = render(template, system="hydra", ratio=25,
                        footprint_bytes=10_000_000, ib_device="mlx5_0",
                        server_endpoints=endpoints)
        self.assertIn("ft_method hydra", config)
        self.assertNotIn("remote_backup_budget_bytes 1000000", config)

    def test_legacy_single_api_rejects_multi_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "external.config"
            template.write_text("server_count 2\nserver_addrs 192.0.2.20,192.0.2.21\n"
                                "server_ports 1400,1400\n")
            with self.assertRaises(ValueError):
                render(template, system="starfish", ratio=25,
                       footprint_bytes=10_000_000, server_addr="192.0.2.20",
                       server_port=1998, ib_device="mlx5_0")

    def test_identity_checks_include_starttime_cwd_and_exact_argv(self):
        script = run_case.process_identity_script(
            1234, "/opt/server", "/tmp/run/endpoint-00/server.config", "777")
        self.assertIn('start=${20}', script)
        self.assertIn('fields=${snapshot##*) }', script)
        self.assertIn("/proc/$pid/cwd", script)
        self.assertIn("argv_real", script)
        self.assertIn("server.config", script)
        checked = subprocess.run(["bash", "-n"], input=script,
                                 text=True, capture_output=True, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_cleanup_reports_already_exited_without_signaling(self):
        state = {
            "pid": 1234, "starttime": "777", "server_bin": "/opt/server",
            "remote_dir": "/tmp/run/endpoint-00",
        }
        calls = []

        def fake_ssh(host, script, check=True, timeout=30):
            calls.append((host, script, check))
            return mock.Mock(returncode=10, stdout="", stderr="")

        with mock.patch.object(run_case, "ssh", side_effect=fake_ssh):
            self.assertEqual(run_case.stop_server_process("memory0", state),
                             "already_exited")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][2] is False)

    def test_cleanup_refuses_identity_mismatch(self):
        state = {
            "pid": 1234, "starttime": "777", "server_bin": "/opt/server",
            "remote_dir": "/tmp/run/endpoint-00",
        }
        with mock.patch.object(
                run_case, "ssh",
                return_value=mock.Mock(returncode=11, stdout="", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
                run_case.stop_server_process("memory0", state)

    def test_cleanup_handles_exit_during_identity_read(self):
        state = {
            "pid": 1234, "starttime": "777", "server_bin": "/opt/server",
            "remote_dir": "/tmp/run/endpoint-00",
        }
        responses = [mock.Mock(returncode=11, stdout="", stderr=""),
                     mock.Mock(returncode=10, stdout="", stderr="")]
        with mock.patch.object(run_case, "ssh", side_effect=responses) as ssh:
            self.assertEqual(run_case.stop_server_process("memory0", state),
                             "already_exited")
        self.assertNotIn("kill -", ssh.call_args_list[1].args[1])

    def test_partial_cleanup_script_is_bounded(self):
        state = {
            "pid": 1234, "starttime": "777", "server_bin": "/opt/server",
            "remote_dir": "/tmp/run/endpoint-00",
        }
        calls = []

        def fake_ssh(host, script, check=True, timeout=30):
            calls.append((host, script, check, timeout))
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(run_case, "ssh", side_effect=fake_ssh):
            self.assertEqual(run_case.stop_server_process("memory0", state), "stopped")
        script = calls[0][1]
        self.assertIn("kill -TERM", script)
        self.assertIn("kill -KILL", script)
        self.assertIn("seq 1 100", script)
        self.assertIn("seq 1 200", script)
        self.assertEqual(calls[0][3], 60)
        self.assertEqual(script.count("stat 2>/dev/null)\" = Z"), 2)
        self.assertIn('start=${20}', script)
        self.assertIn("seq 1 40", script)

    def test_ssh_does_not_load_login_logout_hooks(self):
        with mock.patch.object(run_case.subprocess, "run",
                               return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            run_case.ssh("memory0", "exit 0")
        command = run.call_args.args[0][-1]
        self.assertIn("env -u BASH_ENV -u ENV bash --noprofile --norc -c", command)
        self.assertNotIn("bash -lc", command)


if __name__ == "__main__":
    unittest.main()
