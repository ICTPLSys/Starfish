"""Mocked Figure 12 batch planning/execution tests."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

HERE = Path(__file__).resolve()
AE_ROOT = HERE.parents[3]
sys.path.insert(0, str(AE_ROOT / "scripts" / "figure12"))
import run as figure12_run


def _args(root: str, **changes):
    root = Path(root)
    site = changes.get("site", root / "site.json")
    if "site" not in changes:
        site.write_text(json.dumps({"name": "mock-site"}), encoding="utf-8")
    values = dict(apps=",".join(figure12_run.APPS), systems="starfish,carbink",
                  ratio=25, repeats=1, metric="all", site=site, site_map=None,
                  build_root=root / "build", build_jobs=None, timeout=60,
                  out=root / "batch", build=False, execute=True, dry_run=False)
    values.update(changes)
    return argparse.Namespace(**values)


def _valid_metrics():
    return [{"metric": "local_ec_cpu_cycles"},
            {"metric": "metadata_space_pct"}]


class Figure12Planning(unittest.TestCase):
    def test_full_matrix_and_metric_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary)
            paths, provenance = figure12_run._effective_sites(args, output=None)
            self.addCleanup(figure12_run._cleanup_site_provenance, provenance)
            matrix = figure12_run.plans(args, site_paths=paths, dry_run=True)
            self.assertEqual(len(matrix), 16)
            self.assertEqual([(row["system"], row["app"]) for row in matrix[:8]],
                             [("starfish", app) for app in figure12_run.APPS])
            self.assertEqual(matrix[0]["run_id"], "bfs-starfish-25-r1")
            self.assertNotIn("--collect-remote-cpu", matrix[0]["command"])
            self.assertEqual(matrix[0]["site_overlay"],
                             {"runtime_metadata": True, "runtime_ec_cpu": True})

    def test_metadata_only_does_not_request_ec_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site.json"
            site.write_text(json.dumps({"runtime_metadata": True,
                                        "runtime_ec_cpu": True}),
                            encoding="utf-8")
            args = _args(temporary, site=site, apps="nq",
                         systems="starfish", metric="metadata")
            paths, provenance = figure12_run._effective_sites(args, output=None)
            self.addCleanup(figure12_run._cleanup_site_provenance, provenance)
            matrix = figure12_run.plans(args, site_paths=paths, dry_run=True)
            self.assertNotIn("--collect-remote-cpu", matrix[0]["command"])
            self.assertEqual(matrix[0]["site_overlay"], {"runtime_metadata": True})
            payload = json.loads(paths[("nq", "starfish")].read_text())
            self.assertTrue(payload["runtime_metadata"])
            self.assertFalse(payload["runtime_ec_cpu"])

    def test_ec_cpu_selection_clears_stale_metadata_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site.json"
            site.write_text(json.dumps({"runtime_metadata": True,
                                        "runtime_ec_cpu": True}),
                            encoding="utf-8")
            args = _args(temporary, site=site, apps="nq",
                         systems="starfish", metric="ec-cpu")
            paths, provenance = figure12_run._effective_sites(args, output=None)
            self.addCleanup(figure12_run._cleanup_site_provenance, provenance)
            payload = json.loads(paths[("nq", "starfish")].read_text())
            self.assertFalse(payload["runtime_metadata"])
            self.assertTrue(payload["runtime_ec_cpu"])

    def test_site_map_supports_per_app_system_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site_map = root / "sites.json"
            site_map.write_text(json.dumps({"nq/starfish": {
                "name": "nq-sf", "compute_ip": "127.0.0.1",
            }}), encoding="utf-8")
            args = _args(temporary, apps="nq", systems="starfish", site_map=site_map,
                         metric="metadata")
            paths, provenance = figure12_run._effective_sites(args, output=None)
            self.addCleanup(figure12_run._cleanup_site_provenance, provenance)
            self.assertIn("nq/starfish", provenance["cases"])
            payload = json.loads(paths[("nq", "starfish")].read_text())
            self.assertEqual(payload["name"], "nq-sf")
            self.assertTrue(payload["runtime_metadata"])


class Figure12Execution(unittest.TestCase):
    def test_default_mode_executes_with_hierarchical_status_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs", systems="starfish")
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=0, stdout="", stderr="")
            classified = {"status": "passed", "safe_to_continue": True,
                          "reason": "verified", "measurement_usable": True,
                          "exit_status": 0}
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake) as invoke, \
                 mock.patch.object(figure12_run, "classify_case", return_value=classified):
                with mock.patch.object(figure12_run, "_validate_figure12_metrics",
                                       return_value=_valid_metrics()):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(figure12_run.execute(parser, args), 0)
            self.assertEqual(invoke.call_count, 1)
            self.assertTrue(args.out.exists())
            self.assertIn("STARFISH Start...", output.getvalue())
            self.assertIn("STARFISH App BFS START", output.getvalue())
            self.assertIn("STARFISH FINISH!", output.getvalue())

    def test_old_binary_pass_is_rejected_without_figure12_metric_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs", systems="starfish")
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=0, stdout="", stderr="")
            classified = {"status": "passed", "safe_to_continue": True,
                          "reason": "verified", "measurement_usable": True,
                          "exit_status": 0}
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake):
                with mock.patch.object(figure12_run, "classify_case",
                                       return_value=classified):
                    with mock.patch.object(
                            figure12_run, "_validate_figure12_metrics",
                            side_effect=ValueError("missing runtime_ec_cpu record")):
                        self.assertEqual(figure12_run.execute(parser, args), 1)
            status = json.loads((args.out / "batch-status.json").read_text())[0]
            self.assertEqual(status["status"], "error")
            self.assertTrue(status["safe_to_continue"])
            self.assertFalse(status["measurement_usable"])
            self.assertFalse(status["figure12_metrics_valid"])
            self.assertEqual(status["runner_exit_status"], 0)
            self.assertIn("Figure12 metric validation failed", status["reason"])
            self.assertIn("missing runtime_ec_cpu", status["reason"])

    def test_valid_figure12_metrics_preserve_common_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs", systems="starfish")
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=0, stdout="", stderr="")
            classified = {"status": "passed", "safe_to_continue": True,
                          "reason": "verified", "measurement_usable": True,
                          "exit_status": 0}
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake):
                with mock.patch.object(figure12_run, "classify_case",
                                       return_value=classified):
                    with mock.patch.object(
                            figure12_run, "_validate_figure12_metrics",
                            return_value=_valid_metrics()) as validate:
                        self.assertEqual(figure12_run.execute(parser, args), 0)
            status = json.loads((args.out / "batch-status.json").read_text())[0]
            self.assertEqual(status["status"], "passed")
            self.assertTrue(status["measurement_usable"])
            self.assertTrue(status["figure12_metrics_valid"])
            self.assertEqual(status["figure12_metric_count"], 2)
            validate.assert_called_once()

    def test_measurement_warning_keeps_warning_when_metrics_are_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs", systems="starfish")
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=124, stdout="", stderr="")
            classified = {"status": "warning", "safe_to_continue": True,
                          "reason": "verified shutdown warning",
                          "measurement_usable": True, "exit_status": 124}
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake):
                with mock.patch.object(figure12_run, "classify_case",
                                       return_value=classified):
                    with mock.patch.object(
                            figure12_run, "_validate_figure12_metrics",
                            side_effect=ValueError("metadata record missing")):
                        self.assertEqual(figure12_run.execute(parser, args), 1)
            status = json.loads((args.out / "batch-status.json").read_text())[0]
            self.assertEqual(status["status"], "warning")
            self.assertTrue(status["safe_to_continue"])
            self.assertFalse(status["measurement_usable"])
            self.assertFalse(status["figure12_metrics_valid"])
            self.assertIn("verified shutdown warning", status["reason"])
            self.assertIn("metadata record missing", status["reason"])

    def test_explicit_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs", systems="starfish",
                         execute=False, dry_run=True)
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=1,
                             stdout=json.dumps({"available": True}),
                             stderr="fixture endpoint unavailable")
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake) as invoke:
                output = io.StringIO()
                errors = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    self.assertEqual(figure12_run.execute(parser, args), 1)
            self.assertEqual(invoke.call_count, 1)
            self.assertFalse(args.out.exists())
            record = json.loads(output.getvalue())
            self.assertEqual(record["run_id"], "bfs-starfish-25-r1")
            self.assertEqual(record["stderr"], "fixture endpoint unavailable")
            self.assertIn("dry-run stderr: fixture endpoint unavailable",
                          errors.getvalue())

    def test_execute_records_failed_case_without_calling_it_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs", systems="starfish", execute=True,
                         dry_run=False)
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=2, stdout="", stderr="")
            classified = {"status": "error", "safe_to_continue": True,
                          "reason": "mock failure", "measurement_usable": False,
                          "exit_status": 2}
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake), \
                 mock.patch.object(figure12_run, "classify_case", return_value=classified):
                self.assertEqual(figure12_run.execute(parser, args), 1)
            batch = json.loads((args.out / "batch.json").read_text())
            status = json.loads((args.out / "batch-status.json").read_text())
            self.assertEqual(batch["status"], "failed")
            self.assertEqual(status[0]["status"], "error")
            self.assertNotEqual(status[0]["status"], "passed")
            self.assertTrue((args.out / "batch-plan.json").is_file())

    def test_cleanup_failure_aborts_later_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(temporary, apps="bfs,llama", systems="starfish")
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=1, stdout="", stderr="")
            unsafe = {"status": "error", "safe_to_continue": False,
                      "reason": "cleanup incomplete", "measurement_usable": False,
                      "exit_status": 1}
            with mock.patch.object(figure12_run.subprocess, "run", return_value=fake), \
                 mock.patch.object(figure12_run, "classify_case", return_value=unsafe):
                output = io.StringIO()
                errors = io.StringIO()
                with contextlib.redirect_stdout(output), \
                     contextlib.redirect_stderr(errors):
                    self.assertEqual(figure12_run.execute(parser, args), 1)
            self.assertIn("STARFISH App BFS ABORTED!", output.getvalue())
            self.assertIn("BATCH ABORTED", errors.getvalue())
            status = json.loads((args.out / "batch-status.json").read_text())
            self.assertEqual(len(status), 1)


def _complete_endpoint_site(path: Path) -> None:
    endpoints = []
    for index in range(6):
        endpoints.append({
            "memory_ip": "192.0.2.10", "server_port": 1893 + index,
            "memory_numa_node": 0, "memory_ib_device": "mlx5_1",
            "memory_ib_port": 1,
            "memory_server_bins": {
                "starfish": "/tmp/starfish-server",
                "carbink": "/tmp/carbink-server",
            },
        })
    path.write_text(json.dumps({
        "name": "complete-six-endpoint-fixture", "ib_device": "mlx5_1",
        "ib_port": 1, "numa_node": 0, "memory_endpoints": endpoints,
    }), encoding="utf-8")


class Figure12DryRunFixture(unittest.TestCase):
    def test_complete_endpoint_fixture_plans_all_16_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "site.json"
            _complete_endpoint_site(site)
            args = _args(str(root), site=site, execute=False, dry_run=True,
                         build_root=AE_ROOT / "build")
            parser = argparse.ArgumentParser()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(figure12_run.execute(parser, args), 0)
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(
                [(row["system"], row["app"]) for row in rows[:8]],
                [("starfish", app) for app in figure12_run.APPS])
            self.assertTrue(all("--collect-remote-cpu" not in row["command"]
                                for row in rows))
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
