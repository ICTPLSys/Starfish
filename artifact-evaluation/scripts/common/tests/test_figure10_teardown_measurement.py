"""Mocked acceptance tests for verified work retained after teardown failure."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve()
COMMON = HERE.parents[1]
FIGURE10 = COMMON.parent / "figure10"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(FIGURE10))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect = load_module("figure10_teardown_collect", FIGURE10 / "collect.py")
plot = load_module("figure10_teardown_plot", FIGURE10 / "plot.py")


def result(p99=123400):
    return {"p99_latency_ns": p99,
            "phases": {"measurement": {
                "scheduled": 1000, "completed_in_window_ops_s": 999,
                "p99_service_ns": 3000, "p99_dispatch_ns": 4000,
                "realized_offered_ops_s": 1000, "completed": 1000,
                "deadline_dropped": 0, "drop_fraction": 0,
                "completed_after_deadline": 0}}}


def fixture(root, *, exit_status=124, timed_out=True, run_status="failed",
            correctness="fail", endpoint_status="cleanup_failed"):
    directory = Path(root) / "kv-b-starfish-1000ops-r1"
    directory.mkdir()
    (directory / "client.log").write_text("raw fixture\n", encoding="utf-8")
    spec = {"offered_load_ops": 1000, "max_queue_delay_us": 0,
            "histogram_dir": str(directory / "histograms")}
    analysis = {"status": run_status, "exit_status": exit_status,
                "timed_out": timed_out, "correctness": correctness,
                "application": "kv-b", "kv_latency": None,
                "endpoints": [{"status": endpoint_status}]}
    plan = {"app": "kv-b", "system": "starfish", "kv_latency": spec,
            "worker_profile": {"app_workers": 24}, "effective_config": "",
            "client_env": {}, "ratio": 25}
    manifest = {"status": run_status, "plan": plan, "client_sha256": "client"}
    (directory / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, analysis, manifest


class TeardownCollection(unittest.TestCase):
    def test_recovered_starfish_kv_is_measurement_only_and_ignores_stale_saved_p99(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, analysis, manifest = fixture(temporary)
            recovered = {"kv_latency": result(),
                         "measurement_warning": "verified work; shutdown timed out"}
            with mock.patch.object(collect, "recover_measurement", return_value=recovered), \
                 mock.patch.object(collect, "validate_design2"):
                row = collect.point(directory)
            self.assertEqual(row["p99_latency"], 123.4)
            self.assertEqual(row["exit_status"], 124)
            self.assertEqual(row["correctness"], "pass")
            self.assertEqual(row["measurement_usable"], 1)
            self.assertEqual(row["execution_status"], "teardown_failed")
            self.assertEqual(row["warning"], recovered["measurement_warning"])

    def test_normal_pass_still_requires_cleanup_and_does_not_use_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, analysis, manifest = fixture(
                temporary, exit_status=0, timed_out=False, run_status="passed",
                correctness="pass", endpoint_status="cleanup_failed")
            with mock.patch.object(collect, "recover_measurement") as recover:
                with self.assertRaisesRegex(ValueError, "cleanup"):
                    collect.point(directory)
            recover.assert_not_called()

    def test_non_teardown_crash_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, analysis, manifest = fixture(
                temporary, exit_status=-11, timed_out=False)
            with mock.patch.object(collect, "recover_measurement", return_value=None):
                self.assertIsNone(collect.point(directory))


def write_csv(path, **changes):
    row = {
        "workload": "KV-B", "system": "Starfish", "offered_load": "1",
        "p99_latency": "123.4", "load_unit": "Mops", "latency_unit": "us",
        "source_type": "measured", "source": "fixture", "exit_status": "0",
        "correctness": "pass", "measurement_usable": "1",
        "execution_status": "passed", "warning": "",
    }
    row.update(changes)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


class TeardownPlot(unittest.TestCase):
    def test_recovered_row_requires_markers_but_no_extra_cli_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "figure10.csv"
            write_csv(path, exit_status="124", execution_status="teardown_failed",
                      warning="verified work; shutdown timed out")
            data = plot.prepare(path)
            self.assertEqual(data["input_rows"], 1)
            self.assertEqual(data["series"]["kv_b/starfish"][0]["p99_latency"], 123.4)
            self.assertEqual(data["series"]["kv_b/starfish"][0]["exit_status"], "124")
            self.assertEqual(data["series"]["kv_b/starfish"][0]["warning"],
                             "verified work; shutdown timed out")

    def test_wrong_teardown_marker_or_exit_remains_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "figure10.csv"
            write_csv(path, exit_status="-6", execution_status="teardown_failed")
            with self.assertRaisesRegex(ValueError, "failed runs"):
                plot.prepare(path)
            write_csv(path, exit_status="124", execution_status="passed")
            with self.assertRaisesRegex(ValueError, "failed runs"):
                plot.prepare(path)

    def test_normal_pass_remains_valid_without_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "figure10.csv"
            write_csv(path)
            self.assertEqual(plot.prepare(path)["input_rows"], 1)


if __name__ == "__main__":
    unittest.main()
