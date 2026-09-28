"""Regression tests for explicitly accepted Figure 9 teardown measurements."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


AE_ROOT = Path(__file__).resolve().parents[3]
COMMON = AE_ROOT / "scripts" / "common"
sys.path.insert(0, str(COMMON))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collect_module = load_module(
    "figure9_collect_teardown", AE_ROOT / "scripts" / "figure9" / "collect.py"
)
plot_module = load_module(
    "figure9_plot_teardown", AE_ROOT / "scripts" / "figure9" / "plot.py"
)


class Figure9TeardownMeasurement(unittest.TestCase):
    def make_failed_kv_case(self, root, *, system="starfish", app="kv-a",
                            exit_status=-15):
        run = root / "kv-a-starfish-25-r1"
        run.mkdir()
        analysis = {
            "schema_version": 1, "status": "failed",
            "exit_status": exit_status, "correctness": "fail",
            "application": app, "workload": "KV-A", "system": "Starfish",
            "ratio": 25, "run_id": run.name, "environment": "fast-check56",
            "failure_injection": None, "measurement_phase": None,
        }
        manifest = {
            "schema_version": 1, "status": "failed", "input_bytes": 0,
            "input_mtime_ns": None, "input_files": [],
            "plan": {
                "app": app, "system": system, "ratio": 25,
                "run_id": run.name, "input": None,
            },
        }
        (run / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name in ("client.log", "server.log", "effective.config", "server.config"):
            (run / name).touch()
        return run

    def test_collector_reuses_helper_authorized_kv_teardown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_failed_kv_case(root)
            recovered = {
                "elapsed_s": 86.002381545,
                "measurement_phase": "kv_fixed_count_direct_requests",
                "correctness_scope": "request accounting and sampled value check",
                "measurement_warning": "verified measurement retained; client shutdown timed out",
            }
            with patch.object(collect_module, "recover_measurement",
                              return_value=recovered) as recover:
                rows, failed = collect_module.collect(root)
            recover.assert_called_once()
            self.assertEqual(failed, 1, "raw lifecycle failure remains visible")
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["exit_status"], "-15")
            self.assertEqual(row["correctness"], "pass")
            self.assertEqual(row["measurement_usable"], "1")
            self.assertEqual(row["execution_status"], "teardown_failed")
            self.assertEqual(row["warning"], recovered["measurement_warning"])
            self.assertEqual(row["elapsed_s"], "86.0023815")

    def test_collector_does_not_loosen_other_cases_and_surfaces_bad_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_failed_kv_case(root, system="nonft")
            with patch.object(collect_module, "recover_measurement",
                              return_value=None):
                rows, failed = collect_module.collect(root)
            self.assertEqual((rows, failed), ([], 1))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_failed_kv_case(root)
            with patch.object(collect_module, "recover_measurement",
                              side_effect=ValueError("postcheck evidence is invalid")):
                rows, failed = collect_module.collect(root)
            self.assertEqual((rows, failed), ([], 1))

    def write_csv(self, root, fields, row):
        path = root / "figure9.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_plot_requires_explicit_teardown_acceptance_and_keeps_warning(self):
        fields = [
            "workload", "system", "ratio", "elapsed_s", "source_type",
            "source", "run_id", "exit_status", "correctness",
            "measurement_usable", "execution_status", "warning",
        ]
        base = {
            "workload": "KV-A", "system": "Starfish", "ratio": "25",
            "elapsed_s": "86.002381545", "source_type": "measured",
            "source": "runs/kv-a-starfish-25-r1/analysis.json",
            "run_id": "kv-a-starfish-25-r1", "exit_status": "-15",
            "correctness": "pass", "measurement_usable": "1",
            "execution_status": "teardown_failed",
            "warning": "verified measurement retained; client was terminated",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_csv(root, fields, base)
            data = plot_module.prepare(
                path, workloads=("kv_a",), systems=("starfish",), ratios=(25,)
            )
            run = data["points"][0]["runs"][0]
            self.assertEqual(run["warning"], base["warning"])
            self.assertEqual(run["execution_status"], "teardown_failed")

            for key, value in (
                ("measurement_usable", "0"),
                ("execution_status", "passed"),
                ("correctness", "fail"),
                ("exit_status", "1"),
            ):
                invalid = dict(base, **{key: value})
                bad = self.write_csv(root, fields, invalid)
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError, "nonzero exit_status"):
                        plot_module.prepare(
                            bad, workloads=("kv_a",), systems=("starfish",),
                            ratios=(25,),
                        )

    def test_plot_keeps_legacy_zero_exit_csv_compatible(self):
        fields = [
            "workload", "system", "ratio", "elapsed_s", "source_type",
            "source", "run_id", "exit_status", "correctness",
        ]
        row = {
            "workload": "KV-A", "system": "Starfish", "ratio": "25",
            "elapsed_s": "86.0", "source_type": "measured",
            "source": "old/analysis.json", "run_id": "old-r1",
            "exit_status": "0", "correctness": "pass",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_csv(Path(temporary), fields, row)
            data = plot_module.prepare(
                path, workloads=("kv_a",), systems=("starfish",), ratios=(25,)
            )
            self.assertEqual(data["points"][0]["n"], 1)
            self.assertEqual(data["points"][0]["runs"][0]["warning"], "")


if __name__ == "__main__":
    unittest.main()
