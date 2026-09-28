"""Hardware-free metadata observer/collector contract tests; no benchmarks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys

AE = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "figure12_metadata", AE / "scripts/figure12/collect_metadata.py")
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)
sys.path.insert(0, str(AE / "scripts/common"))
from workloads import client_environment, client_process_environment


def record(**overrides):
    fields = {
        **metadata.CONVENTIONS, "schema_version": 1, "system": "starfish",
        "boundary_sequence": 1, "snapshot_us": 123,
        "metadata_bytes": 150, "measurement_aux_bytes": 50, "accounted_bytes": 200,
        **dict(zip(metadata.COMPONENTS, (10, 20, 30, 40, 50))),
        **dict.fromkeys(metadata.COUNTS, 1), **overrides,
    }
    return "runtime_metadata " + " ".join(f"{k}={v}" for k, v in fields.items())


class MetadataContract(unittest.TestCase):
    def test_recorded_site_opt_in_and_shell_isolation(self):
        site = {"ib_device": "mlx5_0", "runtime_metadata": True}
        for system in ("starfish", "carbink"):
            self.assertEqual(client_environment("kv-b", site, system)[
                "FARLIB_RUNTIME_METADATA"], "1")
        with self.assertRaises(ValueError):
            client_environment("kv-b", site, "nonft")
        with self.assertRaises(ValueError):
            client_environment("kv-b", {**site, "runtime_metadata": "1"}, "starfish")
        clean = client_process_environment("kv-b", {"ib_device": "mlx5_0"},
                                           "starfish", {"FARLIB_RUNTIME_METADATA": "1"})
        self.assertNotIn("FARLIB_RUNTIME_METADATA", clean)

    def test_sum_and_auxiliary_separation(self):
        result = metadata.parse_line(record())
        self.assertEqual(result["metadata_bytes"], 150)
        self.assertEqual(result["accounted_bytes"], 200)

    def test_missing_bad_and_wrong_convention_fail(self):
        for line in (record(metadata_bytes=151), record(accounted_bytes=201),
                     record(schema_version=2), record(phase="cleanup"),
                     record(snapshot="peak"), record(group_bytes=-1),
                     record(system="nonft"), record(boundary_sequence=0),
                     record() + " metadata_bytes=150",
                     record().replace(" span_bytes=50", "")):
            with self.subTest(line=line), self.assertRaises(ValueError):
                metadata.parse_line(line)

    def test_unrelated_line_is_not_zero(self):
        self.assertIsNone(metadata.parse_line("Work time 12.3 s"))

    def test_successful_run_and_failure_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nq-starfish-r1"
            path.mkdir()
            manifest = {"status": "passed", "exit_status": 0, "client_sha256": "fixture",
                        "plan": {"app": "nq", "system": "starfish", "ratio": 25,
                                 "site": "fixture", "run_id": path.name,
                                 "workload_footprint_bytes": 10000,
                                 "client_env": {"FARLIB_RUNTIME_METADATA": "1"}}}
            analysis = {"status": "passed", "exit_status": 0,
                        "correctness": "pass", "application": "nq", "ratio": 25,
                        "system": "Starfish", "run_id": path.name, "environment": "fixture"}
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "analysis.json").write_text(json.dumps(analysis))
            (path / "client.log").write_text(record() + "\n")
            rows = metadata.collect_rows(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["value"]), 2.0)
            self.assertEqual(rows[0]["metadata_bytes"], 200)
            self.assertEqual(rows[0]["core_metadata_bytes"], 150)
            self.assertEqual(rows[0]["metadata_numerator"], "accounted_bytes")
            self.assertEqual(rows[0]["measurement_aux_bytes"], 50)
            for key, invalid in (("system", "Carbink"), ("run_id", "another-run"),
                                 ("environment", "another-site")):
                original = analysis[key]
                analysis[key] = invalid
                (path / "analysis.json").write_text(json.dumps(analysis))
                with self.assertRaises(ValueError):
                    metadata.collect_rows(Path(tmp))
                analysis[key] = original
            (path / "analysis.json").write_text(json.dumps(analysis))
            manifest["exit_status"] = 127
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                metadata.collect_rows(Path(tmp))
            manifest["exit_status"] = 0
            (path / "manifest.json").write_text(json.dumps(manifest))
            analysis["exit_status"] = 139
            (path / "analysis.json").write_text(json.dumps(analysis))
            with self.assertRaises(ValueError):
                metadata.collect_rows(Path(tmp))
            analysis["exit_status"] = 0
            (path / "analysis.json").write_text(json.dumps(analysis))
            (path / "client.log").write_text("no completed snapshot\n")
            with self.assertRaises(ValueError):
                metadata.collect_rows(Path(tmp))
            (path / "client.log").write_text(record() + "\n" + record() + "\n")
            with self.assertRaises(ValueError):
                metadata.collect_rows(Path(tmp))

    def test_explicit_retry_selection_preserves_failure_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = []
            for name, system, status in (
                    ("sf-failed", "starfish", "failed"),
                    ("sf-retry", "starfish", "passed"),
                    ("cb", "carbink", "passed")):
                path = root / name
                path.mkdir()
                manifest = {
                    "status": status, "exit_status": 0 if status == "passed" else 127,
                    "client_sha256": "fixture",
                    "plan": {"app": "kv-b", "system": system, "ratio": 25,
                             "site": "fixture", "run_id": name,
                             "workload_footprint_bytes": 10000,
                             "client_env": {"FARLIB_RUNTIME_METADATA": "1"}}}
                analysis = {"status": status, "exit_status": 0 if status == "passed" else 127,
                            "correctness": "pass" if status == "passed" else "fail",
                            "application": "kv-b", "ratio": 25,
                            "system": metadata.SYSTEMS[system], "run_id": name,
                            "environment": "fixture"}
                (path / "manifest.json").write_text(json.dumps(manifest))
                (path / "analysis.json").write_text(json.dumps(analysis))
                (path / "client.log").write_text(record(system=system) + "\n")
                if status == "passed":
                    selected.append(path)
            self.assertEqual(len(metadata.collect_run_dirs(selected)), 2)
            with self.assertRaises(ValueError):
                metadata.collect_rows(root)
            with self.assertRaises(ValueError):
                metadata.collect_run_dirs(selected + [selected[0]])
            with self.assertRaises(ValueError):
                metadata.collect_run_dirs([root / "sf-failed"])
            with self.assertRaises(ValueError):
                metadata.collect_run_dirs(selected, ratio=0)
            output = root / "selected.csv"
            result = subprocess.run(
                [sys.executable, str(AE / "scripts/figure12/collect_metadata.py"),
                 "--run-dir", str(selected[0]), "--run-dir", str(selected[1]),
                 "--output", str(output)],
                check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("wrote 2 measured metadata rows", result.stdout)
            self.assertEqual(len(output.read_text().splitlines()), 3)

    def test_observer_keeps_last_work_not_peak_or_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "metadata-test"
            subprocess.run([
                "g++", "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
                "-I", str(AE / "runtime/common"),
                str(Path(__file__).with_name("runtime_metadata_test.cpp")),
                "-o", str(binary),
            ], check=True, capture_output=True, text=True, timeout=90)
            result = subprocess.run([str(binary)], check=True, capture_output=True,
                                    text=True, timeout=10)
            parsed = metadata.parse_line(result.stdout.splitlines()[0])
            self.assertEqual(parsed["metadata_bytes"], 147)
            self.assertEqual(parsed["boundary_sequence"], 2)


if __name__ == "__main__":
    unittest.main()
