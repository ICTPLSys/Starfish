"""CPU-only contract fixtures. These numbers are synthetic, not experiments."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

AE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(AE / "scripts/common"))
from figure12 import collect, collect_metadata, log_contract
from workloads import client_environment, client_process_environment


def ec_record(system="starfish", sequence=1, cycles=100, scopes=2, **overrides):
    fields = dict(schema_version=1, system=system, phase="work", scope="compute_ec",
                  clock="tsc", boundary_sequence=sequence, cycles=cycles, scopes=scopes)
    fields.update(overrides)
    return "runtime_ec_cpu " + " ".join(f"{k}={v}" for k, v in fields.items())


def metadata_record(system="starfish", sequence=1):
    fields = dict(collect_metadata.CONVENTIONS, schema_version=1, system=system,
                  boundary_sequence=sequence, snapshot_us=1, metadata_bytes=150,
                  measurement_aux_bytes=50, accounted_bytes=200)
    fields.update(zip(collect_metadata.COMPONENTS, (10, 20, 30, 40, 50)))
    fields.update(dict.fromkeys(collect_metadata.COUNTS, 1))
    return "runtime_metadata " + " ".join(f"{k}={v}" for k, v in fields.items())


def make_run(root, app="nq", system="starfish", name=None, intervals=1):
    directory = root / (name or f"{app}-{system}")
    directory.mkdir(parents=True)
    plan = dict(app=app, system=system, ratio=25, run_id=directory.name, site="fixture",
                workload_footprint_bytes=10000,
                client_env={"FARLIB_RUNTIME_METADATA": "1", "FARLIB_RUNTIME_EC_CPU": "1"})
    manifest = dict(status="passed", exit_status=0, client_sha256="synthetic", plan=plan)
    analysis = dict(status="passed", exit_status=0, correctness="pass", application=app,
                    ratio=25, system=log_contract.SYSTEMS[system],
                    run_id=directory.name, environment="fixture")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "analysis.json").write_text(json.dumps(analysis))
    (directory / "client.log").write_text(
        "\n".join(ec_record(system, n, cycles=100 * n) for n in range(1, intervals + 1))
        + "\n" + metadata_record(system, intervals) + "\n")
    return directory


class Figure12Collection(unittest.TestCase):
    def test_optins_all_apps_both_runtimes_and_environment_isolation(self):
        site = dict(ib_device="mlx5_0", runtime_metadata=True, runtime_ec_cpu=True)
        for app in log_contract.WORKLOADS:
            for system in log_contract.SYSTEMS:
                env = client_environment(app, site, system)
                self.assertEqual(env["FARLIB_RUNTIME_EC_CPU"], "1")
                self.assertEqual(env["FARLIB_RUNTIME_METADATA"], "1")
        for value in ("1", 1, None):
            with self.assertRaises(ValueError):
                client_environment("nq", dict(site, runtime_ec_cpu=value), "starfish")
        for system in ("nonft", "hydra"):
            with self.assertRaises(ValueError):
                client_environment("nq", dict(ib_device="mlx5_0", runtime_ec_cpu=True), system)
        clean = client_process_environment("nq", dict(ib_device="mlx5_0"), "starfish",
                                           {"FARLIB_RUNTIME_EC_CPU": "1"})
        self.assertNotIn("FARLIB_RUNTIME_EC_CPU", clean)

    def test_complete_matrix_and_multiwork_sum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for app in log_contract.WORKLOADS:
                for system in log_contract.SYSTEMS:
                    make_run(root, app, system, intervals=3)
            rows = collect.collect_rows(root)
            self.assertEqual(len(rows), 32)
            for row in rows:
                if row["metric"] == "local_ec_cpu_cycles":
                    self.assertEqual(row["value"], 600)
                    self.assertEqual(row["ec_work_intervals"], 3)
                    self.assertEqual(row["cycle_clock"], "tsc")
                else:
                    self.assertEqual(row["metadata_bytes"], 200)
                    self.assertEqual(float(row["value"]), 2)

    def test_explicit_zero_is_valid_but_absence_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_run(Path(temp))
            log = path / "client.log"
            log.write_text(ec_record(cycles=0, scopes=0) + "\n")
            rows = collect.collect_run_dirs([path], metrics="ec-cpu")
            self.assertEqual(rows[0]["value"], 0)
            with self.assertRaisesRegex(ValueError, "metadata record"):
                collect.collect_run_dirs([path])
            log.write_text(metadata_record() + "\n")
            self.assertEqual(len(collect.collect_run_dirs([path], metrics="metadata")), 1)
            with self.assertRaisesRegex(ValueError, "missing runtime_ec_cpu"):
                collect.collect_run_dirs([path])

    def test_bad_ec_records_are_rejected(self):
        for record in (ec_record(clock="pmu"), ec_record(phase="cleanup"),
                       ec_record(scope="server"), ec_record(cycles=-1),
                       ec_record(cycles=2**64), ec_record(cycles=1, scopes=0),
                       ec_record(sequence=0), ec_record(schema_version=2),
                       ec_record() + " cycles=0"):
            with self.subTest(record=record), self.assertRaises(ValueError):
                log_contract.parse_line(record)
        self.assertIsNone(log_contract.parse_line("Work time: 1"))

    def test_ec_sequence_system_boundary_and_flag_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_run(Path(temp))
            log = path / "client.log"
            for text in (ec_record(sequence=2), ec_record() + "\n" + ec_record(),
                         ec_record(system="carbink"),
                         ec_record() + "\n" + metadata_record(sequence=2)):
                log.write_text(text + "\n")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    collect.collect_run_dirs([path])
            log.write_text(ec_record() + "\n" + metadata_record())
            manifest = json.loads((path / "manifest.json").read_text())
            del manifest["plan"]["client_env"]["FARLIB_RUNTIME_EC_CPU"]
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "FARLIB_RUNTIME_EC_CPU"):
                collect.collect_run_dirs([path])

    def test_batch_repetition_is_explicit_and_relocatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = []
            for repeat in (1, 2):
                path = make_run(root / "runs", name=f"nq-starfish-r{repeat}")
                cases.append(dict(app="nq", system="starfish", run_id=path.name,
                                  run_dir="/stale/absolute/path", repeat=repeat))
            (root / "batch-plan.json").write_text(json.dumps(cases))
            rows = collect.collect_rows(root, repeat=2)
            self.assertEqual(rows[0]["run_id"], "nq-starfish-r2")
            self.assertEqual(rows[0]["repeat"], 2)
            cases[1]["system"] = "carbink"
            (root / "batch-plan.json").write_text(json.dumps(cases))
            with self.assertRaisesRegex(ValueError, "batch/manifest"):
                collect.collect_rows(root, repeat=2)

    def test_duplicates_footprint_mismatch_and_incomplete_runs_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sf = make_run(root)
            cb = make_run(root, system="carbink")
            with self.assertRaisesRegex(ValueError, "multiple runs"):
                collect.collect_run_dirs([sf, sf])
            manifest = json.loads((cb / "manifest.json").read_text())
            manifest["plan"]["workload_footprint_bytes"] = 10001
            (cb / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "footprint"):
                collect.collect_run_dirs([sf, cb])
            (cb / "analysis.json").unlink()
            with self.assertRaises(OSError):
                collect.collect_rows(root)
            with self.assertRaisesRegex(ValueError, "batch-plan"):
                collect.collect_rows(root, repeat=2)

    def test_uncovered_fixed_six_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_run(Path(temp))
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["plan"]["client_env"]["FARLIB_FIXED_SIX_GROUPS"] = "1"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fixed-six"):
                collect.collect_run_dirs([path], metrics="metadata")
            self.assertEqual(len(collect.collect_run_dirs([path], metrics="ec-cpu")), 1)

    def test_selected_ratio_and_repeat_cannot_be_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = make_run(root)
            self.assertEqual(collect.collect_run_dirs([path])[0]["repeat"], "unindexed")
            with self.assertRaisesRegex(ValueError, "batch-plan"):
                collect.collect_run_dirs([path], repeat=2)
            with self.assertRaisesRegex(ValueError, "ratio/runtime"):
                collect.collect_run_dirs([path], ratio=50)

    def test_recovered_teardown_keeps_failure_and_needs_real_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_run(Path(temp), app="kv-b")
            for name in ("analysis.json", "manifest.json"):
                record = json.loads((path / name).read_text())
                record.update(status="failed", exit_status=124)
                if name == "analysis.json":
                    record["correctness"] = "fail"
                (path / name).write_text(json.dumps(record))
            recovered = {"measurement_warning": "verified measurement retained; client shutdown timed out"}
            with patch("measurement_acceptance.recover_measurement", return_value=recovered):
                rows = collect.collect_run_dirs([path])
                self.assertEqual(rows[0]["exit_status"], 124)
                self.assertEqual(rows[0]["execution_status"], "teardown_failed")
                self.assertEqual(rows[0]["correctness"], "pass")
                (path / "client.log").write_text(ec_record() + "\n")
                self.assertEqual(len(collect.collect_run_dirs([path], metrics="ec-cpu")), 1)
                with self.assertRaisesRegex(ValueError, "metadata record"):
                    collect.collect_run_dirs([path])
            with patch("measurement_acceptance.recover_measurement", return_value=None):
                with self.assertRaisesRegex(ValueError, "verified"):
                    collect.collect_run_dirs([path])

    def test_cli_writes_raw_data_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = make_run(root)
            output, summary = root / "figure12.csv", root / "figure12.json"
            command = [sys.executable, str(AE / "scripts/figure12/collect.py"),
                       "--run-dir", str(path), "--output", str(output),
                       "--summary", str(summary)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["value"], "100")
            self.assertEqual(rows[0]["cycle_clock"], "tsc")
            self.assertEqual(json.loads(summary.read_text())["schema_version"], 3)
            again = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("already exists", again.stderr)


if __name__ == "__main__":
    unittest.main()
