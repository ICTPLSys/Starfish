import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AE = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("figure9_collect", AE / "scripts/figure9/collect.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeneratedInputCollection(unittest.TestCase):
    def collect_case(self, app="mg", input_files=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "case"
            run.mkdir()
            record = {
                "schema_version": 1, "status": "passed", "exit_status": 0,
                "correctness": "pass", "application": app,
                "workload": "MG" if app == "mg" else "LLM", "system": "Non-FT",
                "ratio": 25, "run_id": "case", "measurement_phase": "work",
                "elapsed_s": 1.0, "environment": "test",
            }
            manifest = {
                "schema_version": 1, "status": "passed", "input_bytes": 0,
                "input_mtime_ns": None, "input_files": input_files or [],
                "plan": {"app": app, "system": "nonft", "ratio": 25,
                         "run_id": "case", "input": None},
            }
            (run / "analysis.json").write_text(json.dumps(record))
            (run / "manifest.json").write_text(json.dumps(manifest))
            for name in ("client.log", "server.log", "effective.config", "server.config"):
                (run / name).touch()
            return module.collect(root)

    def test_generated_mg_has_no_file_timestamp(self):
        rows, failed = self.collect_case()
        self.assertEqual((len(rows), failed), (1, 0))
        self.assertEqual(rows[0]["workload"], "MG")
        self.assertEqual(rows[0]["baseline_variant"], "canonicalruntime")
        self.assertEqual(rows[0]["backup_enabled"], "")

    def test_file_backed_workload_requires_timestamp(self):
        with self.assertRaisesRegex(ValueError, "input timestamp"):
            self.collect_case("llama")

    def test_generated_mg_rejects_inconsistent_file_list(self):
        with self.assertRaisesRegex(ValueError, "input timestamp"):
            self.collect_case(input_files=["unexpected.input"])

    def write_variant_run(self, root, name, *, variant="nonft-backup-off",
                          backup_enabled=False, config=None):
        run = root / name
        run.mkdir()
        record = {
            "schema_version": 1, "status": "passed", "exit_status": 0,
            "correctness": "pass", "application": "mg",
            "workload": "MG", "system": "Non-FT", "ratio": 25,
            "run_id": name, "measurement_phase": "work", "elapsed_s": 1.0,
            "environment": "test", "baseline_variant": variant,
            "backup_enabled": backup_enabled,
        }
        manifest = {
            "schema_version": 1, "status": "passed", "input_bytes": 0,
            "input_mtime_ns": None, "input_files": [],
            "baseline_variant": variant, "backup_enabled": backup_enabled,
            "plan": {"app": "mg", "system": "nonft", "ratio": 25,
                      "run_id": name, "input": None,
                      "baseline_variant": variant,
                      "backup_enabled": backup_enabled},
        }
        if config is None:
            config = ("enable_selective_backup 0\n"
                      "remote_backup_budget_bytes 0\n"
                      "remote_backup_budget_pct 0\n")
        (run / "analysis.json").write_text(json.dumps(record))
        (run / "manifest.json").write_text(json.dumps(manifest))
        for name_ in ("client.log", "server.log", "server.config"):
            (run / name_).touch()
        (run / "effective.config").write_text(config)

    def test_backup_off_is_a_distinct_csv_system(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_variant_run(root, "off")
            rows, failed = module.collect(root)
        self.assertEqual((len(rows), failed), (1, 0))
        self.assertEqual(rows[0]["system"], "Non-FT (backup off)")
        self.assertEqual(rows[0]["baseline_variant"], "nonft-backup-off")
        self.assertEqual(rows[0]["backup_enabled"], "false")

    def test_backup_off_rejects_nonzero_rendered_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_variant_run(
                root, "off", config="enable_selective_backup 0\n"
                "remote_backup_budget_bytes 123\n")
            with self.assertRaisesRegex(ValueError, "zero byte budget"):
                module.collect(root)

    def test_backup_off_rejects_missing_percentage_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_variant_run(
                root, "off", config="enable_selective_backup 0\n"
                "remote_backup_budget_bytes 0\n")
            with self.assertRaisesRegex(ValueError, "percentage budget"):
                module.collect(root)

    def test_backup_off_rejects_true_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_variant_run(root, "off", backup_enabled=True)
            with self.assertRaisesRegex(ValueError, "backup-OFF"):
                module.collect(root)

    def test_canonical_and_backup_off_rows_are_not_merged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_variant_run(
                root, "on", variant="canonicalruntime", backup_enabled=True,
                config="enable_selective_backup 1\n"
                "remote_backup_budget_bytes 123\n")
            self.write_variant_run(root, "off")
            rows, failed = module.collect(root)
        self.assertEqual((len(rows), failed), (2, 0))
        self.assertEqual({row["system"] for row in rows},
                         {"Non-FT", "Non-FT (backup off)"})
