"""Focused checks for the generated, fixed-size Non-FT MG workload."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figure9_support
import run_case
from workloads import client_command, client_process_environment, parse_result


LOG = """Size: 1024x1024x1024 (class_npb U)
Iterations: 5
iter 1
iter 5
perf result
runtime: 57095.976038 ms
Benchmark completed
FINAL_RESIDUAL rnm2=1.53587137116806717e-05 rnmu=9.88992103639334819e-02
MG Benchmark Completed
"""


class MgAdapter(unittest.TestCase):
    def test_all_runtime_profiles_are_script_integrated(self):
        for system in ("nonft", "starfish", "hydra", "carbink"):
            self.assertTrue(
                figure9_support.support("mg", system)["script_supported"])

    def test_generated_input_command(self):
        command, stdin = client_command("mg", Path("/bin/mg"), Path("/run/config"),
                                        None, Path("/ae"))
        self.assertEqual(command, ["/bin/mg", "/run/config"])
        self.assertIsNone(stdin)

    def test_inherited_mg_flags_do_not_change_workload(self):
        env = client_process_environment("mg", {"ib_device": "mlx5_1"}, "nonft",
            {"PATH": "/usr/bin", "MG_DESIGN1_PHASE_REPLAN": "1",
             "MG_DESIGN1_PHASE_REPLAN_ITERATIONS": "5"})
        self.assertEqual(env["MG_DESIGN1_PHASE_REPLAN"], "1")
        self.assertEqual(env["FARLIB_FIBRE_HEAP"], "1")
        self.assertEqual(env["FARLIB_ALLOC_SCOPE_CHECKPOINT"], "1")
        self.assertEqual(env["FARLIB_LEGACY_SCAN_CURSORS"], "1")
        self.assertNotIn("MG_DESIGN1_PHASE_REPLAN_ITERATIONS", env)

    def test_plan_uses_mg_capacity_and_no_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / "site.json"
            site.write_text(json.dumps({"compute_ip": "10.208.130.56",
                                        "memory_ip": "10.208.130.54"}))
            args = argparse.Namespace(app="mg", system="nonft", ratio=25,
                site=site, out=root/"case", timeout=1800, dry_run=True)
            with patch.object(run_case, "ssh") as ssh:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(run_case.run(args), 0)
                ssh.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertIsNone(plan["input"])
            self.assertTrue(plan["client_bin"].endswith("/benchmark/mg/mg"))
            self.assertIn("client_buffer_size 7130185728", plan["effective_config"])
            self.assertIn("local_resident_budget_bytes 5704148582", plan["effective_config"])
            self.assertIn("enable_selective_backup 1", plan["effective_config"])
            self.assertIn("remote_backup_budget_bytes 2852074291",
                          plan["effective_config"])
            self.assertFalse(args.out.exists())

    def test_native_work_and_residual(self):
        result = parse_result("mg", LOG)
        self.assertAlmostEqual(result["elapsed_s"], 57.095976038)
        self.assertEqual(result["measurement_phase"], "mg_5_iterations_work")

    def test_short_or_incorrect_runs_are_rejected(self):
        for log in (LOG.replace("Iterations: 5", "Iterations: 1"),
                    LOG.replace("1.53587137116806717e-05", "1e-05"),
                    LOG.replace("MG Benchmark Completed", "")):
            with self.assertRaises(ValueError):
                parse_result("mg", log)


if __name__ == "__main__":
    unittest.main()
