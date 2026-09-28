import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_case


class FeatureOverrides(unittest.TestCase):
    def plan(self, app, overrides=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "site.json"
            settings = {"compute_ip": "10.208.130.56", "memory_ip": "10.208.130.76"}
            if overrides is not None:
                settings["feature_overrides"] = overrides
            site.write_text(json.dumps(settings))
            args = argparse.Namespace(app=app, system="nonft", ratio=25,
                                      site=site, out=root/"case", timeout=1800, dry_run=True)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(run_case.run(args), 0)
            self.assertFalse(args.out.exists())
            return json.loads(output.getvalue())

    def test_test_override_does_not_change_defaults(self):
        for app in ("wordcount", "nq", "kv-b", "kv-a", "kv-s"):
            off = self.plan(app, {"backup": False, "resident": False})
            self.assertIn("enable_selective_backup 0\n", off["effective_config"])
            self.assertIn("local_resident_budget_bytes 0\n", off["effective_config"])
            self.assertIn("enable_region_resident_placement 0\n", off["effective_config"])
            default = self.plan(app)
            self.assertIn("enable_selective_backup 1\n", default["effective_config"])
            self.assertIn("enable_region_resident_placement 1\n", default["effective_config"])
            self.assertNotIn("local_resident_budget_bytes 0\n", default["effective_config"])

    def test_kv_generated_input_and_cli_capacity(self):
        plan = self.plan("kv-b")
        self.assertIsNone(plan["input"])
        self.assertEqual(plan["client_command"][-1], "4")
        self.assertEqual(plan["client_env"]["FARLIB_KVS_MAX_SERVE_COUNT"], "1000000000")
        self.assertEqual(plan["client_env"]["FARLIB_KVS_PUT_RATIO"], "0.05")

    def test_unknown_or_nonboolean_overrides_are_rejected(self):
        for overrides in ({"backup": "false"}, {"threads": 1}, []):
            with self.assertRaises(ValueError):
                self.plan("wordcount", overrides)
