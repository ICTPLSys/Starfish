"""Figure 9 baseline-variant planning checks; no SSH or benchmark launch."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_case


def config_values(text):
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            values[fields[0]] = fields[1]
    return values


class BaselineVariant(unittest.TestCase):
    def plan(self, root, *, baseline_variant=None, system="nonft"):
        site = root / "site.json"
        site.write_text(json.dumps({
            "memory_ip": "10.208.130.76",
            "memory_server_count": 1,
            "server_port": 1893,
        }), encoding="utf-8")
        args = argparse.Namespace(
            app="kv-b", system=system, baseline_variant=baseline_variant,
            ratio=25, site=site, out=root / "case", timeout=1800,
            dry_run=True,
        )
        with patch.object(run_case, "ssh") as ssh, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run_case.run(args), 0)
        ssh.assert_not_called()
        self.assertFalse(args.out.exists())
        return json.loads(output.getvalue())

    def test_nonft_backup_off_is_distinct_and_forces_only_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            on = self.plan(root)
            off = self.plan(root, baseline_variant="nonft-backup-off")
        on_values = config_values(on["effective_config"])
        off_values = config_values(off["effective_config"])
        self.assertEqual(on["baseline_variant"], "nonft")
        self.assertTrue(on["backup_enabled"])
        self.assertEqual(off["baseline_variant"], "nonft-backup-off")
        self.assertFalse(off["backup_enabled"])
        self.assertEqual(off["feature_overrides"]["backup"], False)
        self.assertEqual(off_values["enable_selective_backup"], "0")
        self.assertEqual(off_values["remote_backup_budget_bytes"], "0")
        self.assertEqual(off_values["remote_backup_budget_pct"], "0")
        self.assertNotEqual(on_values["enable_selective_backup"], "0")
        self.assertEqual(
            on_values["local_resident_budget_bytes"],
            off_values["local_resident_budget_bytes"])

    def test_off_variant_requires_nonft_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "requires --system nonft"):
                self.plan(Path(temporary), baseline_variant="nonft-backup-off",
                          system="starfish")


if __name__ == "__main__":
    unittest.main()
