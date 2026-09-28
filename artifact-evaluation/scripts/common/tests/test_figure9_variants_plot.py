"""Offline Figure 9 checks that backup-OFF rows stay out of the four-system plot."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


AE_ROOT = Path(__file__).resolve().parents[3]
PLOT_PATH = AE_ROOT / "scripts/figure9/plot.py"
spec = importlib.util.spec_from_file_location("figure9_plot_variants", PLOT_PATH)
plot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(plot)


FIELDS = [
    "workload", "system", "baseline_variant", "backup_enabled", "ratio",
    "elapsed_s", "source_type", "source", "run_id", "exit_status",
    "correctness", "environment", "measurement_phase",
]


class Figure9VariantPlot(unittest.TestCase):
    def write_csv(self, root, rows, fields=FIELDS):
        path = root / "figure9.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def row(system, run_id, elapsed, *, variant="canonicalruntime",
            backup="1"):
        return {
            "workload": "MG", "system": system,
            "baseline_variant": variant, "backup_enabled": backup,
            "ratio": "25", "elapsed_s": str(elapsed),
            "source_type": "measured", "source": f"runs/{run_id}/analysis.json",
            "run_id": run_id, "exit_status": "0", "correctness": "pass",
            "environment": "test", "measurement_phase": "work",
        }

    def test_mixed_backup_off_is_ignored_after_provenance_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_csv(Path(temporary), [
                self.row("Non-FT", "on", 1.0),
                self.row("Non-FT (backup off)", "off", 2.0,
                         variant="nonft-backup-off", backup="0"),
            ])
            data = plot.prepare(
                path, workloads=("mg",), systems=("nonft",), ratios=(25,),
            )
        self.assertEqual(
            {(point["system"], point["n"], point["mean_s"])
             for point in data["points"]},
            {("nonft", 1, 1.0)},
        )
        self.assertEqual(len(data["ignored_backup_off_rows"]), 1)

    def test_default_matrix_has_four_series_and_clear_legend(self):
        self.assertEqual(
            plot.DEFAULT_SYSTEMS,
            ("hydra", "carbink", "starfish", "nonft"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_csv(Path(temporary), [
                self.row("Non-FT", "on", 1.0),
                self.row("Non-FT (backup off)", "off", 2.0,
                         variant="nonft-backup-off", backup="0"),
            ])
            data = plot.prepare(
                path, workloads=("mg",), systems=plot.DEFAULT_SYSTEMS,
                ratios=(25,),
            )
            fig = plot.draw(data)
            try:
                labels = [text.get_text() for legend in fig.legends
                          for text in legend.get_texts()]
            finally:
                plot.get_pyplot().close(fig)
        self.assertEqual(
            labels,
            ["Hydra", "Carbink", "Starfish", "Non-FT"],
        )

    def test_backup_off_requires_provenance_even_when_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_csv(Path(temporary), [
                self.row("Non-FT (backup off)", "bad", 2.0,
                         variant="nonft-backup-off", backup="1"),
            ])
            with self.assertRaisesRegex(ValueError, "backup_enabled=false"):
                plot.prepare(path, workloads=("mg",),
                             systems=("nonft",), ratios=(25,))

    def test_explicit_backup_off_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_csv(Path(temporary), [
                self.row("Non-FT (backup off)", "off", 2.0,
                         variant="nonft-backup-off", backup="0"),
            ])
            with self.assertRaisesRegex(ValueError, "does not plot"):
                plot.prepare(path, workloads=("mg",),
                             systems=("nonft-backup-off",), ratios=(25,))

    def test_legacy_four_system_csv_remains_compatible(self):
        fields = ["workload", "system", "ratio", "elapsed_s", "source_type",
                  "source", "run_id", "exit_status", "correctness"]
        row = {
            "workload": "MG", "system": "Non-FT", "ratio": "25",
            "elapsed_s": "1.0", "source_type": "measured",
            "source": "old/analysis.json", "run_id": "old",
            "exit_status": "0", "correctness": "pass",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_csv(Path(temporary), [row], fields)
            data = plot.prepare(path, workloads=("mg",), systems=("nonft",),
                                ratios=(25,))
        self.assertEqual(data["points"][0]["n"], 1)


if __name__ == "__main__":
    unittest.main()
