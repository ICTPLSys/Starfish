"""Figure12 parsing/layout unit checks: synthetic fixtures, no benchmarks."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_figure12_collection import make_run
from figure12 import collect, plot, log_contract


class Figure12Plot(unittest.TestCase):
    def test_direct_run_directory_pipeline_all_32_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for app in log_contract.WORKLOADS:
                for system in log_contract.SYSTEMS:
                    make_run(root, app=app, system=system)
            data = plot.prepare_logs(root)
            self.assertEqual(len(data["values"]), 32)
            self.assertEqual(data["input_format"], "ae_run_directories")
            self.assertEqual(len(data["collected_rows"]), 32)

    def rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for system in ("starfish", "carbink"):
                make_run(root, system=system)
            return collect.collect_rows(root)

    def prepare(self, rows, **options):
        return plot.prepare_rows(list(enumerate(rows, 1)), "fixture", "synthetic fixture",
                                 apps=("NQ",), **options)

    def test_raw_cycles_and_inclusive_metadata(self):
        data = self.prepare(self.rows())
        self.assertEqual(data["values"]["NQ/starfish/local_ec_cpu_cycles"]["value"], 100)
        self.assertEqual(data["values"]["NQ/carbink/metadata_space_pct"]["value"], 2)
        self.assertEqual(data["missing_conditions"], [])
        self.assertEqual(data["cycle_clock"], "tsc")

    def test_missing_is_not_zero_and_selection_is_explicit(self):
        rows = self.rows()
        metadata = [row for row in rows if row["metric"] == "metadata_space_pct"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.prepare(metadata)
        data = self.prepare(metadata, metrics="metadata")
        self.assertEqual(len(data["values"]), 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare(rows + [rows[0]])

    def test_reject_old_cpu_and_bad_units_clock_and_metadata(self):
        original = self.rows()
        for field, value in (("metric", "local_ec_cpu_norm"), ("unit", "seconds"),
                             ("value", "nan"), ("value", "1.2"),
                             ("cycle_clock", "pmu"), ("exit_status", 1),
                             ("correctness", "fail"), ("ratio", 50),
                             ("measurement_usable", 0), ("execution_status", "failed"),
                             ("measurement_phase", "initialization"),
                             ("scope", "server"), ("ec_work_intervals", 0)):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.prepare(rows)
        for field, value in (("metadata_numerator", "metadata_bytes"),
                             ("measurement_phase", "peak"), ("accounted_bytes", 201),
                             ("value", "3"), ("app_memory_bytes", 10001)):
            rows = copy.deepcopy(original)
            row = next(row for row in rows if row["metric"] == "metadata_space_pct")
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare(rows)

    def test_authorized_teardown_warning_is_retained_only_for_kv(self):
        rows = self.rows()
        for row in rows:
            row["workload"] = "KV-B"
            if row["system"] == "Starfish":
                row.update(exit_status=124, execution_status="teardown_failed",
                           measurement_warning="verified completed Work")
        data = plot.prepare_rows(list(enumerate(rows, 1)), "fixture", "fixture", apps=("KV-B",))
        self.assertEqual(data["values"]["KV-B/starfish/local_ec_cpu_cycles"]["measurement_warning"],
                         "verified completed Work")
        next(row for row in rows if row["system"] == "Starfish")["measurement_usable"] = 0
        with self.assertRaises(ValueError):
            plot.prepare_rows(list(enumerate(rows, 1)), "fixture", "fixture", apps=("KV-B",))

    def test_draw_preserves_paper_labels_geometry_and_normalizes_raw_cycles(self):
        data = self.prepare(self.rows())
        data["values"]["NQ/starfish/local_ec_cpu_cycles"]["value"] = 50
        figure = plot.draw(data)
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(tuple(figure.get_size_inches()), (4.1, 3.05))
            self.assertEqual(figure.axes[0].get_title(), "Local EC computation CPU")
            self.assertEqual(figure.axes[0].get_ylabel(), "Norm. to Carbink")
            self.assertEqual(figure.axes[1].get_ylabel(), "% app memory")
            self.assertEqual(figure.axes[1].get_title(), "Metadata space")
            self.assertEqual(len(figure.axes[0].lines), 1)
            self.assertEqual(figure.axes[0].yaxis.label.get_fontsize(), 11)
            self.assertEqual(figure.axes[0].yaxis.label.get_fontname(), "Times New Roman")
            self.assertEqual(figure.axes[0].get_ylim(), (0, 1.2))
            self.assertEqual(figure.axes[1].get_ylim(), (0, 10))
            self.assertAlmostEqual(figure.axes[0].patches[0].get_height(), 1)
            self.assertAlmostEqual(figure.axes[0].patches[1].get_height(), .5)
            self.assertEqual(data["values"]["NQ/starfish/local_ec_cpu_cycles"]["value"], 50)
            self.assertTrue(figure.axes[0].spines["top"].get_visible())
            self.assertFalse(figure.texts)
        finally:
            plot.get_pyplot().close(figure)

    def test_paper_normalization_rejects_zero_baseline_and_silent_clipping(self):
        for carbink, starfish in ((0, 0), (100, 130)):
            data = self.prepare(self.rows())
            data["values"]["NQ/carbink/local_ec_cpu_cycles"]["value"] = carbink
            data["values"]["NQ/starfish/local_ec_cpu_cycles"]["value"] = starfish
            with self.assertRaises(ValueError):
                plot.draw(data)


if __name__ == "__main__":
    unittest.main()
