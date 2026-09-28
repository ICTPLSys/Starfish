from pathlib import Path
import sys
import tempfile
import unittest

AE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(AE / "scripts/figure11"))
from figure11 import collect, plot
from test_figure11_collection import record, write_records


class Figure11Plot(unittest.TestCase):
    def rows(self, *, unmatched=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_records(root, [
                record(system, system + "-r",
                       environment="site-b" if unmatched and system == "carbink" else "site-a")
                for system in ("nonft", "starfish", "hydra", "carbink")
            ])
            return collect.collect_rows(root, allow_unmatched_environments=unmatched)

    def prepare(self, rows, **kwargs):
        return plot.prepare_rows(list(enumerate(rows, 1)), "test", "fixture", **kwargs)

    def test_component_selection_preserves_provenance(self):
        data = self.prepare(self.rows(), components="remote_memory", workloads="kv-b")
        self.assertEqual(len(data["values"]), 4)
        self.assertEqual(data["missing_conditions"], [])
        self.assertEqual(data["workloads"], ("kv_b",))
        item = data["values"]["kv_b/carbink/remote_memory"]
        self.assertEqual(item["aggregation"], "mean")
        self.assertEqual(item["samples"], "5")
        self.assertEqual(item["metric"], "live_stripe_allocated_bytes")
        self.assertIn("raw_value", item)

    def test_legacy_peak_and_work_only_cpu_are_rejected(self):
        for component, field, wrong in (
            ("remote_memory", "aggregation", "peak"),
            ("remote_memory", "samples", "0"),
            ("remote_memory", "metric", "rss"),
            ("remote_cpu_cores", "window", "work"),
        ):
            rows = self.rows()
            row = next(r for r in rows if r["component"] == component)
            row[field] = wrong
            with self.subTest(field=field, wrong=wrong), self.assertRaises(ValueError):
                self.prepare(rows)

    def test_raw_arithmetic_and_missing_provenance_are_rejected(self):
        for field, wrong in (("value", "123"), ("normalizer", 0),
                             ("run_id", ""), ("environment_match", "")):
            rows = self.rows()
            rows[0][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare(rows)

    def test_explicit_zero_is_not_a_missing_cpu_sample(self):
        rows = self.rows()
        row = next(r for r in rows if r["component"] == "remote_cpu_cores")
        row.update(value="0", raw_value=0)
        data = self.prepare(rows, components="remote_cpu_cores", workloads="kv-b")
        self.assertEqual(len(data["values"]), 4)
        self.assertEqual(data["blank_rows"], [])

    def test_unmatched_comparison_remains_diagnostic(self):
        data = self.prepare(self.rows(unmatched=True), workloads="kv-b")
        self.assertTrue(data["diagnostic_only"])
        self.assertTrue(data["warnings"])
        self.assertEqual(
            data["values"]["kv_b/carbink/remote_memory"]["baseline_environment"],
            "site-a")

    def test_bad_or_empty_selection_fails(self):
        for kwargs in ({"components": "unknown"}, {"workloads": "kv-b,kv_b"},
                       {"workloads": "bfs"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(self.rows(), **kwargs)

    def test_log_path_passes_subset_and_environment_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_records(root, [
                record(s, s, components=("remote_memory",),
                       environment="b" if s == "carbink" else "a")
                for s in ("nonft", "starfish", "hydra", "carbink")
            ])
            data = plot.prepare_logs(root, components="remote_memory", workloads="kv-b",
                                     allow_unmatched_environments=True)
            self.assertEqual(len(data["values"]), 4)
            self.assertTrue(data["diagnostic_only"])

    def test_selected_panel_renders_only_selected_bars(self):
        for components, expected_axes in (("remote_memory", 1),
                                          ("fetch_traffic,eviction_traffic,remote_memory", 2)):
            data = self.prepare(self.rows(), workloads="kv-b", components=components)
            fig = plot.draw(data)
            try:
                self.assertEqual(len(fig.axes), expected_axes)
                self.assertEqual(len(fig.axes[-1].patches), 4)
                fig.canvas.draw()
            finally:
                plot.get_pyplot().close(fig)


if __name__ == "__main__":
    unittest.main()
