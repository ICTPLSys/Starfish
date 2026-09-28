"""Offline regression checks for Figure 10's NQ load-axis coverage."""
import importlib.util
from pathlib import Path
import unittest


FIGURE10 = Path(__file__).resolve().parents[2] / "figure10"
spec = importlib.util.spec_from_file_location(
    "figure10_axis_plot", FIGURE10 / "plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class NqLoadAxis(unittest.TestCase):
    def draw(self, max_load, *, latency=10.0):
        series = {f"{workload}/{system}": []
                  for workload in plot.UNITS for system in plot.ORDER}
        if max_load is not None:
            series["nq/nonft"] = [
                {"offered_load": 1.0, "p99_latency": 2.0},
                {"offered_load": max_load, "p99_latency": latency},
            ]
        figure = plot.draw({"series": series})
        self.addCleanup(plot.get_pyplot().close, figure)
        return figure

    def test_current_80k_grid_is_visible_and_kv_axis_is_unchanged(self):
        figure = self.draw(80.0)
        kv, nq = figure.axes
        self.assertEqual(list(nq.get_xticks()), [0, 20, 40, 60, 80])
        self.assertGreater(nq.get_xlim()[1], 80.0)
        self.assertEqual(kv.get_xlim(), (0.0, 20.5))
        self.assertEqual(list(kv.get_xticks()), [0, 5, 10, 15, 20])
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(nq.get_xlabel(), "Offered load (Kops)")

    def test_future_grid_beyond_80k_is_not_clipped(self):
        self.assertGreater(self.draw(125.0).axes[1].get_xlim()[1], 125.0)

    def test_missing_latency_keeps_its_load_position_visible(self):
        nq = self.draw(80.0, latency=None).axes[1]
        self.assertGreater(nq.get_xlim()[1], 80.0)

    def test_empty_nq_panel_has_a_valid_axis(self):
        nq = self.draw(None).axes[1]
        self.assertEqual(nq.get_xlim()[0], 0.0)
        self.assertGreater(nq.get_xlim()[1], 0.0)
        self.assertEqual(len(nq.lines), 0)

    def test_paper_labels_font_and_geometry_are_preserved(self):
        figure = self.draw(20.0)
        kv, nq = figure.axes
        self.assertEqual(tuple(figure.get_size_inches()), (4.55, 2.35))
        self.assertEqual(kv.get_title(), r"$\mathbf{(a)}$ $\mathbf{KVS}$ YCSB-B")
        self.assertEqual(nq.get_title(), r"$\mathbf{(b)}$ $\mathbf{Nhop}$")
        self.assertEqual(kv.get_ylabel(), "P99 latency (us)")
        self.assertEqual(nq.get_ylabel(), "P99 latency (ms)")
        self.assertEqual(kv.get_xlabel(), "Offered load (Mops)")
        self.assertEqual(nq.get_xlabel(), "Offered load (Kops)")
        self.assertEqual(kv.yaxis.label.get_fontname(), "Times New Roman")
        self.assertEqual(kv.yaxis.label.get_fontsize(), 12.5)
        self.assertEqual(kv.title.get_fontsize(), 13.8)
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            ["Hydra", "Carbink", "Starfish", "Non-FT"],
        )


if __name__ == "__main__":
    unittest.main()
