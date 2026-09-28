from pathlib import Path
import sys
import unittest


AE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AE / "scripts"))
sys.path.insert(0, str(AE / "scripts/figure11"))
from figure11 import plot


class Figure11PaperStyle(unittest.TestCase):
    def synthetic_data(self):
        rows = []
        for workload in plot.WORKLOADS:
            for system in plot.SYSTEMS:
                for component, unit in plot.COMPONENTS.items():
                    value = {
                        "fetch_traffic": 1.0,
                        "eviction_traffic": 2.0,
                        "remote_cpu_cores": 3.0,
                        "remote_memory": 4.0,
                    }[component]
                    rows.append({
                        "workload": workload, "system": system,
                        "component": component, "value": str(value),
                        "unit": unit, "source_type": "synthetic",
                        "source": "paper-style-fixture",
                    })
        return plot.prepare_rows(
            list(enumerate(rows, 1)), "synthetic", "fixture",
            source_type="synthetic",
        )

    def test_default_layout_and_paper_labels(self):
        data = self.synthetic_data()
        fig = plot.draw(data)
        try:
            self.assertEqual(tuple(fig.get_size_inches()), (17.2, 6.35))
            self.assertEqual(len(fig.axes), 24)
            self.assertEqual(
                [legend.get_texts()[0].get_text() for legend in fig.legends],
                ["Hydra"],
            )
            self.assertEqual(
                [text.get_text() for text in fig.legends[0].get_texts()],
                ["Hydra", "Carbink", "Starfish", "Non-FT"],
            )
            self.assertEqual(
                [text.get_text() for text in fig.texts],
                ["(a) Network Traffic", "(b) Remote CPU Cores",
                 "(c) Remote Memory Usage"],
            )
            self.assertAlmostEqual(fig.subplotpars.left, .065)
            self.assertAlmostEqual(fig.subplotpars.right, .995)
            self.assertAlmostEqual(fig.subplotpars.bottom, .085)
            self.assertAlmostEqual(fig.subplotpars.top, .76)
            self.assertAlmostEqual(fig.subplotpars.wspace, .30)
            self.assertAlmostEqual(fig.subplotpars.hspace, .62)
        finally:
            plot.get_pyplot().close(fig)

    def test_titles_rows_fonts_and_nonft_texture(self):
        fig = plot.draw(self.synthetic_data())
        try:
            expected_titles = [
                r"$\mathbf{BFS}$", r"$\mathbf{LLM}$", r"$\mathbf{MG}$",
                r"$\mathbf{WC}$", "$\\mathbf{KVS}$\nYCSB-B",
                "$\\mathbf{KVS}$\nYCSB-A", "$\\mathbf{KVS}$\nSynthetic",
                r"$\mathbf{NQ}$",
            ]
            self.assertEqual(
                [ax.get_title() for ax in fig.axes[:8]], expected_titles
            )
            self.assertEqual(fig.axes[0].get_ylabel(), "Traffic\n(x)")
            self.assertEqual(fig.axes[8].get_ylabel(), "CPU\ncores")
            self.assertEqual(fig.axes[16].get_ylabel(), "Memory\n(x)")
            self.assertEqual(plot.SYSTEM_STYLES["nonft"]["label"], "Non-FT")
            self.assertEqual(plot.SYSTEM_STYLES["nonft"]["hatch"], "")
            self.assertEqual(fig.axes[0].title.get_fontsize(), 24)
            self.assertEqual(fig.axes[0].yaxis.label.get_fontsize(), 30)
            self.assertEqual(fig.axes[0].yaxis.get_ticklabels()[0].get_fontsize(), 22)
            self.assertEqual(fig.legends[0].get_texts()[0].get_fontsize(), 22)
            self.assertEqual(fig.texts[0].get_fontsize(), 24)
            self.assertEqual(len(fig.axes[0].lines), 6)
            self.assertTrue(all(not patch.get_hatch() for patch in fig.axes[0].patches[-2:]))
            self.assertEqual(fig.axes[0].spines["top"].get_linewidth(), 1.4)
            edge = fig.axes[0].spines["right"].get_edgecolor()
            self.assertAlmostEqual(edge[0], 0x7a / 255)
            self.assertAlmostEqual(edge[1], 0x7a / 255)
            self.assertAlmostEqual(edge[2], 0x7a / 255)
        finally:
            plot.get_pyplot().close(fig)


if __name__ == "__main__":
    unittest.main()
