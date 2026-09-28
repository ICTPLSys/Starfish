"""Figure 10 offered-load points, expressed as integer requests per second."""

# Paper point counts and KV-B grids: Sponge-Data/figures/paper_figure_mockups.py,
# fig2_tail_latency().
# Cross-checked against papers/starfish-nsdi/images/eval/eval_figure_02.pdf.
# Copy ONLY the x-axis points, not the paper's P99 values or measurement claims.
# KV-B: paper Mops * 1_000_000; axis ticks are not load points.
# NQ: retain each paper point count, space evenly from 1k QPS to the selected
# scan limit, and round to 100 QPS (half up). These are not paper load points.
# Hydra/Carbink scan limits are provisional, pending offered-load measurements.
OFFERED_LOAD_OPS = {
    "kv-b": {
        "hydra": [
            169000, 673000, 1339000, 1972000, 2515000, 2960000,
            3054000, 3297000, 3610000, 3733000, 3883000, 4011000,
        ],
        "carbink": [
            117000, 458000, 913000, 1345000, 1743000, 2222000, 2623000,
            3676000, 3977000, 4092000, 4211000, 4295000, 4438000,
        ],
        "starfish": [
            324000, 1297000, 2562000, 3819000, 5934000,
            7666000, 9468000, 12902000, 15147000, 17455000,
        ],
        "nonft": [
            362000, 1449000, 2862000, 4266000, 6630000,
            8562000, 10578000, 14415000, 16922000, 19500000,
        ],
    },
    "nq": {
        "hydra": [1000, 2400, 3900, 5300, 6800, 8200, 9700, 11100, 12600, 14000],
        "carbink": [
            1000, 2600, 4200, 5800, 7300, 8900, 10500,
            12100, 13700, 15300, 16800, 18400, 20000,
        ],
        "starfish": [
            1000, 4700, 8400, 12100, 15800, 19400, 23100, 26800, 30500,
            34200, 37900, 41600, 45300, 48900, 52600, 56300, 60000,
        ],
        "nonft": [
            1000, 5900, 10900, 15800, 20800, 25700, 30600, 35600, 40500,
            45400, 50400, 55300, 60300, 65200, 70100, 75100, 80000,
        ],
    },
}

PLOT_UNITS = {"kv-b": ("Mops", 1_000_000), "nq": ("Kops", 1_000)}
