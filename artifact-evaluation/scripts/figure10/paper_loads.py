"""Figure 10 offered-load points, expressed as integer requests per second."""

# Source: Sponge-Data/figures/paper_figure_mockups.py, fig2_tail_latency().
# Cross-checked against papers/starfish-nsdi/images/eval/eval_figure_02.pdf.
# Copy ONLY the x-axis points, not the paper's P99 values or measurement claims.
# KV-B: paper Mops * 1_000_000. NQ: paper Kops * 1_000.
# Each system keeps its own nonuniform point set; axis ticks are not load points.
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
        "hydra": [678, 1356, 2034, 2713, 3052, 3391, 4069, 4340, 4612, 4883],
        "carbink": [
            780, 1550, 2300, 3000, 3350, 3700, 4150,
            4450, 4700, 5050, 5450, 5850, 6230,
        ],
        "starfish": [
            910, 1821, 2731, 3642, 4096, 4552, 5462, 6827, 7283,
            8194, 9104, 10925, 13657, 16388, 18209, 19600, 20800,
        ],
        "nonft": [
            1081, 2162, 3243, 4324, 4864, 5405, 6486, 8107, 8648,
            9729, 10810, 12971, 16214, 19457, 20538, 21619, 22700,
        ],
    },
}

PLOT_UNITS = {"kv-b": ("Mops", 1_000_000), "nq": ("Kops", 1_000)}
