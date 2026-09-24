"""Plot appendix benchmark-suite figures from explicit CSV files."""

import argparse
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "benchmark_suites")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

ACCESS_AMPLIFICATION_CSV = os.path.join(DATA_DIR, "access_amplification.csv")
DC_PERF_CDF_CSV = os.path.join(DATA_DIR, "dcperf_object_size_cdf_combined_fine.csv")
TAILBENCH_CDF_CSV = os.path.join(DATA_DIR, "tailbench_object_size_cdf_raw.csv")

DC_PERF_AMP_ORDER = ["MediaWiki", "Django", "FeedSim", "TaoBench", "Spark", "VideoTrans"]
TAILBENCH_AMP_ORDER = ["Xapian", "Masstree", "Moses", "Sphinx", "img-dnn", "SpecJBB", "Silo", "Shore"]

DC_PERF_CDF_ORDER = [
    "MediaWiki",
    "DjangoBench",
    "FeedSim",
    "TaoBench",
    "SparkBench",
    "VideoTranscodeBench",
]
DC_PERF_CDF_LABELS = {
    "MediaWiki": "MediaWiki",
    "DjangoBench": "Django",
    "FeedSim": "FeedSim",
    "TaoBench": "TaoBench",
    "SparkBench": "Spark",
    "VideoTranscodeBench": "VideoTrans",
}

TAILBENCH_CDF_ORDER = ["xapian", "masstree", "moses", "sphinx", "img-dnn", "specjbb", "silo", "shore"]
TAILBENCH_CDF_LABELS = {
    "xapian": "Xapian",
    "masstree": "Masstree",
    "moses": "Moses",
    "sphinx": "Sphinx",
    "img-dnn": "img-dnn",
    "specjbb": "SpecJBB",
    "silo": "Silo",
    "shore": "Shore",
}


def setup_style():
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_rate(value):
    return f"{value:.1f}x"


def plot_amplification_group(ax, data, labels, color, title):
    y = np.arange(len(labels))
    values = np.array([data.get(label, np.nan) for label in labels], dtype=float)
    valid = np.isfinite(values)

    ax.set_xscale("log")
    ax.set_xlim(0.9, 140)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()

    ax.hlines(y[valid], 1.0, values[valid], color=color, linewidth=5.0, alpha=0.82, zorder=3)
    ax.scatter(values[valid], y[valid], s=58, color=color, edgecolor="black", linewidth=0.7, zorder=4)
    ax.axvline(1.0, color="#444444", linewidth=1.0, zorder=2)

    for yi, value in zip(y, values):
        if not np.isfinite(value):
            continue  # A missing application is blank, not a made-up 1.0x.
        ax.text(value * 1.13, yi, format_rate(value), ha="left", va="center", fontsize=10.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=3)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12, direction="in", length=4)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def generate_access_amplification():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(ACCESS_AMPLIFICATION_CSV)
    if not {"suite", "application", "amplification_rate"}.issubset(df.columns):
        raise ValueError("access CSV lacks suite/application/amplification_rate")
    if df.duplicated(["suite", "application"]).any():
        raise ValueError("access CSV has duplicate suite/application rows")
    dcperf = dict(
        zip(
            df[df["suite"] == "DCPerf"]["application"],
            df[df["suite"] == "DCPerf"]["amplification_rate"],
        )
    )
    tailbench = dict(
        zip(
            df[df["suite"] == "TailBench"]["application"],
            df[df["suite"] == "TailBench"]["amplification_rate"],
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.75, 2.55), sharex=True)
    plot_amplification_group(axes[0], dcperf, DC_PERF_AMP_ORDER, "#4c78a8", "DCPerf")
    plot_amplification_group(axes[1], tailbench, TAILBENCH_AMP_ORDER, "#e7c66a", "TailBench")

    ticks = [1, 2, 4, 8, 16, 32, 64]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
    axes[0].set_ylabel("Application", fontsize=13, labelpad=3)
    fig.supxlabel("Amplification rate (log scale, x)", fontsize=13, y=0.02)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.24, wspace=0.38)

    pdf_path = os.path.join(OUTPUT_DIR, "casestudy_amprate.pdf")
    png_path = os.path.join(OUTPUT_DIR, "casestudy_amprate.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def format_bytes(value, _pos=None):
    value = float(value)
    if value < 1024:
        return f"{int(value)}B"
    if value < 1024**2:
        return f"{int(value / 1024)}KB"
    return f"{int(value / 1024**2)}MB"


def plot_cdf_group(ax, df, app_col, size_col, cdf_col, order, labels, title, colors):
    for app, color in zip(order, cycle(colors)):
        sub = df[df[app_col] == app].sort_values(size_col)
        if sub.empty:
            continue
        x = sub[size_col].to_numpy(dtype=float)
        y = sub[cdf_col].to_numpy(dtype=float)
        x_plot = np.r_[x[0], x]
        y_plot = np.r_[0.0, y]
        ax.step(x_plot, y_plot, where="post", linewidth=1.7, color=color, label=labels.get(app, app))

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.axvline(4096, color="#c92a2a", linestyle=(0, (4, 2)), linewidth=2.2, alpha=0.95, zorder=6)
    ax.set_title(title, fontsize=14.5, fontweight="bold", pad=42)
    ax.tick_params(axis="both", labelsize=10, direction="in", length=3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
        frameon=False,
        ncol=3 if len(order) <= 6 else 4,
        handlelength=1.5,
        columnspacing=0.8,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def generate_object_size_memory_cdf():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dcperf = pd.read_csv(DC_PERF_CDF_CSV)
    tailbench = pd.read_csv(TAILBENCH_CDF_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
    colors = plt.get_cmap("tab10").colors

    plot_cdf_group(
        axes[0],
        dcperf,
        app_col="application",
        size_col="object_size",
        cdf_col="byte_cdf",
        order=DC_PERF_CDF_ORDER,
        labels=DC_PERF_CDF_LABELS,
        title="DCPerf",
        colors=colors,
    )
    plot_cdf_group(
        axes[1],
        tailbench,
        app_col="app",
        size_col="bin_size",
        cdf_col="memory_bytes_cdf",
        order=TAILBENCH_CDF_ORDER,
        labels=TAILBENCH_CDF_LABELS,
        title="TailBench",
        colors=colors,
    )

    xticks = [16, 256, 4096, 65536, 1048576, 67108864]
    for ax in axes:
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
        ax.set_xlim(12, 100_000_000)
    axes[0].set_ylabel("CDF of total memory", fontsize=12)
    fig.supxlabel("Object size", fontsize=12, y=0.035)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.70, wspace=0.12)

    pdf_path = os.path.join(OUTPUT_DIR, "object_size_memory_cdf.pdf")
    png_path = os.path.join(OUTPUT_DIR, "object_size_memory_cdf.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    global DATA_DIR, OUTPUT_DIR, ACCESS_AMPLIFICATION_CSV, DC_PERF_CDF_CSV, TAILBENCH_CDF_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("access", "cdf", "both"))
    parser.add_argument("--data-dir", required=True,
                        help="CSV directory; no measurements are embedded in the plotter")
    parser.add_argument("--source-type", choices=("measured", "paper_reference"),
                        default="measured")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    DATA_DIR = os.path.abspath(args.data_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir or os.path.join(
        ROOT_DIR, "results", "figures", "appendix", args.source_type))
    ACCESS_AMPLIFICATION_CSV = os.path.join(DATA_DIR, "access_amplification.csv")
    DC_PERF_CDF_CSV = os.path.join(DATA_DIR, "dcperf_object_size_cdf_combined_fine.csv")
    TAILBENCH_CDF_CSV = os.path.join(DATA_DIR, "tailbench_object_size_cdf_raw.csv")
    if args.figure in ("access", "both"):
        print(f"Saved ({args.source_type}): {generate_access_amplification()}")
    if args.figure in ("cdf", "both"):
        print(f"Saved ({args.source_type}): {generate_object_size_memory_cdf()}")


if __name__ == "__main__":
    main()
