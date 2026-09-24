#!/usr/bin/env python3
"""Plot evaluation compute-node overhead from explicit CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
AE_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "common"))
from plotting import SYSTEM_STYLES, export_figure, get_pyplot, read_csv

REQUIRED = {"workload", "system", "metric", "value", "unit", "source_type", "source"}
WORKLOADS = ("BFS", "LLM", "MG", "WC", "KV-B", "KV-A", "KV-S", "NQ")
ALIASES = {"bfs": "BFS", "llm": "LLM", "llama": "LLM", "mg": "MG",
           "wc": "WC", "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S",
           "kvs ycsb-b": "KV-B", "kvs ycsb-a": "KV-A",
           "kvs synthetic": "KV-S", "nq": "NQ"}
SYSTEMS = ("carbink", "starfish")
METRICS = {"local_ec_cpu_norm": "x", "metadata_space_pct": "% app memory"}
SOURCE_TYPES = {"measured", "paper_reference", "synthetic"}


def prepare(path: Path, source_type="measured"):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    numbered_rows, digest = read_csv(path, REQUIRED)
    values = {}
    blanks = []
    for line, row in numbered_rows:
        try:
            workload = ALIASES[row["workload"].lower()]
            system = row["system"].lower()
            metric = row["metric"].lower()
            if system not in SYSTEMS or metric not in METRICS:
                raise ValueError(f"unknown system/metric: {system}/{metric}")
            if row["unit"] != METRICS[metric]:
                raise ValueError(f"{metric} requires unit {METRICS[metric]}")
            key = workload, system, metric
            if key in values or key in blanks:
                raise ValueError(f"duplicate condition: {key}")
            if not row["value"]:
                blanks.append(key)
                continue
            value = float(row["value"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("value must be finite and nonnegative")
            if row["source_type"] != source_type or not row["source"]:
                raise ValueError("nonblank values require the selected source_type and source")
            if "exit_status" in row and row["exit_status"] != "0":
                raise ValueError("failed runs cannot be plotted")
            if "correctness" in row and row["correctness"].lower() != "pass":
                raise ValueError("correctness must be pass")
            values[key] = {"value": value, "source": row["source"]}
        except (KeyError, ValueError) as exc:
            raise ValueError(f"CSV line {line}: {exc}") from exc
    return {"figure": "figure12", "input": str(path.resolve()), "input_sha256": digest,
            "source_type": source_type, "metric_units": METRICS,
            "input_rows": len(numbered_rows), "blank_rows": blanks,
            "missing_conditions": [[w, s, m] for w in WORKLOADS for s in SYSTEMS
                                   for m in METRICS if (w, s, m) not in values],
            "values": {"/".join(key): record for key, record in values.items()}}


def draw(data):
    plt = get_pyplot()
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.0), sharex=True)
    x = np.arange(len(WORKLOADS))
    width = .34
    labels = (("local_ec_cpu_norm", "Local EC computation CPU",
               "Norm. to Carbink", 1.2, [0, .5, 1]),
              ("metadata_space_pct", "Metadata space",
               "% app memory", 10.0, [0, 5, 10]))
    for metric_idx, (ax, (metric, title, ylabel, ymax, yticks)) in enumerate(zip(axes, labels)):
        observed = []
        for s_idx, system in enumerate(SYSTEMS):
            for app_idx, workload in enumerate(WORKLOADS):
                record = data["values"].get(f"{workload}/{system}/{metric}")
                if record is None:
                    continue  # Preserve an empty bar slot, never turn missing into zero.
                observed.append(record["value"])
                ax.bar(x[app_idx] + (-width / 2 if s_idx == 0 else width / 2),
                       record["value"], width=width,
                       color=SYSTEM_STYLES[system]["facecolor"],
                       edgecolor="black", linewidth=.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
        actual_max = max(observed, default=0)
        ax.set_ylim(0, max(ymax, actual_max * 1.1))
        if observed:
            if actual_max <= ymax:
                ax.set_yticks(yticks)
            else:
                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.grid(axis="y", linestyle="--", alpha=.5, linewidth=.8)
            ax.axhline(1.0, color="#555555", linewidth=.8, linestyle="--", zorder=0)
        else:
            ax.set_yticks([])
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=3)
        ax.set_title(title, fontsize=11, pad=2)
        ax.tick_params(axis="y", labelsize=9, direction="in", length=3, pad=1.5)
        ax.tick_params(axis="x", direction="in", length=3, pad=1.5)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(WORKLOADS, rotation=35, ha="right", fontsize=8.5)
    handles = [Patch(facecolor=SYSTEM_STYLES[s]["facecolor"], edgecolor="black",
                     label=SYSTEM_STYLES[s]["label"]) for s in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=2, frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=.20, right=.995, bottom=.20, top=.80, hspace=.58)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure12")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="measured")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = prepare(args.input, args.source_type)
        if args.validate_only:
            print(f"validated {data['input_rows']} rows")
            return 0
        data["plot_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fig = draw(data)
        try:
            stem = "figure12" + ("" if args.source_type == "measured"
                                else "-" + args.source_type)
            for output in export_figure(fig, args.output_dir, stem, data):
                print(f"wrote {output}")
        finally:
            get_pyplot().close(fig)
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
