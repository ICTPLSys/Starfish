#!/usr/bin/env python3
"""Plot evaluation fault-tolerance metrics from explicit CSV data."""

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

REQUIRED = {"workload", "system", "component", "value", "unit", "source_type", "source"}
WORKLOADS = ("bfs", "llama", "mg", "wc", "kv_b", "kv_a", "kv_s", "nq")
SYSTEMS = ("hydra", "carbink", "starfish", "nonft")
ALIASES = {"bfs": "bfs", "llm": "llama", "llama": "llama", "mg": "mg",
           "wc": "wc", "kv-b": "kv_b", "kv-a": "kv_a", "kv-s": "kv_s",
           "kv_b": "kv_b", "kv_a": "kv_a", "kv_s": "kv_s",
           "kvs ycsb-b": "kv_b", "kvs ycsb-a": "kv_a",
           "kvs synthetic": "kv_s", "nq": "nq"}
SYSTEM_ALIASES = {"hydra": "hydra", "carbink": "carbink", "starfish": "starfish",
                  "non-ft": "nonft", "nonft": "nonft"}
COMPONENTS = {"fetch_traffic": "x", "eviction_traffic": "x",
              "remote_cpu_cores": "cores", "remote_memory": "x"}
SOURCE_TYPES = {"measured", "paper_reference", "synthetic"}
TITLES = {"bfs": "BFS", "llama": "LLM", "mg": "MG", "wc": "WC",
          "kv_b": "KV-B", "kv_a": "KV-A", "kv_s": "KV-S", "nq": "NQ"}


def prepare(path: Path, source_type="measured"):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    numbered_rows, digest = read_csv(path, REQUIRED)
    values = {}
    blanks = []
    for line, row in numbered_rows:
        try:
            workload = ALIASES[row["workload"].lower()]
            system = SYSTEM_ALIASES[row["system"].lower()]
            component = row["component"].lower()
            if component not in COMPONENTS:
                raise ValueError(f"unknown component: {component}")
            if row["unit"] != COMPONENTS[component]:
                raise ValueError(f"{component} must have unit {COMPONENTS[component]}")
            key = workload, system, component
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
    missing = [[w, s, c] for w in WORKLOADS for s in SYSTEMS for c in COMPONENTS
               if (w, s, c) not in values]
    return {"figure": "figure11", "input": str(path.resolve()), "input_sha256": digest,
            "source_type": source_type, "components": COMPONENTS,
            "input_rows": len(numbered_rows), "blank_rows": blanks,
            "missing_conditions": missing,
            "values": {"/".join(key): record for key, record in values.items()}}


def panel_axis(max_value):
    """Original paper's y-axis choices, including per-app heterogeneous ranges."""
    if max_value <= 2:
        return (0, 2), [0, 1, 2]
    if max_value <= 3:
        return (0, 3), [0, 1.5, 3]
    if max_value <= 4:
        return (0, 4), [0, 2, 4]
    if max_value <= 8:
        return (0, 8), [0, 4, 8]
    if max_value <= 16:
        return (0, 16), [0, 8, 16]
    if max_value <= 45:
        return (0, 45), [0, 20, 40]
    return (0, max_value * 1.15), [0, max_value * 0.5, max_value]


def blend_color(color, amount=0.62):
    base = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    mixed = np.clip(base * (1 - amount) + 255 * amount, 0, 255).astype(int)
    return "#" + "".join(f"{part:02x}" for part in mixed)


def draw(data):
    plt = get_pyplot()
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, len(WORKLOADS), figsize=(17.2, 6.35),
                             sharex=False, sharey=False)
    x_positions = (np.arange(len(SYSTEMS)) - (len(SYSTEMS) - 1) / 2) * 0.20
    colors = [SYSTEM_STYLES[s]["facecolor"] for s in SYSTEMS]
    row_metrics = (("fetch_traffic", "eviction_traffic"),
                   ("remote_cpu_cores",), ("remote_memory",))
    row_labels = ("Traffic\n(x)", "CPU\ncores", "Memory\n(x)")
    for metric_idx, components in enumerate(row_metrics):
        for app_idx, workload in enumerate(WORKLOADS):
            ax = axes[metric_idx, app_idx]
            totals = []
            for s_idx, system in enumerate(SYSTEMS):
                records = [data["values"].get(f"{workload}/{system}/{component}")
                           for component in components]
                # Traffic must have both components; a partial stack is not a total.
                if any(record is None for record in records):
                    continue
                bottom = 0.0
                for component_idx, record in enumerate(records):
                    value = record["value"]
                    facecolor = (blend_color(colors[s_idx])
                                 if metric_idx == 0 and component_idx == 0
                                 else colors[s_idx])
                    bars = ax.bar(x_positions[s_idx], value, bottom=bottom,
                                  width=0.15, facecolor=facecolor,
                                  edgecolor="black", linewidth=0.65,
                                  hatch=SYSTEM_STYLES[system]["hatch"])
                    if system == "nonft":
                        bar = bars[0]
                        y0, y1 = bar.get_y(), bar.get_y() + bar.get_height()
                        x0, x1 = bar.get_x(), bar.get_x() + bar.get_width()
                        for y in np.linspace(y0 + (y1 - y0) * .25,
                                             y0 + (y1 - y0) * .75, 3):
                            ax.plot([x0, x1], [y, y], color="black", linewidth=.72)
                    bottom += value
                totals.append(bottom)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.4)
            ax.set_xlim(-.52, .52)
            ax.set_xticks([])
            if metric_idx == 0:
                ax.set_title(TITLES[workload], fontsize=22, fontweight="bold", pad=4)
            if totals:
                ax.set_ylim(*panel_axis(max(totals))[0])
                ax.set_yticks(panel_axis(max(totals))[1])
                ax.grid(axis="y", linestyle="--", alpha=.5, linewidth=.8)
                ax.set_axisbelow(True)
            else:
                ax.set_yticks([])
            ax.tick_params(axis="y", labelsize=17, direction="in", length=0, pad=1.5)
            if app_idx == 0:
                ax.set_ylabel(row_labels[metric_idx], fontsize=22, labelpad=4)
    handles = [Patch(facecolor=colors[i], edgecolor="black",
                     hatch=SYSTEM_STYLES[s]["hatch"],
                     label=SYSTEM_STYLES[s]["label"])
               for i, s in enumerate(SYSTEMS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=4, frameon=False, fontsize=18)
    fig.subplots_adjust(left=.065, right=.995, bottom=.085, top=.76,
                        wspace=.30, hspace=.62)
    for row_idx, caption in enumerate(("(a) Network Traffic",
                                       "(b) Remote CPU Cores",
                                       "(c) Remote Memory Usage")):
        left = axes[0, 0].get_position().x0
        right = axes[0, -1].get_position().x1
        center = (left + right) / 2
        row_bottom = axes[row_idx, 0].get_position().y0
        fig.text(center, row_bottom - .032, caption, ha="center", va="top",
                 fontsize=20, fontweight="bold")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure11")
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
            stem = "figure11" + ("" if args.source_type == "measured"
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
