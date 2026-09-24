#!/usr/bin/env python3
"""Plot evaluation tail latency from explicit CSV measurements."""

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

REQUIRED = {"workload", "system", "offered_load", "p99_latency", "load_unit",
            "latency_unit", "source_type", "source"}
WORKLOADS = {"kv-b": "kv_b", "kvs ycsb-b": "kv_b", "nq": "nq", "nhop": "nq"}
SYSTEMS = {"hydra": "hydra", "carbink": "carbink", "starfish": "starfish",
           "non-ft": "nonft", "nonft": "nonft"}
SOURCE_TYPES = {"measured", "paper_reference", "synthetic"}
UNITS = {"kv_b": ("Mops", "us"), "nq": ("Kops", "ms")}
ORDER = ("hydra", "carbink", "starfish", "nonft")


def canonical(value, mapping, field):
    try:
        return mapping[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown {field}: {value!r}") from exc


def prepare(path: Path, source_type="measured"):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    numbered_rows, digest = read_csv(path, REQUIRED)
    series = {(workload, system): [] for workload in UNITS for system in ORDER}
    keys = set()
    for line, row in numbered_rows:
        try:
            workload = canonical(row["workload"], WORKLOADS, "workload")
            system = canonical(row["system"], SYSTEMS, "system")
            if (row["load_unit"], row["latency_unit"]) != UNITS[workload]:
                raise ValueError(f"{workload} requires load/latency units {UNITS[workload]}")
            load = float(row["offered_load"])
            if not math.isfinite(load) or load < 0:
                raise ValueError("offered_load must be finite and nonnegative")
            key = workload, system, load
            if key in keys:
                raise ValueError(f"duplicate condition {key}; aggregate repeats explicitly")
            keys.add(key)
            latency = None
            if row["p99_latency"]:
                latency = float(row["p99_latency"])
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError("p99_latency must be finite and positive")
                if row["source_type"] != source_type or not row["source"]:
                    raise ValueError("nonblank values require the selected source_type and source")
                if "exit_status" in row and row["exit_status"] != "0":
                    raise ValueError("failed runs cannot be plotted")
                if "correctness" in row and row["correctness"].lower() != "pass":
                    raise ValueError("correctness must be pass")
            series[workload, system].append({"offered_load": load, "p99_latency": latency,
                                             "source": row["source"]})
        except ValueError as exc:
            raise ValueError(f"CSV line {line}: {exc}") from exc
    for points in series.values():
        points.sort(key=lambda point: point["offered_load"])
    return {"figure": "figure10", "input": str(path.resolve()), "input_sha256": digest,
            "source_type": source_type, "metric": "P99 latency", "units": UNITS,
            "series": {f"{workload}/{system}": series[workload, system]
                       for workload in UNITS for system in ORDER},
            "input_rows": len(numbered_rows),
            "missing_series": [f"{w}/{s}" for w in UNITS for s in ORDER
                               if not any(p["p99_latency"] is not None for p in series[w, s])]}


def draw(data):
    """Preserve the paper's two-panel axes, system order, markers and line styles."""
    plt = get_pyplot()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(4.55, 2.35), sharey=False)
    markers = ["o", "s", "D", "^"]
    linestyles = ["--", "-.", "-", ":"]
    colors = ["#e7c66a", "#b86b43", "#4c78a8", "#7aa37a"]
    for idx, workload in enumerate(UNITS):
        ax = axes[idx]
        has_values = any(point["p99_latency"] is not None
                         for system in ORDER
                         for point in data["series"][f"{workload}/{system}"])
        for s_idx, system in enumerate(ORDER):
            points = data["series"][f"{workload}/{system}"]
            if not any(point["p99_latency"] is not None for point in points):
                continue
            ax.plot([p["offered_load"] for p in points],
                    [p["p99_latency"] if p["p99_latency"] is not None else np.nan
                     for p in points],
                    marker=markers[s_idx], linestyle=linestyles[s_idx],
                    linewidth=1.55, markersize=4.0, color=colors[s_idx],
                    label=SYSTEM_STYLES[system]["label"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
        ax.set_axisbelow(True)
        if has_values:
            ax.grid(True, axis="y", linestyle="--", alpha=0.5, linewidth=1.0)
        ax.set_box_aspect(0.82)
        ax.set_title("(a) KV-B" if idx == 0 else "(b) NQ",
                     fontsize=13.8, fontweight="bold", pad=2)
        ax.tick_params(axis="x", labelsize=10.5, direction="in", length=2.5, pad=1.2)
        ax.tick_params(axis="y", labelsize=10.5, direction="in", length=2.5, pad=1.2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Offered load (Mops)" if idx == 0 else "Offered load (Kops)",
                      fontsize=12.5, labelpad=2)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(0, 20.5 if idx == 0 else 23.5)
        ax.set_ylim(1, 1000 if idx == 0 else 3000)
        if has_values:
            ax.set_yticks([1, 10, 100, 1000])
            ax.set_yticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"])
        else:
            ax.set_yticks([])
        ax.set_ylabel("P99 latency (us)" if idx == 0 else "P99 latency (ms)",
                      fontsize=12.5, labelpad=2)
    handles = [Line2D([0], [0], marker=markers[i], linestyle=linestyles[i],
                      color=colors[i], linewidth=1.55, markersize=4,
                      label=SYSTEM_STYLES[system]["label"])
               for i, system in enumerate(ORDER)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=10.5,
               handlelength=1.0, columnspacing=0.75)
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.21, top=0.76, wspace=0.46)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure10")
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
            stem = "figure10" + ("" if args.source_type == "measured"
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
