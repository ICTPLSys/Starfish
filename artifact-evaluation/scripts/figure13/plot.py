#!/usr/bin/env python3
"""Plot evaluation failure recovery from explicit CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
AE_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "common"))
from plotting import SYSTEM_STYLES, export_figure, get_pyplot, read_csv

SYSTEMS = ("hydra", "carbink", "starfish")
ALIASES = {"hydra": "hydra", "carbink": "carbink", "starfish": "starfish"}
SCENARIOS = ("1-node", "2-node")
SOURCE_TYPES = {"measured", "paper_reference", "synthetic"}
RECOVERY_FIELDS = {"scenario", "system", "recovery_s", "source_type", "source"}
TRACE_FIELDS = {"system", "elapsed_s", "throughput_ops_per_s", "event",
                "normalization", "reference_ops_per_s", "source_type", "source"}
NORMALIZATION = "starfish_pre_failure"


def check_source(row, source_type):
    if row["source_type"] != source_type or not row["source"]:
        raise ValueError("nonblank values/events require the selected source_type and source")
    if "exit_status" in row and row["exit_status"] != "0":
        raise ValueError("failed runs cannot be plotted")
    if "correctness" in row and row["correctness"].lower() != "pass":
        raise ValueError("correctness must be pass")


def prepare(recovery_csv: Path, throughput_csv: Path, source_type="measured"):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    recovery_rows, recovery_digest = read_csv(recovery_csv, RECOVERY_FIELDS)
    trace_rows, trace_digest = read_csv(throughput_csv, TRACE_FIELDS)
    recovery = {}
    blank_recovery = []
    for line, row in recovery_rows:
        try:
            scenario = row["scenario"].lower()
            system = ALIASES[row["system"].lower()]
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown scenario: {scenario}")
            key = scenario, system
            if key in recovery or key in blank_recovery:
                raise ValueError(f"duplicate recovery condition: {key}")
            if not row["recovery_s"]:
                blank_recovery.append(key)
                continue
            value = float(row["recovery_s"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("recovery_s must be finite and positive")
            check_source(row, source_type)
            recovery[key] = {"recovery_s": value, "source": row["source"]}
        except (KeyError, ValueError) as exc:
            raise ValueError(f"recovery CSV line {line}: {exc}") from exc

    traces = {system: [] for system in SYSTEMS}
    events = {"failure": None, "recovered": {}}
    seen_samples = set()
    reference_ops_per_s = None
    for line, row in trace_rows:
        try:
            system = row["system"].lower()
            if system not in (*SYSTEMS, "all"):
                raise ValueError(f"unknown system: {system}")
            event = row["event"].lower() or "none"
            if event not in ("none", "failure", "recovered"):
                raise ValueError(f"unknown event: {event}")
            if row["normalization"] != NORMALIZATION:
                raise ValueError(f"normalization must be {NORMALIZATION}")
            denominator = float(row["reference_ops_per_s"])
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("reference_ops_per_s must be finite and positive")
            if reference_ops_per_s is None:
                reference_ops_per_s = denominator
            elif not math.isclose(denominator, reference_ops_per_s,
                                  rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("all series must use one normalization denominator")
            elapsed = float(row["elapsed_s"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("elapsed_s must be finite and nonnegative")
            if event == "failure":
                if system != "all" or events["failure"] is not None:
                    raise ValueError("one global failure event requires system=all")
                check_source(row, source_type)
                events["failure"] = elapsed
            elif event == "recovered":
                if system == "all" or system in events["recovered"]:
                    raise ValueError("one recovered event per system is allowed")
                check_source(row, source_type)
                events["recovered"][system] = elapsed
            if system == "all":
                if event != "failure":
                    raise ValueError("system=all is reserved for the failure event")
                if row["throughput_ops_per_s"]:
                    raise ValueError("a global event cannot carry a throughput sample")
                continue
            key = system, elapsed
            if key in seen_samples:
                raise ValueError(f"duplicate throughput sample: {key}")
            seen_samples.add(key)
            value = None
            raw = None
            if row["throughput_ops_per_s"]:
                raw = float(row["throughput_ops_per_s"])
                if not math.isfinite(raw) or raw < 0:
                    raise ValueError("throughput_ops_per_s must be finite and nonnegative")
                check_source(row, source_type)
                value = raw / denominator
            traces[system].append({"elapsed_s": elapsed,
                                   "throughput_ops_per_s": raw,
                                   "normalized_throughput": value,
                                   "source": row["source"]})
        except (KeyError, ValueError) as exc:
            raise ValueError(f"throughput CSV line {line}: {exc}") from exc
    for system in SYSTEMS:
        traces[system].sort(key=lambda p: p["elapsed_s"])
    if events["failure"] is not None:
        for system, recovered_at in events["recovered"].items():
            if recovered_at <= events["failure"]:
                raise ValueError(f"{system} recovered before/at failure")
    return {"figure": "figure13", "source_type": source_type,
            "recovery_input": str(recovery_csv.resolve()),
            "recovery_sha256": recovery_digest, "recovery_rows": len(recovery_rows),
            "throughput_input": str(throughput_csv.resolve()),
            "throughput_sha256": trace_digest, "throughput_rows": len(trace_rows),
            "normalization": NORMALIZATION,
            "reference_ops_per_s": reference_ops_per_s,
            "recovery": {"/".join(key): value for key, value in recovery.items()},
            "missing_recovery": [[scenario, system] for scenario in SCENARIOS
                                 for system in SYSTEMS if (scenario, system) not in recovery],
            "blank_recovery": blank_recovery, "traces": traces, "events": events}


def draw(data):
    plt = get_pyplot()
    from matplotlib.patches import Patch
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.1))
    line_ax, bar_ax = axes
    colors = [SYSTEM_STYLES[s]["facecolor"] for s in SYSTEMS]
    markers = ["o", "s", "D"]
    linestyles = [":", "--", "-"]
    x = np.arange(len(SCENARIOS))
    width = .22
    offsets = (np.arange(len(SYSTEMS)) - (len(SYSTEMS) - 1) / 2) * width
    observed_recovery = []
    for s_idx, system in enumerate(SYSTEMS):
        for scenario_idx, scenario in enumerate(SCENARIOS):
            record = data["recovery"].get(f"{scenario}/{system}")
            if record is None:
                continue
            observed_recovery.append(record["recovery_s"])
            bar_ax.bar(x[scenario_idx] + offsets[s_idx], record["recovery_s"],
                       width=width, color=colors[s_idx], edgecolor="black",
                       linewidth=.6, hatch=SYSTEM_STYLES[system]["hatch"])
    bar_ax.set_box_aspect(.54)
    bar_ax.set_title("(b)", fontsize=15, fontweight="bold", pad=1.5)
    bar_ax.set_ylabel("Time (s)", fontsize=14, labelpad=2)
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(SCENARIOS, fontsize=12)
    if observed_recovery:
        upper = max(8, max(observed_recovery) * 1.15)
        bar_ax.set_ylim(0, upper)
        if upper == 8:
            bar_ax.set_yticks([0, 2, 4, 6, 8])
        bar_ax.grid(axis="y", linestyle="--", alpha=.5)
    else:
        bar_ax.set_yticks([])

    failure = data["events"]["failure"]
    recovered = data["events"]["recovered"]
    if failure is not None:
        line_ax.axvline(failure, color="#666666", linestyle="--", linewidth=.9)
        if recovered:
            line_ax.axvspan(failure, max(recovered.values()),
                            color="#f0f0f0", alpha=.85, zorder=0)
    for system, timestamp in recovered.items():
        line_ax.axvline(timestamp, color="#888888", linestyle="--", linewidth=.75)
    observed_t = []
    observed_y = []
    for s_idx, system in enumerate(SYSTEMS):
        points = data["traces"][system]
        if not any(p["normalized_throughput"] is not None for p in points):
            continue
        observed_t.extend(p["elapsed_s"] for p in points)
        observed_y.extend(p["normalized_throughput"] for p in points
                          if p["normalized_throughput"] is not None)
        line_ax.plot([p["elapsed_s"] for p in points],
                     [p["normalized_throughput"] if p["normalized_throughput"] is not None
                      else np.nan for p in points],
                     marker=markers[s_idx], linestyle=linestyles[s_idx],
                     linewidth=1.75, markersize=3.0, color=colors[s_idx],
                     label=SYSTEM_STYLES[system]["label"], zorder=2 + s_idx)
    line_ax.set_box_aspect(.54)
    line_ax.set_title("(a)", fontsize=15, fontweight="bold", pad=1.5)
    line_ax.set_xlabel("Time (s)", fontsize=14, labelpad=2)
    line_ax.set_ylabel("Norm.", fontsize=14, labelpad=2)
    if observed_t:
        low, high = min(observed_t), max(observed_t)
        if failure is not None:
            low, high = min(low, failure), max(high, failure)
        if recovered:
            low, high = min(low, *recovered.values()), max(high, *recovered.values())
        line_ax.set_xlim(low, high if high > low else low + 1)
        line_ax.set_ylim(0, max(1.08, max(observed_y) * 1.10))
        line_ax.grid(axis="y", linestyle="--", alpha=.5)
    else:
        line_ax.set_yticks([])
        event_times = ([failure] if failure is not None else []) + list(recovered.values())
        if event_times:
            line_ax.set_xlim(min(event_times) - 1, max(event_times) + 1)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
        ax.tick_params(axis="both", labelsize=12, direction="in", length=2.5, pad=1.2)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=colors[i], edgecolor="black",
                     hatch=SYSTEM_STYLES[system]["hatch"],
                     label=SYSTEM_STYLES[system]["label"])
               for i, system in enumerate(SYSTEMS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, frameon=False, fontsize=13.5,
               handlelength=1.05, handletextpad=.45, columnspacing=.72)
    fig.subplots_adjust(left=.125, right=.995, bottom=.20, top=.70, wspace=.40)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-csv", required=True, type=Path)
    parser.add_argument("--throughput-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure13")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="measured")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = prepare(args.recovery_csv, args.throughput_csv, args.source_type)
        if args.validate_only:
            print(f"validated {data['recovery_rows']} recovery and "
                  f"{data['throughput_rows']} throughput rows")
            return 0
        data["plot_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fig = draw(data)
        try:
            stem = "figure13" + ("" if args.source_type == "measured"
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
