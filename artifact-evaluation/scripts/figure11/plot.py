#!/usr/bin/env python3
"""Plot evaluation fault-tolerance metrics from final logs or explicit CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
AE_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))
from plotting import SYSTEM_STYLES, export_figure, get_pyplot, read_csv
from figure11 import log_contract
SYSTEM_STYLES = dict(SYSTEM_STYLES)
SYSTEM_STYLES["nonft"] = dict(
    SYSTEM_STYLES.get("nonft-backup-off", SYSTEM_STYLES["nonft"]),
    label="Non-FT", hatch="")

REQUIRED = {"workload", "system", "component", "value", "unit", "source_type", "source"}
WORKLOADS = ("bfs", "llama", "mg", "wc", "kv_b", "kv_a", "kv_s", "nq")
SYSTEMS = ("hydra", "carbink", "starfish", "nonft")
ALIASES = {"bfs": "bfs", "llm": "llama", "llama": "llama", "mg": "mg",
           "wc": "wc", "kv-b": "kv_b", "kv-a": "kv_a", "kv-s": "kv_s",
           "kv_b": "kv_b", "kv_a": "kv_a", "kv_s": "kv_s",
           "kvs ycsb-b": "kv_b", "kvs ycsb-a": "kv_a",
           "kvs synthetic": "kv_s", "nq": "nq"}
SYSTEM_ALIASES = {"hydra": "hydra", "carbink": "carbink", "starfish": "starfish",
                  "non-ft": "nonft", "nonft": "nonft",
                  "nonft-backup-off": "nonft", "non-ft (backup off)": "nonft"}
COMPONENTS = {"fetch_traffic": "x", "eviction_traffic": "x",
              "remote_cpu_cores": "cores", "remote_memory": "x"}
SOURCE_TYPES = {"measured", "paper_reference", "synthetic"}
TITLES = {
    "bfs": r"$\mathbf{BFS}$",
    "llama": r"$\mathbf{LLM}$",
    "mg": r"$\mathbf{MG}$",
    "wc": r"$\mathbf{WC}$",
    "kv_b": "$\\mathbf{KVS}$\nYCSB-B",
    "kv_a": "$\\mathbf{KVS}$\nYCSB-A",
    "kv_s": "$\\mathbf{KVS}$\nSynthetic",
    "nq": r"$\mathbf{NQ}$",
}
EVAL_BASE_FIG_WIDTH = 14.4
EVAL_FONT = {
    "legend": 18,
    "panel_title": 20,
    "tick": 18,
    "axis_label": 25,
}


def scaled_eval_font(size, fig_width):
    # Avoid a binary 21.499999999999996 at the paper's 17.2-inch width.
    return int(round(size * fig_width / EVAL_BASE_FIG_WIDTH + 1e-9))


def _selection(value, allowed, aliases=None):
    if value is None:
        return tuple(allowed)
    parts = value.split(",") if isinstance(value, str) else list(value)
    parts = [str(part).strip().lower() for part in parts]
    if aliases:
        parts = [aliases.get(part, part) for part in parts]
    if not parts or any(part not in allowed for part in parts):
        raise ValueError("selection contains an empty or unknown entry")
    if len(parts) != len(set(parts)):
        raise ValueError("selection contains duplicates")
    return tuple(parts)


def prepare(path: Path, source_type="measured", *, components=None, workloads=None):
    numbered_rows, digest = read_csv(path, REQUIRED)
    return prepare_rows(numbered_rows, digest, str(path.resolve()), source_type,
                        components=components, workloads=workloads)


def prepare_logs(logs_root: Path, *, pattern="*.log", ratio=25, repeat=1,
                 components=None, workloads=None, allow_unmatched_environments=False):
    from figure11.collect import collect_rows
    rows = collect_rows(logs_root, pattern=pattern, ratio=ratio, repeat=repeat,
                        components=components,
                        allow_unmatched_environments=allow_unmatched_environments)
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    data = prepare_rows(list(enumerate(rows, 1)), digest,
                        str(logs_root.resolve()), "measured",
                        components=components, workloads=workloads)
    data.update(input_format="figure11_measured_logs",
                input_hash_scope="canonical_collected_rows",
                ratio=ratio, repeat=repeat, log_pattern=pattern,
                collected_rows=rows)
    return data


def _validate_measured(row, system, component):
    if system == "nonft":
        if (row.get("baseline_variant") != "nonft-backup-off"
                or log_contract.backup_flag(row.get("backup_enabled"))):
            raise ValueError("Figure 11 requires the explicit NonFT backup-off baseline")
    if component != "remote_cpu_cores" and row.get("normalizer_variant") != "nonft-backup-off":
        raise ValueError("Figure 11 normalization must use NonFT backup off")
    if str(row.get("exit_status")) != "0" or row.get("correctness") != "pass":
        raise ValueError("measured rows require successful, verified runs")
    expected_window = (log_contract.REMOTE_CPU_WINDOW
                       if component == "remote_cpu_cores" else
                       log_contract.REMOTE_MEMORY_WINDOW
                       if component == "remote_memory" else "work")
    if row.get("window") != expected_window:
        raise ValueError(f"{component} requires window={expected_window}")
    expected_aggregation = "mean" if component == "remote_memory" else "sum"
    if row.get("aggregation") != expected_aggregation:
        raise ValueError(f"{component} requires aggregation={expected_aggregation}")
    if component == "remote_memory":
        if (str(row.get("samples")) not in ("1", "2", "3", "4", "5")
                or row.get("sampling") != log_contract.REMOTE_MEMORY_SAMPLING
                or row.get("metric") != log_contract.MEMORY_METRICS[system]):
            raise ValueError("remote memory requires 1..5 benchmark-time samples")
        scheduled = log_contract._sample_times(
            row.get("scheduled_times_s"), "scheduled_times_s")
        observed = log_contract._sample_times(
            row.get("sample_times_s"), "sample_times_s")
        if (len(scheduled) != int(row["samples"]) or len(observed) != int(row["samples"])
                or scheduled != sorted(set(scheduled))
                or any(t not in (10, 20, 30, 40, 50) for t in scheduled)
                or observed != sorted(set(observed))
                or any(actual < target for actual, target in zip(observed, scheduled))):
            raise ValueError("invalid scheduled/observed memory sample times")
    for key in ("run_id", "environment", "workload_id"):
        if not row.get(key):
            raise ValueError(f"measured row is missing {key}")
    if row.get("environment_match") not in ("matched", "unmatched_allowed"):
        raise ValueError("measured row requires environment-match provenance")
    raw = float(row["raw_value"])
    normalizer = float(row["normalizer"])
    if not math.isfinite(raw) or raw < 0 or not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("invalid raw value or normalizer")
    if not math.isclose(float(row["value"]), raw / normalizer, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("plotted value disagrees with raw value / normalizer")


def prepare_rows(numbered_rows, digest, input_name, source_type="measured",
                 *, components=None, workloads=None):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    values = {}
    blanks = []
    selected_components = _selection(components, COMPONENTS)
    selected_workloads = _selection(workloads, WORKLOADS, ALIASES)
    warnings = set()
    for line, row in numbered_rows:
        try:
            workload = ALIASES[row["workload"].lower()]
            system = SYSTEM_ALIASES[row["system"].lower()]
            component = row["component"].lower()
            if component not in COMPONENTS:
                raise ValueError(f"unknown component: {component}")
            if component not in selected_components or workload not in selected_workloads:
                continue
            if row["unit"] != COMPONENTS[component]:
                raise ValueError(f"{component} must have unit {COMPONENTS[component]}")
            key = workload, system, component
            if key in values or key in blanks:
                raise ValueError(f"duplicate condition: {key}")
            if row["value"] is None or str(row["value"]).strip() == "":
                blanks.append(key)
                continue
            value = float(row["value"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("value must be finite and nonnegative")
            if row["source_type"] != source_type or not row["source"]:
                raise ValueError("nonblank values require the selected source_type and source")
            if "exit_status" in row and str(row["exit_status"]) != "0":
                raise ValueError("failed runs cannot be plotted")
            if "correctness" in row and row["correctness"].lower() != "pass":
                raise ValueError("correctness must be pass")
            if source_type == "measured":
                _validate_measured(row, system, component)
            if row.get("warning"):
                warnings.add(row["warning"])
            values[key] = dict(row, value=value)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"input row {line}: {exc}") from exc
    if not values:
        raise ValueError("no nonblank values in the selected workloads/components")
    missing = [[w, s, c] for w in selected_workloads for s in SYSTEMS for c in selected_components
               if (w, s, c) not in values]
    return {"figure": "figure11", "input": input_name, "input_sha256": digest,
            "source_type": source_type,
            "components": {c: COMPONENTS[c] for c in selected_components},
            "workloads": selected_workloads,
            "warnings": sorted(warnings),
            "diagnostic_only": any(record.get("environment_match") == "unmatched_allowed"
                                   for record in values.values()),
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


def style_axis(ax, grid_y=True):
    """Match the paper evaluation panels without changing plotted data."""
    for name, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        if name in ("top", "right"):
            spine.set_color("#7a7a7a")
    ax.set_axisbelow(True)
    if grid_y:
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, linewidth=1.0)
    ax.tick_params(direction="in", length=5, width=1)


def _mark_nonft(ax, system, bar):
    if system != "nonft" or not np.isfinite(bar.get_height()):
        return
    y0 = bar.get_y()
    y1 = y0 + bar.get_height()
    x0 = bar.get_x()
    x1 = x0 + bar.get_width()
    for y in np.linspace(y0 + (y1 - y0) * 0.25,
                         y0 + (y1 - y0) * 0.75, 3):
        ax.plot([x0, x1], [y, y], color="black", linewidth=0.72,
                clip_on=True)


def _row_caption(components):
    if components[0] in ("fetch_traffic", "eviction_traffic"):
        return "(a) Network Traffic"
    if components[0] == "remote_cpu_cores":
        return "(b) Remote CPU Cores"
    return "(c) Remote Memory Usage"


def draw(data):
    plt = get_pyplot()
    from matplotlib.patches import Patch

    workloads = data.get("workloads", WORKLOADS)
    selected = data["components"]
    row_metrics = tuple(tuple(c for c in group if c in selected) for group in
                        (("fetch_traffic", "eviction_traffic"),
                         ("remote_cpu_cores",), ("remote_memory",)))
    row_metrics = tuple(group for group in row_metrics if group)
    width = (17.2 if len(workloads) == len(WORKLOADS)
             else max(6.0, 2.15 * len(workloads)))
    height = (6.35 if len(workloads) == len(WORKLOADS) and len(row_metrics) == 3
              else max(4.0, 2.5 * len(row_metrics) + 1.1))
    fonts = {
        key: scaled_eval_font(value, width)
        for key, value in EVAL_FONT.items()
    }
    fig, axes = plt.subplots(len(row_metrics), len(workloads),
                             figsize=(width, height), squeeze=False,
                             sharex=False, sharey=False)
    x_positions = (np.arange(len(SYSTEMS)) - (len(SYSTEMS) - 1) / 2) * 0.20
    colors = [SYSTEM_STYLES[s]["facecolor"] for s in SYSTEMS]
    for metric_idx, components in enumerate(row_metrics):
        traffic = components[0] in ("fetch_traffic", "eviction_traffic")
        row_label = (
            "Traffic\n(x)" if traffic else
            "CPU\ncores" if components[0] == "remote_cpu_cores" else
            "Memory\n(x)"
        )
        for app_idx, workload in enumerate(workloads):
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
                                 if traffic and components[component_idx] == "fetch_traffic"
                                 else colors[s_idx])
                    bars = ax.bar(x_positions[s_idx], value, bottom=bottom,
                                  width=0.15, facecolor=facecolor,
                                  edgecolor="black", linewidth=0.65,
                                  hatch=SYSTEM_STYLES[system]["hatch"])
                    for bar in bars:
                        _mark_nonft(ax, system, bar)
                    bottom += value
                totals.append(bottom)
            style_axis(ax)
            ax.set_xlim(-.52, .52)
            ax.set_xticks([])
            if metric_idx == 0:
                ax.set_title(TITLES[workload], fontsize=fonts["panel_title"],
                             fontweight="normal", pad=4)
            if totals:
                ax.set_ylim(*panel_axis(max(totals))[0])
                ax.set_yticks(panel_axis(max(totals))[1])
            else:
                ax.set_yticks([])
            ax.tick_params(axis="y", labelsize=fonts["tick"], direction="in",
                           length=0, pad=1.5)
            ax.tick_params(axis="x", direction="in", length=2.5, pad=1.0)
            if app_idx == 0:
                ax.set_ylabel(row_label, fontsize=fonts["axis_label"], labelpad=4)
    handles = [Patch(facecolor=colors[i], edgecolor="black",
                     hatch=SYSTEM_STYLES[s]["hatch"],
                     label=SYSTEM_STYLES[s]["label"])
               for i, s in enumerate(SYSTEMS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=4, frameon=False, fontsize=fonts["legend"])
    fig.subplots_adjust(left=.065, right=.995, bottom=.085, top=.76,
                        wspace=.30, hspace=.62)
    row_captions = [_row_caption(components) for components in row_metrics]
    left = axes[0, 0].get_position().x0
    right = axes[0, -1].get_position().x1
    center = (left + right) / 2
    for row_idx, caption in enumerate(row_captions):
        row_bottom = axes[row_idx, 0].get_position().y0
        fig.text(center, row_bottom - 0.032, caption, ha="center", va="top",
                 fontsize=fonts["panel_title"], fontweight="bold")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="read an existing measurement CSV")
    inputs.add_argument("--logs-root", type=Path, help="read figure11_result records directly")
    parser.add_argument("--pattern", default="*.log", help="log filename glob for --logs-root")
    parser.add_argument("--ratio", type=int, default=25, help="local-memory percentage in log mode")
    parser.add_argument("--repeat", type=int, default=1, help="one repeat to select in log mode")
    parser.add_argument("--components", help="comma-separated measured subset; default all four")
    parser.add_argument("--workloads", help="comma-separated workload panels; default all eight")
    parser.add_argument("--allow-unmatched-environments", action="store_true",
                        help="log mode only: preserve unmatched diagnostic comparisons")
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure11")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="measured")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.logs_root is not None:
            if args.source_type != "measured":
                raise ValueError("--logs-root accepts measured records only")
            data = prepare_logs(args.logs_root, pattern=args.pattern,
                                ratio=args.ratio, repeat=args.repeat,
                                components=args.components, workloads=args.workloads,
                                allow_unmatched_environments=args.allow_unmatched_environments)
        else:
            if args.allow_unmatched_environments:
                raise ValueError("--allow-unmatched-environments applies only to --logs-root")
            data = prepare(args.input, args.source_type,
                           components=args.components, workloads=args.workloads)
        if data["warnings"]:
            for warning in data["warnings"]:
                print(f"warning: {warning}", file=sys.stderr)
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
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
