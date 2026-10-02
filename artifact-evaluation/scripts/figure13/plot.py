#!/usr/bin/env python3
"""Plot evaluation failure recovery from final logs or explicit CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
AE_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
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
DEFINITIONS = {"background_start_to_done", "failure_to_reconstruction"}


def check_source(row, source_type):
    if row["source_type"] != source_type or not row["source"]:
        raise ValueError("nonblank values/events require the selected source_type and source")
    if "exit_status" in row and str(row["exit_status"]) != "0":
        raise ValueError("failed runs cannot be plotted")
    if "correctness" in row and row["correctness"].lower() != "pass":
        raise ValueError("correctness must be pass")


def prepare(recovery_csv: Path | None, throughput_csv: Path | None, source_type="measured",
            **selection):
    recovery_rows, recovery_digest = (read_csv(recovery_csv, RECOVERY_FIELDS)
                                      if recovery_csv else ([], None))
    trace_rows, trace_digest = (read_csv(throughput_csv, TRACE_FIELDS)
                               if throughput_csv else ([], None))
    return prepare_rows(recovery_rows, trace_rows, recovery_digest, trace_digest,
                        str(recovery_csv.resolve()) if recovery_csv else None,
                        str(throughput_csv.resolve()) if throughput_csv else None,
                        source_type, **selection)


def prepare_logs(logs_root: Path, *, pattern="*.log", ratio=25, repeat=1,
                 trace_scenario="1-node", failure_at_s=20.0, run_results=None,
                 steady_before_s=10.0, systems=SYSTEMS, scenarios=SCENARIOS, panels="both"):
    from figure13.collect import collect_rows, collect_native
    options = dict(ratio=ratio, repeat=repeat, trace_scenario=trace_scenario,
                   failure_at_s=failure_at_s, panels=panels)
    if run_results:
        recovery, trace, info = collect_native(run_results, steady_before_s=steady_before_s,
                                               **options)
    else:
        recovery, trace, info = collect_rows(logs_root, pattern=pattern, **options)
    def digest(rows):
        return hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    name = ";".join(str(Path(p).resolve()) for p in run_results) if run_results else str(logs_root.resolve())
    data = prepare_rows(list(enumerate(recovery, 1)), list(enumerate(trace, 1)),
                        digest(recovery), digest(trace), name, name, "measured",
                        systems=systems, scenarios=scenarios, panels=panels)
    data.update(info)
    data.update(input_format="native_recovery_result" if run_results else "figure13_result_and_sample_logs",
                input_hash_scope="canonical_collected_rows", log_pattern=pattern,
                collected_recovery_rows=recovery, collected_throughput_rows=trace)
    return data


def prepare_rows(recovery_rows, trace_rows, recovery_digest, trace_digest,
                 recovery_name, trace_name, source_type="measured",
                 *, systems=SYSTEMS, scenarios=SCENARIOS, panels="both"):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    systems, scenarios = tuple(systems), tuple(scenarios)
    for values, allowed, name in ((systems, SYSTEMS, "systems"), (scenarios, SCENARIOS, "scenarios")):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError(f"select distinct known {name}")
    if panels not in ("both", "recovery", "throughput"):
        raise ValueError("unknown panel selection")
    recovery, definitions, comparison_kinds = {}, set(), set()
    blank_recovery = []
    for line, row in recovery_rows:
        try:
            scenario = row["scenario"].lower()
            system = ALIASES[row["system"].lower()]
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown scenario: {scenario}")
            if system not in systems or scenario not in scenarios or panels == "throughput":
                continue
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
            comparison_kinds.add(row.get("comparison_kind") or "unspecified")
            definition = row.get("recovery_definition") or "failure_to_reconstruction"
            if definition not in DEFINITIONS:
                raise ValueError("unknown recovery timing definition")
            definitions.add(definition)
            start_key = ("rebuild_start_elapsed_s" if definition == "background_start_to_done"
                         else "failure_elapsed_s")
            if definition == "background_start_to_done" and (
                    not row.get(start_key) and str(row.get(start_key)) != "0"):
                raise ValueError("background duration requires rebuild_start_elapsed_s")
            if start_key in row and row.get(start_key) != "":
                duration = float(row["recovered_elapsed_s"]) - float(row[start_key])
                if not math.isclose(value, duration, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("recovery duration disagrees with recorded native boundaries")
            recovery[key] = {"recovery_s": value, "source": row["source"]}
        except (KeyError, ValueError) as exc:
            raise ValueError(f"recovery CSV line {line}: {exc}") from exc

    traces = {system: [] for system in systems}
    events = {"failure": None, "recovered": {}, "rebuild_start": {}}
    seen_samples = set()
    reference_ops_per_s = None
    time_alignment = None
    for line, row in trace_rows:
        try:
            system = row["system"].lower()
            if system not in (*SYSTEMS, "all"):
                raise ValueError(f"unknown system: {system}")
            if panels == "recovery" or (system != "all" and system not in systems):
                continue
            comparison_kinds.add(row.get("comparison_kind") or "unspecified")
            if row.get("recovery_definition"):
                if row["recovery_definition"] not in DEFINITIONS:
                    raise ValueError("unknown recovery timing definition")
                definitions.add(row["recovery_definition"])
            event = row["event"].lower() or "none"
            if event not in ("none", "failure", "recovered", "rebuild_start"):
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
            alignment = row.get("time_alignment") or "work_start"
            if alignment not in ("work_start", "failure_aligned"):
                raise ValueError("unknown time_alignment")
            if time_alignment is None:
                time_alignment = alignment
            elif alignment != time_alignment:
                raise ValueError("mixed trace time alignments")
            elapsed = float(row["elapsed_s"])
            if (not math.isfinite(elapsed)
                    or (elapsed < 0 and alignment != "failure_aligned")):
                raise ValueError("negative elapsed_s requires explicit failure_aligned coordinates")
            if event == "failure":
                if system != "all" or events["failure"] is not None:
                    raise ValueError("one global failure event requires system=all")
                check_source(row, source_type)
                events["failure"] = elapsed
            elif event in ("recovered", "rebuild_start"):
                if system == "all" or system in events[event]:
                    raise ValueError(f"one {event} event per system is allowed")
                check_source(row, source_type)
                events[event][system] = elapsed
            if system == "all":
                if event != "failure":
                    raise ValueError("system=all is reserved for the failure event")
                if row["throughput_ops_per_s"]:
                    raise ValueError("a global event cannot carry a throughput sample")
                continue
            if event in ("recovered", "rebuild_start") and not row["throughput_ops_per_s"]:
                # Event-only annotations are not missing throughput samples.
                continue
            key = system, elapsed
            if key in seen_samples:
                raise ValueError(f"duplicate throughput sample: {key}")
            seen_samples.add(key)
            value = None
            raw = None
            duration = None
            if row.get("original_window_start_s") and row.get("original_window_end_s"):
                duration = float(row["original_window_end_s"]) - float(row["original_window_start_s"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("sample window duration must be positive")
            if row["throughput_ops_per_s"]:
                raw = float(row["throughput_ops_per_s"])
                if not math.isfinite(raw) or raw < 0:
                    raise ValueError("throughput_ops_per_s must be finite and nonnegative")
                check_source(row, source_type)
                value = raw / denominator
                if not math.isfinite(value):
                    raise ValueError("nonfinite normalized throughput")
            traces[system].append({"elapsed_s": elapsed,
                                   "throughput_ops_per_s": raw,
                                   "normalized_throughput": value,
                                   "window_duration_s": duration,
                                   "source": row["source"]})
        except (KeyError, ValueError) as exc:
            raise ValueError(f"throughput CSV line {line}: {exc}") from exc
    for system in systems:
        traces[system].sort(key=lambda p: p["elapsed_s"])
    if events["failure"] is not None:
        for system, recovered_at in events["recovered"].items():
            if recovered_at <= events["failure"]:
                raise ValueError(f"{system} recovered before/at failure")
            if (system in events["rebuild_start"] and
                    events["rebuild_start"][system] >= recovered_at):
                raise ValueError("rebuild must start before completion")
    if len(definitions) > 1:
        raise ValueError("do not mix background duration and failure-to-completion bars")
    if len(comparison_kinds) > 1:
        raise ValueError("mixed comparison kinds")
    missing = [[scenario, system] for scenario in scenarios for system in systems
               if (scenario, system) not in recovery] if panels != "throughput" else []
    if missing:
        raise ValueError(f"missing recovery conditions: {missing}; select --systems/--scenarios "
                         "explicitly; missing is not zero")
    if panels != "recovery":
        if events["failure"] is None:
            raise ValueError("throughput panel requires the observed failure event")
        for system in systems:
            points = [p for p in traces[system] if p["normalized_throughput"] is not None]
            if not points or system not in events["recovered"]:
                raise ValueError(f"missing verified throughput/recovery events for {system}")
            if not (any(p["elapsed_s"] <= events["failure"] for p in points) and
                    any(p["elapsed_s"] >= events["recovered"][system] for p in points)):
                raise ValueError(f"{system} trace must include pre-failure and post-rebuild samples")
    return {"figure": "figure13", "source_type": source_type,
            "systems": systems, "scenarios": scenarios, "panels": panels,
            "recovery_definition": next(iter(definitions), None),
            "comparison_kind": next(iter(comparison_kinds), "unspecified"),
            "recovery_input": recovery_name,
            "recovery_sha256": recovery_digest, "recovery_rows": len(recovery_rows),
            "throughput_input": trace_name,
            "throughput_sha256": trace_digest, "throughput_rows": len(trace_rows),
            "normalization": NORMALIZATION,
            "time_alignment": time_alignment,
            "reference_ops_per_s": reference_ops_per_s,
            "recovery": {"/".join(key): value for key, value in recovery.items()},
            "missing_recovery": missing,
            "blank_recovery": blank_recovery, "traces": traces, "events": events}


def display_points(points, reference, average_windows):
    from collections import deque
    if average_windows == 1:
        return [(p["elapsed_s"], p["normalized_throughput"]) for p in points]
    history, result = deque(), []
    for point in points:
        if point["normalized_throughput"] is None:
            history.clear()
            result.append((point["elapsed_s"], None))
            continue
        duration = point["window_duration_s"]
        if duration is None:
            raise ValueError("weighted averaging requires original sample window durations")
        history.append((point["throughput_ops_per_s"] * duration, duration))
        if len(history) > average_windows:
            history.popleft()
        value = sum(item[0] for item in history) / sum(item[1] for item in history) / reference
        result.append((point["elapsed_s"], value))
    return result


def binned_points(points, reference, windows, boundaries):
    """Non-overlapping duration-weighted bins; never pool across events/gaps."""
    if windows == 1:
        return display_points(points, reference, 1)
    result, group = [], []
    previous_end, previous_segment = None, None

    def flush():
        if not group:
            return
        seconds = math.fsum(p["window_duration_s"] for p in group)
        completed = math.fsum(p["throughput_ops_per_s"] * p["window_duration_s"]
                              for p in group)
        result.append((group[-1]["elapsed_s"], completed / seconds / reference))
        group.clear()

    for point in points:
        end = point["elapsed_s"]
        if point["normalized_throughput"] is None:
            flush()
            result.append((end, None))
            previous_end = previous_segment = None
            continue
        duration = point["window_duration_s"]
        if duration is None or duration <= 0:
            raise ValueError("binning requires positive original window durations")
        start = end - duration
        segment = sum(start >= boundary - 1e-9 for boundary in boundaries)
        crosses = any(start + 1e-9 < boundary < end - 1e-9 for boundary in boundaries)
        if ((previous_end is not None and not math.isclose(start, previous_end, abs_tol=1e-8, rel_tol=0))
                or segment != previous_segment or crosses):
            flush()
        group.append(point)
        if len(group) == windows or crosses:
            flush()
        previous_end, previous_segment = end, segment
    flush()
    return result


def fixed_count_points(points, reference, count, low, high, boundaries):
    """Allocate complete native windows to exactly count visible means."""
    selected = [p for p in points if low <= p["elapsed_s"] <= high]
    segments, current, gaps = [], [], []
    previous_end, previous_segment = None, None

    def flush():
        if current:
            segments.append(list(current))
            current.clear()

    for point in selected:
        end = point["elapsed_s"]
        if point["normalized_throughput"] is None:
            flush()
            gaps.append((end, None))
            previous_end = previous_segment = None
            continue
        duration = point["window_duration_s"]
        if duration is None or duration <= 0:
            raise ValueError("fixed-count aggregation requires native window durations")
        start = end - duration
        segment = sum(start >= event - 1e-9 for event in boundaries)
        crosses = any(start + 1e-9 < event < end - 1e-9 for event in boundaries)
        if (segment != previous_segment or crosses or
                (previous_end is not None and not math.isclose(start, previous_end, abs_tol=1e-8, rel_tol=0))):
            flush()
        current.append(point)
        if crosses:
            flush()
        previous_end, previous_segment = end, segment
    flush()
    if not len(segments) <= count <= sum(map(len, segments)):
        raise ValueError("requested point count cannot preserve the observed events/gaps")
    bins = [1] * len(segments)
    for _ in range(count - len(segments)):
        index = max((i for i, segment in enumerate(segments) if bins[i] < len(segment)),
                    key=lambda i: len(segments[i]) / bins[i])
        bins[index] += 1
    result = list(gaps)
    for segment, parts in zip(segments, bins):
        for index in range(parts):
            group = segment[index * len(segment) // parts:(index + 1) * len(segment) // parts]
            seconds = math.fsum(p["window_duration_s"] for p in group)
            completed = math.fsum(p["throughput_ops_per_s"] * p["window_duration_s"] for p in group)
            result.append((group[-1]["elapsed_s"], completed / seconds / reference))
    return sorted(result, key=lambda point: point[0])


def draw_recovery(ax, data, scale):
    import numpy as np
    systems, scenarios = data["systems"], data["scenarios"]
    x, width = np.arange(len(scenarios)), .22
    offsets = (np.arange(len(systems)) - (len(systems) - 1) / 2) * width
    observed = []
    for s_idx, system in enumerate(systems):
        values = [data["recovery"][f"{scenario}/{system}"]["recovery_s"] for scenario in scenarios]
        observed.extend(values)
        ax.bar(x + offsets[s_idx], values, width=width,
               color=SYSTEM_STYLES[system]["facecolor"], edgecolor="black",
               linewidth=.6 * scale, hatch=SYSTEM_STYLES[system]["hatch"])
    ax.set_title(r"$\mathbf{(b)}$", fontsize=15 * scale, pad=1.5 * scale)
    ax.set_ylabel("Time (s)", fontsize=14 * scale, labelpad=2 * scale)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12 * scale)
    upper = max(8, max(observed) * 1.15)
    ax.set_ylim(0, upper)
    tick_step = 2 if upper <= 12 else math.ceil(upper / 5)
    ax.set_yticks(np.arange(0, upper + 1e-9, tick_step))


def draw_trace(ax, data, scale, view, average_windows, bin_windows=1, display_points_count=None):
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    failure, recovered = data["events"]["failure"], data["events"]["recovered"]
    all_times = [p["elapsed_s"] for points in data["traces"].values() for p in points]
    if view == "zoom":
        # Preserve the paper's 18..28 display around a failure at 20 when it
        # fits; extend in 2-second steps for longer observed recovery.
        low = failure - 2
        high = failure + max(8, 2 * math.ceil((max(recovered.values()) - failure + 1) / 2))
    else:
        low, high = min(all_times), max(all_times)
    ax.axvspan(failure, max(recovered.values()), color="#f0f0f0", alpha=.85, zorder=0)
    ax.axvline(failure, color="#666666", linestyle="--", linewidth=.9 * scale)
    ax.text(failure + .35, .58, "failure", fontsize=8.4 * scale, va="center")
    for system, timestamp in recovered.items():
        ax.axvline(timestamp, color="#666666" if system == "carbink" else "#888888",
                   linestyle="--", linewidth=(.9 if system == "carbink" else .75) * scale)
    visible = []
    markers, linestyles = ("o", "s", "D"), (":", "--", "-")
    for system in data["systems"]:
        points = (fixed_count_points(data["traces"][system], data["reference_ops_per_s"],
                                     display_points_count, low, high, (failure, recovered[system]))
                  if display_points_count is not None else
                  binned_points(data["traces"][system], data["reference_ops_per_s"],
                                bin_windows, (failure, recovered[system]))
                  if bin_windows > 1 else
                  display_points(data["traces"][system], data["reference_ops_per_s"], average_windows))
        data.setdefault("display_traces", {})[system] = [
            {"elapsed_s": time, "normalized_throughput": value} for time, value in points]
        visible.extend(value for time, value in points if low <= time <= high and value is not None)
        index = SYSTEMS.index(system)
        ax.plot([time for time, _ in points],
                [value if value is not None else np.nan for _, value in points],
                marker=markers[index], linestyle=linestyles[index],
                linewidth=(1.85 if system == "starfish" else 1.75) * scale,
                markersize=(3.2 if system == "hydra" else 3.0) * scale,
                color=SYSTEM_STYLES[system]["facecolor"],
                label=SYSTEM_STYLES[system]["label"], zorder=2 + index)
        event_time = recovered[system]
        for left, right in zip(points, points[1:]):
            if left[0] <= event_time <= right[0] and left[1] is not None and right[1] is not None:
                fraction = (event_time - left[0]) / (right[0] - left[0])
                marker_y = left[1] + fraction * (right[1] - left[1])
                ax.scatter([event_time], [marker_y], s=28 * scale * scale,
                           marker="o", color=SYSTEM_STYLES[system]["facecolor"],
                           edgecolor="black", linewidth=.45 * scale, zorder=6)
                ax.text(event_time + .35, marker_y + (.055 if system == "carbink" else -.095),
                        {"hydra": "H", "starfish": "S", "carbink": "C"}[system],
                        fontsize=8 * scale, color="black", va="center", zorder=7)
                break
    if not visible:
        raise ValueError("selected display window has no observed throughput")
    ax.set_title(r"$\mathbf{(a)}$", fontsize=15 * scale, pad=1.5 * scale)
    ax.set_xlabel("Time (s)", fontsize=14 * scale, labelpad=2 * scale)
    ax.set_ylabel("Norm.", fontsize=14 * scale, labelpad=2 * scale)
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    upper = max(1.08, max(visible) * 1.10)
    ax.set_ylim(0, upper)
    if upper <= 1.5:
        ax.set_yticks([0, .5, 1])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    data["display_xlim_s"] = [low, high]
    data["displayed_event_meaning"] = "failure; H/S/C: native full-rebuild completion"
    data["event_marker_y"] = "interpolated display-line position only; not a recovery-time estimator"


def draw(data, *, view="zoom", average_windows=1, bin_windows=1, display_points_count=None):
    plt = get_pyplot()
    from matplotlib.patches import Patch
    if (view not in ("zoom", "full") or average_windows < 1 or bin_windows < 1
            or (average_windows > 1 and bin_windows > 1)):
        raise ValueError("invalid view or averaging window")
    if display_points_count is not None and (
            display_points_count < 2 or average_windows != 1 or bin_windows != 1):
        raise ValueError("fixed-count display cannot be combined with another smoothing mode")
    scale = 1
    panels = data["panels"]
    count = 2 if panels == "both" else 1
    # Keep the original two-panel geometry. Standalone panels need enough
    # plotting area for the three-system legend, not a narrow axis under it.
    size = (4.55, 1.72) if count == 2 else (4.0, 2.8)
    fig, array = plt.subplots(1, count, squeeze=False,
                              figsize=(size[0] * scale, size[1] * scale))
    axes = array[0]
    if panels != "recovery":
        draw_trace(axes[0], data, scale, view, average_windows, bin_windows, display_points_count)
    if panels != "throughput":
        draw_recovery(axes[-1], data, scale)
    for ax in axes:
        ax.set_box_aspect(.54)
        for name, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.4 * scale)
            if name in ("top", "right"):
                spine.set_color("#7a7a7a")
        ax.tick_params(axis="both", labelsize=12 * scale, direction="in",
                       length=2.5 * scale, pad=1.2 * scale)
        ax.grid(axis="y", linestyle="--", alpha=.5, linewidth=scale)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=SYSTEM_STYLES[system]["facecolor"], edgecolor="black",
                     hatch=SYSTEM_STYLES[system]["hatch"],
                     label=SYSTEM_STYLES[system]["label"])
               for system in data["systems"]]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(.43 if count == 2 else .52, .995),
               ncol=len(handles), frameon=False, fontsize=13.5 * scale,
               handlelength=1.05, handletextpad=.45, columnspacing=.72)
    fig.subplots_adjust(left=.125 if count == 2 else .18, right=.995,
                        bottom=.18 if count == 2 else .19,
                        top=.66 if count == 2 else .74, wspace=.40)
    data.update(plot_style="paper", view=view, average_windows=average_windows,
                display_points_target=display_points_count,
                point_aggregation="none" if display_points_count is None else
                "fixed_count_nonoverlapping_native_windows_duration_weighted_reset_at_events_and_gaps",
                bin_windows=bin_windows,
                bin_method="none" if bin_windows == 1 else
                "nonoverlapping_completed_ops_over_actual_seconds_reset_at_events_and_gaps",
                bin_time_coordinate="last_native_window_end",
                average_method="none" if average_windows == 1 else
                "trailing_completed_ops_over_actual_window_seconds",
                reference_style_source="paper_figure_mockups.py:fig5_failure_recovery_onerow")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-csv", type=Path)
    parser.add_argument("--throughput-csv", type=Path)
    parser.add_argument("--logs-root", type=Path, help="read Figure 13 result/sample logs directly")
    parser.add_argument("--run-result", type=Path, action="append",
                        help="Explicit native recovery/revalidation result JSON; repeat per run.")
    parser.add_argument("--pattern", default="*.log", help="log filename glob in log mode")
    parser.add_argument("--ratio", type=int, default=25, help="local-memory percentage in log mode")
    parser.add_argument("--repeat", type=int, default=1, help="one repeat in log mode")
    parser.add_argument("--trace-scenario", choices=SCENARIOS, default="1-node")
    parser.add_argument("--failure-at-s", type=float, default=20,
                        help="display coordinate for aligned failures; never injection time")
    parser.add_argument("--steady-before-s", type=float, default=10)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--panels", choices=("both", "recovery", "throughput"), default="both")
    parser.add_argument("--view", choices=("zoom", "full"), default="zoom")
    parser.add_argument("--average-windows", type=int, default=1,
                        help="Display-only trailing duration-weighted average; 1 preserves raw windows.")
    parser.add_argument("--bin-windows", type=int, default=1)
    parser.add_argument("--display-points", type=int,
                        help="Exact number of event-aware display means per curve.")
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure13")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="measured")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        selection = dict(systems=tuple(value.strip().lower() for value in args.systems.split(",")),
                         scenarios=tuple(value.strip().lower() for value in args.scenarios.split(",")),
                         panels=args.panels)
        if args.average_windows < 1:
            raise ValueError("--average-windows must be positive")
        if args.logs_root and args.run_result:
            parser.error("use --logs-root or --run-result, not both")
        if args.logs_root is not None or args.run_result:
            if args.recovery_csv is not None or args.throughput_csv is not None:
                parser.error("use raw inputs or CSV inputs, not both")
            if args.source_type != "measured":
                raise ValueError("raw inputs accept measured records only")
            data = prepare_logs(
                args.logs_root, pattern=args.pattern, ratio=args.ratio, repeat=args.repeat,
                trace_scenario=args.trace_scenario, failure_at_s=args.failure_at_s,
                run_results=args.run_result, steady_before_s=args.steady_before_s, **selection)
        else:
            if args.panels != "throughput" and args.recovery_csv is None:
                parser.error("selected recovery panel requires --recovery-csv")
            if args.panels != "recovery" and args.throughput_csv is None:
                parser.error("selected throughput panel requires --throughput-csv")
            data = prepare(args.recovery_csv, args.throughput_csv, args.source_type, **selection)
        if args.average_windows > 1 and args.panels != "recovery":
            for points in data["traces"].values():
                display_points(points, data["reference_ops_per_s"], args.average_windows)
        if args.validate_only:
            print(f"validated {data['recovery_rows']} recovery and "
                  f"{data['throughput_rows']} throughput rows")
            return 0
        data["plot_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fig = draw(data, view=args.view, average_windows=args.average_windows,
                   bin_windows=args.bin_windows, display_points_count=args.display_points)
        try:
            stem = "figure13" + ("" if args.source_type == "measured"
                                else "-" + args.source_type)
            if data.get("comparison_kind") == "diagnostic":
                stem += "-diagnostic"
            if args.panels != "both":
                stem += "-" + args.panels
            if args.view == "full":
                stem += "-full"
            if args.average_windows != 1:
                stem += f"-avg{args.average_windows}"
            if args.bin_windows != 1:
                stem += f"-bin{args.bin_windows}"
            if args.display_points is not None:
                stem += f"-points{args.display_points}"
            for output in export_figure(fig, args.output_dir, stem, data):
                print(f"wrote {output}")
        finally:
            get_pyplot().close(fig)
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError, KeyError, TypeError,
            RuntimeError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
