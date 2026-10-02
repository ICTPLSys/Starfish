#!/usr/bin/env python3
"""Plot paper Figure 9 from explicit CSV measurements."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import statistics
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
AE_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "common"))
from plotting import SYSTEM_STYLES, export_figure, get_pyplot, read_csv

REQUIRED = {"workload", "system", "ratio", "elapsed_s", "source_type", "source"}
WORKLOADS = {
    "bfs": "bfs", "llm": "llama", "llama": "llama", "mg": "mg", "wc": "wc",
    "kv-b": "kv_b", "kv_b": "kv_b", "kvs ycsb-b": "kv_b",
    "kv-a": "kv_a", "kv_a": "kv_a", "kvs ycsb-a": "kv_a",
    "kv-s": "kv_s", "kv_s": "kv_s", "kvs synthetic": "kv_s", "nq": "nq",
}
DEFAULT_WORKLOADS = ("bfs", "llama", "mg", "wc", "kv_b", "kv_a", "kv_s", "nq")
TITLES = {"bfs": "BFS", "llama": "LLM", "mg": "MG", "wc": "WC",
          "kv_b": "KV-B", "kv_a": "KV-A", "kv_s": "KV-S", "nq": "NQ"}
BAR_SYSTEMS = {
    "nonft": "nonft", "non-ft": "nonft",
    "nonft-backup-off": "nonft-backup-off",
    "non-ft (backup off)": "nonft-backup-off",
    "nonft backup off": "nonft-backup-off",
    "starfish": "starfish", "carbink": "carbink", "hydra": "hydra",
}
SYSTEMS = dict(BAR_SYSTEMS, native="native", **{"native linux all local": "native"})
DEFAULT_SYSTEMS = ("hydra", "carbink", "starfish", "nonft")
SOURCE_TYPES = {"run": "measured", "measured": "measured",
                "paper_reference": "paper_reference", "synthetic": "synthetic"}
DEFAULT_RATIOS = (13, 25, 50, 75, 100)
TEARDOWN_EXIT_STATUSES = frozenset(("124", "-15", "-9"))
CANONICAL_VARIANT = "canonicalruntime"
OFF_VARIANT = "nonft-backup-off"
NONFT_SOURCE_SYSTEMS = {
    "bfs": "nonft", "llama": "nonft", "mg": "nonft",
    "wc": OFF_VARIANT, "kv_b": OFF_VARIANT, "kv_a": OFF_VARIANT,
    "kv_s": OFF_VARIANT, "nq": OFF_VARIANT,
}


def canonical(value, names, label):
    try:
        return names[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown {label}: {value!r}") from exc


def validate_variant_provenance(row, system):
    """Keep a backup-OFF row from being silently merged with canonical Non-FT."""
    variant = row.get("baseline_variant", "").strip().lower()
    backup = row.get("backup_enabled", "").strip().lower()
    if system == OFF_VARIANT:
        if variant != OFF_VARIANT:
            raise ValueError("backup-OFF system requires baseline_variant=nonft-backup-off")
        if backup not in {"0", "false"}:
            raise ValueError("backup-OFF system requires backup_enabled=false")
        return
    if variant == OFF_VARIANT:
        raise ValueError("backup-OFF provenance must use the backup-OFF system label")
    if variant:
        accepted = {CANONICAL_VARIANT, system}
        if variant not in accepted:
            raise ValueError(f"baseline_variant={variant!r} disagrees with system={system!r}")
    if backup:
        if backup not in {"0", "1", "true", "false"}:
            raise ValueError("backup_enabled must be a boolean value")
        if system == "nonft" and backup in {"0", "false"}:
            raise ValueError("Non-FT backup=false requires the backup-OFF system label")


def run_metadata(row):
    """Keep lifecycle and teardown warnings in the exported provenance JSON."""
    return {
        "run_id": row["run_id"],
        "elapsed_s": row["elapsed_s"],
        "source": row["source"],
        "source_system": row["system"],
        "exit_status": row.get("exit_status", "0"),
        "measurement_usable": row.get("measurement_usable", ""),
        "execution_status": row.get("execution_status", ""),
        "warning": row.get("warning", ""),
        "baseline_variant": row.get("baseline_variant", ""),
        "backup_enabled": row.get("backup_enabled", ""),
    }


def prepare(path: Path, workloads=DEFAULT_WORKLOADS, systems=DEFAULT_SYSTEMS,
            ratios=DEFAULT_RATIOS, source_type="measured"):
    """Validate every row, select the requested matrix, and summarize repetitions."""
    if source_type not in set(SOURCE_TYPES.values()):
        raise ValueError(f"unknown source type: {source_type}")
    workloads = [canonical(x, WORKLOADS, "workload") for x in workloads]
    systems = [canonical(x, BAR_SYSTEMS, "bar system") for x in systems]
    if OFF_VARIANT in systems:
        raise ValueError(
            "Figure 9 has one Non-FT series with a fixed per-workload ON/OFF source; "
            "select nonft, not a separate backup-OFF series")
    if not workloads or len(set(workloads)) != len(workloads):
        raise ValueError("workloads must be nonempty and unique")
    if not systems or len(set(systems)) != len(systems):
        raise ValueError("systems must be nonempty and unique")
    if (not ratios or len(set(ratios)) != len(ratios)
            or any(not isinstance(x, int) or not 1 <= x <= 100 for x in ratios)):
        raise ValueError("ratios must be unique integer percentages in [1, 100]")
    ratios = sorted(ratios)
    selected_conditions = {
        (w, NONFT_SOURCE_SYSTEMS[w] if s == "nonft" else s, r)
        for w in workloads for s in systems for r in ratios
    } | {(w, "native", 100) for w in workloads}
    numbered_rows, digest = read_csv(path, REQUIRED)
    groups = defaultdict(list)
    blank_rows = []
    ignored_backup_off_rows = []
    contexts = defaultdict(set)
    for line, row in numbered_rows:
        try:
            workload = canonical(row["workload"], WORKLOADS, "workload")
            system = canonical(row["system"], SYSTEMS, "system")
            validate_variant_provenance(row, system)
            kind = (canonical(row["source_type"], SOURCE_TYPES, "source_type")
                    if row["source_type"] else None)
            if kind is not None and kind != source_type:
                raise ValueError(f"source_type={kind}, expected {source_type}; do not mix data kinds")
            ratio_value = float(row["ratio"])
            if not math.isfinite(ratio_value) or not ratio_value.is_integer() or not 1 <= ratio_value <= 100:
                raise ValueError("ratio must be an integer percentage in [1, 100]")
            ratio = int(ratio_value)
            if system == "native" and ratio != 100:
                raise ValueError("Native Linux All local must use ratio=100")
            if not row["elapsed_s"]:
                record = {
                    "workload": workload, "system": system, "ratio": ratio,
                    "csv_line": line, "source": row["source"],
                }
                if system == OFF_VARIANT and NONFT_SOURCE_SYSTEMS[workload] != system:
                    ignored_backup_off_rows.append(record)
                else:
                    blank_rows.append(record)
                continue
            if kind is None:
                raise ValueError("source_type is required when elapsed_s has a value")
            elapsed = float(row["elapsed_s"])
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("elapsed_s must be finite and greater than zero")
            if not row["source"]:
                raise ValueError("source must identify the raw result or reference")
            exit_status = row.get("exit_status", "0")
            correctness = row.get("correctness", "")
            teardown_measurement = (
                exit_status in TEARDOWN_EXIT_STATUSES
                and row.get("measurement_usable") == "1"
                and row.get("execution_status") == "teardown_failed"
                and correctness.lower() == "pass"
            )
            if exit_status != "0" and not teardown_measurement:
                raise ValueError(
                    "nonzero exit_status requires measurement_usable=1, "
                    "execution_status=teardown_failed, correctness=pass, "
                    "and exit_status in 124,-15,-9"
                )
            if correctness and correctness.lower() != "pass":
                raise ValueError("correctness must be pass")
            record = dict(row, workload=workload, system=system, ratio=ratio,
                          elapsed_s=elapsed, source_type=kind, run_id=row.get("run_id", ""),
                          exit_status=exit_status, correctness=correctness,
                          measurement_usable=row.get("measurement_usable", ""),
                          execution_status=row.get("execution_status", ""),
                          warning=row.get("warning", ""))
            if system == OFF_VARIANT and NONFT_SOURCE_SYSTEMS[workload] != system:
                ignored_backup_off_rows.append({
                    "workload": workload, "system": system, "ratio": ratio,
                    "csv_line": line, "source": row["source"],
                })
            groups[workload, system, ratio].append(record)
        except ValueError as exc:
            raise ValueError(f"CSV line {line}: {exc}") from exc

    # Never turn duplicate summary rows into hidden averaging. Repetitions
    # require explicit, unique run IDs even in an unselected series.
    for key, records in groups.items():
        if source_type == "paper_reference" and len(records) > 1:
            raise ValueError(f"{key}: a reference point must have only one row")
        ids = [r["run_id"] for r in records]
        if len(records) > 1 and (any(not x for x in ids) or len(ids) != len(set(ids))):
            raise ValueError(f"{key}: repeated conditions need unique, nonempty run_id values")

    points = []
    native_baselines = {}
    for w in workloads:
        native = groups.get((w, "native", 100), [])
        for row in native:
            for field in ("experiment_id", "environment", "measurement_phase"):
                if field in row:
                    if not row[field]:
                        raise ValueError(f"{(w, 'native', 100)}: {field} cannot be empty")
                    contexts[w, field].add(row[field])
        native_values = [row["elapsed_s"] for row in native]
        native_baselines[w] = {
            "mean_s": statistics.mean(native_values) if native_values else None,
            "stddev_s": statistics.stdev(native_values) if len(native_values) > 1 else None,
            "n": len(native),
            "runs": [run_metadata(row) for row in native],
        }
        for s in systems:
            # Select by the approved application policy, never by speed or availability.
            source_system = NONFT_SOURCE_SYSTEMS[w] if s == "nonft" else s
            for r in ratios:
                records = groups.get((w, source_system, r), [])
                for row in records:
                    for field in ("experiment_id", "environment", "measurement_phase"):
                        if field in row:
                            if not row[field]:
                                raise ValueError(f"{(w, s, r)}: {field} cannot be empty")
                            contexts[w, field].add(row[field])
                values = [row["elapsed_s"] for row in records]
                points.append({
                    "workload": w, "system": s, "ratio": r,
                    "source_system": source_system,
                    "mean_s": statistics.mean(values) if values else None,
                    "stddev_s": statistics.stdev(values) if len(values) > 1 else None,
                    "n": len(values),
                    "runs": [run_metadata(row) for row in records],
                })
    for (workload, field), values in contexts.items():
        if len(values) > 1:
            raise ValueError(f"{workload}: mixed {field} values: {sorted(values)}")
    return {
        "schema_version": 3, "figure": "figure9",
        "input": str(path.resolve()), "input_sha256": digest,
        "source_type": source_type,
        "metric": "elapsed_s", "unit": "seconds",
        "x_metric": "local memory capacity / application footprint * 100",
        "aggregation": "arithmetic mean; error bars = sample standard deviation (ddof=1)",
        "workloads": workloads, "systems": systems, "ratios": ratios,
        "nonft_source_systems": {w: NONFT_SOURCE_SYSTEMS[w] for w in workloads},
        "input_rows": len(numbered_rows),
        "blank_rows": blank_rows,
        "ignored_backup_off_rows": ignored_backup_off_rows,
        "unselected_conditions": [list(key) for key in sorted(
            set(groups) | {(row["workload"], row["system"], row["ratio"])
                           for row in blank_rows})
            if key not in selected_conditions],
        "selected_rows": sum(p["n"] for p in points)
                         + sum(v["n"] for v in native_baselines.values()),
        "missing_conditions": [[p["workload"], p["system"], p["ratio"]]
                               for p in points if p["n"] == 0],
        "context": {w: {field: next(iter(values)) for (work, field), values in contexts.items()
                        if work == w} for w in workloads},
        "native_baselines": native_baselines,
        "points": points,
    }


def draw(data):
    """Keep the paper's geometry, colors, hatches and guides; replace its arrays with CSV."""
    if OFF_VARIANT in data["systems"]:
        raise ValueError(
            "Figure 9 uses backup-OFF data within its Non-FT series, not as a fifth series")
    plt = get_pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    columns = min(4, len(data["workloads"]))
    rows = math.ceil(len(data["workloads"]) / columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(14.4 if columns == 4 else 4.3 * columns,
                                      6.8 if rows == 2 else 3.7),
                             sharex=(rows == 2), sharey=False, squeeze=False)
    indexed = {(p["workload"], p["system"], p["ratio"]): p for p in data["points"]}
    x = np.arange(len(data["ratios"]))
    width = 0.22
    offsets = (np.arange(len(data["systems"])) - (len(data["systems"]) - 1) / 2) * width
    guide_lines = {"llama": (50, 100), "mg": (50, 100)}
    panel_titles = {"bfs": r"$\mathbf{BFS}$", "llama": r"$\mathbf{LLM}$",
                    "mg": r"$\mathbf{MG}$", "wc": r"$\mathbf{WC}$",
                    "kv_b": r"$\mathbf{KVS}$ YCSB-B",
                    "kv_a": r"$\mathbf{KVS}$ YCSB-A",
                    "kv_s": r"$\mathbf{KVS}$ Synthetic", "nq": r"$\mathbf{NQ}$"}
    for idx, workload in enumerate(data["workloads"]):
        ax = axes.flat[idx]
        ax.set_box_aspect(0.62)
        finite_values = []
        for s_idx, system in enumerate(data["systems"]):
            style = SYSTEM_STYLES[system]
            for ratio_idx, ratio in enumerate(data["ratios"]):
                point = indexed[workload, system, ratio]
                if point["mean_s"] is None:
                    continue  # No row or blank value: leave this bar slot empty.
                finite_values.append(point["mean_s"])
                bars = ax.bar(x[ratio_idx] + offsets[s_idx], point["mean_s"],
                              width=width, facecolor=style["facecolor"],
                              edgecolor="black", linewidth=0.8,
                              hatch=style["hatch"],
                              yerr=point["stddev_s"],
                              capsize=2 if point["n"] > 1 else 0,
                              error_kw={"elinewidth": 1})
                # Paper's three sparse horizontal marks on the Non-FT bars.
                if system == "nonft":
                    bar = bars[0]
                    y0, y1 = bar.get_y(), bar.get_y() + bar.get_height()
                    x0, x1 = bar.get_x(), bar.get_x() + bar.get_width()
                    for y in np.linspace(y0 + (y1 - y0) * 0.25,
                                         y0 + (y1 - y0) * 0.75, 3):
                        ax.plot([x0, x1], [y, y], color="black", linewidth=0.9,
                                clip_on=True)
        native = data["native_baselines"][workload]["mean_s"]
        if native is not None:
            finite_values.append(native)
            ax.axhline(native, color="#c73535", linestyle=(0, (4, 2)),
                       linewidth=1.55, alpha=0.95, zorder=3)
        panel_letter = chr(97 + idx)
        ax.set_title(rf"$\mathbf{{({panel_letter})}}$ {panel_titles[workload]}",
                     fontsize=20, fontweight="normal", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ratio}%" for ratio in data["ratios"]])
        if finite_values:
            ylimit = max(finite_values) * 1.18
            if workload in guide_lines:
                ylimit = max(ylimit, max(guide_lines[workload]) * 1.12)
            ax.set_ylim(0, ylimit)
            ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=1.0)
            ax.set_axisbelow(True)
            for y in guide_lines.get(workload, ()):
                ax.axhline(y, color="#9a9a9a", linestyle="--", linewidth=1.15,
                           alpha=0.8, zorder=0)
        else:
            ax.set_yticks([])  # An empty panel must not imply a 0-to-1 s measurement.
        ax.tick_params(axis="x", labelsize=18, direction="in", length=4, pad=4.0)
        ax.tick_params(axis="y", labelsize=18, direction="in", length=0, pad=2.0)
        for name, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
            if name in ("top", "right"):
                spine.set_color("#7a7a7a")
    for ax in list(axes.flat)[len(data["workloads"]):]:
        ax.set_visible(False)
    handles = [Patch(facecolor=SYSTEM_STYLES[s]["facecolor"],
                     hatch=SYSTEM_STYLES[s]["hatch"], edgecolor="black",
                     label=SYSTEM_STYLES[s]["label"]) for s in data["systems"]]
    if any(p["n"] for p in data["native_baselines"].values()):
        handles.append(Line2D([0], [0], color="#c73535", linestyle=(0, (4, 2)),
                              linewidth=1.55, label="Native Linux All local"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=18)
    fig.supylabel("Elapsed Time (s)", fontsize=25, x=0.005)
    fig.supxlabel("Local memory ratio (%)", fontsize=25, y=0.055)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.105, top=0.885,
                        wspace=0.12, hspace=0.01)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=AE_ROOT / "data/figure9.csv")
    parser.add_argument("--output-dir", type=Path, default=AE_ROOT / "results/figures/figure9")
    parser.add_argument("--source-type", choices=sorted(set(SOURCE_TYPES.values())), default="measured")
    parser.add_argument("--workloads", nargs="+", default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--ratios", type=int, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = prepare(args.input, args.workloads, args.systems, args.ratios, args.source_type)
        available = len(data["points"]) - len(data["missing_conditions"])
        print(f"validated {data['selected_rows']}/{data['input_rows']} rows, "
              f"{available}/{len(data['points'])} points available, "
              f"source_type={args.source_type}")
        if args.validate_only:
            return 0
        stem = "figure9" + ("" if args.source_type == "measured" else "-" + args.source_type)
        # Do not overwrite input data even if the caller chooses an odd filename.
        if args.input.resolve() in [(args.output_dir / (stem + ext)).resolve()
                                    for ext in (".png", ".pdf", ".json")]:
            raise ValueError("input and output paths must be different")
        data["plot_code_sha256"] = {
            str(path.relative_to(AE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), SCRIPTS / "common/plotting.py")
        }
        fig = draw(data)
        try:
            for output in export_figure(fig, args.output_dir, stem, data):
                print(f"wrote {output}")
        finally:
            get_pyplot().close(fig)
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
