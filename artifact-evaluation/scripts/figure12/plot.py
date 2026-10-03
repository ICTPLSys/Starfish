#!/usr/bin/env python3
"""Plot evaluation compute-node overhead from final logs or explicit CSV data."""

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
from figure12.log_contract import (EC_BOUNDARY_SEMANTICS, METRIC_UNITS,
                                   METRIC_SELECTIONS, nonnegative_integer)

REQUIRED = {"workload", "system", "metric", "value", "unit", "source_type", "source"}
WORKLOADS = ("BFS", "LLM", "MG", "WC", "KV-B", "KV-A", "KV-S", "NQ")
ALIASES = {"bfs": "BFS", "llm": "LLM", "llama": "LLM", "mg": "MG",
           "wc": "WC", "wordcount": "WC", "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S",
           "kvs ycsb-b": "KV-B", "kvs ycsb-a": "KV-A",
           "kvs synthetic": "KV-S", "nq": "NQ"}
SYSTEMS = ("carbink", "starfish")
METRICS = METRIC_UNITS
SOURCE_TYPES = {"measured", "paper_reference", "synthetic"}
EC_BASIS_LEGACY = "work_only_legacy"
EC_BASIS_INIT_WORK = "initialization_plus_work"
EC_COMPONENT_FIELDS = (
    "ec_initialization_cycles", "ec_work_cycles", "ec_total_cycles",
    "ec_initialization_scopes", "ec_work_scopes", "ec_total_scopes",
)


def prepare(path: Path, source_type="measured", *, apps=WORKLOADS, metrics="all"):
    numbered_rows, digest = read_csv(path, REQUIRED)
    return prepare_rows(numbered_rows, digest, str(path.resolve()), source_type,
                        apps=apps, metrics=metrics)


def prepare_logs(logs_root: Path, *, ratio=25, repeat=1, apps=WORKLOADS, metrics="all"):
    from figure12.collect import collect_rows
    rows = collect_rows(logs_root, ratio=ratio, repeat=repeat, metrics=metrics)
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    data = prepare_rows(list(enumerate(rows, 1)), digest,
                        str(logs_root.resolve()), "measured", apps=apps, metrics=metrics)
    data.update(input_format="ae_run_directories",
                input_hash_scope="canonical_collected_rows",
                ratio=ratio, repeat=repeat,
                collected_rows=rows)
    return data


def prepare_rows(numbered_rows, digest, input_name, source_type="measured",
                 *, apps=WORKLOADS, metrics="all"):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type: {source_type}")
    selected = METRIC_SELECTIONS[metrics]
    apps = tuple(apps)
    if not apps or len(set(apps)) != len(apps) or set(apps) - set(WORKLOADS):
        raise ValueError("select distinct known applications")
    values, footprints, contexts = {}, {}, {}
    ec_bases, ec_schema_versions, ec_boundary_semantics = set(), set(), set()
    blanks = []
    for line, row in numbered_rows:
        try:
            workload = ALIASES[row["workload"].lower()]
            system = row["system"].lower()
            metric = row["metric"].lower()
            if system not in SYSTEMS or metric not in METRICS:
                raise ValueError(f"unknown or legacy system/metric: {system}/{metric}; "
                                 "CPU must be raw local_ec_cpu_cycles")
            if workload not in apps or metric not in selected:
                continue
            if row["unit"] != METRICS[metric]:
                raise ValueError(f"{metric} requires unit {METRICS[metric]}")
            key = workload, system, metric
            if key in values or key in blanks:
                raise ValueError(f"duplicate condition: {key}")
            raw_value = row["value"]
            if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
                blanks.append(key)
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("value must be finite and nonnegative")
            if row["source_type"] != source_type or not row["source"]:
                raise ValueError("nonblank values require the selected source_type and source")
            if source_type == "measured":
                retained = (str(row.get("exit_status")) in {"124", "-15", "-9"}
                            and str(row.get("measurement_usable")) == "1"
                            and row.get("execution_status") == "teardown_failed"
                            and bool(row.get("measurement_warning"))
                            and system == "starfish" and workload in ("KV-B", "KV-A", "KV-S"))
                if str(row.get("exit_status")) != "0" and not retained:
                    raise ValueError("failed/unverified runs cannot be plotted")
                if not retained and (
                        str(row.get("measurement_usable", 1)) != "1" or
                        row.get("execution_status", "passed") != "passed"):
                    raise ValueError("measurement is explicitly unusable or execution failed")
                if row.get("correctness", "").lower() != "pass":
                    raise ValueError("correctness must be pass")
                footprint = nonnegative_integer(str(row.get("app_memory_bytes", "")),
                                                "app_memory_bytes")
                if not footprint or footprints.setdefault(workload, footprint) != footprint:
                    raise ValueError("paired runtimes need the same positive app footprint")
                for field in ("ratio", "repeat"):
                    current = str(row.get(field, "1" if field == "repeat" else ""))
                    if not current or contexts.setdefault(field, current) != current:
                        raise ValueError(f"mixed/missing {field}; select one condition")
            ec_record = {}
            if metric == "local_ec_cpu_cycles":
                value = nonnegative_integer(str(row["value"]), "EC cycles")
                if row.get("cycle_clock") != "tsc":
                    raise ValueError("EC cycles must explicitly record cycle_clock=tsc")
                if source_type == "measured":
                    intervals = nonnegative_integer(str(row.get("ec_work_intervals", "")),
                                                    "ec_work_intervals")
                    scopes = nonnegative_integer(str(row.get("ec_scopes", "")), "ec_scopes")
                    verified_scopes = scopes
                    def text(key):
                        raw = row.get(key, "")
                        if raw is None:
                            return ""
                        return raw.strip() if isinstance(raw, str) else str(raw)
                    basis = text("ec_measurement_basis")
                    schema_text = text("ec_schema_version")
                    boundary_semantics = text("ec_boundary_semantics")
                    phase = text("measurement_phase")
                    if not basis:
                        if phase != "all_completed_work":
                            raise ValueError("EC measurement basis is missing")
                        basis = EC_BASIS_LEGACY
                    if basis not in (EC_BASIS_LEGACY, EC_BASIS_INIT_WORK):
                        raise ValueError(f"unknown EC measurement basis: {basis}")
                    if basis == EC_BASIS_LEGACY:
                        if phase != "all_completed_work":
                            raise ValueError("legacy EC rows require all_completed_work")
                        if schema_text and schema_text != "1":
                            raise ValueError("legacy EC rows require schema_version=1")
                        if boundary_semantics:
                            raise ValueError("legacy EC rows must not declare boundary semantics")
                        if any(text(field) for field in
                               ("ec_initialization_cycles", "ec_initialization_scopes")):
                            raise ValueError("legacy EC initialization components must be empty")
                        optional = ("ec_work_cycles", "ec_total_cycles",
                                    "ec_work_scopes", "ec_total_scopes")
                        present = [bool(text(field)) for field in optional]
                        if any(present) and not all(present):
                            raise ValueError("legacy EC component fields are incomplete")
                        if all(present):
                            work_cycles = nonnegative_integer(text("ec_work_cycles"),
                                                              "ec_work_cycles")
                            total_cycles = nonnegative_integer(text("ec_total_cycles"),
                                                               "ec_total_cycles")
                            work_scopes = nonnegative_integer(text("ec_work_scopes"),
                                                              "ec_work_scopes")
                            total_scopes = nonnegative_integer(text("ec_total_scopes"),
                                                               "ec_total_scopes")
                            if (work_cycles != value or total_cycles != value or
                                    work_scopes != scopes or total_scopes != scopes):
                                raise ValueError("legacy EC component totals do not close")
                            verified_scopes = total_scopes
                        schema_version = int(schema_text) if schema_text else 1
                    else:
                        if phase != "initialization_and_all_completed_work":
                            raise ValueError("initialization-plus-Work EC phase is required")
                        if schema_text != "2":
                            raise ValueError("initialization-plus-Work EC rows require schema_version=2")
                        if boundary_semantics != EC_BOUNDARY_SEMANTICS:
                            raise ValueError("unknown or missing EC boundary semantics")
                        init_cycles = nonnegative_integer(
                            text("ec_initialization_cycles"), "ec_initialization_cycles")
                        work_cycles = nonnegative_integer(
                            text("ec_work_cycles"), "ec_work_cycles")
                        total_cycles = nonnegative_integer(
                            text("ec_total_cycles"), "ec_total_cycles")
                        init_scopes = nonnegative_integer(
                            text("ec_initialization_scopes"), "ec_initialization_scopes")
                        work_scopes = nonnegative_integer(
                            text("ec_work_scopes"), "ec_work_scopes")
                        total_scopes = nonnegative_integer(
                            text("ec_total_scopes"), "ec_total_scopes")
                        if (init_cycles and not init_scopes or
                                work_cycles and not work_scopes or
                                total_cycles != init_cycles + work_cycles or
                                total_scopes != init_scopes + work_scopes or
                                total_cycles != value or work_scopes != scopes):
                            raise ValueError("initialization-plus-Work EC totals do not close")
                        verified_scopes = total_scopes
                        schema_version = 2
                    ec_bases.add(basis)
                    ec_schema_versions.add(schema_version)
                    if boundary_semantics:
                        ec_boundary_semantics.add(boundary_semantics)
                    ec_record = {field: row.get(field, "") for field in
                                 ("ec_schema_version", "ec_measurement_basis",
                                  "ec_boundary_semantics", *EC_COMPONENT_FIELDS)}
                    ec_record.update(ec_schema_version=schema_version,
                                     ec_measurement_basis=basis,
                                     ec_boundary_semantics=boundary_semantics)
                    expected_phase = ("initialization_and_all_completed_work"
                                      if basis == EC_BASIS_INIT_WORK
                                      else "all_completed_work")
                    if (row.get("measurement_phase") != expected_phase or
                            row.get("scope") != "compute_ec" or not intervals or
                            str(row.get("boundary_sequence")) != str(intervals) or
                            str(row.get("measurement_usable")) != "1" or
                            row.get("execution_status") not in ("passed", "teardown_failed") or
                            (value and not verified_scopes)):
                        raise ValueError("EC requires verified compute-only completed Work scopes")
            elif source_type == "measured":
                if (row.get("metadata_numerator") != "accounted_bytes" or
                        row.get("measurement_phase") != "final_work_end"):
                    raise ValueError("metadata requires inclusive final-Work accounting")
                total = nonnegative_integer(str(row.get("accounted_bytes", "")), "accounted_bytes")
                core = nonnegative_integer(str(row.get("core_metadata_bytes", "")), "core_metadata_bytes")
                aux = nonnegative_integer(str(row.get("measurement_aux_bytes", "")), "measurement_aux_bytes")
                if (total != core + aux or
                        int(row.get("metadata_bytes", -1)) != total or
                        not math.isclose(value, 100 * total / footprint, rel_tol=1e-9, abs_tol=1e-10)):
                    raise ValueError("metadata numerator/component/percentage mismatch")
            values[key] = {"value": value, "source": row["source"],
                           "measurement_warning": row.get("measurement_warning", ""),
                           **ec_record}
        except (KeyError, ValueError) as exc:
            raise ValueError(f"input row {line}: {exc}") from exc
    missing = [[w, s, m] for w in apps for s in SYSTEMS
               for m in selected if (w, s, m) not in values]
    if missing:
        raise ValueError(f"missing {len(missing)} selected conditions: {missing}; "
                         "select --apps/--metrics explicitly; missing is not zero")
    result = {"figure": "figure12", "input": input_name, "input_sha256": digest,
              "source_type": source_type, "metric_units": {m: METRICS[m] for m in selected},
              "workloads": apps, "metrics": selected, "cycle_clock": "tsc",
              "input_rows": len(numbered_rows), "blank_rows": blanks,
              "missing_conditions": missing,
              "values": {"/".join(key): record for key, record in values.items()}}
    if "local_ec_cpu_cycles" in selected:
        result.update(
            ec_measurement_basis=(next(iter(ec_bases)) if len(ec_bases) == 1 else None),
            ec_measurement_bases=sorted(ec_bases),
            ec_schema_versions=sorted(ec_schema_versions),
            ec_boundary_semantics=sorted(ec_boundary_semantics),
            ec_scopes_semantics=("completed_work_only" if ec_bases else None),
            ec_total_value=(next(iter(ec_bases)) if len(ec_bases) == 1
                            else "per_record" if ec_bases else None),
        )
    return result


def draw(data):
    plt = get_pyplot(paper_font="Times New Roman")
    from matplotlib.patches import Patch

    apps, metrics = data["workloads"], data["metrics"]
    fig, axes = plt.subplots(len(metrics), 1, squeeze=False,
                             figsize=(4.1, 3.05), sharex=True)
    axes = axes[:, 0]
    x = np.arange(len(apps))
    width = .34
    labels = {
        "local_ec_cpu_cycles": ("Local EC computation CPU", "Norm. to Carbink",
                                1.2, [0, .5, 1]),
        "metadata_space_pct": ("Metadata space", "% app memory", 10., [0, 5, 10]),
    }
    displayed = {}
    denominators = {}
    for ax, metric in zip(axes, metrics):
        title, ylabel, upper, ticks = labels[metric]
        maximum = 0.
        for s_idx, system in enumerate(SYSTEMS):
            for app_idx, workload in enumerate(apps):
                record = data["values"][f"{workload}/{system}/{metric}"]
                value = record["value"]
                if metric == "local_ec_cpu_cycles":
                    denominator = data["values"][f"{workload}/carbink/{metric}"]["value"]
                    if denominator <= 0:
                        plt.close(fig)
                        raise ValueError(f"{workload}: Carbink EC cycles must be positive for normalization")
                    denominators[workload] = denominator
                    value /= denominator
                maximum = max(maximum, value)
                displayed[f"{workload}/{system}/{metric}"] = value
                position = x[app_idx] + (-width / 2 if s_idx == 0 else width / 2)
                ax.bar(position, value, width=width,
                       color=SYSTEM_STYLES[system]["facecolor"],
                       edgecolor="black", linewidth=.7)
        if maximum > upper:
            step = .5 if metric == "local_ec_cpu_cycles" else 5.
            upper = math.ceil(maximum / step) * step
            ticks = np.arange(0., upper + step / 2, step)
        for name, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
            spine.set_color("black")
        ax.set_ylim(0, upper)
        ax.set_yticks(ticks)
        ax.grid(axis="y", linestyle="--", alpha=.5, linewidth=1.)
        ax.axhline(1., color="#555555", linewidth=.8, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=3)
        ax.set_title(title, fontsize=11, pad=2)
        ax.tick_params(axis="y", labelsize=9, direction="in", length=3, pad=1.5)
        ax.tick_params(axis="x", direction="in", length=3, pad=1.5)
    if len(axes) > 1:
        axes[0].tick_params(axis="x", labelbottom=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(apps, rotation=35, ha="right", fontsize=8.5)
    handles = [Patch(facecolor=SYSTEM_STYLES[s]["facecolor"], edgecolor="black",
                     label=SYSTEM_STYLES[s]["label"]) for s in SYSTEMS]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.58),
                   ncol=2, frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=.19, right=.995, bottom=.25, top=.82, hspace=.45)
    data["display_values"] = displayed
    data["ec_display_normalization"] = "same_workload_carbink_ec_cycles"
    data["ec_display_denominator_bases"] = {
        workload: data["values"][f"{workload}/carbink/local_ec_cpu_cycles"].get(
            "ec_measurement_basis")
        for workload in denominators
    }
    data["ec_display_denominators"] = denominators
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="read an existing measurement CSV")
    inputs.add_argument("--logs-root", type=Path, help="read AE run directories directly")
    parser.add_argument("--apps", default=",".join(WORKLOADS),
                        help="Explicit subset, e.g. kv-b,nq; both systems are required.")
    parser.add_argument("--metrics", "--metric", choices=METRIC_SELECTIONS, default="all")
    parser.add_argument("--ratio", type=int, default=25, help="local-memory percentage in log mode")
    parser.add_argument("--repeat", type=int, default=1, help="one repeat to select in log mode")
    parser.add_argument("--output-dir", type=Path,
                        default=AE_ROOT / "results/figures/figure12")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="measured")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        apps = tuple(ALIASES.get(value.strip().lower(), value.strip())
                     for value in args.apps.split(","))
        if args.logs_root is not None:
            if args.source_type != "measured":
                raise ValueError("--logs-root accepts measured records only")
            data = prepare_logs(args.logs_root, ratio=args.ratio, repeat=args.repeat,
                                apps=apps, metrics=args.metrics)
        else:
            data = prepare(args.input, args.source_type, apps=apps, metrics=args.metrics)
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
    except (OSError, UnicodeError, csv.Error, ValueError, KeyError, TypeError,
            RuntimeError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
