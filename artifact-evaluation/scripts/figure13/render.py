#!/usr/bin/env python3
"""Render Figure 13 from an explicit run manifest; never launch experiments."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import warnings

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect
import plot


_UNFINALIZED_RESULT_STATES = frozenset({
    "pending", "running", "started", "in_progress", "unfinalized",
})
_FAILED_RESULT_STATES = frozenset({
    "failed", "failure", "error", "timeout", "timed_out",
    "aborted", "cancelled", "canceled",
})
_SUCCESS_RESULT_STATES = frozenset({
    "passed", "success", "completed", "finished", "done",
})


def _result_skip_reason(path: Path):
    """Return a skip reason only for absent or explicitly unfinalized results.

    A present result with invalid JSON or a failed status raises instead of
    silently dropping evidence.  A pending filename is only a skip marker when
    no terminal failure or success evidence contradicts it.
    """
    if not path.exists():
        return "absent_result"
    if not path.is_file():
        raise ValueError(f"manifest result path is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid present result JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"present result JSON is not an object: {path}")
    status = payload.get("status")
    status_text = status.strip().lower() if isinstance(status, str) else None
    exit_status = payload.get("exit_status")
    if exit_status is not None:
        try:
            exit_code = int(exit_status)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"present result has invalid exit_status: {path}") from exc
        if isinstance(exit_status, bool) or exit_code != 0:
            raise ValueError(f"present result is failed/unverified: {path}")
    if status_text in _FAILED_RESULT_STATES:
        raise ValueError(f"present result is failed/unverified: {path}")
    if payload.get("passed") is True or status_text in _SUCCESS_RESULT_STATES:
        return None
    if status_text in _UNFINALIZED_RESULT_STATES:
        return f"unfinalized_status:{status.strip()}"
    if payload.get("passed") is False:
        raise ValueError(f"present result is failed/unverified: {path}")
    if path.name.endswith(".pending.json"):
        return "unfinalized_pending_filename"
    return None


def read_manifest(path, scenarios, repeat=1):
    from inputs import load_manifest
    manifest, entries, indexed = load_manifest(path, repeat)
    manifest = dict(manifest)
    pending_ids = {str(value) for value in manifest.get("pending", [])
                   if isinstance(value, (str, int))}
    selected, skipped = [], []
    for entry in entries:
        if entry["scenario"] not in scenarios:
            continue
        key = entry["scenario"], entry["system"]
        result_path = Path(entry["result"])
        reason = _result_skip_reason(result_path)
        if reason:
            if (reason == "absent_result" and entry.get("run_id") is not None
                    and str(entry["run_id"]) in pending_ids):
                reason = "not_run_pending_result"
            skipped.append({"scenario": key[0], "system": key[1],
                            "run_id": entry.get("run_id"),
                            "result": str(result_path), "reason": reason})
            continue
        selected.append((key, result_path))
    seen = {key for key, _ in selected}
    missing = [(scenario, system) for scenario in scenarios
               for system in plot.SYSTEMS if (scenario, system) not in seen]
    warnings_text = []
    if skipped:
        warnings_text.append(f"skipped {len(skipped)} absent/unfinalized manifest result(s)")
    if missing:
        warnings_text.append(f"missing {len(missing)} requested condition(s)")
    if warnings_text:
        warnings.warn("Figure 13 partial manifest: " + "; ".join(warnings_text),
                      RuntimeWarning, stacklevel=2)
    manifest["_render_skipped_conditions"] = skipped
    manifest["_render_missing_conditions"] = missing
    manifest["_render_warnings"] = warnings_text
    if not selected:
        raise ValueError("no present finalized Figure 13 result remains for the selected scenarios")
    return manifest, selected, hashlib.sha256(path.read_bytes()).hexdigest(), indexed


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=1, help="indexed batch repetition only; historical runs stay unindexed")
    parser.add_argument("--panel", choices=("a", "b", "both"), default="a",
                        help="a: one-node throughput; b: recovery times (default requires both scenarios)")
    parser.add_argument("--scenarios", nargs="+", choices=plot.SCENARIOS,
                        help="For b/both, explicitly select 1-node for a partial preview.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--view", choices=("zoom", "full"), default="zoom")
    parser.add_argument("--average-windows", type=int, default=1,
                        help="Display-only trailing duration-weighted average; 1 keeps raw 100ms samples.")
    parser.add_argument("--bin-windows", type=int, default=10,
                        help="Non-overlapping display bins (default10, about1s); use1 for raw samples. Events split bins.")
    parser.add_argument("--display-points", type=int,
                        help="Exact visible point count per curve; overrides --bin-windows, preserves event boundaries.")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate selected native records without writing plot outputs")
    args = parser.parse_args()
    try:
        scenarios = tuple(args.scenarios or (("1-node",) if args.panel == "a" else plot.SCENARIOS))
        if args.panel == "a" and scenarios != ("1-node",):
            raise ValueError("panel (a) uses the one-node trace; select panel b for 2-node durations")
        if len(set(scenarios)) != len(scenarios) or args.average_windows < 1:
            raise ValueError("duplicate scenario or invalid averaging window")
        bin_windows = args.bin_windows if args.panel != "b" else 1
        point_count = args.display_points if args.panel != "b" else None
        if point_count is not None:
            if point_count < 2 or args.average_windows != 1:
                raise ValueError("--display-points must be >=2 and cannot use trailing averaging")
            bin_windows = 1
        if bin_windows < 1 or (bin_windows > 1 and args.average_windows > 1):
            raise ValueError("select binning or trailing averaging, not both; windows must be positive")
        manifest_path = args.manifest.resolve()
        manifest, selected, manifest_hash, indexed = read_manifest(manifest_path, scenarios, args.repeat)
        panels = {"a": "throughput", "b": "recovery", "both": "both"}[args.panel]
        native_repeat = (
            args.repeat
            if indexed and any(path.name == "figure13-result.json" for _, path in selected)
            else 1
        )
        selected_paths = [path for _, path in selected]
        collection_warnings = []
        has_trace_reference = ("1-node", "starfish") in {key for key, _ in selected}
        if panels == "throughput" and not has_trace_reference:
            raise ValueError(
                "no verified 1-node Starfish reference run remains for the throughput panel")
        if panels == "both" and not has_trace_reference:
            # collect_native requires Starfish's pre-failure reference to build
            # a throughput trace. Recovery bars remain independently valid when
            # that one native reference condition is absent/unfinalized.
            recovery, trace, info = collect.collect_native(
                selected_paths, panels="recovery", failure_at_s=20.0,
                steady_before_s=10.0, repeat=native_repeat)
            collection_warnings.append(
                "throughput collection skipped: no verified 1-node Starfish reference run")
            info["collection_panels"] = "recovery"
            info["panels"] = panels
        else:
            recovery, trace, info = collect.collect_native(
                selected_paths, panels=panels,
                failure_at_s=20.0, steady_before_s=10.0, repeat=native_repeat)
            info["collection_panels"] = panels
        if indexed:
            for row in recovery + trace:
                row["repeat"] = str(args.repeat)
        observed = {(row["scenario"], row["system"].lower())
                    for row in recovery + trace if row["system"].lower() != "all"}
        if observed != {key for key, _ in selected}:
            raise ValueError("manifest condition labels disagree with verified native records")
        if args.validate_only:
            data = plot.prepare_rows(
                list(enumerate(recovery, 1)), list(enumerate(trace, 1)),
                None, None, str(manifest_path), str(manifest_path), "measured",
                systems=plot.SYSTEMS, scenarios=scenarios, panels=panels)
            data.update(info)
            data.update(
                manifest=str(manifest_path), manifest_sha256=manifest_hash,
                pending_conditions=manifest.get("pending", []),
                manifest_skipped_conditions=manifest.get(
                    "_render_skipped_conditions", []),
                manifest_missing_conditions=manifest.get(
                    "_render_missing_conditions", []),
                manifest_warnings=manifest.get("_render_warnings", []),
                collection_warnings=collection_warnings,
                selected_scenarios=list(scenarios),
                selected_repeat=args.repeat if indexed else None,
                repeat_source="batch-plan.json" if indexed else "historical_unindexed",
                complete_requested_selection=not (
                    manifest.get("_render_skipped_conditions")
                    or manifest.get("_render_missing_conditions")
                ),
                plot_code_sha256=hashlib.sha256((HERE / "plot.py").read_bytes()).hexdigest(),
                renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            print(json.dumps({
                "panel": args.panel,
                "scenarios": list(scenarios),
                "systems": list(plot.SYSTEMS),
                "recovery_rows": len(recovery),
                "throughput_rows": len(trace),
                "rendered_panels": data["render_panels"],
                "trace_systems": list(data.get("trace_systems", ())),
                "missing_recovery": data.get("missing_recovery", []),
                "omitted_traces": data.get("omitted_traces", {}),
                "manifest_skipped_conditions": data["manifest_skipped_conditions"],
                "manifest_missing_conditions": data["manifest_missing_conditions"],
                "collection_warnings": data["collection_warnings"],
                "complete_requested_selection": data["complete_requested_selection"],
                "reference_ops_per_s": data.get("reference_ops_per_s"),
                "manifest_sha256": manifest_hash,
                "plot_code_sha256": data["plot_code_sha256"],
                "renderer_sha256": data["renderer_sha256"],
            }, allow_nan=False, sort_keys=True))
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        recovery_path = args.output_dir / "figure13-recovery.csv"
        trace_path = args.output_dir / "figure13-throughput.csv"
        write_csv(recovery_path, collect.RECOVERY_FIELDS, recovery)
        write_csv(trace_path, collect.TRACE_FIELDS, trace)
        data = plot.prepare(recovery_path if panels != "throughput" else None,
                            trace_path if panels != "recovery" else None,
                            systems=plot.SYSTEMS, scenarios=scenarios, panels=panels)
        # Preserve all provenance without copying raw trace rows into the figure JSON twice.
        data.update(info)
        data.update(manifest=str(manifest_path), manifest_sha256=manifest_hash,
                    pending_conditions=manifest.get("pending", []),
                    manifest_skipped_conditions=manifest.get(
                        "_render_skipped_conditions", []),
                    manifest_missing_conditions=manifest.get(
                        "_render_missing_conditions", []),
                    manifest_warnings=manifest.get("_render_warnings", []),
                    collection_warnings=collection_warnings,
                    selected_scenarios=list(scenarios),
                    selected_repeat=args.repeat if indexed else None,
                    repeat_source="batch-plan.json" if indexed else "historical_unindexed",
                    complete_requested_selection=not (
                        manifest.get("_render_skipped_conditions")
                        or manifest.get("_render_missing_conditions")
                    ),
                    plot_code_sha256=hashlib.sha256((HERE / "plot.py").read_bytes()).hexdigest(),
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    caption=(
                        "Diagnostic KV-B runs, not matched-resource protocol comparisons. "
                        "Panel (a): completed operations / actual window seconds, divided by the "
                        "same Starfish pre-failure reference, from native 100ms target windows. "
                        "Each detected failure is aligned to display time 20s, not command-injection time. "
                        "H/C/S mark native background rebuild completion, not return-to-baseline estimates. "
                        "Panel (b): background rebuild start to completion; excludes failure detection. "
                        "Failures kill logical memory-service processes, not entire physical hosts."))
        data["caption"] += ((f" Exactly {point_count} display means are formed per curve in the visible interval; "
                             if point_count is not None else
                             f" Display bins contain up to {bin_windows} consecutive native windows; ") +
                            "completed counts are divided by their actual total duration. "
                            "Bins reset at failure, rebuild completion and sampling gaps. "
                            "Native event times and raw samples are unchanged.")
        fig = plot.draw(data, view=args.view, average_windows=args.average_windows,
                        bin_windows=bin_windows, display_points_count=point_count)
        stem = "figure13-" + args.panel + "-diagnostic"
        if args.view == "full":
            stem += "-full"
        if args.average_windows != 1:
            stem += f"-avg{args.average_windows}"
        if bin_windows != 1:
            stem += f"-bin{bin_windows}"
        if point_count is not None:
            stem += f"-points{point_count}"
        try:
            for path in plot.export_figure(fig, args.output_dir, stem, data):
                print(path)
        finally:
            plot.get_pyplot().close(fig)
        print(json.dumps({"panel": args.panel, "scenarios": scenarios,
                          "systems": list(plot.SYSTEMS),
                          "recovery_s": {f'{row["scenario"]}/{row["system"]}': float(row["recovery_s"])
                                         for row in recovery},
                          "reference_ops_per_s": data["reference_ops_per_s"],
                          "pending": manifest.get("pending", [])}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
