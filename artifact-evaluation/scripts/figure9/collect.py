#!/usr/bin/env python3
"""Collect verified Figure 9 runs into a measured-only, sparse CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from measurement_acceptance import recover_measurement

FIELDS = ("workload", "system", "baseline_variant", "backup_enabled",
          "ratio", "elapsed_s", "source_type", "source", "run_id",
          "exit_status", "correctness", "environment", "measurement_phase",
          "measurement_usable", "execution_status", "warning")
SYSTEMS = {"Non-FT", "Starfish", "Carbink", "Hydra"}
WORKLOADS = {"BFS", "LLM", "MG", "WC", "KV-B", "KV-A", "KV-S", "NQ"}
AE_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_LABEL = {"nonft": "Non-FT", "starfish": "Starfish",
                "hydra": "Hydra", "carbink": "Carbink"}
TEARDOWN_EXIT_STATUSES = frozenset((124, -15, -9))
CANONICAL_VARIANT = "canonicalruntime"
OFF_VARIANT = "nonft-backup-off"
OFF_SYSTEM_LABEL = "Non-FT (backup off)"
METADATA_OBJECTS = ("analysis", "manifest", "plan")
RESIDENT_CONFIG_KEYS = frozenset((
    "local_resident_budget_bytes", "enable_region_resident_placement",
    "enable_resident_profile_planner", "resident_profile_apply_plan",
    "enable_region_hotness_placement", "enable_region_fetch_hotness_placement",
    "region_placement_bind_groups", "enable_logical_object_profile"))
RESIDENT_REQUIRED_ON_KEYS = frozenset((
    "enable_region_resident_placement",))
KNOWN_VARIANTS = frozenset((CANONICAL_VARIANT, OFF_VARIANT,
                            "nonft", "starfish", "hydra", "carbink"))


def source_path(path: Path) -> str:
    """Use an artifact-root-relative source, even when the CSV is copied."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(AE_ROOT))
    except ValueError:
        return str(resolved)


def _variant_metadata(record: dict, manifest: dict, plan: dict,
                      analysis_path: Path) -> tuple[str, bool | None]:
    """Normalize legacy baseline names and reject partial declarations.

    Runs produced before the variant protocol have no such keys and remain
    canonical by default.  Once a run declares either provenance field, all
    three immutable records must carry the same value; this prevents a
    hand-edited analysis file from turning backup-OFF data into the ON bar.
    """
    objects = {"analysis": record, "manifest": manifest, "plan": plan}
    variant_values = {name: obj["baseline_variant"]
                      for name, obj in objects.items()
                      if "baseline_variant" in obj}
    backup_values = {name: obj["backup_enabled"]
                     for name, obj in objects.items()
                     if "backup_enabled" in obj}
    if variant_values and set(variant_values) != set(METADATA_OBJECTS):
        raise ValueError(f"partial baseline_variant metadata: {analysis_path}")
    if backup_values and set(backup_values) != set(METADATA_OBJECTS):
        raise ValueError(f"partial backup_enabled metadata: {analysis_path}")
    if variant_values:
        raw_values = list(variant_values.values())
        if (any(not isinstance(value, str) for value in raw_values)
                or len(set(raw_values)) != 1):
            raise ValueError(f"conflicting baseline_variant metadata: {analysis_path}")
        raw = raw_values[0]
        if raw not in KNOWN_VARIANTS:
            raise ValueError(f"unknown baseline_variant {raw!r}: {analysis_path}")
        variant = OFF_VARIANT if raw == OFF_VARIANT else CANONICAL_VARIANT
    else:
        variant = CANONICAL_VARIANT
    backup_enabled = None
    if backup_values:
        raw_values = list(backup_values.values())
        if (any(type(value) is not bool for value in raw_values)
                or len(set(raw_values)) != 1):
            raise ValueError(f"conflicting backup_enabled metadata: {analysis_path}")
        backup_enabled = raw_values[0]
    if variant == OFF_VARIANT and (
            record.get("system") != "Non-FT" or plan.get("system") != "nonft"):
        raise ValueError(f"backup-OFF variant requires the Non-FT runtime: {analysis_path}")
    if variant_values and raw != CANONICAL_VARIANT and raw != OFF_VARIANT:
        if raw != plan.get("system"):
            raise ValueError(f"baseline_variant disagrees with runtime system: {analysis_path}")
    if variant == OFF_VARIANT and backup_enabled is not False:
        raise ValueError(f"backup-OFF run must declare backup_enabled=false: {analysis_path}")
    # A Non-FT run with an explicit false backup flag is only valid when it
    # carries the dedicated OFF variant; otherwise it would merge with ON.
    if (record.get("system") == "Non-FT" and backup_enabled is False
            and variant != OFF_VARIANT):
        raise ValueError(f"Non-FT backup=false lacks OFF variant: {analysis_path}")
    return variant, backup_enabled


def _config_values(text: str) -> dict[str, str]:
    """Extract simple ``key value`` settings while ignoring comments."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        fields = body.split()
        if len(fields) < 2:
            continue
        key, value = fields[0], fields[1]
        if key not in ({"enable_selective_backup", "remote_backup_budget_bytes",
                        "remote_backup_budget_pct"} | RESIDENT_CONFIG_KEYS):
            continue
        if key in values and values[key] != value:
            raise ValueError(f"duplicate {key} setting has conflicting values")
        values[key] = value
    return values


def _require_backup_off_config(text: str, source: str) -> None:
    values = _config_values(text)
    if values.get("enable_selective_backup") != "0":
        raise ValueError(f"backup-OFF {source} lacks enable_selective_backup 0")
    if values.get("remote_backup_budget_bytes") != "0":
        raise ValueError(f"backup-OFF {source} lacks zero byte budget")
    if values.get("remote_backup_budget_pct") != "0":
        raise ValueError(f"backup-OFF {source} lacks zero percentage budget")
    resident_budget = values.get("local_resident_budget_bytes")
    try:
        resident_budget_value = int(resident_budget) if resident_budget is not None else 0
    except ValueError as exc:
        raise ValueError(f"backup-OFF {source} has a nonnumeric Resident budget") from exc
    if resident_budget_value <= 0:
        raise ValueError(f"backup-OFF {source} lacks a positive Resident budget")
    missing = sorted(key for key in RESIDENT_REQUIRED_ON_KEYS
                     if values.get(key) != "1")
    if missing:
        raise ValueError(f"backup-OFF {source} lacks verified Resident ON: "
                         + ", ".join(missing))


def _require_backup_on_resident_on_config(text: str, source: str) -> None:
    values = _config_values(text)
    if values.get("enable_selective_backup") != "1":
        raise ValueError(f"canonical NonFT {source} requires backup ON")
    if values.get("enable_region_resident_placement") != "1":
        raise ValueError(f"canonical NonFT {source} requires Resident ON")
    for key in ("remote_backup_budget_bytes", "local_resident_budget_bytes"):
        value = values.get(key, "")
        if not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"canonical NonFT {source} requires positive {key}")


def _validate_backup_declaration(analysis_path: Path,
                                 backup_enabled: bool | None) -> None:
    """When declared, ensure the rendered enable bit agrees with the bytes."""
    if backup_enabled is None:
        return
    try:
        values = _config_values(
            analysis_path.with_name("effective.config").read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read declared effective.config: {analysis_path}") from exc
    rendered = values.get("enable_selective_backup")
    if rendered not in {"0", "1"} or (rendered == "1") != backup_enabled:
        raise ValueError(f"backup_enabled disagrees with effective.config: {analysis_path}")


def _validate_backup_off(analysis_path: Path, manifest: dict, plan: dict) -> None:
    """Validate both rendered bytes and the runner's declared config text."""
    effective_path = analysis_path.with_name("effective.config")
    try:
        _require_backup_off_config(
            effective_path.read_text(encoding="utf-8"), "effective.config")
    except OSError as exc:
        raise ValueError(f"cannot read backup-OFF effective.config: {analysis_path}") from exc
    declared = plan.get("effective_config")
    if declared is not None:
        if not isinstance(declared, str):
            raise ValueError(f"invalid declared effective_config: {analysis_path}")
        _require_backup_off_config(declared, "declared effective_config")


def collect(runs_dir: Path) -> tuple[list[dict], int]:
    if not runs_dir.is_dir():
        raise ValueError(f"run directory not found: {runs_dir}")
    rows = []
    failed = 0
    seen_ids = set()
    contexts: dict[str, set[tuple[str, str, str, int, int]]] = {}
    for analysis_path in sorted(runs_dir.rglob("analysis.json")):
        record = json.loads(analysis_path.read_text(encoding="utf-8"))
        teardown = record.get("status") != "passed"
        if teardown:
            failed += 1
        manifest_path = analysis_path.with_name("manifest.json")
        if not manifest_path.is_file():
            if teardown:
                continue
            raise ValueError(f"successful run has no manifest: {analysis_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        measurement = record
        if teardown:
            # The helper re-parses immutable logs and authorizes only a
            # complete Starfish KV measurement whose teardown failed.
            try:
                measurement = recover_measurement(
                    analysis_path.parent, record, manifest
                )
            except (OSError, KeyError, TypeError, ValueError):
                # A teardown-shaped case is still a failed execution when
                # work/postcheck/profile evidence is incomplete. Keep it out
                # of measured rows while allowing other runs to collect.
                continue
            if measurement is None:
                continue
            if manifest.get("status") == "passed":
                raise ValueError(
                    f"teardown measurement has a passed manifest: {analysis_path}"
                )
            if record.get("exit_status") not in TEARDOWN_EXIT_STATUSES:
                raise ValueError(
                    f"teardown measurement has an unsupported exit status: {analysis_path}"
                )
        plan = manifest.get("plan", {})
        baseline_variant, backup_enabled = _variant_metadata(
            record, manifest, plan, analysis_path)
        if (record.get("schema_version") != 1 or manifest.get("schema_version") != 1
                or (not teardown and manifest.get("status") != "passed")
                or (not teardown and record.get("exit_status") != 0)
                or (not teardown and record.get("correctness") != "pass")
                or record.get("workload") not in WORKLOADS
                or record.get("system") not in SYSTEMS
                or not isinstance(record.get("ratio"), int)
                or not 1 <= record["ratio"] <= 100
                or record.get("run_id") != analysis_path.parent.name
                or plan.get("app") != record.get("application")
                or SYSTEM_LABEL.get(plan.get("system")) != record.get("system")
                or plan.get("failure_injection_endpoint") is not None
                or record.get("failure_injection")
                or plan.get("ratio") != record["ratio"]
                or plan.get("run_id") != record["run_id"]
                or not measurement.get("measurement_phase")):
            raise ValueError(f"inconsistent measured run: {analysis_path}")
        required_files = ("client.log", "effective.config", "server.config")
        if not teardown:
            required_files += ("server.log",)
        for name in required_files:
            if not analysis_path.with_name(name).is_file():
                raise ValueError(f"measured run has no {name}: {analysis_path}")
        _validate_backup_declaration(analysis_path, backup_enabled)
        if baseline_variant == OFF_VARIANT:
            _validate_backup_off(analysis_path, manifest, plan)
        elif plan.get("system") == "nonft":
            _require_backup_on_resident_on_config(
                analysis_path.with_name("effective.config").read_text(),
                str(analysis_path.parent))
        try:
            elapsed = float(measurement["elapsed_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid measured elapsed_s: {analysis_path}") from exc
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid measured elapsed_s: {analysis_path}")
        run_id = record["run_id"]
        if run_id in seen_ids:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen_ids.add(run_id)
        workload = record["workload"]
        input_mtime = manifest.get("input_mtime_ns", -1)
        if input_mtime is None:
            if (record["application"] not in {"mg", "kv-b", "kv-a", "kv-s"}
                    or plan.get("input") is not None
                    or manifest.get("input_files") != []
                    or manifest.get("input_bytes") != 0):
                raise ValueError(f"missing input timestamp for file-backed run: {analysis_path}")
            input_mtime = -1
        measurement_phase = measurement["measurement_phase"]
        contexts.setdefault(workload, set()).add((
            str(record["environment"]), str(measurement_phase),
            str(plan.get("input")), int(manifest.get("input_bytes", -1)),
            int(input_mtime),
        ))
        csv_system = (OFF_SYSTEM_LABEL if baseline_variant == OFF_VARIANT
                      else record["system"])
        rows.append({
            "workload": workload, "system": csv_system,
            "baseline_variant": baseline_variant,
            "backup_enabled": ("" if backup_enabled is None
                                else str(backup_enabled).lower()),
            "ratio": record["ratio"], "elapsed_s": f"{elapsed:.9g}",
            "source_type": "measured",
            "source": source_path(analysis_path),
            "run_id": run_id, "exit_status": str(record["exit_status"]),
            "correctness": "pass", "environment": record["environment"],
            "measurement_phase": measurement_phase,
            "measurement_usable": "1",
            "execution_status": "teardown_failed" if teardown else "passed",
            "warning": str(measurement.get("measurement_warning", ""))
            if teardown else "",
        })
    for workload, context in contexts.items():
        if len(context) != 1:
            raise ValueError(f"{workload}: mixed input, environment, or measurement phase")
    rows.sort(key=lambda r: (r["workload"], r["system"], r["ratio"], r["run_id"]))
    return rows, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows, failed = collect(args.runs_dir)
        if args.output.exists():
            raise ValueError(f"refusing to overwrite CSV: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}; measured_rows={len(rows)}; failed_runs={failed}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
