"""Schema-v3 contract for Figure 11 measured run records.

Producers emit one whitespace-separated key=value line after a complete Work
interval, verification, process exit, and owned-service cleanup.  Component
availability is declared by the comma-separated components field. Schema-v1
peak-memory and schema-v2 request-progress memory records are rejected.
"""

from __future__ import annotations

import math
import shlex
from collections.abc import Mapping

PREFIX = "figure11_result"
SCHEMA_VERSION = 3
COMPONENTS = (
    "fetch_traffic",
    "eviction_traffic",
    "remote_cpu_cores",
    "remote_memory",
)
ALL_COMPONENTS = COMPONENTS
REMOTE_MEMORY_SAMPLING = "work_start_10_20_30_40_50s"
REMOTE_MEMORY_WINDOW = "work"
REMOTE_MEMORY_ORIGIN = "profile_start_work"
KVS_WORKLOADS = frozenset(("kv-a", "kv-b", "kv-s"))
REMOTE_MEMORY_GOOD_END_STATUSES = frozenset(("complete", "partial-work"))
REMOTE_MEMORY_REJECTED_END_STATUSES = frozenset(
    ("short-work", "unsupported", "work-end-missing"))
REMOTE_MEMORY_PROFILES = {
    "bfs": {"sampling": "work_start_3s", "points": (3,), "expected_samples": 1},
    "default": {
        "sampling": REMOTE_MEMORY_SAMPLING,
        "points": (10, 20, 30, 40, 50),
        "expected_samples": 5,
    },
}
REMOTE_CPU_WINDOW = "initialization_and_requests"
MEMORY_METRICS = {
    "nonft": "allocator_occupied_bytes",
    "starfish": "allocator_occupied_bytes",
    "hydra": "live_encoded_page_bytes",
    "carbink": "live_stripe_allocated_bytes",
}
WORKLOADS = {
    "bfs": "BFS",
    "llama": "LLM",
    "mg": "MG",
    "wc": "WC",
    "kv-b": "KV-B",
    "kv-a": "KV-A",
    "kv-s": "KV-S",
    "nq": "NQ",
}
SYSTEMS = {
    "hydra": "Hydra",
    "carbink": "Carbink",
    "starfish": "Starfish",
    "nonft": "Non-FT (backup off)",
}
ALIASES = {
    "llm": "llama",
    "wordcount": "wc",
    "kv_b": "kv-b",
    "kv_a": "kv-a",
    "kv_s": "kv-s",
    "non-ft": "nonft",
    "nonft-backup-off": "nonft",
    "non-ft (backup off)": "nonft",
}
CONTEXT_FIELDS = (
    "environment",
    "workload_id",
    "ratio",
    "app_workers",
    "repeat",
    "phase",
)
CORE_FIELDS = (
    "schema_version",
    "workload",
    "system",
    "run_id",
    "environment",
    "workload_id",
    "ratio",
    "app_workers",
    "repeat",
    "phase",
    "components",
    "elapsed_s",
    "exit_status",
    "correctness",
    "source_type",
)
COMPONENT_FIELDS = {
    "fetch_traffic": ("fetch_bytes",),
    "eviction_traffic": ("eviction_bytes",),
    "remote_cpu_cores": (
        "remote_cpu_seconds",
        "remote_cpu_elapsed_s",
        "remote_cpu_window",
    ),
    "remote_memory": (
        "remote_memory_mean_bytes",
        "remote_memory_samples",
        "remote_memory_sampling",
        "remote_memory_metric",
        "remote_memory_window",
        "remote_memory_origin",
        "remote_memory_work_window_id",
        "remote_memory_end_monotonic_ns",
        "remote_memory_scheduled_samples",
        "remote_memory_missing_samples",
        "remote_memory_expected_samples",
        "remote_memory_work_duration_ns",
        "remote_memory_missing_times_s",
        "remote_memory_end_status",
        "remote_memory_scheduled_times_s",
        "remote_memory_sample_times_s",
    ),
}
LEGACY_FIELDS = frozenset(("remote_memory_peak_bytes",))


def _required(record, keys):
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError("missing log fields: " + ", ".join(sorted(missing)))


def _nonblank(record, keys):
    for key in keys:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must not be blank")


def _integer(value, key, *, allow_negative=False):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        sign = text.startswith("-")
        digits = text[1:] if sign else text
        if not digits or not digits.isascii() or not digits.isdecimal():
            raise ValueError(f"{key} must be an integer")
        result = int(text)
    else:
        raise ValueError(f"{key} must be an integer")
    if not allow_negative and result < 0:
        raise ValueError(f"{key} must be nonnegative")
    return result


def _number(value, key, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "finite and nonnegative"
        raise ValueError(f"{key} must be {qualifier}")
    return result


def backup_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("0", "false", "off"):
        return False
    if isinstance(value, str) and value.lower() in ("1", "true", "on"):
        return True
    raise ValueError("backup_enabled must explicitly be true or false")


def normalize_components(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(part).strip() for part in value]
    else:
        raise ValueError("components must be a comma-separated list")
    if not parts or any(not part for part in parts):
        raise ValueError("components must not be empty")
    unknown = sorted(set(parts) - set(COMPONENTS))
    if unknown:
        raise ValueError("unknown component(s): " + ", ".join(unknown))
    if len(set(parts)) != len(parts):
        raise ValueError("components must not contain duplicates")
    return tuple(component for component in COMPONENTS if component in parts)


def _sample_times(value, key):
    parts = value.split(",") if isinstance(value, str) else value
    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError(f"{key} must be a nonempty sample-time sequence")
    return [_number(part, key, positive=True) for part in parts]


def component_names(record):
    private = record.get("_components")
    if private is not None:
        return tuple(private)
    return normalize_components(record.get("components"))


def memory_profile(workload):
    profile = dict(REMOTE_MEMORY_PROFILES["bfs" if workload == "bfs" else "default"])
    profile["origin"] = (
        "kvs_request_start" if workload in KVS_WORKLOADS
        else REMOTE_MEMORY_ORIGIN
    )
    return profile


def validate_record(record, *, source=None):
    """Validate and normalize a schema-v3 record dictionary."""
    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    result = dict(record)
    if LEGACY_FIELDS.intersection(result):
        raise ValueError("schema-v1 peak memory fields are not accepted")
    _required(result, CORE_FIELDS)
    _nonblank(result, ("workload", "system", "run_id", "environment",
                       "workload_id", "phase", "correctness", "source_type"))
    result["schema_version"] = _integer(result["schema_version"], "schema_version")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported Figure 11 log schema; schema v1/v2 are rejected")
    result["ratio"] = _integer(result["ratio"], "ratio")
    result["app_workers"] = _integer(result["app_workers"], "app_workers")
    result["repeat"] = _integer(result["repeat"], "repeat")
    result["exit_status"] = _integer(result["exit_status"], "exit_status",
                                      allow_negative=True)
    result["elapsed_s"] = _number(result["elapsed_s"], "elapsed_s", positive=True)
    if not 1 <= result["ratio"] <= 100:
        raise ValueError("ratio must be in 1..100")
    if result["app_workers"] < 1 or result["repeat"] < 1:
        raise ValueError("app_workers and repeat must be positive")
    if result["phase"] != "work":
        raise ValueError("Figure 11 expects the complete Work phase")
    if result["source_type"] != "measured":
        raise ValueError("collector accepts measured records only")

    workload = ALIASES.get(str(result["workload"]).lower(),
                           str(result["workload"]).lower())
    system = ALIASES.get(str(result["system"]).lower(),
                         str(result["system"]).lower())
    if workload not in WORKLOADS or system not in SYSTEMS:
        raise ValueError("unknown workload or system")
    result["workload"] = workload
    result["system"] = system
    if system == "nonft":
        _required(result, ("baseline_variant", "backup_enabled"))
        if result["baseline_variant"] not in ("nonft", "nonft-backup-off"):
            raise ValueError("unknown NonFT baseline_variant")
        result["backup_enabled"] = backup_flag(result["backup_enabled"])
        if (result["baseline_variant"] == "nonft-backup-off"
                and result["backup_enabled"]):
            raise ValueError("NonFT backup-off variant has backup enabled")
        if "off_policy" in result:
            if result["off_policy"] not in (
                    "backup_and_resident_off", "backup_only_off", "resident_unverified"):
                raise ValueError("unknown NonFT off_policy")
            if result.get("local_resident_budget_bytes") is not None:
                result["local_resident_budget_bytes"] = _integer(
                    result["local_resident_budget_bytes"], "local_resident_budget_bytes")
            if (result["off_policy"] == "backup_and_resident_off"
                    and result.get("local_resident_budget_bytes") != 0):
                raise ValueError("double-OFF requires an explicit zero resident budget")

    selected = normalize_components(result["components"])
    result["components"] = ",".join(selected)
    result["_components"] = selected
    for component in selected:
        _required(result, COMPONENT_FIELDS[component])

    if "fetch_traffic" in selected:
        result["fetch_bytes"] = _integer(result["fetch_bytes"], "fetch_bytes")
    if "eviction_traffic" in selected:
        result["eviction_bytes"] = _integer(
            result["eviction_bytes"], "eviction_bytes")
    if "remote_cpu_cores" in selected:
        result["remote_cpu_seconds"] = _number(
            result["remote_cpu_seconds"], "remote_cpu_seconds")
        result["remote_cpu_elapsed_s"] = _number(
            result["remote_cpu_elapsed_s"], "remote_cpu_elapsed_s",
            positive=True)
        if result["remote_cpu_window"] != REMOTE_CPU_WINDOW:
            raise ValueError("remote_cpu_window must be initialization_and_requests")
    if "remote_memory" in selected:
        profile = memory_profile(result["workload"])
        result["remote_memory_mean_bytes"] = _number(
            result["remote_memory_mean_bytes"], "remote_memory_mean_bytes")
        result["remote_memory_samples"] = _integer(
            result["remote_memory_samples"], "remote_memory_samples")
        if result["remote_memory_samples"] < 1:
            raise ValueError(
                "Work window has no successful memory samples; short Work is missing data"
            )
        if result["remote_memory_samples"] > profile["expected_samples"]:
            raise ValueError("remote_memory_samples exceeds workload profile")
        if result["remote_memory_sampling"] != profile["sampling"]:
            raise ValueError(
                f"remote_memory_sampling must be {profile['sampling']} for "
                f"{result['workload']}")
        if result["remote_memory_window"] != REMOTE_MEMORY_WINDOW:
            raise ValueError("remote_memory_window must be work")
        if result["remote_memory_origin"] != profile["origin"]:
            raise ValueError(
                f"remote_memory_origin must be {profile['origin']} for {result['workload']}"
            )
        result["remote_memory_work_window_id"] = _integer(
            result["remote_memory_work_window_id"], "remote_memory_work_window_id")
        if result["remote_memory_work_window_id"] < 1:
            raise ValueError("remote_memory_work_window_id must be positive")
        result["remote_memory_end_monotonic_ns"] = _integer(
            result["remote_memory_end_monotonic_ns"],
            "remote_memory_end_monotonic_ns")
        result["remote_memory_work_duration_ns"] = _integer(
            result["remote_memory_work_duration_ns"],
            "remote_memory_work_duration_ns")
        if result["remote_memory_work_duration_ns"] <= 0:
            raise ValueError("remote_memory_work_duration_ns must be positive")
        result["remote_memory_scheduled_samples"] = _integer(
            result["remote_memory_scheduled_samples"],
            "remote_memory_scheduled_samples")
        result["remote_memory_missing_samples"] = _integer(
            result["remote_memory_missing_samples"],
            "remote_memory_missing_samples")
        result["remote_memory_expected_samples"] = _integer(
            result["remote_memory_expected_samples"],
            "remote_memory_expected_samples")
        if result["remote_memory_expected_samples"] != profile["expected_samples"]:
            raise ValueError("remote_memory_expected_samples disagrees with workload profile")
        if result["remote_memory_scheduled_samples"] > result["remote_memory_expected_samples"]:
            raise ValueError("remote_memory_scheduled_samples exceeds profile")
        duration_ns = result["remote_memory_work_duration_ns"]
        due = sum(point * 1_000_000_000 <= duration_ns
                  for point in profile["points"])
        if result["remote_memory_scheduled_samples"] != due:
            raise ValueError("remote memory scheduled points disagree with Work duration")
        if (result["remote_memory_samples"]
                + result["remote_memory_missing_samples"]
                != result["remote_memory_scheduled_samples"]):
            raise ValueError("remote memory sample/missing counts do not close scheduled points")
        end_status = result["remote_memory_end_status"]
        if not isinstance(end_status, str) or not end_status.strip():
            raise ValueError("remote_memory_end_status must not be blank")
        if end_status in REMOTE_MEMORY_REJECTED_END_STATUSES:
            raise ValueError(
                f"remote_memory_end_status={end_status} cannot enter formal Figure 11"
            )
        if end_status not in REMOTE_MEMORY_GOOD_END_STATUSES:
            raise ValueError(
                f"unsupported remote_memory_end_status={end_status}"
            )
        if end_status == "complete":
            if (result["remote_memory_scheduled_samples"]
                    != profile["expected_samples"]
                    or result["remote_memory_samples"]
                    != profile["expected_samples"]):
                raise ValueError(
                    "remote_memory_end_status=complete disagrees with profile counts"
                )
        elif (result["remote_memory_scheduled_samples"] <= 0
              or result["remote_memory_samples"] >= profile["expected_samples"]):
            raise ValueError(
                "remote_memory_end_status=partial-work disagrees with profile counts"
            )
        scheduled = _sample_times(result["remote_memory_scheduled_times_s"],
                                  "remote_memory_scheduled_times_s")
        observed = _sample_times(result["remote_memory_sample_times_s"],
                                 "remote_memory_sample_times_s")
        missing_value = result["remote_memory_missing_times_s"]
        if result["remote_memory_missing_samples"]:
            missing = _sample_times(missing_value, "remote_memory_missing_times_s")
            result["remote_memory_missing_times_s"] = missing
        else:
            # Direct whitespace-delimited records still need a nonempty token;
            # normalize the explicit no-missing sentinel for CSV consumers.
            result["remote_memory_missing_times_s"] = []
            missing = []
        allowed = profile["points"][:result["remote_memory_scheduled_samples"]]
        successful_targets = [target for target in allowed if target not in missing]
        if (len(scheduled) != result["remote_memory_scheduled_samples"] or
                len(observed) != result["remote_memory_samples"] or
                len(missing) != result["remote_memory_missing_samples"] or
                scheduled != list(allowed) or
                any(value not in allowed for value in missing) or
                len(set(missing)) != len(missing)):
            raise ValueError("memory Work sample/missing times do not match scheduled points")
        if observed != sorted(set(observed)) or any(
                actual < target
                for actual, target in zip(observed, successful_targets)):
            raise ValueError("observed Work memory times must be increasing and not precede targets")
        if any(actual > duration_ns / 1e9 + 1e-6 for actual in observed):
            raise ValueError("observed Work memory times occur after Work end")
        result["remote_memory_scheduled_times_s"] = scheduled
        result["remote_memory_sample_times_s"] = observed
        expected_metric = MEMORY_METRICS[system]
        if result["remote_memory_metric"] != expected_metric:
            raise ValueError(
                f"{system} remote_memory_metric must be {expected_metric}")

    if source is not None:
        result.setdefault("_source", str(source))
    return result


def parse_line(line):
    """Return None for unrelated log output; malformed records fail closed."""
    stripped = line.strip()
    if not stripped or stripped.split(maxsplit=1)[0] != PREFIX:
        return None
    record = {}
    for token in shlex.split(stripped)[1:]:
        key, separator, value = token.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"expected a nonempty key=value token, got {token!r}")
        if key in record:
            raise ValueError(f"duplicate log field: {key}")
        record[key] = value
    return validate_record(record)
