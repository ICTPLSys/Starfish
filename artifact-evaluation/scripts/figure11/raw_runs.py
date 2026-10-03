"""Fail-closed adapter for Figure 11 run directories.

The formal memory evidence is schema-v3 runtime_remote_memory sampled at
10/20/30/40/50 seconds from the explicit profile Work start.  Program-start
schema-v2 and legacy request-progress records are rejected and never converted.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import sys
from pathlib import Path

try:
    from . import log_contract
except ImportError:
    try:
        import log_contract
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import log_contract


TIMED_SCHEMA_VERSION = 3
TIMED_SAMPLING = "work_start_10_20_30_40_50s"
TIMED_POINTS = (10, 20, 30, 40, 50)
TIMED_ORIGIN = "profile_start_work"
TIMED_PROFILES = {
    "bfs": {
        "sampling": "work_start_3s",
        "points": (3,),
        "origin": "profile_start_work",
    },
    "default": {
        "sampling": TIMED_SAMPLING,
        "points": TIMED_POINTS,
        "origin": "profile_start_work",
    },
}
KVS_WORKLOADS = frozenset(("kv-a", "kv-b", "kv-s"))
TIMED_GOOD_END_STATUSES = frozenset(("complete", "partial-work"))
TIMED_REJECTED_END_STATUSES = frozenset(
    ("short-work", "unsupported", "work-end-missing"))
MEMORY_RESULTS = ("remote-memory-result.json", "memory-result.json")
METADATA_NAMES = ("analysis.json", "result.json", "run-result.json",
                  "manifest.json", "plan.json")
SUCCESS = frozenset(("pass", "passed", "success", "succeeded", "ok",
                     "complete", "completed", "true", "1"))
FAILURE = frozenset(("fail", "failed", "failure", "error", "aborted",
                     "incomplete", "running", "false", "0"))


class _SkipRun(Exception):
    """A known failed attempt which should not poison a retry selection."""


def _timed_profile(workload):
    profile = TIMED_PROFILES["bfs" if workload == "bfs" else "default"]
    if workload in KVS_WORKLOADS:
        return {**profile, "origin": "kvs_request_start"}
    return profile


def _load(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _fields(line: str):
    result = {}
    for token in line.strip().split()[1:]:
        key, sep, value = token.partition("=")
        if not sep or not key or not value or key in result:
            raise ValueError(f"malformed key=value line: {line.strip()}")
        result[key] = value
    return result


def _int(value, key: str, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        result = int(value.strip())
    else:
        raise ValueError(f"{key} must be an integer")
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{key} must be {'positive' if positive else 'nonnegative'}")
    return result


def _float(value, key: str, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{key} must be {'positive' if positive else 'finite and nonnegative'}")
    return result


def _first(layers, *keys):
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for key in keys:
            value = layer.get(key)
            if value is not None and value != "":
                return value
    return None


def _layers(documents):
    result = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        result.append(document)
        for key in ("metadata", "run", "result", "summary"):
            value = document.get(key)
            if isinstance(value, dict):
                result.append(value)
        if isinstance(document.get("plan"), dict):
            result.append(document["plan"])
    return result


def _named_documents(documents, files):
    """Associate loaded metadata documents with their canonical filenames."""
    # _documents() appends documents in the same fixed filename order used to
    # populate ``files``.  Keeping the names here lets repeat provenance use
    # only authoritative analysis/manifest/plan fields rather than arbitrary
    # workload metadata layers.
    return {name: document for name, document in zip(files, documents)}


def _repeat_suffix(value, source):
    """Return a positive trailing ``-rN`` index, or None if unindexed."""
    if value is None:
        return None
    match = re.search(r"-r([0-9]+)$", str(value).strip())
    if match is None:
        return None
    return _int(match.group(1), source, positive=True)


def _observed_repeat(documents, files, layers, run_dir):
    """Resolve the batch repeat without confusing workload repetitions.

    New common-run metadata records an explicit positive ``repeat`` in the
    analysis, manifest, and plan.  Older records have no such field, so their
    anchored run-id/directory ``-rN`` suffix is the only safe inference.  A
    legacy unindexed directory remains repeat one.  Application-level
    ``repetitions`` and ``measured_repetitions`` are deliberately excluded.
    """
    named = _named_documents(documents, files)
    candidates = []

    def add(document, source):
        if isinstance(document, dict) and "repeat" in document:
            candidates.append((source, _int(
                document["repeat"], source + ".repeat", positive=True)))

    add(named.get("analysis.json"), "analysis.json")
    manifest = named.get("manifest.json")
    add(manifest, "manifest.json")
    if isinstance(manifest, dict):
        add(manifest.get("plan"), "manifest.plan")
    add(named.get("plan.json"), "plan.json")

    explicit_values = {value for _, value in candidates}
    if len(explicit_values) > 1:
        detail = ", ".join(f"{source}={value}" for source, value in candidates)
        raise ValueError(f"repeat metadata disagrees: {run_dir} ({detail})")
    explicit = next(iter(explicit_values), None)

    run_id = _first(layers, "run_id", "run_name")
    run_id_repeat = _repeat_suffix(run_id, "run_id repeat suffix")
    directory_repeat = _repeat_suffix(run_dir.name, "directory repeat suffix")
    if (run_id_repeat is not None and directory_repeat is not None
            and run_id_repeat != directory_repeat):
        raise ValueError(
            f"run-id and directory repeat suffixes disagree: {run_dir} "
            f"(run_id={run_id_repeat}, directory={directory_repeat})")
    suffix_repeat = (run_id_repeat if run_id_repeat is not None
                     else directory_repeat)
    if explicit is not None:
        if suffix_repeat is not None and explicit != suffix_repeat:
            raise ValueError(
                f"explicit repeat disagrees with run-id/directory suffix: {run_dir} "
                f"(explicit={explicit}, suffix={suffix_repeat})")
        return explicit
    return 1 if suffix_repeat is None else suffix_repeat


def _workload(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    text = {"kvb": "kv-b", "kva": "kv-a", "kvs": "kv-s",
            "llm": "llama", "wordcount": "wc"}.get(text, text)
    aliases = getattr(log_contract, "ALIASES", {})
    text = aliases.get(text, text)
    if text in log_contract.WORKLOADS:
        return text
    match = re.fullmatch(r"kv-?([abs])(?:[-_].*)?", text)
    return "kv-" + match.group(1) if match else None


def _system(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if text in ("nonft", "non-ft", "nonft-backup-off", "non-ft-(backup-off)"):
        return "nonft"
    if text in log_contract.SYSTEMS:
        return text
    for key, label in log_contract.SYSTEMS.items():
        if text == label.lower().replace(" ", "-"):
            return key
    return None


def _discover(root: Path, pattern: str):
    found = {}
    for name in METADATA_NAMES:
        for path in root.rglob(name):
            if not path.is_file():
                continue
            run_dir = path.parent
            logs = sorted(p for p in run_dir.iterdir()
                          if p.is_file() and fnmatch.fnmatch(p.name, pattern))
            if logs:
                found.setdefault(run_dir, {"logs": logs})[name] = path
    return sorted(found.items(), key=lambda item: str(item[0]))


def _documents(run_dir: Path, paths: dict):
    documents, files = [], {}
    for name in ("analysis.json", "result.json", "run-result.json",
                 "manifest.json", "plan.json"):
        path = paths.get(name)
        if path is not None:
            documents.append(_load(path))
            files[name] = path
    if not documents:
        raise ValueError(f"run metadata is missing: {run_dir}")
    if paths.get("analysis.json") is None or paths.get("manifest.json") is None:
        raise ValueError(f"analysis.json and manifest.json are required: {run_dir}")
    analysis = documents[0]
    manifest_index = list(files).index("manifest.json")
    manifest = documents[manifest_index]
    plan = manifest.get("plan") if isinstance(manifest, dict) else None
    plan = plan if isinstance(plan, dict) else {}
    for key in ("run_id", "ratio"):
        left = analysis.get(key)
        right = manifest.get(key, plan.get(key))
        if left is not None and right is not None and str(left) != str(right):
            raise ValueError(f"same-run metadata identity mismatch for {key}: {run_dir}")
    left_workload = _workload(analysis.get("application", analysis.get("app", analysis.get("workload"))))
    right_workload = _workload(plan.get("app", plan.get("workload")))
    if left_workload and right_workload and left_workload != right_workload:
        raise ValueError(f"same-run metadata identity mismatch for workload: {run_dir}")
    left_system = _system(analysis.get("system", analysis.get("runtime")))
    right_system = _system(plan.get("system", plan.get("runtime")))
    if left_system and right_system and left_system != right_system:
        raise ValueError(f"same-run metadata identity mismatch for system: {run_dir}")
    return documents, files


def _manifest_endpoint_count(documents):
    for document in documents:
        for holder in (document, document.get("plan")):
            if not isinstance(holder, dict):
                continue
            endpoints = holder.get("endpoints") or holder.get("memory_endpoints")
            if isinstance(endpoints, list):
                return len(endpoints)
    return None


def _is(value, words):
    if isinstance(value, bool):
        return value if "true" in words else not value
    return str(value).strip().lower() in words


def _eligibility(layers, run_dir: Path):
    statuses = [layer[key] for layer in layers
                for key in ("status", "run_status", "state")
                if key in layer and layer[key] is not None]
    if any(_is(value, FAILURE) for value in statuses):
        raise _SkipRun(f"failed run: {run_dir}")
    passed = [layer["passed"] for layer in layers if "passed" in layer]
    if any(value is False or _is(value, FAILURE) for value in passed):
        raise _SkipRun(f"failed correctness: {run_dir}")
    correctness = [layer[key] for layer in layers
                   for key in ("correctness", "measurement_correctness",
                               "correctness_status")
                   if key in layer and layer[key] is not None]
    if any(not _is(value, SUCCESS) for value in correctness):
        raise _SkipRun(f"failed correctness: {run_dir}")
    exits = [_int(layer[key], key) for layer in layers
             for key in ("exit_status", "exit_code")
             if key in layer and layer[key] is not None]
    if not exits:
        raise ValueError(f"final exit status is missing: {run_dir}")
    if any(value != 0 for value in exits):
        raise _SkipRun(f"nonzero exit status: {run_dir}")
    usable = [layer["measurement_usable"] for layer in layers
              if "measurement_usable" in layer]
    if any(value is False or _is(value, FAILURE) for value in usable):
        raise _SkipRun(f"measurement is not usable: {run_dir}")
    for layer in layers:
        if isinstance(layer.get("error"), str) and layer["error"].strip():
            raise _SkipRun(f"run cleanup/error: {run_dir}")
        endpoints = layer.get("endpoints")
        if isinstance(endpoints, list):
            for endpoint in endpoints:
                if not isinstance(endpoint, dict):
                    raise ValueError(f"malformed endpoint metadata: {run_dir}")
                if str(endpoint.get("status", "")).lower() in {
                        "failed", "start_failed", "cleanup_failed",
                        "log_unavailable", "running", "starting"}:
                    raise _SkipRun(f"incomplete endpoint: {run_dir}")
    if not correctness:
        strong_proof = any(layer.get("correctness_evidence") or
                           layer.get("post_verify") or
                           layer.get("verification") for layer in layers)
        if not (any(_is(value, SUCCESS) for value in passed) and strong_proof):
            raise ValueError(f"correctness pass/proof is missing: {run_dir}")


def _request_evidence(log: str, run_dir: Path):
    starts = re.findall(r"^kvs_phase name=direct event=request_start\b.*$", log, re.M)
    ends = re.findall(r"^kvs_phase name=direct event=request_drain_end\b.*$", log, re.M)
    receipts = re.findall(r"^kvs_receipt name=direct\b.*$", log, re.M)
    verifies = re.findall(r"^kvs_post_verify\b.*$", log, re.M)
    result = {}
    if starts or ends:
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(f"incomplete or duplicated KVS Work markers: {run_dir}")
        begin = _fields(starts[0])
        end = _fields(ends[0])
        t0 = _int(begin.get("monotonic_ns"), "request_start.monotonic_ns")
        t1 = _int(end.get("monotonic_ns"), "request_drain_end.monotonic_ns")
        if t1 <= t0:
            raise ValueError(f"KVS Work markers are not ordered: {run_dir}")
        result["elapsed_s"] = (t1 - t0) / 1e9
        if "completed" in end:
            result["request_count"] = _int(end["completed"], "completed")
    if receipts:
        if len(receipts) != 1:
            raise ValueError(f"duplicated KVS receipts: {run_dir}")
        fields = _fields(receipts[0])
        required = ("generated_get", "generated_put", "generated_remove",
                    "accepted_get", "accepted_put", "accepted_remove",
                    "completed_get", "completed_put", "completed_remove",
                    "accepted_equals_completed", "request_fingerprint")
        if any(key not in fields for key in required):
            raise ValueError(f"incomplete KVS receipt: {run_dir}")
        for op in ("get", "put", "remove"):
            values = [_int(fields[f"{stage}_{op}"], f"{stage}_{op}")
                      for stage in ("generated", "accepted", "completed")]
            if len(set(values)) != 1:
                raise _SkipRun(f"request correctness failed: {run_dir}")
        if fields["accepted_equals_completed"] != "1":
            raise _SkipRun(f"request correctness failed: {run_dir}")
        result["request_fingerprint"] = fields["request_fingerprint"]
        result["request_count"] = sum(
            _int(fields[f"completed_{op}"], f"completed_{op}")
            for op in ("get", "put", "remove"))
    if verifies:
        if len(verifies) != 1:
            raise ValueError(f"duplicated KVS post-verification: {run_dir}")
        fields = _fields(verifies[0])
        if "checked" not in fields or "failures" not in fields:
            raise ValueError(f"incomplete KVS post-verification: {run_dir}")
        if _int(fields["checked"], "post_verify.checked") <= 0:
            raise ValueError(f"post-verification checked no values: {run_dir}")
        if _int(fields["failures"], "post_verify.failures") != 0:
            raise _SkipRun(f"post-verification failed: {run_dir}")
    return result


def _traffic(candidate, source: str):
    if not isinstance(candidate, dict):
        raise ValueError(f"traffic record must be an object: {source}")
    if candidate.get("phase") not in (None, "work"):
        raise ValueError(f"traffic record is not Work: {source}")
    if candidate.get("fetch_bytes") is None or candidate.get("eviction_bytes") is None:
        raise ValueError(f"traffic lacks fetch/eviction bytes: {source}")
    fetch = _int(candidate["fetch_bytes"], "fetch_bytes")
    eviction = _int(candidate["eviction_bytes"], "eviction_bytes")
    if "read_bytes" in candidate and fetch != _int(candidate["read_bytes"], "read_bytes"):
        raise ValueError(f"fetch_bytes disagrees with read_bytes: {source}")
    if "write_bytes" in candidate or "rmw_read_bytes" in candidate:
        write = _int(candidate.get("write_bytes", 0), "write_bytes")
        rmw = _int(candidate.get("rmw_read_bytes", 0), "rmw_read_bytes")
        if eviction != write + rmw:
            raise ValueError(f"eviction_bytes double-count or mismatch: {source}")
    return {"fetch_bytes": fetch, "eviction_bytes": eviction}


def _traffic_log(log: str, run_dir: Path):
    lines = re.findall(r"^runtime_traffic_work\b.*$", log, re.M)
    if not lines:
        return None
    values, intervals, window_ids = [], [], set()
    for index, line in enumerate(lines):
        fields = _fields(line)
        if fields.get("schema_version") != "1" or fields.get("scope") != "client_rdma":
            raise ValueError(f"unsupported Work traffic schema: {run_dir}")
        window_id = fields.get("window_id", str(index))
        if window_id in window_ids:
            raise ValueError(f"duplicate Work traffic window_id: {run_dir}")
        window_ids.add(window_id)
        if "start_monotonic_ns" in fields or "end_monotonic_ns" in fields:
            if "start_monotonic_ns" not in fields or "end_monotonic_ns" not in fields:
                raise ValueError(f"incomplete Work traffic time bounds: {run_dir}")
            start = _int(fields["start_monotonic_ns"], "start_monotonic_ns")
            end = _int(fields["end_monotonic_ns"], "end_monotonic_ns")
            if end <= start:
                raise ValueError(f"invalid Work traffic time bounds: {run_dir}")
            intervals.append((start, end))
        values.append(_traffic(fields, str(run_dir / "client.log")))
    intervals.sort()
    if any(previous_end > start for (_, previous_end), (start, _) in
           zip(intervals, intervals[1:])):
        raise ValueError(f"overlapping Work traffic windows: {run_dir}")
    return {"fetch_bytes": sum(value["fetch_bytes"] for value in values),
            "eviction_bytes": sum(value["eviction_bytes"] for value in values)}


def _traffic_docs(documents, run_dir: Path):
    values = []
    for document in documents:
        for key in ("traffic", "work_traffic", "runtime_traffic_work"):
            item = document.get(key)
            if isinstance(item, dict):
                values.append(_traffic(item, f"{run_dir}:{key}"))
            elif isinstance(item, list):
                values.extend(_traffic(value, f"{run_dir}:{key}") for value in item)
        if {"fetch_bytes", "eviction_bytes"} <= document.keys():
            values.append(_traffic(document, f"{run_dir}:metadata"))
    if len(values) > 1 and any(value != values[0] for value in values[1:]):
        raise ValueError(f"traffic sources disagree: {run_dir}")
    return values[0] if values else None


def _memory_metric(documents):
    values = []
    for document in documents:
        for key in ("metric", "remote_memory_metric", "memory_metric"):
            if document.get(key) is not None:
                values.append(str(document[key]))
        for holder in (document, document.get("plan")):
            if isinstance(holder, dict) and isinstance(holder.get("client_env"), dict):
                value = holder["client_env"].get("FARLIB_KVS_MEMORY_METRIC")
                if value:
                    values.append(str(value))
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError("memory metric sources disagree")
    return values[0]


def _endpoint_sum(value, source, endpoint_count=None):
    if value is None:
        raise ValueError(f"endpoint_bytes is missing: {source}")
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, str):
        pairs = []
        for token in value.split(","):
            index, separator, item = token.partition(":")
            if not separator or not index:
                raise ValueError(f"malformed endpoint_bytes: {source}")
            pairs.append((index, item))
    else:
        raise ValueError(f"malformed endpoint_bytes: {source}")
    indexes = []
    total = 0
    for index, item in pairs:
        try:
            numeric_index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed endpoint index: {source}") from exc
        if numeric_index < 0 or numeric_index in indexes:
            raise ValueError(f"duplicate endpoint index: {source}")
        indexes.append(numeric_index)
        total += _int(item, "endpoint_bytes")
    if endpoint_count is None:
        endpoint_count = len(indexes)
    endpoint_count = _int(endpoint_count, "endpoint_count", positive=True)
    if sorted(indexes) != list(range(endpoint_count)):
        raise ValueError(f"endpoint indexes are not contiguous: {source}")
    return total


def _validate_end_status(status, scheduled_count, sample_count,
                         missing_count, expected_count, source):
    if status is None or not str(status).strip():
        raise ValueError(f"runtime memory end status is missing: {source}")
    status = str(status)
    if status in TIMED_REJECTED_END_STATUSES:
        raise ValueError(
            f"runtime memory end status={status} cannot enter formal Figure 11: {source}"
        )
    if status not in TIMED_GOOD_END_STATUSES:
        raise ValueError(
            f"unsupported runtime memory end status={status}: {source}"
        )
    if status == "complete":
        if (scheduled_count != expected_count or sample_count != expected_count):
            raise ValueError(
                f"runtime memory status=complete disagrees with scheduled/sample counts: {source}"
            )
    elif scheduled_count <= 0 or sample_count >= expected_count:
        raise ValueError(
            f"runtime memory status=partial-work disagrees with Work profile counts: {source}"
        )
    return status


def _timed_samples(samples, missing, metric, system, source, endpoint_count,
                   expected_origin, expected_end_metric, expected_window_id,
                   end_ns, points, scheduled_count, expected_sampling,
                   expected_origin_label):
    if not isinstance(samples, list) or not 1 <= len(samples) <= len(points):
        raise ValueError(
            f"timed Work memory requires at least one successful sample: {source}"
        )
    expected_metric = log_contract.MEMORY_METRICS[system]
    if metric != expected_metric:
        raise ValueError(f"{system} memory metric must be {expected_metric}: {source}")
    if expected_end_metric != expected_metric:
        raise ValueError(f"timed memory end metric mismatch: {source}")
    if not isinstance(missing, list):
        raise ValueError(f"timed Work missing records are malformed: {source}")
    if scheduled_count < 0 or scheduled_count > len(points):
        raise ValueError(f"scheduled sample count is outside the profile: {source}")
    if len(samples) + len(missing) != scheduled_count:
        raise ValueError(
            f"sample/missing count does not close scheduled Work points: {source}"
        )
    seen = set()
    normalized, observed = [], []
    duration_ns = end_ns - expected_origin
    if duration_ns < 0:
        raise ValueError(f"Work end precedes Work origin: {source}")

    def common(item, label):
        if not isinstance(item, dict):
            raise ValueError(f"malformed {label}: {source}")
        if str(item.get("schema_version")) != str(TIMED_SCHEMA_VERSION):
            raise ValueError(f"{label} schema is not 3; old program-start data is rejected: {source}")
        if item.get("origin") != expected_origin_label:
            raise ValueError(
                f"{label} origin is not {expected_origin_label}: {source}"
            )
        if item.get("window") != expected_sampling:
            raise ValueError(f"{label} Work window is wrong: {source}")
        if item.get("sampling") != expected_sampling:
            raise ValueError(f"{label} Work sampling is wrong: {source}")
        origin = _int(item.get("start_monotonic_ns"), f"{label}.start_monotonic_ns")
        if origin != expected_origin:
            raise ValueError(f"timed Work origins disagree: {source}")
        window_id = _int(item.get("work_window_id"), f"{label}.work_window_id",
                         positive=True)
        if window_id != expected_window_id:
            raise ValueError(f"multiple or inconsistent Work window ids: {source}")
        index = _int(item.get("sample"), f"{label}.sample")
        if index >= scheduled_count or index >= len(points):
            raise ValueError(f"{label} sample is outside scheduled Work points: {source}")
        point = _int(item.get("scheduled_elapsed_s"),
                     f"{label}.scheduled_elapsed_s")
        if point != points[index]:
            raise ValueError(f"{label} schedule is wrong: {source}")
        observed_elapsed = _float(item.get("observed_elapsed_s"),
                                  f"{label}.observed_elapsed_s")
        if observed_elapsed > duration_ns / 1e9 + 1e-6:
            raise ValueError(f"{label} occurs after the true Work end: {source}")
        if index in seen:
            raise ValueError(f"duplicate sample/missing index: {source}")
        seen.add(index)
        return index, observed_elapsed

    for item in samples:
        index, actual = common(item, "timed sample")
        if str(item.get("metric")) != expected_metric:
            raise ValueError(f"timed memory sample metric mismatch: {source}")
        snapshot_start = _int(item.get("snapshot_start_monotonic_ns"),
                              "snapshot_start_monotonic_ns")
        snapshot_ns = _int(item.get("snapshot_ns"), "snapshot_ns")
        if snapshot_start < expected_origin or snapshot_start + snapshot_ns > end_ns:
            raise ValueError(f"timed memory snapshot crosses Work end: {source}")
        elapsed_from_clock = (snapshot_start - expected_origin) / 1e9
        if not math.isclose(actual, elapsed_from_clock, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"timed memory observed time disagrees with clock: {source}")
        occupied = _int(item.get("occupied_bytes"), "occupied_bytes")
        item_endpoint_count = _int(item.get("endpoint_count"), "endpoint_count",
                                   positive=True)
        if endpoint_count is not None and item_endpoint_count != endpoint_count:
            raise ValueError(f"endpoint count disagrees with manifest: {source}")
        endpoint_total = _endpoint_sum(item.get("endpoint_bytes"), source,
                                       item_endpoint_count)
        if endpoint_total != occupied:
            raise ValueError(f"timed memory endpoint sum mismatch: {source}")
        normalized.append({"sample": index, "occupied_bytes": occupied})
        observed.append((index, actual))
    for item in missing:
        common(item, "timed missing")
        status = str(item.get("status", ""))
        if not status or status == "observer-error":
            raise ValueError(f"timed Work missing record has invalid status: {source}")
    if seen != set(range(scheduled_count)):
        raise ValueError(f"scheduled Work points are not closed by samples/missing: {source}")
    if not normalized:
        raise ValueError(
            f"work window shorter than first sample; missing data (samples=0, "
            f"scheduled_samples={scheduled_count}): {source}"
        )
    normalized.sort(key=lambda item: item["sample"])
    observed.sort()
    return (
        sum(item["occupied_bytes"] for item in normalized) / len(normalized),
        [points[index] for index in range(scheduled_count)],
        [actual for _, actual in observed],
        normalized,
        {
            "origin": expected_origin_label,
            "work_window_id": expected_window_id,
            "end_monotonic_ns": end_ns,
            "scheduled_samples": scheduled_count,
            "samples": len(normalized),
            "missing_samples": len(missing),
            "expected_samples": len(points),
            "work_duration_ns": duration_ns,
            "missing_times_s": [
                points[_int(item.get("sample"), "timed missing.sample")]
                for item in missing
            ],
        },
    )


def _timed_log(log: str, system: str, run_dir: Path, endpoint_count=None,
               workload=None):
    profile = _timed_profile(workload)
    points = profile["points"]
    sampling = profile["sampling"]
    origin_label = profile["origin"]
    if re.search(r"^runtime_remote_memory(?:_end|_missing)?\b.*schema_version=2",
                 log, re.M):
        raise ValueError(
            f"old program-start schema-v2 memory evidence is rejected; "
            f"require schema-v3 Work-start data: {run_dir}"
        )
    if re.search(r"^kvs_remote_memory(?:_end)?\b", log, re.M):
        raise ValueError(f"legacy request-progress memory evidence is not timed: {run_dir}")
    missing = re.findall(r"^runtime_remote_memory_missing\b.*$", log, re.M)
    for line in missing:
        fields = _fields(line)
        if fields.get("schema_version") != str(TIMED_SCHEMA_VERSION):
            raise ValueError(f"unsupported runtime memory missing schema: {run_dir}")
    lines = re.findall(r"^runtime_remote_memory\b.*$", log, re.M)
    ends = re.findall(r"^runtime_remote_memory_end\b.*$", log, re.M)
    if not lines and not ends:
        if "runtime_remote_memory" in log:
            raise ValueError(
                f"runtime Work memory records are not standalone lines or are missing: "
                f"{run_dir}"
            )
        return None
    if len(ends) != 1:
        raise ValueError(f"missing or duplicated timed memory end marker: {run_dir}")
    end = _fields(ends[0])
    if (end.get("schema_version") != str(TIMED_SCHEMA_VERSION)
            or end.get("sampling") != sampling
            or end.get("window") != sampling
            or end.get("origin") != origin_label):
        raise ValueError(
            f"unsupported program-start/schema-v2 or wrong Work memory profile: {run_dir}"
        )
    end_metric = end.get("metric")
    if end_metric is None:
        raise ValueError(f"timed memory end metric is missing: {run_dir}")
    end_origin = _int(end.get("start_monotonic_ns"),
                      "remote_memory_end.start_monotonic_ns")
    end_ns = _int(end.get("end_monotonic_ns"),
                  "remote_memory_end.end_monotonic_ns")
    if workload in KVS_WORKLOADS:
        starts = re.findall(
            r"^kvs_phase name=direct event=request_start\b.*$", log, re.M
        )
        drains = re.findall(
            r"^kvs_phase name=direct event=request_drain_end\b.*$", log, re.M
        )
        if len(starts) != 1 or len(drains) != 1:
            raise ValueError(
                f"KVS Work memory requires one request_start and request_drain_end: "
                f"{run_dir}"
            )
        request_start = _int(
            _fields(starts[0]).get("monotonic_ns"),
            "kvs request_start.monotonic_ns",
        )
        request_end = _int(
            _fields(drains[0]).get("monotonic_ns"),
            "kvs request_drain_end.monotonic_ns",
        )
        if end_origin != request_start or end_ns != request_end:
            raise ValueError(
                f"KVS Work memory origin/end do not match request Work bounds: "
                f"{run_dir}"
            )
    work_window_id = _int(end.get("work_window_id"),
                          "remote_memory_end.work_window_id", positive=True)
    expected = _int(end.get("expected_samples"),
                    "remote_memory_end.expected_samples")
    if expected != len(points):
        raise ValueError(f"runtime memory expected_samples does not match workload profile: {run_dir}")
    scheduled_count = _int(end.get("scheduled_samples"),
                           "remote_memory_end.scheduled_samples")
    count = _int(end.get("samples"), "remote_memory_end.samples")
    missing_count = _int(end.get("missing"), "remote_memory_end.missing")
    duration_ns = end_ns - end_origin
    if duration_ns < 0:
        raise ValueError(f"runtime memory Work end precedes Work origin: {run_dir}")
    expected_due = sum(point * 1_000_000_000 <= duration_ns for point in points)
    if scheduled_count != expected_due:
        raise ValueError(
            f"scheduled_samples disagrees with true Work duration: {run_dir}"
        )
    if count != len(lines) or missing_count != len(missing):
        raise ValueError(f"timed Work sample/missing count mismatch: {run_dir}")
    if count + missing_count != scheduled_count:
        raise ValueError(f"timed Work samples do not close scheduled points: {run_dir}")
    end_status = _validate_end_status(
        end.get("status"), scheduled_count, count, missing_count,
        expected, str(run_dir / "client.log"))
    if scheduled_count == 0 and count == 0:
        raise ValueError(
            f"work window shorter than first sample; missing data: {run_dir}"
        )
    samples = [_fields(line) for line in lines]
    missing_records = [_fields(line) for line in missing]
    metric_values = {str(item.get("metric")) for item in samples}
    if not metric_values:
        metric_values = {end_metric}
    if len(metric_values) != 1:
        raise ValueError(f"timed memory metric is missing or inconsistent: {run_dir}")
    mean, scheduled, observed, normalized, metadata = _timed_samples(
        samples, missing_records, metric_values.pop(), system,
        str(run_dir / "client.log"), endpoint_count, end_origin,
        end_metric, work_window_id, end_ns, points, scheduled_count, sampling,
        origin_label)
    metadata["end_status"] = end_status
    return mean, scheduled, observed, normalized, metadata


def _timed_json(run_dir: Path, system: str, endpoint_count=None,
                workload=None):
    profile = _timed_profile(workload)
    for name in MEMORY_RESULTS:
        path = run_dir / name
        if not path.is_file():
            continue
        document = _load(path)
        if (document.get("schema_version") != TIMED_SCHEMA_VERSION
                or document.get("sampling") != profile["sampling"]
                or document.get("window") != profile["sampling"]
                or document.get("origin") != profile["origin"]):
            raise ValueError(
                f"old program-start or unsupported runtime memory JSON is rejected: {path}"
            )
        samples = document.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"runtime memory JSON has no successful Work samples: {path}")
        if not all(isinstance(item, dict) and "scheduled_elapsed_s" in item
                   for item in samples):
            raise ValueError(f"runtime memory JSON samples are malformed: {path}")
        missing = document.get("missing_records", document.get("missing", []))
        if not isinstance(missing, list):
            raise ValueError(f"runtime memory JSON missing records are malformed: {path}")
        metric = document.get("metric")
        if metric is None:
            metric_values = {str(item.get("metric")) for item in samples}
            if len(metric_values) != 1:
                raise ValueError(f"timed memory JSON metric is missing: {path}")
            metric = metric_values.pop()
        end = document.get("end", document)
        end_origin = _int(end.get("start_monotonic_ns"),
                          "runtime memory JSON start_monotonic_ns")
        end_ns = _int(end.get("end_monotonic_ns"),
                      "runtime memory JSON end_monotonic_ns")
        work_window_id = _int(end.get("work_window_id"),
                              "runtime memory JSON work_window_id", positive=True)
        scheduled_count = _int(end.get("scheduled_samples"),
                               "runtime memory JSON scheduled_samples")
        expected_count = _int(end.get("expected_samples"),
                              "runtime memory JSON expected_samples")
        if expected_count != len(profile["points"]):
            raise ValueError(
                f"runtime memory JSON expected_samples does not match workload profile: {path}"
            )
        end_count = _int(end.get("samples"), "runtime memory JSON samples")
        end_missing_count = _int(end.get("missing"),
                                 "runtime memory JSON missing")
        if end_count != len(samples) or end_missing_count != len(missing):
            raise ValueError(
                f"runtime memory JSON sample/missing count mismatch: {path}"
            )
        end_status = _validate_end_status(
            end.get("status"), scheduled_count, end_count,
            end_missing_count, expected_count, str(path))
        mean, scheduled, observed, normalized, metadata = _timed_samples(
            samples, missing, str(metric), system, str(path), endpoint_count,
            end_origin, str(metric), work_window_id, end_ns,
            profile["points"], scheduled_count, profile["sampling"],
            profile["origin"])
        metadata["end_status"] = end_status
        if document.get("sample_mean_bytes") is not None and not math.isclose(
                _float(document["sample_mean_bytes"], "sample_mean_bytes"),
                mean, rel_tol=1e-12, abs_tol=0.5):
            raise ValueError(f"timed memory JSON mean disagrees with samples: {path}")
        return mean, scheduled, observed, normalized, metadata
    return None


def _memory_evidence(log: str, run_dir: Path, system: str,
                     endpoint_count=None, workload=None):
    from_log = _timed_log(log, system, run_dir, endpoint_count, workload)
    from_json = _timed_json(run_dir, system, endpoint_count, workload)
    if from_log is None:
        raise ValueError(f"timed runtime memory log is missing: {run_dir}")
    if from_json is not None and from_log[1:] != from_json[1:]:
        raise ValueError(f"timed memory log and JSON disagree: {run_dir}")
    return from_log


def _cpu_path(run_dir: Path):
    candidates = (run_dir / "remote-cpu.json",
                  run_dir / "remote-cpu" / "remote-cpu.json")
    return next((path for path in candidates if path.is_file()), None)


def _endpoint_identities(value, source):
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list) or not value:
        raise ValueError(f"CPU endpoint provenance is malformed: {source}")
    identities = []
    for endpoint in value:
        if not isinstance(endpoint, dict):
            raise ValueError(f"CPU endpoint provenance is malformed: {source}")
        host = endpoint.get("host", endpoint.get("memory_host"))
        pid = endpoint.get("pid", endpoint.get("memory_server_pid"))
        start = endpoint.get("starttime", endpoint.get("start_time"))
        if host is None or pid is None or start is None:
            raise ValueError(f"CPU endpoint identity is incomplete: {source}")
        identities.append((str(host), str(pid), str(start)))
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate CPU endpoint identity: {source}")
    return sorted(identities)


def _manifest_endpoints(documents):
    for document in documents:
        for holder in (document, document.get("plan")):
            if not isinstance(holder, dict):
                continue
            endpoints = holder.get("endpoints") or holder.get("memory_endpoints")
            if isinstance(endpoints, list) and endpoints:
                return endpoints
    return None


def _cpu(run_dir: Path, documents):
    path = _cpu_path(run_dir)
    if path is None:
        raise ValueError(f"remote CPU sidecar is missing: {run_dir}")
    value = _load(path)
    if value.get("schema_version") != 1 or value.get("status") != "passed":
        raise ValueError(f"remote CPU sidecar did not pass: {path}")
    if value.get("remote_cpu_window") != "initialization_and_requests":
        raise ValueError(f"remote CPU sidecar has the wrong window: {path}")
    if not isinstance(value.get("endpoints"), (list, dict)) or not value.get("provenance"):
        raise ValueError(f"remote CPU provenance is missing: {path}")
    manifest_endpoints = _manifest_endpoints(documents)
    if manifest_endpoints is None or _endpoint_identities(value["endpoints"], str(path)) != \
            _endpoint_identities(manifest_endpoints, "manifest"):
        raise ValueError(f"remote CPU sidecars do not match this run: {path}")
    return ({"remote_cpu_seconds": _float(value.get("remote_cpu_seconds"),
                                           "remote_cpu_seconds"),
             "remote_cpu_elapsed_s": _float(value.get("remote_cpu_elapsed_s"),
                                             "remote_cpu_elapsed_s", positive=True),
             "remote_cpu_window": "initialization_and_requests"}, path)


def _workload_id(layers, request, workload, ratio, workers, repeat):
    value = _first(layers, "workload_id", "workload_key", "workload_fingerprint")
    if value is not None:
        return str(value)
    if request.get("request_fingerprint"):
        return f"{workload}:request_fingerprint={request['request_fingerprint']}"
    keys = ("requests", "request_count", "records", "object_bytes", "get_ratio",
            "put_ratio", "remove_ratio", "zipfian", "random_seed", "local_bytes",
            "logical_bytes", "workload_footprint_bytes", "declared_input_sha256",
            "input_bytes", "input_mtime_ns", "input_files", "prompt_sha256",
            "tokenizer_sha256", "input", "iterations", "repetitions",
            "bfs_repetitions", "measured_repetitions")
    values = {key: _first(layers, key) for key in keys}
    values = {key: value for key, value in values.items() if value is not None}
    identity_keys = {"declared_input_sha256", "input_bytes", "input_mtime_ns",
                     "input_files", "prompt_sha256", "tokenizer_sha256", "input",
                     "request_count", "requests", "iterations", "repetitions",
                     "bfs_repetitions", "measured_repetitions"}
    if not values or not identity_keys.intersection(values):
        raise ValueError("workload_id/fingerprint is missing or underspecified")
    values.update(workload=workload, ratio=ratio, app_workers=workers, repeat=repeat)
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _nonft_backup_off(layers, run_dir, *, include_regular_nonft=False):
    variants = {str(layer["baseline_variant"]) for layer in layers
                if layer.get("baseline_variant") is not None}
    if len(variants) > 1:
        raise ValueError(f"NonFT variant metadata disagrees: {run_dir}")
    variant = next(iter(variants), "nonft")
    if variant not in ("nonft", "nonft-backup-off"):
        raise ValueError(f"unknown NonFT variant: {run_dir}")
    if variant == "nonft" and not include_regular_nonft:
        raise _SkipRun(f"Figure 11 excludes the regular NonFT variant: {run_dir}")
    states = [log_contract.backup_flag(layer["backup_enabled"]) for layer in layers
              if "backup_enabled" in layer]
    if variant == "nonft":
        if not states or len(set(states)) != 1:
            raise ValueError(f"NonFT requires consistent explicit backup_enabled: {run_dir}")
        return variant
    if not states or any(states):
        raise ValueError(f"NonFT backup-off requires explicit backup_enabled=false: {run_dir}")
    path = run_dir / "effective.config"
    if not path.is_file():
        raise ValueError(f"NonFT backup-off requires effective.config: {run_dir}")
    required = {"enable_selective_backup", "remote_backup_budget_bytes",
                "remote_backup_budget_pct"}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if parts and parts[0] in required:
            if len(parts) != 2 or parts[0] in values:
                raise ValueError(f"ambiguous NonFT backup configuration: {path}")
            values[parts[0]] = parts[1]
    if set(values) != required or any(value != "0" for value in values.values()):
        raise ValueError(f"NonFT backup-off contradicts effective.config: {run_dir}")
    return variant


def _resident_provenance(run_dir):
    keys = ("local_resident_budget_bytes", "enable_region_resident_placement",
            "enable_resident_profile_planner", "resident_profile_apply_plan",
            "enable_region_hotness_placement", "enable_region_fetch_hotness_placement",
            "region_placement_bind_groups", "enable_logical_object_profile")
    values = {}
    for line in (run_dir / "effective.config").read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if parts and parts[0] in keys:
            if len(parts) != 2 or parts[0] in values:
                raise ValueError(f"ambiguous resident configuration: {run_dir}")
            values[parts[0]] = _int(parts[1], parts[0])
    resident_enabled = (values.get("enable_region_resident_placement") == 1
                        and values.get("local_resident_budget_bytes", 0) > 0)
    policy = (log_contract.NONFT_REFERENCE_POLICY if resident_enabled else
              "backup_and_resident_off" if len(values) == len(keys)
              and not any(values.values()) else "resident_unverified")
    return {"off_policy": policy, "resident_config": values,
            "resident_enabled": resident_enabled,
            "local_resident_budget_bytes": values.get("local_resident_budget_bytes")}


def _parse_run(run_dir: Path, paths: dict, selected, ratio: int, repeat: int,
               *, include_regular_nonft=False):
    documents, files = _documents(run_dir, paths)
    layers = _layers(documents)
    nonft_variant = None
    if _system(_first(layers, "system", "runtime")) == "nonft":
        nonft_variant = _nonft_backup_off(
            layers, run_dir, include_regular_nonft=include_regular_nonft)
    _eligibility(layers, run_dir)
    log_path = run_dir / "client.log" if (run_dir / "client.log").is_file() else paths["logs"][0]
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read client log: {log_path}") from exc
    request = _request_evidence(log, run_dir)
    workload = _workload(_first(layers, "application", "app", "workload"))
    system = _system(_first(layers, "system", "runtime"))
    if workload is None:
        workload = next((_workload(part) for part in reversed(run_dir.parts) if _workload(part)), None)
    if system is None:
        system = next((_system(part) for part in reversed(run_dir.parts) if _system(part)), None)
    if workload is None or system is None:
        raise ValueError(f"workload/system metadata is missing: {run_dir}")
    run_id = _first(layers, "run_id", "run_name") or run_dir.name
    environment = _first(layers, "environment", "environment_name", "site")
    if isinstance(environment, dict):
        environment = environment.get("name")
    if environment is None:
        raise ValueError(f"environment metadata is missing: {run_dir}")
    observed_ratio = _first(layers, "ratio", "put_ratio_percent")
    if observed_ratio is None:
        raise ValueError(f"ratio metadata is missing: {run_dir}")
    observed_ratio = _int(observed_ratio, "ratio")
    if observed_ratio != ratio:
        raise _SkipRun(f"different ratio: {run_dir}")
    workers = _first(layers, "app_workers", "application_workers")
    if workers is None:
        for layer in layers:
            for key in ("worker_profile", "configured_cpu_profile", "cpu_profile"):
                profile = layer.get(key)
                if isinstance(profile, dict) and profile.get("app_workers") is not None:
                    workers = profile["app_workers"]
                    break
            if workers is not None:
                break
    workers = _first(layers, "workers") if workers is None else workers
    if workers is None:
        raise ValueError(f"app_workers metadata is missing: {run_dir}")
    workers = _int(workers, "app_workers", positive=True)
    observed_repeat = _observed_repeat(documents, files, layers, run_dir)
    if observed_repeat != repeat:
        raise _SkipRun(f"different repeat: {run_dir}")
    elapsed = _first(layers, "elapsed_s", "request_seconds", "work_elapsed_s")
    elapsed = request.get("elapsed_s") if elapsed is None else _float(elapsed, "elapsed_s", positive=True)
    if elapsed is None:
        raise ValueError(f"Work elapsed_s is missing: {run_dir}")
    elapsed = _float(elapsed, "elapsed_s", positive=True)
    if request.get("elapsed_s") is not None and not math.isclose(elapsed, request["elapsed_s"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"metadata and client Work elapsed_s disagree: {run_dir}")
    record = {"schema_version": getattr(log_contract, "SCHEMA_VERSION", 3),
              "workload": workload, "system": system, "run_id": str(run_id),
              "environment": str(environment),
              "workload_id": _workload_id(layers, request, workload, observed_ratio, workers, observed_repeat),
              "ratio": observed_ratio, "app_workers": workers, "repeat": observed_repeat,
              "phase": "work", "components": ",".join(selected),
              "elapsed_s": elapsed, "exit_status": 0, "correctness": "pass",
              "source_type": "measured"}
    if system == "nonft":
        if nonft_variant is None:
            nonft_variant = _nonft_backup_off(
                layers, run_dir, include_regular_nonft=include_regular_nonft)
        record.update(baseline_variant=nonft_variant,
                      backup_enabled=log_contract.backup_flag(
                          _first(layers, "backup_enabled")))
        if nonft_variant == "nonft-backup-off":
            record.update(_resident_provenance(run_dir))
    if "fetch_traffic" in selected or "eviction_traffic" in selected:
        traffic = _traffic_log(log, run_dir)
        metadata_traffic = _traffic_docs(documents, run_dir)
        if traffic is None:
            traffic = metadata_traffic
        elif metadata_traffic is not None and traffic != metadata_traffic:
            raise ValueError(f"traffic sources disagree: {run_dir}")
        if traffic is None:
            raise ValueError(f"Work traffic is missing: {run_dir}")
        if "fetch_traffic" in selected:
            record["fetch_bytes"] = traffic["fetch_bytes"]
        if "eviction_traffic" in selected:
            record["eviction_bytes"] = traffic["eviction_bytes"]
    if "remote_memory" in selected:
        endpoint_count = _manifest_endpoint_count(documents)
        mean, scheduled, observed, _, memory_meta = _memory_evidence(
            log, run_dir, system, endpoint_count, workload)
        metric = None
        for line in re.findall(r"^runtime_remote_memory\b.*$", log, re.M):
            metric = _fields(line).get("metric")
            if metric:
                break
        metric = metric or _memory_metric(documents)
        if metric is None:
            raise ValueError(f"timed memory metric is missing: {run_dir}")
        record.update(
            remote_memory_mean_bytes=mean,
            remote_memory_samples=memory_meta["samples"],
            remote_memory_sampling=_timed_profile(workload)["sampling"],
            remote_memory_window="work",
            remote_memory_origin=memory_meta["origin"],
            remote_memory_work_window_id=memory_meta["work_window_id"],
            remote_memory_end_monotonic_ns=memory_meta["end_monotonic_ns"],
            remote_memory_scheduled_samples=memory_meta["scheduled_samples"],
            remote_memory_missing_samples=memory_meta["missing_samples"],
            remote_memory_expected_samples=memory_meta["expected_samples"],
            remote_memory_work_duration_ns=memory_meta["work_duration_ns"],
            remote_memory_missing_times_s=memory_meta["missing_times_s"],
            remote_memory_end_status=memory_meta["end_status"],
            remote_memory_scheduled_times_s=scheduled,
            remote_memory_sample_times_s=observed,
            remote_memory_metric=metric,
        )
    if "remote_cpu_cores" in selected:
        cpu, _ = _cpu(run_dir, documents)
        record.update(cpu)
    source = [str(path.resolve()) for path in files.values()] + [str(log_path.resolve())]
    for name in MEMORY_RESULTS:
        path = run_dir / name
        if path.is_file():
            source.append(str(path.resolve()))
    cpu_path = _cpu_path(run_dir)
    if cpu_path is not None:
        source.append(str(cpu_path.resolve()))
    record["_source"] = "; ".join(dict.fromkeys(source))
    record["_raw_run_dir"] = str(run_dir.resolve())
    return log_contract.validate_record(record, source=record["_source"])


def collect_records(logs_root, *, pattern="*.log", ratio=25, repeat=1, components=None,
                    include_regular_nonft=False):
    """Collect complete measured records; failed retries are skipped."""
    root = Path(logs_root).resolve()
    if not root.is_dir():
        raise ValueError(f"log directory not found: {root}")
    selected = log_contract.normalize_components(
        log_contract.ALL_COMPONENTS if components is None else components)
    if not 1 <= ratio <= 100 or repeat < 1:
        raise ValueError("ratio must be 1..100 and repeat must be positive")
    records, skipped = [], []
    for run_dir, paths in _discover(root, pattern):
        try:
            records.append(_parse_run(
                run_dir, paths, selected, ratio, repeat,
                include_regular_nonft=include_regular_nonft))
        except _SkipRun as exc:
            skipped.append(str(exc))
    if not records:
        detail = f" ({'; '.join(skipped[:3])})" if skipped else ""
        raise ValueError(f"no complete raw Figure 11 records under {root}{detail}")
    return records
