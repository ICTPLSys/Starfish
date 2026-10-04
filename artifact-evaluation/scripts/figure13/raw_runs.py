"""Read native Figure 13 recovery runs without re-running any workload.

The Figure 13 runner predates the figure13_result log contract used by
collect.py. This adapter reconstructs that contract from the native result,
plan, client log, and completion-series files. It does not trust a requested
fault time or a single passed bit: request markers, fault identity, native
rebuild events, completion windows, receipts, and post-verification are
checked independently.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from typing import Any


NANOSECONDS = 1_000_000_000
SYSTEMS = {"starfish", "carbink", "hydra"}
SCENARIOS = {"1-node", "2-node"}
REQUEST_EVENTS = {"request_start", "request_generation_end", "request_drain_end"}
REBUILD_EVENTS = {"start", "done", "joined"}
COUNT_FIELDS = (
    "generated_get", "generated_put", "generated_remove",
    "accepted_get", "accepted_put", "accepted_remove",
    "completed_get", "completed_put", "completed_remove",
)
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class RawRunError(ValueError):
    """Raised when a native run lacks a complete, auditable contract."""


def _fail(message: str) -> None:
    raise RawRunError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing JSON evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON evidence {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON evidence is not an object: {path}")
    return value


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        _fail(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        return int(value.strip())
    _fail(f"{field} must be an integer, got {value!r}")


def _nonnegative_int(value: Any, field: str) -> int:
    result = _int(value, field)
    if result < 0:
        _fail(f"{field} must be nonnegative")
    return result


def _finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        _fail(f"{field} must be numeric")
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _count(value: Any, field: str) -> int:
    return _nonnegative_int(value, field)


def _bool_true(value: Any, field: str) -> None:
    if value is not True:
        _fail(f"{field} must be true")


def _source(path: Path, line_number: int) -> str:
    return f"{path}:{line_number}"


def _tokens(line: str, source: str) -> dict[str, str]:
    try:
        words = shlex.split(line, comments=False, posix=True)
    except ValueError as exc:
        _fail(f"cannot parse native event at {source}: {exc}")
    result: dict[str, str] = {}
    for word in words:
        key, separator, value = word.partition("=")
        if not separator or not _KEY_RE.fullmatch(key):
            continue
        if key in result:
            _fail(f"duplicate event field {key} at {source}")
        result[key] = value
    return result


def _event_record(path: Path, line_number: int, line: str) -> dict[str, Any]:
    return {
        "line": line.rstrip("\n"),
        "line_number": line_number,
        "source": _source(path, line_number),
        "tokens": _tokens(line, _source(path, line_number)),
    }


def _token_int(record: dict[str, Any], key: str, field: str | None = None) -> int:
    name = field or key
    tokens = record["tokens"]
    if key not in tokens:
        _fail(f"missing {name} in {record['source']}")
    return _int(tokens[key], f"{name} at {record['source']}")


def _optional_token_int(
    record: dict[str, Any], keys: tuple[str, ...], field: str
) -> int | None:
    for key in keys:
        if key in record["tokens"]:
            return _int(record["tokens"][key], f"{field} at {record['source']}")
    return None


def _single(records: list[dict[str, Any]], what: str) -> dict[str, Any]:
    if len(records) != 1:
        _fail(f"expected exactly one {what}, found {len(records)}")
    return records[0]


def _safe_reference(value: Any, expected: str, run_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"missing same-directory reference for {expected}")
    ref = Path(value)
    if ref.is_absolute() or ref.name != value or value in {".", ".."}:
        _fail(f"unsafe {expected} reference: {value!r}")
    if value != expected:
        _fail(f"{expected} reference is {value!r}, expected {expected!r}")
    path = run_dir / value
    if not path.is_file():
        _fail(f"missing {expected} evidence: {path}")
    return path

def _dual_source_result(
    result: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], Path | None]:
    """Return native evidence for a dual result, including Hydra revalidation."""
    if result.get("_result_filename") != "revalidated-dual-result.json":
        return result, None
    original_path = _safe_reference(
        result.get("original_result"), "dual-result.json", run_dir
    )
    original = _read_json(original_path)
    if original.get("run") != result.get("run"):
        _fail("revalidated dual result run disagrees with dual-result.json")
    return original, original_path


def _declared_endpoints(plan: dict[str, Any]) -> list[int]:
    inject = plan.get("inject")
    if not isinstance(inject, dict) or inject.get("enabled") is not True:
        _fail("plan does not declare an enabled fault injection")
    raw = inject.get("endpoints")
    if raw is None:
        raw = inject.get("endpoint")
    if raw is None:
        _fail("plan does not declare inject.endpoint(s)")
    if isinstance(raw, dict):
        if "endpoint" in raw:
            raw = [raw]
        else:
            raw = list(raw.values())
    elif not isinstance(raw, (list, tuple)):
        raw = [raw]
    result: list[int] = []
    for item in raw:
        if isinstance(item, dict):
            for key in ("endpoint", "index", "id"):
                if key in item:
                    item = item[key]
                    break
            else:
                _fail(f"inject endpoint object lacks endpoint/index/id: {item!r}")
        endpoint = _int(item, "inject endpoint")
        if endpoint in result:
            _fail(f"duplicate declared inject endpoint: {endpoint}")
        result.append(endpoint)
    if not result:
        _fail("plan declares no inject endpoints")
    return result


def _scenario(plan: dict[str, Any], endpoints: list[int]) -> str:
    explicit = plan.get("scenario")
    if explicit is None and isinstance(plan.get("inject"), dict):
        explicit = plan["inject"].get("scenario")
    if len(endpoints) not in (1, 2):
        _fail(f"Figure 13 supports exactly one or two injected endpoints, got {endpoints}")
    inferred = "2-node" if len(endpoints) == 2 else "1-node"
    if explicit is None:
        return inferred
    if explicit not in SCENARIOS:
        _fail(f"unsupported Figure 13 scenario: {explicit!r}")
    if explicit != inferred:
        _fail(f"scenario {explicit!r} disagrees with declared endpoints {endpoints}")
    return explicit

def _validate_dual_result(
    result: dict[str, Any], evidence: dict[str, Any], expected_requests: int
) -> dict[str, Any]:
    """Validate the newer two-endpoint result/revalidation contract."""
    if result.get("passed") is not True:
        _fail("selected dual result passed=false")
    for key in ("coverage_exit", "cleanup_exit", "client_exit"):
        if key in result and _int(result[key], key) != 0:
            _fail(f"dual result {key} is not zero")
    validation = result.get("validation")
    if not isinstance(validation, dict):
        _fail("selected dual result lacks validation object")
    _bool_true(validation.get("passed"), "dual validation passed")
    series = validation.get("completion_series")
    if not isinstance(series, dict):
        _fail("dual validation lacks completion_series")
    if _count(series.get("completed"), "dual completion completed") != expected_requests:
        _fail("dual completion completed differs from plan requests")
    meta = series.get("meta")
    if not isinstance(meta, dict) or meta.get("enabled") is not True:
        _fail("dual completion series is not enabled")
    if _int(meta.get("interval_ns"), "dual completion interval_ns") != 100_000_000:
        _fail("dual completion series interval is not 100ms")
    receipt = validation.get("request_receipt")
    if not isinstance(receipt, dict):
        _fail("dual validation lacks request_receipt")
    raw_record = _single(evidence["receipts"], "kvs_receipt")
    raw = raw_record["tokens"]
    for key in COUNT_FIELDS + ("accepted_equals_completed", "request_fingerprint"):
        if str(receipt.get(key)) != str(raw.get(key)):
            _fail("dual validation receipt disagrees with client.log: " + key)
    if raw.get("accepted_equals_completed") != "1":
        _fail("dual raw receipt does not claim request closure")
    counts = {key: _count(raw.get(key), key) for key in COUNT_FIELDS}
    for phase in ("generated", "accepted", "completed"):
        if sum(counts[phase + "_" + op] for op in ("get", "put", "remove")) != expected_requests:
            _fail("dual raw request total does not close: " + phase)
    for op in ("get", "put", "remove"):
        if len({counts[phase + "_" + op] for phase in ("generated", "accepted", "completed")}) != 1:
            _fail("dual raw operation conservation failed: " + op)
    generated = _single(evidence["request"]["request_generation_end"], "request_generation_end")
    drained = _single(evidence["request"]["request_drain_end"], "request_drain_end")
    for record, key in ((generated, "generated"), (generated, "accepted"), (drained, "completed")):
        if _token_int(record, key) != expected_requests:
            _fail("dual phase request total does not close: " + key)
    post = validation.get("post_verify")
    if not isinstance(post, dict):
        _fail("dual validation lacks post_verify")
    checked = _count(post.get("checked"), "dual post_verify.checked")
    failures = _count(post.get("failures"), "dual post_verify.failures")
    if checked != 4096 or failures != 0:
        _fail(f"dual post verification is checked={checked}, failures={failures}")
    post_record = _single(evidence["post_verify"], "kvs_post_verify")
    if _count(post_record["tokens"].get("checked"), "kvs_post_verify.checked") != checked:
        _fail("dual post verification checked count disagrees with client.log")
    if _count(post_record["tokens"].get("failures"), "kvs_post_verify.failures") != failures:
        _fail("dual post verification failure count disagrees with client.log")
    proof = validation.get("endpoint_proof")
    if not isinstance(proof, dict):
        _fail("dual validation lacks endpoint_proof")
    _bool_true(proof.get("passed"), "dual endpoint proof passed")
    if proof.get("schema") != "starfish-kv-dual-endpoint-proof-v1":
        _fail("dual endpoint proof schema is not v1")
    cleanup = result.get("cleanup")
    if not isinstance(cleanup, dict):
        _fail("dual result lacks cleanup evidence")
    if _int(cleanup.get("clean_stops"), "dual cleanup clean_stops") != 6:
        _fail("dual cleanup did not stop six healthy endpoints")
    if _int(cleanup.get("skipped_dead_stops"), "dual cleanup skipped_dead_stops") != 2:
        _fail("dual cleanup did not record two dead endpoints")
    stopped = cleanup.get("stopped_endpoints")
    if not isinstance(stopped, list) or len(stopped) != 6:
        _fail("dual cleanup lacks six stopped endpoint identities")
    return {
        "validation": validation,
        "receipt": raw,
        "post_verify": {"checked": checked, "failures": failures},
        "native_result_source": result.get("_result_filename"),
    }


def _validate_dual_topology(
    result: dict[str, Any], plan: dict[str, Any], endpoints: list[int], system: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Check the recorded 8-service, two-fault topology without inventing roles."""
    if len(endpoints) != 2:
        _fail("dual topology requires exactly two failed endpoints")
    if _int(plan.get("remote_endpoints"), "plan remote_endpoints") != 8:
        _fail("dual topology does not declare eight remote endpoints")
    if _int(plan.get("server_count"), "plan server_count") != 8:
        _fail("dual topology does not declare eight servers")
    failed = plan.get("failed_endpoints")
    if isinstance(failed, list) and sorted(_int(item, "plan failed endpoint") for item in failed) != sorted(endpoints):
        _fail("plan failed_endpoints disagrees with inject endpoints")
    source, source_path = _dual_source_result(result, run_dir)
    source_failed = source.get("failed_endpoints")
    if isinstance(source_failed, list) and sorted(_int(item, "result failed endpoint") for item in source_failed) != sorted(endpoints):
        _fail("result failed_endpoints disagrees with inject endpoints")
    layout = source.get("server_layout")
    if not isinstance(layout, dict):
        _fail("dual result lacks server_layout")
    if _int(layout.get("server_count"), "server_layout.server_count") != 8:
        _fail("dual result server_layout does not record eight servers")
    active = 7 if system == "starfish" else 6
    if _int(plan.get("active_remote_endpoints"), "plan active_remote_endpoints") != active:
        _fail("plan active endpoint count disagrees with dual topology")
    if _int(layout.get("active_endpoint_count"), "server_layout.active_endpoint_count") != active:
        _fail("result active endpoint count disagrees with dual topology")
    if system == "starfish":
        if _int(plan.get("standby_endpoint"), "plan standby_endpoint") != 7:
            _fail("Starfish dual plan lacks endpoint 7 as standby")
        if plan.get("standby_endpoint2") is not None:
            _fail("Starfish dual plan unexpectedly declares a second standby")
        if _int(layout.get("standby_endpoint"), "server_layout.standby_endpoint") != 7:
            _fail("Starfish server layout lacks healthy endpoint-7 spare role")
        proofs = plan.get("endpoint_proofs")
        if not isinstance(proofs, list) or len(proofs) != 8:
            _fail("Starfish dual plan lacks eight endpoint proofs")
        roles = { _int(item.get("endpoint"), "endpoint proof endpoint"): item.get("role")
                  for item in proofs if isinstance(item, dict) }
        if roles.get(7) != "standby" or any(roles.get(ep) != "active" for ep in range(7)):
            _fail("Starfish dual endpoint roles are not seven active plus one standby")
    else:
        if _int(plan.get("standby_endpoint"), "plan standby_endpoint") != 6 or _int(plan.get("standby_endpoint2"), "plan standby_endpoint2") != 7:
            _fail("dual plan lacks endpoints 6/7 as two spares")
    cleanup = source.get("cleanup")
    if not isinstance(cleanup, dict):
        _fail("dual result lacks cleanup evidence")
    if _int(cleanup.get("clean_stops"), "cleanup clean_stops") != 6 or _int(cleanup.get("skipped_dead_stops"), "cleanup skipped_dead_stops") != 2:
        _fail("dual cleanup STOP evidence is not six healthy/two dead")
    stopped = cleanup.get("stopped_endpoints")
    if sorted(_int(item, "cleanup stopped endpoint") for item in stopped or []) != [ep for ep in range(8) if ep not in endpoints]:
        _fail("dual cleanup stopped endpoint set disagrees with failed endpoints")
    return {
        "validated": True,
        "server_count": 8,
        "active_endpoints": active,
        "failed_endpoints": list(endpoints),
        "standby_endpoints": [7] if system == "starfish" else [6, 7],
        "healthy_stop_count": 6,
        "dead_stop_count": 2,
        "result_source": source_path.name if source_path else result.get("_result_filename"),
    }
def _workload_value(
    plan: dict[str, Any],
    key: str,
    *,
    env_keys: tuple[str, ...] = (),
    default: Any = None,
) -> Any:
    if key in plan:
        return plan[key]
    env = plan.get("env")
    if isinstance(env, dict):
        for env_key in env_keys:
            if env_key in env:
                return env[env_key]
    if default is not None:
        return default
    _fail(f"plan lacks common workload field {key}")


def _workload_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """Build the baseline-independent workload/placement identity payload."""
    payload: dict[str, Any] = {
        "workload": "kv-b",
        "records": _workload_value(plan, "records",
                                   env_keys=("FARLIB_KVS_INITIAL_DATA_COUNT",)),
        "object_bytes": _workload_value(plan, "object_bytes",
                                        env_keys=("FARLIB_KVS_OBJECT_BYTES",)),
        "requests": _workload_value(plan, "requests",
                                    env_keys=("FARLIB_KVS_MAX_SERVE_COUNT",)),
        "get_ratio": _workload_value(plan, "get_ratio"),
        "put_ratio": _workload_value(plan, "put_ratio",
                                     env_keys=("FARLIB_KVS_PUT_RATIO",)),
        "remove_ratio": _workload_value(
            plan, "remove_ratio", env_keys=("FARLIB_KVS_REMOVE_RATIO",), default=0
        ),
        "zipfian": _workload_value(
            plan, "zipfian", env_keys=("FARLIB_KVS_ZIPFIAN_CONSTANT",)
        ),
        "random_seed": _workload_value(
            plan, "random_seed", env_keys=("FARLIB_KVS_RANDOM_SEED",)
        ),
        "local_bytes": _workload_value(plan, "local_bytes"),
        "logical_bytes": _workload_value(plan, "logical_bytes"),
        "app_workers": _workload_value(
            plan, "app_workers", env_keys=("FibreWorkerCount",)
        ),
        "app_fibres": _workload_value(
            plan, "app_fibres", env_keys=("FARLIB_KVS_DIRECT_FIBRES",)
        ),
        "compute": _workload_value(plan, "compute"),
        "memory": _workload_value(plan, "memory"),
        "app_cpus": plan.get("app_cpus"),
        "background_cpus": plan.get("background_cpus"),
    }
    if payload["app_cpus"] is None or payload["background_cpus"] is None:
        _fail("plan lacks common CPU placement for workload identity")
    return payload


def _workload_identity(plan: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    payload = _workload_payload(plan)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload, canonical


def _read_client_log(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.is_file():
        _fail(f"missing client log: {path}")
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"client log is not valid UTF-8: {path}: {exc}")
    except OSError as exc:
        _fail(f"cannot read client log {path}: {exc}")
    request: dict[str, list[dict[str, Any]]] = {
        "request_start": [], "request_generation_end": [], "request_drain_end": []
    }
    packing: list[dict[str, Any]] = []
    endpoint_dead: list[dict[str, Any]] = []
    rebuild: dict[str, list[dict[str, Any]]] = {
        "start": [], "done": [], "joined": []
    }
    receipts: list[dict[str, Any]] = []
    post_verify: list[dict[str, Any]] = []
    worker_roles: list[dict[str, Any]] = []
    configuration: dict[str, Any] = {}
    def set_configuration(key: str, value: Any, source: str) -> None:
        if key in configuration and configuration[key] != value:
            _fail(f"conflicting client startup field {key} at {source}")
        configuration[key] = value
    physical_lines = text.splitlines()
    for line_number, line in enumerate(physical_lines, 1):
        if "kvs_phase" in line and "event=" in line:
            record = _event_record(path, line_number, line)
            tokens = record["tokens"]
            if tokens.get("name") == "direct" and tokens.get("event") in REQUEST_EVENTS:
                request[tokens["event"]].append(record)
        if "runtime_packing_ready" in line:
            packing.append(_event_record(path, line_number, line))
        if "ec_recovery endpoint_dead" in line:
            # Concurrent INFO records can share one physical line. Parse only
            # this explicit event span; retain the original line as evidence.
            for match in re.finditer(r"ec_recovery endpoint_dead\b", line):
                fragment = line[match.start():].split("INFO:", 1)[0]
                continuation = None
                if fragment.rstrip().endswith("endpoint="):
                    # iostream endpoint reporting may be interrupted by complete
                    # rebuild worker-start records. Join only this exact native
                    # suffix; never infer an endpoint or timestamp from a plan.
                    for offset in range(line_number, min(line_number + 8, len(physical_lines))):
                        candidate = physical_lines[offset]
                        if re.fullmatch(
                            r"[0-9]+ qp_num=[0-9]+ wc_status=[0-9]+ wr_id=[0-9]+ monotonic_ns=[0-9]+",
                            candidate,
                        ):
                            fragment += candidate
                            continuation = offset + 1
                            break
                        if not re.fullmatch(
                            r"INFO: ec_background_rebuild event=worker_start worker=[0-9]+ tid=[0-9]+ cpu=[0-9]+",
                            candidate,
                        ):
                            break
                record = _event_record(path, line_number, fragment)
                record["physical_line"] = line
                if continuation is not None:
                    record["continuation_line_number"] = continuation
                    record["physical_lines"] = physical_lines[line_number - 1:continuation]
                endpoint_dead.append(record)
        if "ec_background_rebuild" in line and "event=" in line:
            # Ignore diagnostic worker/progress events before tokenization.
            # Only the explicit measured-event span belongs to this record;
            # unrelated concurrent ERROR fields are not part of its contract.
            for match in re.finditer(
                r"ec_background_rebuild event=(start|done|joined)\b", line
            ):
                fragment = re.split(r"(?:INFO|ERROR):", line[match.start():], maxsplit=1)[0]
                record = _event_record(path, line_number, fragment)
                record["physical_line"] = line
                event = record["tokens"]["event"]
                rebuild[event].append(record)
        if "kvs_receipt" in line:
            receipts.append(_event_record(path, line_number, line))
        if "kvs_post_verify" in line:
            post_verify.append(_event_record(path, line_number, line))
        if "kvs_direct_config" in line:
            direct_record = _event_record(path, line_number, line)
            if "initial_count" in direct_record["tokens"]:
                set_configuration(
                    "records", _int(direct_record["tokens"]["initial_count"], "initial_count"),
                    _source(path, line_number)
                )
            if "fibres" in direct_record["tokens"]:
                set_configuration(
                    "app_fibres", direct_record["tokens"]["fibres"],
                    _source(path, line_number)
                )
        if "worker_role" in line:
            record = _event_record(path, line_number, line)
            if record["tokens"].get("role") in {"app", "background"}:
                worker_roles.append(record)
        stripped = line.strip()
        records_match = re.match(r"Initialize:\s+loading\s+([0-9]+)\s+K-V\s+pairs", stripped)
        if records_match:
            set_configuration("records", int(records_match.group(1)),
                              _source(path, line_number))
        for key in ("object_bytes", "put_ratio", "remove_ratio", "zipfian", "random_seed"):
            match = re.match(rf"kvs\.{re.escape(key)}:\s+(\S+)", stripped)
            if match:
                set_configuration(key, match.group(1), _source(path, line_number))
    return text, {
        "request": request,
        "packing": packing,
        "endpoint_dead": endpoint_dead,
        "rebuild": rebuild,
        "receipts": receipts,
        "post_verify": post_verify,
        "worker_roles": worker_roles,
        "configuration": configuration,
        "raw_bytes": raw_bytes,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _event_endpoint(record: dict[str, Any]) -> int | None:
    for key in ("endpoint", "failed_endpoint", "failed"):
        if key in record["tokens"]:
            return _int(record["tokens"][key], f"{key} at {record['source']}")
    return None


def _group_rebuild_events(
    evidence: dict[str, Any], endpoints: list[int]
) -> dict[int, dict[str, dict[str, Any]]]:
    grouped = {endpoint: {} for endpoint in endpoints}
    global_joined: list[dict[str, Any]] = []
    for event_name, records in (
        ("endpoint_dead", evidence["endpoint_dead"]),
        ("start", evidence["rebuild"]["start"]),
        ("done", evidence["rebuild"]["done"]),
        ("joined", evidence["rebuild"]["joined"]),
    ):
        for record in records:
            event_endpoint = _event_endpoint(record)
            if event_endpoint is None:
                if event_name == "joined" and len(endpoints) > 1:
                    global_joined.append(record)
                    continue
                if len(endpoints) != 1:
                    _fail(
                        f"{event_name} event lacks endpoint identity at "
                        f"{record['source']}"
                    )
                event_endpoint = endpoints[0]
            if event_endpoint not in grouped:
                _fail(
                    f"{event_name} event references undeclared endpoint "
                    f"{event_endpoint} at {record['source']}"
                )
            if event_name in grouped[event_endpoint]:
                _fail(f"duplicate {event_name} event for endpoint {event_endpoint}")
            grouped[event_endpoint][event_name] = record
    if global_joined:
        if len(global_joined) != 1:
            _fail("dual rebuild has more than one global joined event")
        for endpoint in endpoints:
            if "joined" in grouped[endpoint]:
                _fail(f"dual endpoint {endpoint} has both global and endpoint joined events")
            grouped[endpoint]["joined"] = global_joined[0]
    for endpoint, events in grouped.items():
        missing = {"endpoint_dead", "start", "done", "joined"} - set(events)
        if missing:
            _fail(
                f"endpoint {endpoint} lacks native event(s): "
                + ", ".join(sorted(missing))
            )
    return grouped

def _validate_dual_rebuild_traffic(
    grouped: dict[int, dict[str, dict[str, Any]]], *, system: str | None = None
) -> dict[str, Any] | None:
    """Validate shared dual-node traffic accounting from native event tokens."""
    if len(grouped) != 2:
        return None
    endpoints = sorted(grouped)
    starts = [grouped[ep]["start"] for ep in endpoints]
    dones = [grouped[ep]["done"] for ep in endpoints]
    start_ns = [_token_int(record, "monotonic_ns") for record in starts]
    done_ns = [_token_int(record, "monotonic_ns") for record in dones]
    if len(set(start_ns)) != 1 or len(set(done_ns)) != 1:
        _fail("dual rebuild start/done events do not share global timestamps")
    for label, records in (("start", starts), ("done", dones)):
        for key in ("expected_failures", "failed_mask", "read_policy", "read_accounting"):
            values = [record["tokens"].get(key) for record in records]
            if any(value is None for value in values) or len(set(values)) != 1:
                _fail(f"dual rebuild {label} {key} is not shared")
        if records[0]["tokens"].get("expected_failures") != "2":
            _fail("dual rebuild expected_failures is not 2")
        if records[0]["tokens"].get("failed_mask") != "3":
            _fail("dual rebuild failed_mask is not 3")
        if records[0]["tokens"].get("read_policy") != "four_survivors_shared":
            _fail("dual rebuild read_policy is not four_survivors_shared")
        if records[0]["tokens"].get("read_accounting") != "first_missing_role":
            _fail("dual rebuild read accounting is not first_missing_role")
    done_values = [record["tokens"] for record in dones]
    common_keys = ("global_read_bytes", "global_write_bytes", "union_stripes")
    common: dict[str, int] = {}
    for key in common_keys:
        values = [_nonnegative_int(tokens.get(key), f"dual done {key}") for tokens in done_values]
        if len(set(values)) != 1:
            _fail(f"dual rebuild {key} is not shared")
        common[key] = values[0]
    union_stripes = common["union_stripes"]
    if union_stripes <= 0:
        _fail("dual rebuild union_stripes is not positive")
    write_values = [_nonnegative_int(tokens.get("write_bytes"), "dual done write_bytes") for tokens in done_values]
    if any(value <= 0 for value in write_values):
        _fail("dual rebuild endpoint write_bytes is not positive")
    expected_write = sum(write_values)
    if common["global_write_bytes"] != expected_write:
        _fail("dual global_write_bytes does not equal endpoint write_bytes sum")
    range_policies = [tokens.get("range_policy") for tokens in done_values]
    if len(set(range_policies)) != 1:
        _fail("dual rebuild range policies disagree")
    if range_policies[0] == "live_groups":
        selected_values = [_nonnegative_int(tokens.get("selected_bytes"), "dual selected_bytes") for tokens in done_values]
        live_values = [_nonnegative_int(tokens.get("live_groups_at_copy"), "dual live_groups_at_copy") for tokens in done_values]
        union_selected = [_nonnegative_int(tokens.get("union_selected_bytes"), "dual union_selected_bytes") for tokens in done_values]
        if any(selected != live * 8192 for selected, live in zip(selected_values, live_values)):
            _fail("dual selected_bytes does not equal live_groups_at_copy*8192")
        if system == "carbink":
            # New stripes may exclude an already-dead endpoint, so a dual
            # failure can leave one or two missing shards in each stripe.
            if len(set(union_selected)) != 1:
                _fail("Carbink dual union_selected_bytes is not shared")
            if not max(selected_values) <= union_selected[0] <= sum(selected_values):
                _fail("Carbink dual union_selected_bytes is outside endpoint bounds")
        elif len(set(selected_values)) != 1 or any(value != selected_values[0] for value in union_selected):
            _fail("dual union_selected_bytes does not close across endpoints")
        expected_read = 4 * union_selected[0]
        read_formula = "4*union_selected_bytes"
        selected_bytes = union_selected[0]
    elif range_policies[0] in (None, ""):
        expected_read = 4 * union_stripes * 256 * 1024
        read_formula = "4*union_stripes*256KiB"
        selected_bytes = None
    else:
        _fail("dual rebuild has an unsupported range_policy")
    if common["global_read_bytes"] != expected_read:
        _fail(f"dual global_read_bytes does not satisfy {read_formula}")
    return {
        "schema": "dual_endpoint_shared_traffic_v1",
        "endpoints": endpoints,
        "global_start_ns": start_ns[0],
        "global_done_ns": done_ns[0],
        "union_stripes": union_stripes,
        "global_read_bytes": common["global_read_bytes"],
        "global_write_bytes": common["global_write_bytes"],
        "endpoint_write_bytes": dict(zip(endpoints, write_values)),
        "read_formula": read_formula,
        "selected_bytes": selected_bytes,
    }


def _validate_rebuild_events(
    grouped: dict[int, dict[str, dict[str, Any]]],
    request_start_ns: int,
    request_end_ns: int,
    *,
    system: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized: dict[str, Any] = {}
    dead_ns: list[int] = []
    start_ns: list[int] = []
    done_ns: list[int] = []
    for endpoint, events in grouped.items():
        dead = _token_int(events["endpoint_dead"], "monotonic_ns")
        start = _token_int(events["start"], "monotonic_ns")
        done = _token_int(events["done"], "monotonic_ns")
        done_start = _token_int(events["done"], "start_ns")
        duration = _token_int(events["done"], "duration_ns")
        if done_start != start or duration != done - start or duration <= 0:
            _fail(
                f"endpoint {endpoint} rebuild duration is not closed: "
                f"start={start} done={done} start_ns={done_start} "
                f"duration_ns={duration}"
            )
        if not request_start_ns <= dead <= start <= done <= request_end_ns:
            _fail(
                f"endpoint {endpoint} native recovery events fall outside "
                f"request window: {dead}, {start}, {done}"
            )
        done_record = events["done"]
        completed = _optional_token_int(
            done_record, ("completed_stripes",), "completed_stripes"
        )
        write_completed = _optional_token_int(
            done_record, ("write_completed_stripes",), "write_completed_stripes"
        )
        if completed is None or write_completed is None:
            _fail(f"endpoint {endpoint} done event lacks stripe completion counters")
        if completed <= 0 or write_completed != completed:
            _fail(
                f"endpoint {endpoint} done event is not a full write completion: "
                f"completed={completed} write_completed={write_completed}"
            )
        for key in ("remaining", "failed", "in_flight"):
            value = _optional_token_int(done_record, (key, f"{key}_stripes"), key)
            if value is None:
                _fail(f"endpoint {endpoint} done event lacks {key}")
            if value != 0:
                _fail(f"endpoint {endpoint} rebuild {key} is {value}, expected 0")
        redirected = _optional_token_int(
            events["joined"], ("redirected_reads",), "redirected_reads"
        )
        if redirected is None or redirected <= 0:
            _fail(
                f"endpoint {endpoint} joined event lacks positive redirected_reads"
            )
        dead_ns.append(dead)
        start_ns.append(start)
        done_ns.append(done)
        normalized[str(endpoint)] = {
            "endpoint_dead": events["endpoint_dead"],
            "start": events["start"],
            "done": events["done"],
            "joined": events["joined"],
            "start_ns": start,
            "done_ns": done,
            "duration_ns": duration,
            "completed_stripes": completed,
            "write_completed_stripes": write_completed,
            "redirected_reads": redirected,
        }
    aggregate = {
        "failure_ns": max(dead_ns),
        "start_ns": min(start_ns),
        "done_ns": max(done_ns),
        "duration_ns": max(done_ns) - min(start_ns),
        "endpoint_count": len(grouped),
    }
    traffic = _validate_dual_rebuild_traffic(grouped, system=system)
    if traffic is not None:
        aggregate["traffic"] = traffic
    return aggregate, normalized


def _validate_revalidation(
    result: dict[str, Any],
    run_dir: Path,
    client_bytes: bytes,
) -> dict[str, Any]:
    if result.get("_result_filename") == "revalidated-dual-result.json":
        original_path = _safe_reference(
            result.get("original_result"), "dual-result.json", run_dir
        )
        validation_path = _safe_reference(
            result.get("original_validation"), "dual-validation.json", run_dir
        )
        original = _read_json(original_path)
        original_validation = _read_json(validation_path)
        if original.get("run") != result.get("run"):
            _fail("revalidated dual result run disagrees with original result")
        if original.get("passed") is not False:
            _fail("revalidated dual result does not retain original passed=false result")
        if original_validation.get("passed") is not False:
            _fail("dual-validation.json is not the retained failed validation")
        errors = original_validation.get("errors")
        if errors != ["client.config ft_rmw_read_failure_fallback=None, expected '1'"]:
            _fail("retained Hydra dual failure is not the documented RMW-config check")
        _bool_true(result.get("passed"), "revalidated dual result passed")
        validation = result.get("validation")
        if not isinstance(validation, dict):
            _fail("revalidated dual result lacks validation")
        _bool_true(validation.get("passed"), "revalidated dual validation passed")
        correction = result.get("validation_correction")
        if not isinstance(correction, str) or not correction.strip():
            _fail("revalidated dual result lacks its correction explanation")
        return {
            "selected_result": result.get("_result_filename"),
            "original_result_file": original_path.name,
            "original_validation_file": validation_path.name,
            "original_result_passed": original.get("passed"),
            "original_validation_passed": original_validation.get("passed"),
            "original_validation_errors": errors,
            "validation_correction": correction,
        }
    is_revalidation = (
        result.get("_result_filename") == "revalidation-background-result.json"
    )
    if not is_revalidation:
        return {"selected_result": result.get("_result_filename")}
    if str(result.get("runtime", "hydra")).lower() != "hydra":
        _fail("revalidation-background-result.json is only supported for Hydra")
    coverage_path = _safe_reference(
        result.get("coverage_revalidation_file"),
        "coverage-revalidation.json",
        run_dir,
    )
    original_coverage_path = _safe_reference(
        result.get("original_coverage_file"), "run-result.json", run_dir
    )
    original_result_path = _safe_reference(
        result.get("original_result_file"), "recovery-result.json", run_dir
    )
    proof_path = _safe_reference(
        result.get("interleave_proof"), "stream-interleave-proof-v2.json", run_dir
    )
    coverage_revalidation = _read_json(coverage_path)
    _bool_true(coverage_revalidation.get("passed"), "coverage revalidation passed")
    if coverage_revalidation.get("raw_modified") is not False:
        _fail("coverage revalidation does not prove raw_modified=false")
    if coverage_revalidation.get("original_coverage_file") != "run-result.json":
        _fail("coverage revalidation has an unsafe original coverage reference")
    if coverage_revalidation.get("coverage") != result.get("coverage"):
        _fail("selected coverage differs from coverage-revalidation.json")
    proof = _read_json(proof_path)
    if proof.get("run") != result.get("run"):
        _fail("stream interleave proof run disagrees with selected result")
    _bool_true(proof.get("raw_block_unique"), "stream raw_block_unique")
    if proof.get("raw_modified") is not False:
        _fail("stream interleave proof does not prove raw_modified=false")
    raw_block = proof.get("raw_block")
    if not isinstance(raw_block, str) or not raw_block:
        _fail("stream interleave proof lacks raw_block")
    needle = raw_block.encode("utf-8")
    if client_bytes.count(needle) != 1:
        _fail("stream proof raw_block bytes do not uniquely match original client.log")
    return {
        "selected_result": result.get("_result_filename"),
        "coverage_revalidation_file": coverage_path.name,
        "original_coverage_file": original_coverage_path.name,
        "original_result_file": original_result_path.name,
        "stream_interleave_proof": proof_path.name,
        "coverage_revalidation": coverage_revalidation,
        "stream_interleave_proof_data": proof,
        "raw_block_match_count": client_bytes.count(needle),
        "original_result_passed": _read_json(original_result_path).get("passed"),
    }


def _inject_endpoint_ports(plan: dict[str, Any], endpoints: list[int]) -> dict[int, int]:
    inject = plan.get("inject")
    if not isinstance(inject, dict):
        _fail("plan lacks inject object")
    server_ports = plan.get("server_ports") or plan.get("active_server_ports")
    if not isinstance(server_ports, list):
        server_ports = []
    raw_items = inject.get("endpoints")
    if raw_items is None:
        raw_items = inject.get("endpoint")
    if isinstance(raw_items, dict):
        raw_items = [raw_items] if "endpoint" in raw_items else list(raw_items.values())
    elif not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]
    item_by_endpoint: dict[int, dict[str, Any]] = {}
    for item in raw_items:
        if isinstance(item, dict):
            endpoint = None
            for key in ("endpoint", "index", "id"):
                if key in item:
                    endpoint = _int(item[key], "inject endpoint")
                    break
            if endpoint is None:
                _fail(f"inject endpoint object lacks endpoint/index/id: {item!r}")
            item_by_endpoint[endpoint] = item
    ports: dict[int, int] = {}
    for endpoint in endpoints:
        item = item_by_endpoint.get(endpoint, {})
        port = item.get("port")
        if port is None and len(endpoints) == 1:
            port = inject.get("port")
        if port is None:
            declared_ports = inject.get("ports")
            if isinstance(declared_ports, dict):
                port = declared_ports.get(str(endpoint), declared_ports.get(endpoint))
            elif isinstance(declared_ports, (list, tuple)) and endpoint < len(declared_ports):
                port = declared_ports[endpoint]
        if port is None and endpoint < len(server_ports):
            port = server_ports[endpoint]
        if port is None:
            _fail(f"cannot map injected endpoint {endpoint} to a server port")
        ports[endpoint] = _int(port, f"inject endpoint {endpoint} port")
    if len(set(ports.values())) != len(ports):
        _fail("injected endpoints do not have unique server ports")
    return ports


def _validate_fault(
    result: dict[str, Any], plan: dict[str, Any], endpoints: list[int],
    *, run_dir: Path | None = None,
) -> dict[str, Any]:
    plan_inject = plan.get("inject")
    source_result = result
    source_path: Path | None = None
    if len(endpoints) > 1:
        if run_dir is None:
            _fail("dual fault validation lacks run directory")
        source_result, source_path = _dual_source_result(result, run_dir)
    inject = source_result.get("inject")
    if len(endpoints) > 1 and inject is None:
        inject = plan_inject
    if not isinstance(plan_inject, dict) or not isinstance(inject, dict):
        _fail("selected result and plan must both record inject objects")
    if inject != plan_inject:
        _fail("selected result inject declaration disagrees with plan")
    if inject.get("enabled") is not True:
        _fail("selected result does not record an enabled fault")
    fault = source_result.get("fault")
    if not isinstance(fault, dict):
        fault = {}
    if len(endpoints) > 1:
        receipts = fault.get("receipts")
        if not isinstance(receipts, list) or len(receipts) != len(endpoints):
            _fail("dual fault receipt lacks one kill receipt per endpoint")
        endpoint_ports = _inject_endpoint_ports(plan, endpoints)
        expected_binary = plan.get("server_binary")
        if expected_binary is None and isinstance(plan.get("server_remote_root"), str):
            expected_binary = plan["server_remote_root"].rstrip("/") + "/bin/server"
        if not isinstance(expected_binary, str) or not expected_binary:
            _fail("plan lacks expected server binary identity")
        identities: list[dict[str, Any]] = []
        seen_endpoints: set[int] = set()
        for receipt in receipts:
            if not isinstance(receipt, dict) or receipt.get("action") != "kill":
                _fail("dual fault receipt is not an owned process kill")
            _bool_true(receipt.get("identity_verified"), "dual fault identity_verified")
            _bool_true(receipt.get("process_stopped"), "dual fault process_stopped")
            processes = receipt.get("processes")
            if not isinstance(processes, list) or len(processes) != 1:
                _fail("dual fault receipt must identify exactly one process")
            process = processes[0]
            if not isinstance(process, dict):
                _fail("dual fault process identity is not an object")
            port = _int(process.get("port"), "dual fault process port")
            endpoint = next((candidate for candidate, expected_port in endpoint_ports.items()
                             if expected_port == port), None)
            if endpoint is None or endpoint in seen_endpoints:
                _fail("dual fault process port does not map uniquely to an injected endpoint")
            seen_endpoints.add(endpoint)
            if "endpoint" in process and _int(process["endpoint"], "dual fault process endpoint") != endpoint:
                _fail("dual fault process endpoint/port mismatch")
            pid = _int(process.get("pid"), "dual fault process pid")
            start_ticks = _nonnegative_int(process.get("start_ticks"), "dual fault process start_ticks")
            binary = process.get("binary")
            binary_matches = binary == expected_binary
            if (not binary_matches and isinstance(binary, str)
                    and expected_binary.startswith("/home/xiayanwen/research/")):
                binary_alias = "/mnt/nfs/xiayanwen/research/" + expected_binary.split(
                    "/research/", 1
                )[1]
                binary_matches = binary == binary_alias
            if pid <= 0 or not isinstance(binary, str) or not binary_matches:
                _fail("dual fault process binary identity disagrees with plan")
            identities.append({
                "endpoint": endpoint, "port": port, "pid": pid,
                "start_ticks": start_ticks, "binary": binary,
                "memory_numa": process.get("memory_numa"),
            })
        batch = fault.get("batch_receipt")
        if not isinstance(batch, dict) or batch.get("action") != "kill-many":
            _fail("dual fault lacks kill-many batch receipt")
        _bool_true(batch.get("identity_verified"), "dual batch identity_verified")
        _bool_true(batch.get("process_stopped"), "dual batch process_stopped")
        if sorted(_int(port, "dual batch port") for port in batch.get("ports", [])) != sorted(endpoint_ports.values()):
            _fail("dual batch ports disagree with inject endpoints")
        return {
            "model": inject.get("model"),
            "scope": "logical service process",
            "receipt": fault,
            "declared": inject,
            "declared_endpoints": endpoints,
            "endpoint_ports": endpoint_ports,
            "process_identities": identities,
            "source_result": source_path.name if source_path else result.get("_result_filename"),
        }
    receipt = fault.get("receipt")
    if not isinstance(receipt, dict):
        receipt = result.get("stop_receipt")
    if not isinstance(receipt, dict):
        _fail("selected result lacks fault receipt")
    if receipt.get("action") != "kill":
        _fail("fault receipt is not an owned process kill")
    _bool_true(receipt.get("identity_verified"), "fault identity_verified")
    _bool_true(receipt.get("process_stopped"), "fault process_stopped")
    processes = receipt.get("processes")
    if not isinstance(processes, list) or not processes:
        _fail("fault receipt lacks stopped process identities")
    endpoint_ports = _inject_endpoint_ports(plan, endpoints)
    expected_binary = plan.get("server_binary")
    if expected_binary is None and isinstance(plan.get("server_remote_root"), str):
        expected_binary = plan["server_remote_root"].rstrip("/") + "/bin/server"
    if not isinstance(expected_binary, str) or not expected_binary:
        _fail("plan lacks expected server binary identity")
    if len(processes) != len(endpoints):
        _fail(
            f"fault receipt process count {len(processes)} does not match "
            f"injected endpoint count {len(endpoints)}"
        )
    identities = []
    seen_ports: set[int] = set()
    for process in processes:
        if not isinstance(process, dict):
            _fail("fault receipt process identity is not an object")
        port = _int(process.get("port"), "fault receipt process port")
        endpoint = next(
            (candidate for candidate, expected_port in endpoint_ports.items()
             if expected_port == port), None
        )
        if endpoint is None:
            _fail(f"fault receipt process port {port} is not an injected endpoint")
        if port in seen_ports:
            _fail(f"duplicate fault receipt process port {port}")
        seen_ports.add(port)
        if "endpoint" in process and _int(
                process["endpoint"], "fault receipt process endpoint") != endpoint:
            _fail(f"fault receipt endpoint/port mismatch for port {port}")
        pid = _int(process.get("pid"), "fault receipt process pid")
        start_ticks = _nonnegative_int(
            process.get("start_ticks"), "fault receipt process start_ticks"
        )
        binary = process.get("binary")
        if pid <= 0 or not isinstance(binary, str) or not binary:
            _fail("fault receipt lacks pid/start_ticks/binary identity")
        if binary != expected_binary:
            _fail(
                f"fault receipt binary {binary!r} does not match "
                f"plan server binary {expected_binary!r}"
            )
        identities.append({
            "endpoint": endpoint, "port": port, "pid": pid,
            "start_ticks": start_ticks, "binary": binary,
            "memory_numa": process.get("memory_numa"),
        })
    return {
        "model": inject.get("model"),
        "scope": "logical service process",
        "receipt": receipt,
        "declared": inject,
        "declared_endpoints": endpoints,
        "endpoint_ports": endpoint_ports,
        "process_identities": identities,
    }


def _validate_client_configuration(
    plan: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any]:
    required = ("records", "object_bytes", "random_seed", "zipfian",
                "put_ratio", "remove_ratio", "app_fibres")
    missing = [key for key in required if key not in configuration]
    if missing:
        _fail("client.log lacks startup configuration: " + ", ".join(missing))
    plan_records = _int(plan.get("records"), "plan records")
    plan_object_bytes = _int(plan.get("object_bytes"), "plan object_bytes")
    plan_seed = _int(plan.get("random_seed"), "plan random_seed")
    plan_fibres = _int(plan.get("app_fibres"), "plan app_fibres")
    plan_zipfian = _finite_number(plan.get("zipfian"), "plan zipfian")
    plan_put = _finite_number(plan.get("put_ratio"), "plan put_ratio")
    plan_remove = _finite_number(
        _workload_value(plan, "remove_ratio",
                        env_keys=("FARLIB_KVS_REMOVE_RATIO",), default=0),
        "plan remove_ratio",
    )
    if _int(configuration["records"], "client records") != plan_records:
        _fail("client.log records disagree with plan")
    if _int(configuration["object_bytes"], "client object_bytes") != plan_object_bytes:
        _fail("client.log object_bytes disagree with plan")
    if _int(configuration["random_seed"], "client random_seed") != plan_seed:
        _fail("client.log random_seed disagree with plan")
    if _int(configuration["app_fibres"], "client app_fibres") != plan_fibres:
        _fail("client.log fibres disagree with plan")
    checks = (
        ("zipfian", plan_zipfian),
        ("put_ratio", plan_put),
        ("remove_ratio", plan_remove),
    )
    for key, expected in checks:
        actual = _finite_number(configuration[key], f"client {key}")
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            _fail(f"client.log {key} disagrees with plan")
    expected_get = 1.0 - plan_put - plan_remove
    plan_get = _finite_number(plan.get("get_ratio"), "plan get_ratio")
    if not math.isclose(plan_get, expected_get, rel_tol=0, abs_tol=1e-12):
        _fail("plan get/put/remove ratios do not close")
    return {
        "validated": True,
        "records": plan_records,
        "object_bytes": plan_object_bytes,
        "random_seed": plan_seed,
        "app_fibres": plan_fibres,
        "zipfian": plan_zipfian,
        "put_ratio": plan_put,
        "remove_ratio": plan_remove,
        "derived_get_ratio": expected_get,
        "observed_operation_share_not_used": True,
    }


def _validate_placement(
    plan: dict[str, Any], worker_roles: list[dict[str, Any]]
) -> dict[str, Any]:
    app_cpus = [_int(cpu, "plan app CPU") for cpu in plan.get("app_cpus", [])]
    background_cpus = [
        _int(cpu, "plan background CPU") for cpu in plan.get("background_cpus", [])
    ]
    if not app_cpus or not background_cpus:
        _fail("plan lacks app/background CPU placement")
    app_workers = _int(plan.get("app_workers"), "plan app_workers")
    expected_counts = {"app": app_workers, "background": len(background_cpus)}
    if len(worker_roles) != sum(expected_counts.values()):
        _fail("client.log lacks complete worker_role CPU placement evidence")
    seen: set[tuple[str, int]] = set()
    actual_cpus = {"app": [], "background": []}
    actual_indices = {"app": [], "background": []}
    actual_roles = []
    for record in worker_roles:
        tokens = record["tokens"]
        role = tokens.get("role")
        if role not in expected_counts:
            _fail(f"unknown worker_role at {record['source']}")
        index = _int(tokens.get("index"), f"worker_role index at {record['source']}")
        tid = _int(tokens.get("tid"), f"worker_role tid at {record['source']}")
        cpu = _int(tokens.get("cpu"), f"worker_role cpu at {record['source']}")
        allowed = _int(tokens.get("allowed"), f"worker_role allowed at {record['source']}")
        allowed_count = _int(
            tokens.get("allowed_count"), f"worker_role allowed_count at {record['source']}"
        )
        if tid <= 0 or allowed_count != 1 or allowed != cpu:
            _fail(f"worker_role affinity evidence is incomplete at {record['source']}")
        identity = (role, index)
        if identity in seen:
            _fail(f"duplicate worker_role index {identity}")
        seen.add(identity)
        actual_cpus[role].append(cpu)
        actual_indices[role].append(index)
        actual_roles.append({
            "role": role, "index": index, "tid": tid, "cpu": cpu,
            "allowed": allowed, "source": record["source"],
        })
    for role, expected_cpus in (("app", app_cpus), ("background", background_cpus)):
        if len(actual_cpus[role]) != expected_counts[role]:
            _fail(f"worker_role count for {role} disagrees with plan")
        if sorted(actual_indices[role]) != list(range(expected_counts[role])):
            _fail(f"worker_role indices for {role} are not complete")
        if sorted(actual_cpus[role]) != sorted(expected_cpus):
            _fail(f"worker_role CPUs for {role} disagree with plan")
    return {
        "validated": True,
        "plan_app_cpus": app_cpus,
        "plan_background_cpus": background_cpus,
        "roles": actual_roles,
    }


def _validate_receipt_and_coverage(
    result: dict[str, Any],
    evidence: dict[str, Any],
    expected_requests: int,
) -> dict[str, Any]:
    if isinstance(result.get("validation"), dict) and "endpoint_proof" in result["validation"]:
        return _validate_dual_result(result, evidence, expected_requests)
    if result.get("passed") is not True:
        _fail("selected result passed=false")
    if _int(result.get("coverage_exit"), "coverage_exit") != 0:
        _fail("coverage_exit is not zero")
    if _int(result.get("cleanup_exit"), "cleanup_exit") != 0:
        _fail("cleanup_exit is not zero")
    coverage = result.get("coverage")
    if not isinstance(coverage, dict):
        _fail("selected result lacks coverage object")
    _bool_true(coverage.get("passed"), "coverage passed")
    if _int(coverage.get("exit_code"), "coverage.exit_code") != 0:
        _fail("coverage exit_code is not zero")
    series = coverage.get("completion_series")
    if not isinstance(series, dict):
        _fail("coverage lacks completion_series")
    _bool_true(series.get("passed"), "completion_series passed")
    completed = _count(
        series.get("completed", coverage.get("completed")), "coverage completed"
    )
    if completed != expected_requests:
        _fail(
            f"coverage completed {completed} does not equal plan requests "
            f"{expected_requests}"
        )
    generation_record = _single(
        evidence["request"]["request_generation_end"], "request_generation_end"
    )
    generation_tokens = generation_record["tokens"]
    for key in ("generated", "accepted"):
        if key not in generation_tokens:
            _fail(f"request_generation_end lacks {key}")
        if _count(generation_tokens[key], f"request_generation_end.{key}") != expected_requests:
            _fail(f"request_generation_end {key} differs from plan requests")
    drain_record = _single(
        evidence["request"]["request_drain_end"], "request_drain_end"
    )
    if "completed" not in drain_record["tokens"]:
        _fail("request_drain_end lacks completed")
    if _count(drain_record["tokens"]["completed"], "request_drain_end.completed") != expected_requests:
        _fail("request_drain_end completed differs from plan requests")
    receipt_record = _single(evidence["receipts"], "kvs_receipt")
    raw_receipt = receipt_record["tokens"]
    if any(key not in raw_receipt for key in COUNT_FIELDS):
        _fail(
            "kvs_receipt lacks one or more request counters at "
            f"{receipt_record['source']}"
        )
    if raw_receipt.get("accepted_equals_completed") != "1":
        _fail("kvs_receipt accepted_equals_completed is not 1")
    fingerprint = raw_receipt.get("request_fingerprint")
    if not fingerprint:
        _fail("kvs_receipt lacks request_fingerprint")
    raw_counts = {key: _count(raw_receipt[key], key) for key in COUNT_FIELDS}
    generated_total = sum(
        raw_counts[f"generated_{op}"] for op in ("get", "put", "remove")
    )
    accepted_total = sum(
        raw_counts[f"accepted_{op}"] for op in ("get", "put", "remove")
    )
    completed_total = sum(
        raw_counts[f"completed_{op}"] for op in ("get", "put", "remove")
    )
    if generated_total != expected_requests or accepted_total != expected_requests:
        _fail("kvs_receipt generated/accepted total differs from plan requests")
    if completed_total != expected_requests:
        _fail("kvs_receipt completed total differs from plan requests")
    if any(
        raw_counts[f"accepted_{op}"] != raw_counts[f"completed_{op}"]
        for op in ("get", "put", "remove")
    ):
        _fail("kvs_receipt has accepted/completed per-operation mismatch")
    coverage_receipt = coverage.get("receipt")
    if not isinstance(coverage_receipt, dict):
        _fail("coverage lacks receipt")
    for key in COUNT_FIELDS + ("accepted_equals_completed", "request_fingerprint"):
        if key not in coverage_receipt:
            _fail(f"coverage receipt lacks {key}")
        if str(coverage_receipt[key]) != str(raw_receipt[key]):
            _fail(f"coverage receipt disagrees with client.log for {key}")
    post_record = _single(evidence["post_verify"], "kvs_post_verify")
    checked = _count(post_record["tokens"].get("checked"), "kvs_post_verify.checked")
    failures = _count(post_record["tokens"].get("failures"), "kvs_post_verify.failures")
    if checked != 4096 or failures != 0:
        _fail(f"post verification is checked={checked}, failures={failures}")
    coverage_post = coverage.get("post_verify")
    if not isinstance(coverage_post, dict):
        _fail("coverage lacks post_verify")
    if _count(coverage_post.get("checked"), "coverage.post_verify.checked") != checked:
        _fail("coverage post verification checked count disagrees with client.log")
    if _count(coverage_post.get("failures"), "coverage.post_verify.failures") != failures:
        _fail("coverage post verification failure count disagrees with client.log")
    return {
        "receipt": raw_receipt,
        "post_verify": {"checked": checked, "failures": failures},
        "coverage": coverage,
        "request_fingerprint": fingerprint,
    }


def _read_windows(
    run_dir: Path,
    request_start_ns: int,
    request_end_ns: int,
    expected_requests: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = run_dir / "kvs.completed-100ms.csv"
    json_path = run_dir / "kvs.completed-100ms.json"
    summary = _read_json(json_path)
    if summary.get("enabled") is not True:
        _fail("completion-series JSON is not enabled")
    if _int(summary.get("request_start_ns"), "completion JSON request_start_ns") != request_start_ns:
        _fail("completion JSON request_start_ns disagrees with client.log")
    if _int(summary.get("request_end_ns"), "completion JSON request_end_ns") != request_end_ns:
        _fail("completion JSON request_end_ns disagrees with client.log")
    if _count(summary.get("completed"), "completion JSON completed") != expected_requests:
        _fail("completion JSON completed differs from plan requests")
    if not csv_path.is_file():
        _fail(f"missing completion-series CSV: {csv_path}")
    try:
        stream = csv_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot read completion-series CSV {csv_path}: {exc}")
    windows: list[dict[str, Any]] = []
    try:
        reader = csv.DictReader(stream)
        required = {
            "window_start_ns", "window_end_ns", "completed",
            "cumulative_completed", "final",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            _fail("completion CSV lacks required window fields")
        for row in reader:
            line_number = reader.line_num
            start = _nonnegative_int(
                row.get("window_start_ns"), f"{csv_path}:{line_number} window_start_ns"
            )
            end = _nonnegative_int(
                row.get("window_end_ns"), f"{csv_path}:{line_number} window_end_ns"
            )
            completed = _count(
                row.get("completed"), f"{csv_path}:{line_number} completed"
            )
            cumulative = _count(
                row.get("cumulative_completed"),
                f"{csv_path}:{line_number} cumulative_completed",
            )
            final = _int(row.get("final"), f"{csv_path}:{line_number} final")
            if end <= start:
                _fail(f"completion window is not positive at {csv_path}:{line_number}")
            windows.append({
                "window_start_ns": start,
                "window_end_ns": end,
                "completed_ops": completed,
                "cumulative_completed": cumulative,
                "final": final,
                "window_start_s": (start - request_start_ns) / NANOSECONDS,
                "window_end_s": (end - request_start_ns) / NANOSECONDS,
                "window_source": _source(csv_path, line_number),
                "_source": _source(csv_path, line_number),
            })
    finally:
        stream.close()
    if not windows:
        _fail("completion CSV has no windows")
    if _count(summary.get("rows"), "completion JSON rows") != len(windows):
        _fail("completion JSON rows differs from CSV rows")
    previous_end = request_start_ns
    previous_cumulative = 0
    for index, window in enumerate(windows):
        if window["window_start_ns"] != previous_end:
            _fail(
                f"completion windows are not continuous at row {index + 2}: "
                f"{window['window_start_ns']} != {previous_end}"
            )
        expected_cumulative = previous_cumulative + window["completed_ops"]
        if window["cumulative_completed"] != expected_cumulative:
            _fail(
                "completion cumulative count does not close at "
                f"{window['window_source']}"
            )
        if window["final"] not in (0, 1):
            _fail(f"completion final must be 0/1 at {window['window_source']}")
        if index < len(windows) - 1 and window["final"] != 0:
            _fail(
                f"non-final completion window marked final at "
                f"{window['window_source']}"
            )
        previous_end = window["window_end_ns"]
        previous_cumulative = window["cumulative_completed"]
    if windows[-1]["window_end_ns"] != request_end_ns:
        _fail("last completion window does not end at request_drain_end")
    if windows[-1]["final"] != 1:
        _fail("last completion window is not marked final")
    if previous_cumulative != expected_requests:
        _fail("completion windows total differs from plan requests")
    return windows, {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "rows": len(windows),
        "completed": previous_cumulative,
        "request_start_ns": request_start_ns,
        "request_end_ns": request_end_ns,
        "summary": summary,
    }


def _native_request_times(
    evidence: dict[str, Any], window_json: dict[str, Any]
) -> tuple[int, int]:
    start_record = _single(evidence["request"]["request_start"], "request_start")
    end_record = _single(evidence["request"]["request_drain_end"], "request_drain_end")
    start = _token_int(start_record, "monotonic_ns")
    end = _token_int(end_record, "monotonic_ns")
    if end <= start:
        _fail("native request_drain_end is not after request_start")
    generation_record = _single(
        evidence["request"]["request_generation_end"], "request_generation_end"
    )
    generation_ns = _token_int(generation_record, "monotonic_ns")
    if generation_ns != end:
        _fail("request_generation_end disagrees with request_drain_end")
    if _int(window_json.get("request_start_ns"), "completion JSON request_start_ns") != start:
        _fail("completion JSON request_start_ns disagrees with native request_start")
    if _int(window_json.get("request_end_ns"), "completion JSON request_end_ns") != end:
        _fail("completion JSON request_end_ns disagrees with native request_drain_end")
    return start, end


def read_run(result_path: str | Path, *, steady_before_s: float = 10.0):
    """Read and validate one native Figure 13 result."""
    steady_before_s = _finite_number(steady_before_s, "steady_before_s")
    if steady_before_s < 0:
        _fail("steady_before_s must be nonnegative")
    selected_path = Path(result_path).expanduser()
    if not selected_path.is_file():
        _fail(f"result file not found: {selected_path}")
    selected_path = selected_path.resolve()
    run_dir = selected_path.parent
    result = _read_json(selected_path)
    if result.get("schema") == "figure13-site-result-v1":
        try:
            from .site_runs import read_site_run
        except ImportError:
            from site_runs import read_site_run
        return read_site_run(selected_path, steady_before_s=steady_before_s)
    result["_result_filename"] = selected_path.name
    plan_path = run_dir / "plan.json"
    plan = _read_json(plan_path)
    system = str(result.get("runtime", plan.get("runtime", ""))).lower()
    if system not in SYSTEMS:
        _fail(f"unsupported Figure 13 runtime: {system!r}")
    plan_system = str(plan.get("runtime", system)).lower()
    if plan_system != system:
        _fail("selected result runtime disagrees with plan runtime")
    # Carbink v11 uses the native runner/validator JSON contract rather than
    # the older S/H coverage+inject contract.  Keep the established path
    # untouched for S/H and legacy non-background Carbink records.
    if system == "carbink" and result.get("background_rebuild") is True:
        try:
            from .carbink_native import read_carbink_run
        except ImportError:
            from carbink_native import read_carbink_run
        return read_carbink_run(
            selected_path, result, plan, steady_before_s=steady_before_s
        )
    run_id = result.get("run", plan.get("run"))
    if not isinstance(run_id, str) or not run_id:
        _fail("native result lacks run id")
    if plan.get("run", run_id) != run_id:
        _fail("selected result run id disagrees with plan")
    endpoints = _declared_endpoints(plan)
    scenario = _scenario(plan, endpoints)
    workload_id, workload_payload, workload_canonical = _workload_identity(plan)
    client_path = run_dir / "client.log"
    client_text, evidence = _read_client_log(client_path)
    window_json = _read_json(run_dir / "kvs.completed-100ms.json")
    request_start_ns, request_end_ns = _native_request_times(evidence, window_json)
    expected_requests = _nonnegative_int(plan.get("requests"), "plan requests")
    if expected_requests <= 0:
        _fail("plan requests must be positive")
    windows, window_details = _read_windows(
        run_dir, request_start_ns, request_end_ns, expected_requests
    )
    if len(endpoints) == 2:
        native_coverage = _read_json(run_dir / "run-result.json")
        if native_coverage.get("passed") is not True or native_coverage.get("exit_code") != 0:
            _fail("dual native client coverage is not passed/exit0")
    topology_details = (
        _validate_dual_topology(result, plan, endpoints, system, run_dir)
        if len(endpoints) == 2 else None
    )
    receipt_details = _validate_receipt_and_coverage(
        result, evidence, expected_requests
    )
    fault_details = _validate_fault(result, plan, endpoints, run_dir=run_dir)
    configuration_details = _validate_client_configuration(
        plan, evidence["configuration"]
    )
    placement_details = _validate_placement(plan, evidence["worker_roles"])
    aggregate, event_details = _validate_rebuild_events(
        _group_rebuild_events(evidence, endpoints), request_start_ns, request_end_ns,
        system=system,
    )
    packing_records = evidence["packing"]
    packing_ns: int | None = None
    if packing_records:
        packing_record = _single(packing_records, "runtime_packing_ready")
        packing_ns = _token_int(packing_record, "monotonic_ns")
    result_packing_ns = None
    fault = result.get("fault")
    if isinstance(fault, dict) and fault.get("packing_ready_ns") is not None:
        result_packing_ns = _nonnegative_int(
            fault["packing_ready_ns"], "fault.packing_ready_ns"
        )
    if (
        packing_ns is not None
        and result_packing_ns is not None
        and packing_ns != result_packing_ns
    ):
        _fail("native packing_ready timestamp disagrees with result evidence")
    if packing_ns is None:
        packing_ns = result_packing_ns
    if packing_ns is not None and not request_start_ns <= packing_ns <= request_end_ns:
        _fail("packing_ready timestamp falls outside native request window")
    failure_s = (aggregate["failure_ns"] - request_start_ns) / NANOSECONDS
    rebuild_start_s = (aggregate["start_ns"] - request_start_ns) / NANOSECONDS
    recovered_s = (aggregate["done_ns"] - request_start_ns) / NANOSECONDS
    packing_s = (
        (packing_ns - request_start_ns) / NANOSECONDS
        if packing_ns is not None else 0.0
    )
    steady_start_s = max(0.0, failure_s - steady_before_s, packing_s)
    if not steady_start_s < failure_s:
        _fail("no steady pre-failure interval remains after native packing boundary")
    ratio_value = _finite_number(plan.get("local_bytes"), "plan local_bytes")
    logical_bytes = _finite_number(plan.get("logical_bytes"), "plan logical_bytes")
    if logical_bytes <= 0 or ratio_value < 0:
        _fail("plan local/logical bytes are invalid")
    ratio = ratio_value / logical_bytes * 100.0
    if ratio.is_integer():
        ratio = int(ratio)
    revalidation_details = _validate_revalidation(
        result, run_dir, evidence["raw_bytes"]
    )
    if plan.get("compute") is None or plan.get("memory") is None:
        _fail("plan lacks compute/memory environment")
    environment = f"{plan['compute']}->{plan['memory']}"
    details: dict[str, Any] = {
        "adapter": "figure13.raw_runs",
        "source_type": "native_result_json",
        "result_path": str(selected_path),
        "plan_path": str(plan_path),
        "client_log_path": str(client_path),
        "client_log_sha256": evidence["sha256"],
        "raw_result_source": selected_path.name,
        "recovery_definition": "background_start_to_done",
        "steady_before_s": steady_before_s,
        "packing_ready_ns": packing_ns,
        "packing_ready_elapsed_s": packing_s if packing_ns is not None else None,
        "workload_id_payload": workload_payload,
        "workload_id_canonical_json": workload_canonical,
        "plan": plan,
        "selected_result": result,
        "baseline_config": {
            key: plan.get(key)
            for key in (
                "runtime", "variant", "compute", "memory", "app_cpus",
                "background_cpus", "remote_endpoints", "active_remote_endpoints",
                "server_count", "server_ports", "backup", "planner",
                "regionResident", "runtime_hot_packing", "remote_bytes_each",
                "remote_total_bytes", "standby_endpoint", "feature_arm",
            )
        },
        "fault": fault_details,
        "fault_physical_model": "logical service process",
        "topology_validation": topology_details,
        "startup_configuration": configuration_details,
        "placement_validation": placement_details,
        "rebuild_events": event_details,
        "rebuild_aggregate": aggregate,
        "rebuild_traffic": {
            endpoint: data["done"]["tokens"]
            for endpoint, data in event_details.items()
        },
        "native_rebuild_duration_s": aggregate["duration_ns"] / NANOSECONDS,
        "receipt_and_correctness": receipt_details,
        "windows": window_details,
        "revalidation": revalidation_details,
        "source_files": {
            "result": selected_path.name,
            "plan": plan_path.name,
            "client_log": client_path.name,
            "windows_csv": Path(window_details["csv_path"]).name,
            "windows_json": Path(window_details["json_path"]).name,
        },
    }
    run = {
        "run_id": run_id,
        "system": system,
        "scenario": scenario,
        "workload": "kv-b",
        "environment": environment,
        "workload_id": workload_id,
        "ratio": ratio,
        "app_workers": _int(plan.get("app_workers"), "plan app_workers"),
        "repeat": "unindexed",
        "phase": "work",
        "time_origin": "work_start",
        "steady_start_s": steady_start_s,
        "failure_elapsed_s": failure_s,
        "rebuild_start_elapsed_s": rebuild_start_s,
        "recovered_elapsed_s": recovered_s,
        "run_end_s": (request_end_ns - request_start_ns) / NANOSECONDS,
        "exit_status": 0,
        "correctness": "pass",
        "source_type": "measured",
        "failure_confirmed": 1,
        "recovery_verified": 1,
        "panel": "both",
        "_source": str(selected_path),
        "recovery_definition": "background_start_to_done",
    }
    return run, windows, details
