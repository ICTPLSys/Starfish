#!/usr/bin/env python3
"""Independent adapter for normal AE Figure13 site-run evidence.

The common runner's passed/status fields are only a hint. This module
revalidates the manifest, analysis, owned endpoint cleanup, fault receipts,
client request receipt, completion windows, and native rebuild events before
returning the legacy (run, windows, details) tuple consumed by collect.py.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

NANOSECONDS = 1_000_000_000
SYSTEMS = {"hydra", "carbink", "starfish"}
SCENARIOS = {"1-node", "2-node"}


def _rr():
    try:
        from . import raw_runs as rr
    except ImportError:
        import raw_runs as rr
    return rr


def _fail(message: str) -> None:
    raise _rr().RawRunError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return _rr()._read_json(path)


def _ref(run_dir: Path, value: Any, expected: str) -> Path:
    if not isinstance(value, str) or value != expected:
        _fail(f"site result must reference {expected!r}")
    path = run_dir / value
    if not path.is_file():
        _fail(f"missing {expected} evidence: {path}")
    return path


def _int(value: Any, field: str) -> int:
    return _rr()._int(value, field)


def _nonnegative(value: Any, field: str) -> int:
    return _rr()._nonnegative_int(value, field)


def _number(value: Any, field: str) -> float:
    return _rr()._finite_number(value, field)


def _bool_true(value: Any, field: str) -> None:
    _rr()._bool_true(value, field)


def _first(sources: list[Any], *keys: str, default: Any = None) -> Any:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return default


def _as_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if value is None:
        return []
    return [value]


def _endpoint_ids(value: Any, field: str) -> list[int]:
    if isinstance(value, dict):
        if "endpoint" in value:
            value = [value]
        else:
            result = []
            for key in value:
                try:
                    result.append(_int(key, field))
                except Exception:
                    pass
            if result:
                return result
    result = []
    for item in _as_list(value, field):
        if isinstance(item, dict):
            item = item.get("endpoint", item.get("index", item.get("id")))
        result.append(_int(item, field))
    return result


def _figure_plan(manifest):
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("figure13"), dict):
        _fail("manifest lacks plan.figure13")
    figure = plan["figure13"]
    required = ("system", "scenario", "repeat", "run_id", "app", "ratio",
                "observer_cpu", "logical_endpoints", "active_endpoints",
                "standby_endpoints", "failed_endpoints", "requests")
    if figure.get("enabled") is not True or any(key not in figure for key in required):
        _fail("manifest Figure13 plan is incomplete/disabled")
    return figure


def _merge_plan(manifest, analysis, result):
    # Only the producer's recorded plan defines the experiment. Analysis and
    # result summaries must agree with it, never override its configuration.
    plan = dict(manifest["plan"])
    figure = _figure_plan(manifest)
    for key in ("system", "run_id", "app", "ratio"):
        if plan.get(key) != figure.get(key):
            _fail("common/Figure13 plan identity mismatch: " + key)
    plan.update(figure)
    return plan


def _config_text(run_dir: Path, value: Any) -> str:
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            return value
        path = Path(value)
        try:
            if not path.is_absolute():
                candidate = run_dir / path
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8", errors="replace")
            if value.strip().startswith("/") and path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return value
        return value
    if isinstance(value, dict):
        return "\n".join(f"{key} {val}" for key, val in value.items()) + "\n"
    _fail("site manifest lacks effective_config text")


def _config_values(run_dir: Path, plan: dict[str, Any],
                   manifest: dict[str, Any]) -> dict[str, str]:
    value = _first(
        [plan, manifest],
        "effective_config", "effective_config_text", "config",
    )
    text = _config_text(run_dir, value)
    values: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and not fields[0].startswith("#"):
            values[fields[0]] = fields[1]
    if not values:
        _fail("effective_config contains no key/value settings")
    return values


def _saved_effective_config(run_dir: Path, plan: dict[str, Any],
                            manifest: dict[str, Any]) -> str:
    path = run_dir / "effective.config"
    if not path.is_file():
        _fail("site run lacks saved effective.config")
    text = path.read_text(encoding="utf-8", errors="replace")
    declared = _first([plan, manifest], "effective_config_sha256")
    if declared is not None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != declared:
            _fail("effective.config hash disagrees with manifest")
    declared_text = _first([plan], "effective_config")
    if isinstance(declared_text, str) and ("\n" in declared_text or "\r" in declared_text):
        if declared_text != text:
            _fail("saved effective.config disagrees with manifest plan")
    return text


def _worker_profile(plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    profile = _first([plan, manifest], "worker_profile", default={})
    if not isinstance(profile, dict):
        _fail("worker_profile must be an object")
    configured = _first([plan, manifest], "configured_cpu_profile", default={})
    if not isinstance(configured, dict):
        _fail("configured_cpu_profile must be an object")
    client_env = _first([plan, manifest], "client_env", default={})
    if not isinstance(client_env, dict):
        client_env = {}
    cpu_sources = [configured, profile, plan, manifest]
    app_cpus = _first(
        cpu_sources, "app_cpus", "application_cpus",
    )
    background_cpus = _first(
        cpu_sources, "background_cpus", "evacuator_cpus",
    )
    app_workers = _first(
        [profile, plan, manifest], "app_workers", "application_workers",
    )
    app_fibres = _first(
        [profile, plan, manifest, client_env], "app_fibres", "fibres",
        "FARLIB_KVS_DIRECT_FIBRES",
    )
    if not isinstance(app_cpus, list) or not app_cpus:
        _fail("worker_profile lacks app_cpus")
    if not isinstance(background_cpus, list) or not background_cpus:
        _fail("worker_profile lacks background_cpus")
    if app_workers is None or app_fibres is None:
        _fail("worker_profile lacks app_workers/app_fibres")
    return {
        "app_cpus": [_int(cpu, "app CPU") for cpu in app_cpus],
        "background_cpus": [_int(cpu, "background CPU") for cpu in background_cpus],
        "app_workers": _int(app_workers, "app_workers"),
        "app_fibres": _int(app_fibres, "app_fibres"),
        "raw": profile,
    }


def _workload_payload(plan: dict[str, Any], config: dict[str, str],
                      receipt_requests: int, worker: dict[str, Any],
                      manifest: dict[str, Any]) -> dict[str, Any]:
    def pick(*keys: str, default: Any = None) -> Any:
        value = _first([plan, manifest], *keys, default=None)
        if value is not None:
            return value
        env = _first([plan, manifest], "client_env", default={})
        if isinstance(env, dict):
            aliases = {
                "records": ("FARLIB_KVS_INITIAL_DATA_COUNT",),
                "object_bytes": ("FARLIB_KVS_OBJECT_BYTES",),
                "requests": ("FARLIB_KVS_MAX_SERVE_COUNT",),
                "put_ratio": ("FARLIB_KVS_PUT_RATIO",),
                "remove_ratio": ("FARLIB_KVS_REMOVE_RATIO",),
                "zipfian": ("FARLIB_KVS_ZIPFIAN_CONSTANT",),
                "random_seed": ("FARLIB_KVS_RANDOM_SEED",),
            }
            for env_key in aliases.get(keys[0], ()):
                if env_key in env:
                    return env[env_key]
        config_aliases = {
            "records": "records", "object_bytes": "object_bytes",
            "put_ratio": "put_ratio", "remove_ratio": "remove_ratio",
            "zipfian": "zipfian", "random_seed": "random_seed",
            "local_bytes": "client_buffer_size",
            "logical_bytes": "workload_footprint_bytes",
            "get_ratio": "get_ratio",
            "requests": "requests",
            "request_count": "requests",
            "expected_requests": "requests",
        }
        config_key = config_aliases.get(keys[0])
        if config_key in config:
            return config[config_key]
        return default

    records = pick("records")
    object_bytes = pick("object_bytes")
    requests = pick("requests", "request_count", "expected_requests",
                    default=receipt_requests)
    put_ratio = pick("put_ratio")
    remove_ratio = pick("remove_ratio", default=0)
    zipfian = pick("zipfian")
    random_seed = pick("random_seed")
    get_ratio = pick("get_ratio")
    if get_ratio is None and put_ratio is not None and remove_ratio is not None:
        get_ratio = 1.0 - float(put_ratio) - float(remove_ratio)
    local_bytes = pick("local_bytes", "local_memory_bytes")
    logical_bytes = pick("logical_bytes", "logical_memory_bytes")
    compute = pick("compute", "compute_ip", default=manifest.get("compute_ip"))
    memory = pick("memory", "memory_host")
    endpoint_list = _as_list(manifest.get("memory_endpoints"), "memory_endpoints")
    if memory is None and endpoint_list:
        memory = endpoint_list[0].get("memory_host", endpoint_list[0].get("host"))
    inventory = _as_list(
        _first([plan, manifest], "memory_endpoint_inventory", default=[]),
        "memory_endpoint_inventory",
    )
    physical_inventory = []
    for item in inventory:
        if not isinstance(item, dict):
            _fail("memory endpoint inventory item is not an object")
        physical_inventory.append({
            "inventory_index": item.get("inventory_index", item.get("index")),
            "memory_host": item.get("memory_host", item.get("host")),
            "memory_addr": item.get("memory_addr", item.get("address")),
        })
    physical_inventory.sort(key=lambda item: (
        -1 if item["inventory_index"] is None else int(item["inventory_index"])
    ))
    values = {
        "workload": "kv-b",
        "records": records, "object_bytes": object_bytes, "requests": requests,
        "get_ratio": get_ratio, "put_ratio": put_ratio,
        "remove_ratio": remove_ratio, "zipfian": zipfian,
        "random_seed": random_seed, "local_bytes": local_bytes,
        "logical_bytes": logical_bytes, "app_workers": worker["app_workers"],
        "app_fibres": worker["app_fibres"],
        "physical_compute": compute,
        "physical_memory_inventory": physical_inventory,
        "app_cpus": worker["app_cpus"],
        "background_cpus": worker["background_cpus"],
    }
    missing = [key for key, value in values.items()
               if value is None and key not in {"workload"}]
    if missing:
        _fail("site workload identity lacks: " + ", ".join(missing))
    return values


def _validate_workload(payload: dict[str, Any]) -> None:
    expected = {
        "records": 33_554_432, "object_bytes": 512, "requests": 1_000_000_000,
        "get_ratio": 0.95, "put_ratio": 0.05, "remove_ratio": 0.0,
        "zipfian": 0.99, "random_seed": 20_260_917,
        "local_bytes": 4 * 1024**3, "logical_bytes": 16 * 1024**3,
        "app_workers": 24, "app_fibres": 48,
        "local_bytes": 4 * 1024**3, "logical_bytes": 16 * 1024**3,
    }
    for key, target in expected.items():
        actual = payload[key]
        if key in {"get_ratio", "put_ratio", "remove_ratio", "zipfian"}:
            if not math.isclose(float(actual), target, rel_tol=0, abs_tol=1e-12):
                _fail(f"site workload {key}={actual!r}, expected {target!r}")
        elif _int(actual, key) != target:
            _fail(f"site workload {key}={actual!r}, expected {target!r}")


def _endpoint_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("endpoints")
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list) or not raw:
        _fail("manifest lacks final endpoint records")
    return [item for item in raw if isinstance(item, dict)]


def _validate_cleanup(manifest, plan, scenario):
    endpoints = _endpoint_records(manifest)
    expected = 7 if scenario == "1-node" else 8
    if _int(plan.get("logical_endpoints"), "logical_endpoints") != expected:
        _fail("Figure13 logical endpoint count disagrees with scenario")
    planned = plan.get("memory_endpoints")
    if not isinstance(planned, list) or len(planned) != expected or len(endpoints) != expected:
        _fail("planned/final endpoint count disagrees with scenario")
    seen = set()
    for item in endpoints:
        index = _int(item.get("index"), "endpoint index")
        if index in seen or not 0 <= index < expected:
            _fail("duplicate/invalid endpoint index")
        seen.add(index)
        declared = planned[index]
        for key in ("index", "memory_host", "memory_addr", "server_port", "server_bin"):
            if item.get(key) != declared.get(key):
                _fail("final endpoint differs from deployment plan: " + key)
        if _int(item.get("pid"), "endpoint pid") <= 0 or _int(item.get("starttime"), "endpoint starttime") <= 0:
            _fail("endpoint lacks owned process identity")
        # Healthy services normally receive TERM; killed services are already
        # exited. The common runner verifies PID/starttime/exe/cwd/uid before
        # signaling. Do not invent a SIGKILL receipt for healthy cleanup.
        if item.get("status") not in ("stopped", "already_exited") or item.get("error"):
            _fail("owned endpoint cleanup is not confirmed")
    return {"endpoint_count": expected, "endpoints": endpoints}


def _failed_endpoints(plan: dict[str, Any], scenario: str) -> list[int]:
    raw = _first(
        [plan], "failed_endpoints", "fault_endpoints", "inject_endpoints",
        "failure_endpoints",
    )
    if raw is None:
        raw = [0] if scenario == "1-node" else [0, 1]
    result = _endpoint_ids(raw, "failed endpoint")
    if len(result) != (1 if scenario == "1-node" else 2) or len(set(result)) != len(result):
        _fail("failed endpoint set disagrees with scenario")
    return result


def _validate_failure(manifest, analysis, result, plan, endpoints):
    proof = manifest.get("failure_injection")
    if not isinstance(proof, dict) or proof != analysis.get("failure_injection"):
        _fail("manifest/analysis fault proof is missing or inconsistent")
    for key in ("injected", "owned_identity_verified", "process_stopped"):
        if proof.get(key) is not True:
            _fail("fault proof lacks " + key)
    if proof.get("signal") != "SIGKILL" or sorted(proof.get("endpoints", [])) != sorted(endpoints):
        _fail("fault signal/endpoint set disagrees with plan")
    if proof.get("trigger_event") != plan.get("trigger_event"):
        _fail("fault trigger disagrees with plan")
    begin = _int(proof.get("command_begin_monotonic_ns"), "fault command begin")
    end = _int(proof.get("command_end_monotonic_ns"), "fault command end")
    observed = _int(proof.get("trigger_observed_monotonic_ns"), "fault trigger observed")
    delay = _number(proof.get("scheduled_delay_s"), "fault delay")
    if delay != 20 or begin < observed + 20 * NANOSECONDS or end < begin:
        _fail("fault command is not at least 20 seconds after its observed trigger")
    receipts = proof.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != len(endpoints):
        _fail("fault proof lacks one receipt per failed endpoint")
    states = {item["index"]: item for item in manifest["endpoints"]}
    seen = set()
    for receipt in receipts:
        index = _int(receipt.get("endpoint"), "fault endpoint")
        if index not in endpoints or index in seen:
            _fail("duplicate/undeclared fault receipt")
        seen.add(index)
        state = states[index]
        if (receipt.get("owned_identity_verified") is not True
                or receipt.get("process_stopped") is not True
                or receipt.get("signal") != "SIGKILL"):
            _fail("fault receipt ownership/termination is unverified")
        for key in ("pid", "starttime", "memory_host", "memory_addr", "server_port"):
            if receipt.get(key) != state.get(key):
                _fail("fault receipt differs from owned endpoint: " + key)
        rbegin = _int(receipt.get("command_begin_monotonic_ns"), "receipt begin")
        rend = _int(receipt.get("command_end_monotonic_ns"), "receipt end")
        if not begin <= rbegin <= rend <= end:
            _fail("per-endpoint fault command interval is not closed")
    return {"verified": True, "failed_endpoints": endpoints, "proof": proof}


def _validate_config(run_dir: Path, manifest: dict[str, Any], plan: dict[str, Any],
                     system: str, scenario: str, failed: list[int],
                     endpoint_count: int) -> dict[str, Any]:
    config = _config_values(run_dir, plan, manifest)
    dual = scenario == "2-node"
    expected_failed = [0, 1] if dual else [0]
    scenario_standby = ([7] if dual else [6]) if system == "starfish" else ([6, 7] if dual else [6])
    if failed != expected_failed or plan.get("standby_endpoints") != scenario_standby:
        _fail("Figure13 planned fault/standby topology differs from its scenario")
    if plan.get("active_endpoints") != endpoint_count - len(scenario_standby):
        _fail("Figure13 active endpoint count is inconsistent")
    if "server_count" not in config:
        _fail("effective config lacks required server_count")
    if _int(config["server_count"], "server_count") != endpoint_count:
        _fail("effective config server_count disagrees with endpoint inventory")
    for key, expected in (("ft_background_rebuild", "1"),
                          ("ft_background_rebuild_failures", str(len(failed))),
                          ("ft_rebuild_bandwidth_mbps", "2500"),
                          ("qp_timeout", "4"), ("qp_retry_cnt", "0")):
        if key not in config:
            _fail(f"effective config lacks required Figure13 override {key}")
        if config[key].lower() != expected.lower():
            _fail(f"effective config {key} is not Figure13 override {expected}")
    standby = []
    for key in ("ft_standby_endpoint", "ft_standby_endpoint2"):
        if key in config:
            value = _int(config[key], key)
            if value >= 0:
                standby.append(value)
    if len(standby) != len(set(standby)):
        _fail("effective config standby endpoints are not distinct")
    if any(endpoint in failed for endpoint in standby):
        _fail("effective config standby endpoint is also failed")
    expected_standby = _first([plan], "standby_endpoints")
    expected = sorted(_int(x, "standby endpoint") for x in _as_list(expected_standby, "standby_endpoints"))
    if system in ("hydra", "carbink") and "ft_standby_endpoint2" not in config:
        _fail("effective config lacks required dual-standby setting")
    if "ft_standby_endpoint" not in config:
        _fail("effective config lacks required standby setting")
    if sorted(standby) != expected:
        _fail("effective config standby set disagrees with Figure13 plan")
    if system == "starfish":
        if "ft_rmw_read_failure_fallback" not in config or _int(config["ft_rmw_read_failure_fallback"], "ft_rmw_read_failure_fallback") != 1:
            _fail("Starfish effective config lacks Figure13 fallback=1")
    if config.get("ft_method") != {"starfish": "ec_batch", "hydra": "hydra", "carbink": "carbink"}[system]:
        _fail("Figure13 FT method differs from selected runtime")
    if config.get("enable_selective_backup") != ("1" if system == "starfish" else "0"):
        _fail("Figure13 backup policy differs from selected runtime")
    return {"values": config, "standby_endpoints": standby}


def _validate_receipt(rr, evidence: dict[str, Any], expected: int) -> dict[str, Any]:
    receipt = rr._single(evidence["receipts"], "kvs_receipt")
    tokens = receipt["tokens"]
    for key in rr.COUNT_FIELDS:
        if key not in tokens:
            _fail(f"kvs_receipt lacks {key}")
    if tokens.get("accepted_equals_completed") != "1":
        _fail("kvs_receipt accepted_equals_completed is not 1")
    counts = {key: rr._count(tokens[key], key) for key in rr.COUNT_FIELDS}
    for phase in ("generated", "accepted", "completed"):
        total = sum(counts[f"{phase}_{op}"] for op in ("get", "put", "remove"))
        if total != expected:
            _fail(f"kvs_receipt {phase} total is {total}, expected {expected}")
    for op in ("get", "put", "remove"):
        if counts[f"accepted_{op}"] != counts[f"completed_{op}"]:
            _fail(f"kvs_receipt accepted/completed mismatch for {op}")
    if not tokens.get("request_fingerprint"):
        _fail("kvs_receipt lacks request_fingerprint")
    post = rr._single(evidence["post_verify"], "kvs_post_verify")
    checked = rr._count(post["tokens"].get("checked"), "post_verify.checked")
    failures = rr._count(post["tokens"].get("failures"), "post_verify.failures")
    if checked != 4096 or failures != 0:
        _fail(f"post verification is checked={checked}, failures={failures}")
    return {"receipt": tokens, "post_verify": {"checked": checked, "failures": failures}}


def _validate_rebuild_bytes(event_details: dict[str, Any],
                            expected: dict[str, Any] | None = None) -> dict[str, Any]:
    totals = {}
    for endpoint, event in event_details.items():
        tokens = event["done"]["tokens"]
        read = _first([tokens], "read_bytes", "global_read_bytes")
        write = _first([tokens], "write_bytes", "global_write_bytes")
        if read is None or write is None:
            _fail(f"rebuild endpoint {endpoint} lacks read/write byte counts")
        read_i, write_i = _nonnegative(read, "rebuild read_bytes"), _nonnegative(write, "rebuild write_bytes")
        if write_i <= 0 or (len(event_details) == 1 and read_i <= 0):
            _fail(f"rebuild endpoint {endpoint} has nonpositive byte count")
        totals[str(endpoint)] = {"read_bytes": read_i, "write_bytes": write_i}
    if len(event_details) > 1 and sum(item["read_bytes"] for item in totals.values()) <= 0:
        _fail("dual rebuild has no global read traffic")
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key in ("read_bytes", "write_bytes", "global_read_bytes", "global_write_bytes"):
                observed = sum(item[key.replace("global_", "")] for item in totals.values())
                if _nonnegative(value, f"expected {key}") != observed:
                    _fail(f"rebuild byte count disagrees with analysis for {key}")
    return totals


def read_site_run(result_path: str | Path, *, steady_before_s: float = 10.0):
    """Validate one normal AE site run and return the legacy collector tuple."""
    rr = _rr()
    selected = Path(result_path).expanduser().resolve()
    if not selected.is_file():
        _fail(f"site result file not found: {selected}")
    run_dir = selected.parent
    result = _read_json(selected)
    if result.get("schema") != "figure13-site-result-v1":
        _fail("unsupported site result schema")
    manifest = _read_json(_ref(run_dir, result.get("manifest"), "manifest.json"))
    analysis = _read_json(_ref(run_dir, result.get("analysis"), "analysis.json"))
    figure = _figure_plan(manifest)
    plan = _merge_plan(manifest, analysis, result)
    system = str(result.get("system", plan.get("system", ""))).lower()
    scenario = str(result.get("scenario", plan.get("scenario", "")))
    if system not in SYSTEMS or scenario not in SCENARIOS:
        _fail("site result has unknown system/scenario")
    run_id = result.get("run_id", plan.get("run_id"))
    repeat = _int(result.get("repeat", plan.get("repeat")), "repeat")
    if not isinstance(run_id, str) or not run_id:
        _fail("site result lacks run_id")
    if str(figure["system"]).lower() != system:
        _fail("site result system disagrees with manifest plan.figure13")
    if figure["scenario"] != scenario or _int(figure["repeat"], "figure13 repeat") != repeat:
        _fail("site result scenario/repeat disagrees with manifest plan.figure13")
    if figure["app"] != "kv-b" or _int(figure["ratio"], "figure13 ratio") != 25:
        _fail("manifest plan.figure13 is not the complete KV-B ratio-25 case")
    if manifest.get("run_id") not in (None, run_id) or plan.get("run_id") not in (None, run_id):
        _fail("site result run_id disagrees with manifest plan")
    if manifest.get("run_id") not in (None, run_id) or analysis.get("run_id") not in (None, run_id):
        _fail("site result run_id disagrees with manifest/analysis")
    for source, name in ((manifest, "manifest"), (analysis, "analysis")):
        named_system = source.get("system")
        if ((named_system is not None and str(named_system).lower() != system)
                or source.get("scenario") not in (None, scenario)):
            _fail(f"{name} system/scenario disagrees with site result")
        if source.get("repeat") not in (None, repeat):
            _fail(f"{name} repeat disagrees with site result")
    analysis_system = analysis.get("system")
    if isinstance(analysis_system, str) and analysis_system.lower() != system:
        _fail("analysis system disagrees with manifest plan.figure13")
    manifest_system = manifest.get("system")
    if isinstance(manifest_system, str) and manifest_system.lower() != system:
        _fail("manifest system disagrees with manifest plan.figure13")
    if result.get("passed") is not True or result.get("status") not in ("passed", "completed"):
        _fail("site result passed/status is not successful; raw validation still required")
    revalidation = result.get("revalidation", {})
    reparsed = revalidation.get("mode") == "revalidated_current_parser"
    if reparsed:
        import re
        old_error = analysis.get("error", "")
        if not re.fullmatch(
            r"(?:duplicate event field [a-z_]+ at .+/client\.log:[0-9]+|endpoint at .+/client\.log:[0-9]+ must be an integer, got '')",
            old_error,
        ) or revalidation.get("original_error") != old_error:
            _fail("revalidation is restricted to an explicitly recorded native-log parser error")
        for source in (manifest, analysis):
            if source.get("status") != "failed" or source.get("error") != old_error:
                _fail("original parser failure evidence disagrees")
        # The same common validator is rerun, in addition to all independent
        # checks below. Original manifest/analysis files remain unchanged.
        import sys
        common = Path(__file__).resolve().parents[1] / "common"
        sys.path.insert(0, str(common))
        import kv_recovery
        kv_recovery.validate_run(run_dir, figure, analysis["failure_injection"])
    if not reparsed and manifest.get("status") != "passed":
        _fail("manifest status is not successful")
    if not reparsed and analysis.get("status") != "passed":
        _fail("analysis status is not successful")
    for source, name in ((manifest, "manifest"), (analysis, "analysis")):
        if source.get("exit_status") != 0:
            _fail(f"{name} exit_status is nonzero")
        if not reparsed and name == "analysis" and source.get("correctness") != "pass":
            _fail("analysis correctness is not pass")
        if not reparsed and source.get("error") not in (None, "", [], {}):
            _fail(f"{name} records an error")

    client_path = run_dir / "client.log"
    client_text, evidence = rr._read_client_log(client_path)
    config = evidence["configuration"]
    receipt_record = rr._single(evidence["receipts"], "kvs_receipt")
    receipt_total = sum(
        rr._count(receipt_record["tokens"].get(f"generated_{op}"), f"generated_{op}")
        for op in ("get", "put", "remove")
    )
    worker = _worker_profile(plan, manifest)
    observer_cpu = _int(figure["observer_cpu"], "observer_cpu")
    if observer_cpu in worker["app_cpus"] or observer_cpu in worker["background_cpus"]:
        _fail("observer_cpu overlaps worker CPU placement")
    client_env = _first([plan, manifest], "client_env", default={})
    if (not isinstance(client_env, dict)
            or client_env.get("FARLIB_KVS_COMPLETION_SERIES") != "1"
            or client_env.get("FARLIB_KVS_COMPLETION_SERIES_CPU") != str(observer_cpu)):
        _fail("completion-series observer CPU is not recorded in client_env")
    saved_config = _saved_effective_config(run_dir, plan, manifest)
    effective_values = _config_values(run_dir, {**plan, "effective_config": saved_config}, manifest)
    payload_plan = dict(plan)
    payload_plan.setdefault("local_bytes", effective_values.get("client_buffer_size"))
    payload_plan.setdefault("logical_bytes", _first(
        [plan, manifest], "workload_footprint_bytes", default=16 * 1024**3))
    merged_config = dict(effective_values)
    merged_config.update(config)
    payload = _workload_payload(payload_plan, merged_config, receipt_total, worker, manifest)
    rr._validate_client_configuration(payload, config)
    placement_plan = {
        "app_cpus": worker["app_cpus"], "background_cpus": worker["background_cpus"],
        "app_workers": worker["app_workers"],
    }
    placement = rr._validate_placement(placement_plan, evidence["worker_roles"])
    _validate_workload(payload)
    endpoints = _failed_endpoints(plan, scenario)
    cleanup = _validate_cleanup(manifest, plan, scenario)
    failure = _validate_failure(manifest, analysis, result, plan, endpoints)
    cfg = _validate_config(run_dir, manifest, plan, system, scenario, endpoints,
                           cleanup["endpoint_count"])
    expected_requests = _int(payload["requests"], "requests")
    windows_json = run_dir / "kvs.completed-100ms.json"
    request_start_ns, request_end_ns = rr._native_request_times(
        evidence, _read_json(windows_json)
    )
    windows, window_details = rr._read_windows(
        run_dir, request_start_ns, request_end_ns, expected_requests
    )
    receipt = _validate_receipt(rr, evidence, expected_requests)
    grouped = rr._group_rebuild_events(evidence, endpoints)
    aggregate, event_details = rr._validate_rebuild_events(
        grouped, request_start_ns, request_end_ns
    )
    for receipt in failure["proof"]["receipts"]:
        begin = _int(receipt["command_begin_monotonic_ns"], "fault begin")
        dead = rr._token_int(grouped[receipt["endpoint"]]["endpoint_dead"], "monotonic_ns")
        if not request_start_ns <= begin <= dead <= request_end_ns:
            _fail("fault command/detection falls outside the request interval")
    rebuild_bytes = _validate_rebuild_bytes(event_details)
    steady_before_s = _number(steady_before_s, "steady_before_s")
    failure_s = (aggregate["failure_ns"] - request_start_ns) / NANOSECONDS
    rebuild_start_s = (aggregate["start_ns"] - request_start_ns) / NANOSECONDS
    recovered_s = (aggregate["done_ns"] - request_start_ns) / NANOSECONDS
    steady_start_s = max(0.0, failure_s - steady_before_s)
    if not steady_start_s < failure_s:
        _fail("no steady pre-failure interval remains")
    ratio = _first([plan, manifest, analysis], "ratio", default=25)
    ratio = _int(ratio, "ratio")
    environment = str(_first([plan, manifest], "site", "environment", default="site"))
    workload_id_payload = payload
    canonical = json.dumps(workload_id_payload, sort_keys=True, separators=(",", ":"))
    workload_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    details = {
        "adapter": "figure13.site_runs",
        "source_type": "site_result_json",
        "result_path": str(selected), "manifest_path": str(run_dir / "manifest.json"),
        "analysis_path": str(run_dir / "analysis.json"), "client_log_path": str(client_path),
        "client_log_sha256": evidence["sha256"], "workload_id_payload": workload_id_payload,
        "workload_id_canonical_json": canonical, "plan": plan,
        "selected_result": result, "manifest": manifest, "analysis": analysis,
        "config": cfg, "cleanup": cleanup, "failure": failure,
        "receipt_and_correctness": receipt, "windows": window_details,
        "configuration_validation": {
            "client": True, "placement": placement,
            "observer_cpu": observer_cpu,
        },
        "rebuild_events": event_details, "rebuild_aggregate": aggregate,
        "rebuild_bytes": rebuild_bytes, "native_rebuild_duration_s": aggregate["duration_ns"] / NANOSECONDS,
        "source_files": {
            "result": selected.name, "manifest": "manifest.json",
            "analysis": "analysis.json", "client_log": "client.log",
            "windows_csv": "kvs.completed-100ms.csv",
            "windows_json": "kvs.completed-100ms.json",
        },
    }
    run = {
        "run_id": run_id, "system": system, "scenario": scenario, "workload": "kv-b",
        "environment": environment, "workload_id": workload_id, "ratio": ratio,
        "app_workers": worker["app_workers"], "repeat": repeat, "phase": "work",
        "time_origin": "work_start", "steady_start_s": steady_start_s,
        "failure_elapsed_s": failure_s, "rebuild_start_elapsed_s": rebuild_start_s,
        "recovered_elapsed_s": recovered_s,
        "run_end_s": (request_end_ns - request_start_ns) / NANOSECONDS,
        "exit_status": 0, "correctness": "pass", "source_type": "measured",
        "failure_confirmed": 1, "recovery_verified": 1, "panel": "both",
        "_source": str(selected), "recovery_definition": "background_start_to_done",
    }
    return run, windows, details
