"""Figure 10 KV arrival-window configuration and strict measurement checks."""
from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess

SCHEMA = 1
DEFAULT_MAX_QUEUE_DELAY_US = 500
PHASES = ("warmup", "measurement")
HISTOGRAMS = ("total", "service", "dispatch")


def specification(args):
    rate = getattr(args, "offered_load_ops", None)
    queue_us = getattr(args, "max_queue_delay_us", None)
    values = [getattr(args, key, None)
              for key in ("latency_warmup_ms", "latency_measure_ms", "latency_drain_ms")]
    if rate is None:
        if any(value is not None for value in values) or queue_us not in (None, 0):
            raise ValueError("latency windows require --offered-load-ops")
        return None
    if args.app != "kv-b":
        raise ValueError("offered-load measurement requires the KV-B adapter")
    warmup, measure, drain = [default if value is None else value
                             for value, default in zip(values, (10000, 10000, 60000))]
    for name, value in (("offered load", rate), ("warmup", warmup),
                        ("measurement", measure), ("drain timeout", drain)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if max(warmup, measure) + drain > (2**63 - 1) // 1000000:
        raise ValueError("latency window exceeds the nanosecond clock range")
    if rate > 2**63 - 1:
        raise ValueError("offered load exceeds the integer range")
    queue_us = DEFAULT_MAX_QUEUE_DELAY_US if queue_us is None else queue_us
    if type(queue_us) is not int or not 0 <= queue_us <= 60_000_000:
        raise ValueError("queue deadline must be 0..60000000 microseconds")
    return {"schema": SCHEMA, "offered_load_ops": rate, "warmup_ms": warmup,
            "measure_ms": measure, "drain_ms": drain,
            "max_queue_delay_us": queue_us,
            "arrival": "poisson_per_fibre", "include_queue_wait": True,
            "hist_sample_period": 1,
            "histogram_dir": str((args.out / "histograms").resolve())}


def environment(base, spec):
    if spec is None:
        return dict(base)
    env = dict(base)
    for key in ("FARLIB_KVS_MAX_SERVE_COUNT", "FARLIB_KVS_MAX_RUNTIME_MS",
                "FARLIB_KVS_HOTSET_SHIFT_MS"):
        env.pop(key, None)
    env.update({
        "FARLIB_KVS_OFFERED_LOAD_OPS": str(spec["offered_load_ops"]),
        "FARLIB_KVS_LATENCY_WARMUP_MS": str(spec["warmup_ms"]),
        "FARLIB_KVS_LATENCY_MEASURE_MS": str(spec["measure_ms"]),
        "FARLIB_KVS_DRAIN_TIMEOUT_MS": str(spec["drain_ms"]),
        "FARLIB_KVS_LATENCY_OUTPUT_DIR": spec["histogram_dir"],
        "FARLIB_KVS_MAX_QUEUE_DELAY_US": str(spec.get(
            "max_queue_delay_us", DEFAULT_MAX_QUEUE_DELAY_US)),
        "FARLIB_KVS_FIXED_REQUEST_COUNT": "0",
        "FARLIB_KVS_DRAIN_ALL_REQUESTS": "1",
        "FARLIB_KVS_HIST_SAMPLE_PERIOD": "1",
        "FARLIB_KVS_EXECUTION_MODE": "direct",
    })
    return env


def fields(line):
    result = {}
    for token in line.split()[1:]:
        key, sep, value = token.partition("=")
        if not sep or not key or key in result:
            raise ValueError("invalid or duplicate latency record field")
        result[key] = value
    return result


def records(log, prefix):
    return [(match.start(), fields(match.group(0)))
            for match in re.finditer(r"^" + re.escape(prefix) + r" [^\n\r]+$", log, re.M)]


def one(log, prefix, **selectors):
    rows = [(position, row) for position, row in records(log, prefix)
            if all(row.get(key) == value for key, value in selectors.items())]
    if len(rows) != 1:
        raise ValueError(f"expected one {prefix} record for {selectors}, got {len(rows)}")
    return rows[0]


def number(row, key, *, positive=False):
    text = row.get(key, "")
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("missing or invalid latency integer: " + key)
    value = int(text)
    if positive and value == 0:
        raise ValueError("latency integer must be positive: " + key)
    return value


def probe(binary, env):
    result = subprocess.run([str(binary), "--describe-latency-mode"], env=env,
                            capture_output=True, text=True, timeout=5, check=False)
    if result.returncode != 0:
        raise ValueError("KV binary lacks Figure 10 latency mode; rebuild the selected system")
    _, row = one(result.stdout + "\n" + result.stderr, "kvs_latency_capability")
    required = {"schema": str(SCHEMA), "arrival": "poisson_per_fibre",
                "include_queue_wait": "1", "hist_sample_period": "1"}
    if int(env.get("FARLIB_KVS_MAX_QUEUE_DELAY_US", "0")):
        required["queue_deadline"] = "drop_before_execution"
    if any(row.get(key) != value for key, value in required.items()):
        raise ValueError("KV latency capability contract differs")
    return row


def parse_result(log, spec, *, expected_workers, histogram_dir=None):
    config_pos, config = one(log, "kvs_latency_config")
    queue_ns = spec.get("max_queue_delay_us", 0) * 1000
    if queue_ns or "max_queue_delay_ns" in config:
        if (number(config, "max_queue_delay_ns") != queue_ns
                or config.get("deadline_action") != (
                    "drop_before_execution" if queue_ns else "disabled")):
            raise ValueError("KV queue deadline differs from the plan")
    fixed = {"arrival": "poisson_per_fibre", "queue_model": "fifo_independent_lanes",
             "include_queue_wait": "1",
             "hist_sample_period": "1", "locked_get": "1", "real_put": "1",
             "get_content_validation": "0"}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("KV latency arrival, timing, sampling or operation contract differs")
    expected = {"offered_load_ops": spec["offered_load_ops"], "fibres": 48,
                "os_workers": expected_workers, "warmup_ns": spec["warmup_ms"] * 1000000,
                "measurement_ns": spec["measure_ms"] * 1000000,
                "drain_timeout_ns": spec["drain_ms"] * 1000000,
                "object_bytes": 512, "initial_count": 33554432, "random_seed": 20260917,
                "arrival_seed": 20260917}
    if any(number(config, key) != value for key, value in expected.items()):
        raise ValueError("KV latency effective workload or window differs from the plan")
    try:
        distribution_matches = (
            math.isclose(float(config["put_ratio"]), 0.05, rel_tol=0, abs_tol=1e-12)
            and math.isclose(float(config["zipf"]), 0.99, rel_tol=0, abs_tol=1e-12))
    except (KeyError, ValueError):
        distribution_matches = False
    if not distribution_matches:
        raise ValueError("KV-B read/write ratio or Zipf distribution differs")
    if len(records(log, "kvs_latency_phase")) != 4:
        raise ValueError("latency phase markers are missing or duplicated")
    if (len(records(log, "kvs_latency_receipt")) != 2
            or len(records(log, "kvs_latency_result")) != 2):
        raise ValueError("latency phase receipts/results are missing or duplicated")
    validated = {}
    previous_end_pos = config_pos
    previous_end_ns = None
    for name in PHASES:
        start_pos, start = one(log, "kvs_latency_phase", name=name, event="start")
        end_pos, end = one(log, "kvs_latency_phase", name=name, event="end")
        receipt_pos, receipt = one(log, "kvs_latency_receipt", name=name)
        result_pos, result = one(log, "kvs_latency_result", name=name)
        start_ns = number(start, "monotonic_ns", positive=True)
        end_ns = number(end, "monotonic_ns", positive=True)
        if (not previous_end_pos < start_pos < end_pos < receipt_pos < result_pos
                or end_ns <= start_ns
                or (previous_end_ns is not None and start_ns < previous_end_ns)):
            raise ValueError("latency phase ordering or clock is invalid")
        previous_end_pos = max(end_pos, receipt_pos, result_pos)
        previous_end_ns = end_ns
        if result.get("status") != "passed":
            raise ValueError(name + " did not drain successfully")
        count = number(receipt, "scheduled", positive=True)
        done = number(receipt, "completed", positive=True)
        dropped = (number(receipt, "deadline_dropped")
                   if queue_ns or "deadline_dropped" in receipt else 0)
        if (number(receipt, "started") != done or done + dropped != count
                or number(receipt, "skipped") != 0
                or number(receipt, "completed_in_window") > done
                or (not queue_ns and dropped)):
            raise ValueError(name + " has missing, skipped or incomplete scheduled requests")
        if number(receipt, "generated_remove") or number(receipt, "completed_remove"):
            raise ValueError("KV-B must not issue REMOVE")
        generated = [number(receipt, "generated_" + op) for op in ("get", "put", "remove")]
        completed = [number(receipt, "completed_" + op) for op in ("get", "put", "remove")]
        expired = [number(receipt, "dropped_" + op)
                   if queue_ns or "deadline_dropped" in receipt else 0
                   for op in ("get", "put", "remove")]
        if (generated != [a + b for a, b in zip(completed, expired)]
                or sum(completed) != done or sum(expired) != dropped or expired[2]):
            raise ValueError(name + " operation counts do not close")
        for op in ("get", "put", "remove"):
            if "skipped_" + op in receipt and number(receipt, "skipped_" + op):
                raise ValueError(name + " skipped operations after drain timeout")
        late = (number(receipt, "completed_after_deadline")
                if queue_ns or "completed_after_deadline" in receipt else 0)
        if late > done or (not queue_ns and late):
            raise ValueError(name + " completed-after-deadline count differs")
        number(receipt, "request_fingerprint")
        window = spec["warmup_ms" if name == "warmup" else "measure_ms"] * 1000000
        if end_ns < start_ns + window:
            raise ValueError(name + " ended before its arrival window closed")
        if number(result, "arrival_window_ns") != window:
            raise ValueError(name + " arrival window differs")
        for key in ("drain_timeout", "hist_dropped", "merge_dropped", "hist_write_errors"):
            if number(result, key) != 0:
                raise ValueError(name + " contains timeout or dropped latency observations")
        for key in ("hist_count", "hist_service_count", "hist_dispatch_count"):
            if number(result, key) != done:
                raise ValueError(name + " histogram population differs from completed requests")
        if number(result, "drain_elapsed_ns") != max(0, end_ns - start_ns - window):
            raise ValueError(name + " drain time is inconsistent with its phase clock")
        p99 = number(result, "p99_ns", positive=True)
        service = number(result, "p99_service_ns")
        dispatch = number(result, "p99_dispatch_ns")
        if p99 < max(service, dispatch):
            raise ValueError("end-to-end P99 cannot be below a component P99")
        expected_arrivals = spec["offered_load_ops"] * window / 1e9
        # A generous Poisson sanity bound also catches a per-worker/aggregate mixup.
        if abs(count - expected_arrivals) > max(
                10, 8 * math.sqrt(expected_arrivals) + 0.001 * expected_arrivals):
            raise ValueError(name + " scheduled arrival population differs from aggregate offered load")
        if histogram_dir is not None:
            for kind in HISTOGRAMS:
                path = Path(histogram_dir) / f"{name}.{kind}.hgrm"
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("missing raw latency histogram: " + str(path))
        validated[name] = {**receipt, **result, "start_ns": start_ns, "end_ns": end_ns,
                           "scheduled": count, "completed": done,
                           "deadline_dropped": dropped, "drop_fraction": dropped / count,
                           "completed_after_deadline": late, "p99_ns": p99,
                           "realized_offered_ops_s": count * 1e9 / window,
                           "completed_in_window_ops_s":
                               number(receipt, "completed_in_window") * 1e9 / window}
    post_pos, post = one(log, "kvs_post_verify")
    if (number(post, "checked") != 4096 or number(post, "failures") != 0
            or post_pos <= previous_end_pos):
        raise ValueError("KV latency postcheck missing, failed or before the completed window")
    measured = validated["measurement"]
    return {
        "elapsed_s": (measured["end_ns"] - measured["start_ns"]) / 1e9,
        "request_count": measured["scheduled"],
        "measurement_phase": "kv_open_loop_measurement_and_drain",
        "measurement_method": "P99 of completion minus scheduled Poisson arrival; includes queue delay",
        "correctness_scope": "full arrival accounting, zero histogram loss and 4096 sampled value checks",
        "correctness_evidence": (
            f"scheduled={measured['scheduled']}; completed={measured['completed']}; "
            f"deadline_dropped={measured['deadline_dropped']}; hist_dropped=0; postcheck=4096/0"),
        "kv_latency": {"schema": SCHEMA, "config": config, "phases": validated,
                       "offered_load_ops": spec["offered_load_ops"],
                       "p99_latency_ns": measured["p99_ns"],
                       "histogram_dir": str(histogram_dir) if histogram_dir is not None else None},
    }
