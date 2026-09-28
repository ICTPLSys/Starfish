"""NQ offered-load configuration and measured-result checks."""
from pathlib import Path
import math
import re
import subprocess

from kv_latency import fields, records, one, number, PHASES, HISTOGRAMS

DEFAULT_MAX_QUEUE_DELAY_US = 1_500_000  # NQ: 1500 ms before query admission.


def specification(args):
    rate = getattr(args, "offered_load_ops", None)
    durations = [getattr(args, key, None) for key in (
        "latency_warmup_ms", "latency_measure_ms", "latency_drain_ms")]
    queue = getattr(args, "max_queue_delay_us", None)
    if rate is None:
        if any(value is not None for value in durations) or queue not in (None, 0):
            raise ValueError("NQ latency windows require --offered-load-ops")
        return None
    if args.app != "nq":
        raise ValueError("NQ latency adapter requires the NQ application")
    durations = [default if value is None else value
                 for value, default in zip(durations, (10000, 10000, 60000))]
    queue = DEFAULT_MAX_QUEUE_DELAY_US if queue is None else queue
    if any(type(value) is not int or value <= 0 for value in [rate, *durations]):
        raise ValueError("NQ rate and durations must be positive integers")
    if type(queue) is not int or not 0 <= queue <= 60_000_000:
        raise ValueError("NQ queue deadline must be 0..60000000 microseconds")
    if rate > 2**63 - 1 or max(durations[:2]) + durations[2] > (2**63 - 1) // 1000000:
        raise ValueError("NQ latency configuration exceeds clock/integer range")
    return dict(schema=1, offered_load_ops=rate, warmup_ms=durations[0],
                measure_ms=durations[1], drain_ms=durations[2], max_queue_delay_us=queue,
                arrival="poisson_per_fibre", include_queue_wait=True, hist_sample_period=1,
                histogram_dir=str((args.out / "histograms").resolve()))


def environment(base, spec):
    result = dict(base)
    if spec is None:
        return result
    result.update(
        NHOP_OFFERED_LOAD_OPS=str(spec["offered_load_ops"]),
        NHOP_LATENCY_WARMUP_MS=str(spec["warmup_ms"]),
        NHOP_LATENCY_MEASURE_MS=str(spec["measure_ms"]),
        NHOP_DRAIN_TIMEOUT_MS=str(spec["drain_ms"]),
        NHOP_MAX_QUEUE_DELAY_US=str(spec["max_queue_delay_us"]),
        NHOP_LATENCY_OUTPUT_DIR=spec["histogram_dir"],
        NHOP_WORK_PROGRESS="0")
    return result


def probe(binary, env):
    result = subprocess.run([str(binary), "--describe-latency-mode"], env=env,
                            capture_output=True, text=True, timeout=5, check=False)
    if result.returncode != 0:
        raise ValueError("NQ binary lacks latency mode; rebuild the selected system")
    _, row = one(result.stdout + "\n" + result.stderr, "nq_latency_capability")
    required = dict(schema="1", arrival="poisson_per_fibre", include_queue_wait="1",
                    hist_sample_period="1", queue_deadline="drop_before_execution")
    if any(row.get(key) != value for key, value in required.items()):
        raise ValueError("NQ latency capability contract differs")
    return row


def parse_result(log, spec, *, expected_workers, histogram_dir=None):
    position, config = one(log, "nq_latency_config")
    queue_ns = spec["max_queue_delay_us"] * 1000
    fixed = dict(arrival="poisson_per_fibre", queue_model="fifo_independent_lanes",
                 include_queue_wait="1", hist_sample_period="1",
                 query="benchmark_lite", vertex_distribution="uniform",
                 deadline_action="drop_before_execution" if queue_ns else "disabled")
    expected = dict(offered_load_ops=spec["offered_load_ops"], fibres=48,
                    os_workers=expected_workers, warmup_ns=spec["warmup_ms"] * 1000000,
                    measurement_ns=spec["measure_ms"] * 1000000,
                    drain_timeout_ns=spec["drain_ms"] * 1000000,
                    max_queue_delay_ns=queue_ns, vertices=65608366,
                    adjacency_lists=65608367, arrival_seed=20260917)
    if (any(config.get(key) != value for key, value in fixed.items())
            or any(number(config, key) != value for key, value in expected.items())):
        raise ValueError("NQ effective graph, query, placement or timing contract differs")
    if (len(records(log, "nq_latency_phase")) != 4
            or len(records(log, "nq_latency_receipt")) != 2
            or len(records(log, "nq_latency_result")) != 2):
        raise ValueError("NQ phase records are missing or duplicated")
    phases, previous_ns = {}, None
    for name in PHASES:
        start_pos, start = one(log, "nq_latency_phase", name=name, event="start")
        end_pos, end = one(log, "nq_latency_phase", name=name, event="end")
        receipt_pos, receipt = one(log, "nq_latency_receipt", name=name)
        result_pos, result = one(log, "nq_latency_result", name=name)
        start_ns, end_ns = number(start, "monotonic_ns", positive=True), number(end, "monotonic_ns")
        window = spec["warmup_ms" if name == "warmup" else "measure_ms"] * 1000000
        if (not position < start_pos < end_pos < receipt_pos < result_pos
                or end_ns < start_ns + window
                or (previous_ns is not None and start_ns < previous_ns)):
            raise ValueError("NQ phase ordering or clock is invalid")
        position, previous_ns = result_pos, end_ns
        scheduled = number(receipt, "scheduled", positive=True)
        completed = number(receipt, "completed", positive=True)
        dropped = number(receipt, "deadline_dropped")
        inside = number(receipt, "completed_in_window")
        late = number(receipt, "completed_after_deadline")
        if (number(receipt, "started") != completed or scheduled != completed + dropped
                or number(receipt, "skipped") or inside > completed or late > completed
                or (not queue_ns and (dropped or late))):
            raise ValueError("NQ arrived/completed/dropped counts do not close")
        number(receipt, "request_fingerprint")
        number(receipt, "result_checksum")
        if result.get("status") != "passed":
            raise ValueError("NQ phase did not finish successfully")
        if (number(result, "arrival_window_ns") != window
                or number(result, "drain_elapsed_ns") != end_ns - start_ns - window):
            raise ValueError("NQ phase duration differs from the plan")
        if any(number(result, key) for key in (
                "drain_timeout", "hist_dropped", "merge_dropped", "hist_write_errors")):
            raise ValueError("NQ timed out or lost histogram observations")
        if any(number(result, key) != completed for key in (
                "hist_count", "hist_service_count", "hist_dispatch_count")):
            raise ValueError("NQ histogram populations differ from completed queries")
        p99 = number(result, "p99_ns", positive=True)
        if p99 < max(number(result, "p99_service_ns"), number(result, "p99_dispatch_ns")):
            raise ValueError("NQ total P99 is below a component")
        mean = spec["offered_load_ops"] * window / 1e9
        if abs(scheduled - mean) > max(10, 8 * math.sqrt(mean) + 0.001 * mean):
            raise ValueError("NQ arrivals differ from the requested aggregate load")
        if histogram_dir is not None:
            for kind in HISTOGRAMS:
                path = Path(histogram_dir) / f"{name}.{kind}.hgrm"
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("missing NQ raw histogram: " + str(path))
        phases[name] = {**receipt, **result, "start_ns": start_ns, "end_ns": end_ns,
                        "scheduled": scheduled, "completed": completed,
                        "deadline_dropped": dropped, "drop_fraction": dropped / scheduled,
                        "completed_after_deadline": late, "p99_ns": p99,
                        "realized_offered_ops_s": scheduled * 1e9 / window,
                        "completed_in_window_ops_s": inside * 1e9 / window}
    post_pos, post = one(log, "nq_latency_post_verify")
    if (post_pos <= position or number(post, "checked") != 4096
            or number(post, "failures") != 0):
        raise ValueError("NQ independent host/far query postcheck did not pass")
    allocation_ends = re.findall(r"^exact used bytes: (\d+)\s*$", log, re.M)
    if not allocation_ends or int(allocation_ends[-1]) != 0:
        raise ValueError("NQ final remote allocations were not released")
    measured = phases["measurement"]
    return dict(
        elapsed_s=(measured["end_ns"] - measured["start_ns"]) / 1e9,
        request_count=measured["scheduled"],
        measurement_phase="nq_open_loop_measurement_and_drain",
        measurement_method="P99 of completion minus planned Poisson arrival; includes queue delay",
        correctness_scope="full arrival accounting and 4096 independent host/far query checks",
        correctness_evidence=f"scheduled={measured['scheduled']}; completed={measured['completed']}; "
                             f"deadline_dropped={measured['deadline_dropped']}; postcheck=4096/0",
        nq_latency=dict(schema=1, config=config, phases=phases,
                        offered_load_ops=spec["offered_load_ops"], p99_latency_ns=measured["p99_ns"],
                        histogram_dir=str(histogram_dir) if histogram_dir is not None else None))
