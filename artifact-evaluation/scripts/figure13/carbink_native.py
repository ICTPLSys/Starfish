"""Adapter for the passed Carbink native background-rebuild runner schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NANOSECONDS = 1_000_000_000


def _common():
    try:
        from . import raw_runs as rr
    except ImportError:
        import raw_runs as rr
    return rr


def _fail(message):
    _common()._fail(message)


def _tokens(rr, record, key, field=None):
    return rr._token_int(record, key, field or key)


def _record_fields(rr, text, needle):
    records = [
        rr._event_record(Path("<client.log>"), index, line)
        for index, line in enumerate(text.splitlines(), 1)
        if needle in line
    ]
    return records


def _validate_runner(rr, run_dir, result, plan):
    runner = rr._read_json(run_dir / "runner-result.json")
    if runner.get("client_exit_code") != 0:
        _fail("Carbink runner client_exit_code is not zero")
    if runner.get("cleanup_returncode") != 0:
        _fail("Carbink runner cleanup_returncode is not zero")
    # The runner retains a pre-validation false placeholder. The separate
    # run-result is the post-run validator output; do not rewrite either file.
    if "runner_error" in runner:
        _fail("Carbink runner recorded an error")
    if runner.get("owner_token") != plan.get("owner_token"):
        _fail("Carbink runner ownership token differs from plan")
    if result.get("passed") is not True:
        _fail("Carbink run-result passed is not true")
    if result.get("background_rebuild") is not True:
        _fail("Carbink result does not declare background_rebuild=true")
    stop = rr._read_json(run_dir / "controller-stop.json")
    if stop.get("returncode") != 0:
        _fail("Carbink controller cleanup did not return zero")
    return runner, stop


def _validate_receipt(rr, result, evidence, expected_requests):
    receipt_record = rr._single(evidence["receipts"], "kvs_receipt")
    raw = receipt_record["tokens"]
    missing = [key for key in rr.COUNT_FIELDS if key not in raw]
    if missing:
        _fail("Carbink receipt lacks: " + ", ".join(missing))
    if raw.get("accepted_equals_completed") != "1":
        _fail("Carbink accepted_equals_completed is not 1")
    generated = sum(rr._count(raw["generated_" + op], "generated_" + op)
                    for op in ("get", "put", "remove"))
    accepted = sum(rr._count(raw["accepted_" + op], "accepted_" + op)
                   for op in ("get", "put", "remove"))
    completed = sum(rr._count(raw["completed_" + op], "completed_" + op)
                   for op in ("get", "put", "remove"))
    if generated != expected_requests or accepted != expected_requests:
        _fail("Carbink generated/accepted receipt total differs from plan")
    if completed != expected_requests:
        _fail("Carbink completed receipt total differs from plan")
    for op in ("get", "put", "remove"):
        if rr._count(raw["accepted_" + op], "accepted_" + op) != rr._count(
            raw["completed_" + op], "completed_" + op
        ):
            _fail("Carbink accepted/completed mismatch for " + op)
    selected = result.get("receipt")
    if not isinstance(selected, dict):
        _fail("Carbink result lacks receipt")
    for key in rr.COUNT_FIELDS + ("accepted_equals_completed",):
        if str(selected.get(key)) != str(raw.get(key)):
            _fail("Carbink result receipt disagrees with client.log: " + key)
    post = rr._single(evidence["post_verify"], "kvs_post_verify")
    checked = rr._count(post["tokens"].get("checked"), "post_verify.checked")
    failures = rr._count(post["tokens"].get("failures"), "post_verify.failures")
    if checked != 4096 or failures != 0:
        _fail(f"Carbink post verification is checked={checked}, failures={failures}")
    return {
        "receipt": raw,
        "post_verify": {"checked": checked, "failures": failures},
        "request_fingerprint": raw.get("request_fingerprint"),
    }


def _validate_fault(rr, run_dir, plan, packing_ns, failure_ns, request_end_ns):
    fault = rr._read_json(run_dir / "fault-command.json")
    endpoint = rr._int(plan.get("fault_endpoint"), "plan fault_endpoint")
    if fault.get("injected") is not True or rr._int(fault.get("endpoint"), "fault endpoint") != endpoint:
        _fail("Carbink fault receipt does not identify the planned endpoint")
    begin = rr._nonnegative_int(
        fault.get("command_begin_monotonic_ns"), "fault command begin"
    )
    end = rr._nonnegative_int(
        fault.get("command_end_monotonic_ns"), "fault command end"
    )
    delay = rr._finite_number(
        plan.get("fault_delay_after_packing_ready_s"),
        "fault_delay_after_packing_ready_s",
    )
    if begin < packing_ns + int(delay * NANOSECONDS):
        _fail("Carbink fault command preceded packing-ready delay")
    if not begin <= failure_ns < request_end_ns or end < begin:
        _fail("Carbink fault timing is outside request boundary")
    controller = fault.get("controller")
    if not isinstance(controller, dict) or controller.get("returncode") != 0:
        _fail("Carbink fault controller did not return zero")
    try:
        remote = json.loads(controller.get("stdout", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        _fail(f"invalid Carbink fault controller receipt: {exc}")
    if remote.get("action") != "fail" or rr._int(remote.get("endpoint"), "fault endpoint") != endpoint:
        _fail("Carbink fault controller receipt has wrong action/endpoint")
    if remote.get("process_stopped") is not True or remote.get("port_cleared") is not True:
        _fail("Carbink fault controller did not prove process/port cleanup")
    return {
        "endpoint": endpoint,
        "command_begin_monotonic_ns": begin,
        "command_end_monotonic_ns": end,
        "receipt": remote,
    }


def _validate_background(rr, plan, done_record, start_record, result, completed, write_bytes):
    start = start_record["tokens"]
    done = done_record["tokens"]
    policy = start.get("range_policy", done.get("range_policy", "whole_shard"))
    configured = plan.get("background_rebuild_range_policy")
    if configured is not None and configured != policy:
        _fail("Carbink plan/result range policy disagrees")
    if policy == "live_groups":
        if start.get("range_policy") != "live_groups" or done.get("range_policy") != "live_groups":
            _fail("Carbink live-groups run lacks explicit start/done range_policy")
        for key in ("selected_bytes", "skipped_bytes", "completed_ranges",
                    "empty_reactivated"):
            if key not in done:
                _fail("Carbink live-groups done lacks " + key)
        selected = rr._int(done["selected_bytes"], "selected_bytes")
        skipped = rr._int(done["skipped_bytes"], "skipped_bytes")
        ranges = rr._int(done["completed_ranges"], "completed_ranges")
        empty = rr._int(done["empty_reactivated"], "empty_reactivated")
        live_groups = rr._int(done["live_groups_at_copy"], "live_groups_at_copy")
        if selected != live_groups * 8192 or selected != write_bytes:
            _fail("Carbink selected live-group bytes do not close")
        if skipped != completed * 262144 - selected or skipped < 0:
            _fail("Carbink skipped bytes do not close")
        if not completed <= ranges <= live_groups:
            _fail("Carbink completed range count is outside live-group bounds")
        if rr._int(done["post_attempts"], "post_attempts") < 5 * ranges:
            _fail("Carbink post attempts are below five per range")
        if completed + empty != rr._int(done["scan_limit"], "scan_limit"):
            _fail("Carbink completed/empty scan accounting does not close")
    elif policy == "whole_shard":
        selected = skipped = ranges = None
    else:
        _fail("unknown Carbink range policy: " + policy)
    reported = result.get("background_rebuild_events", {})
    reported_done = reported.get("background_done") if isinstance(reported, dict) else None
    if isinstance(reported_done, dict):
        for key in ("completed_stripes", "write_bytes", "read_bytes"):
            if key in reported_done and str(reported_done[key]) != str(done.get(key)):
                _fail("Carbink result background counters disagree for " + key)
    return {
        "range_policy": policy,
        "selected_bytes": selected,
        "skipped_bytes": skipped,
        "completed_ranges": ranges,
        "done": done,
    }


def _dual_endpoints(rr, plan):
    declared = plan.get("failed_endpoints")
    if not isinstance(declared, list):
        declared = plan.get("fault_endpoints")
    if not isinstance(declared, list):
        _fail("Carbink dual plan lacks failed_endpoints/fault_endpoints")
    endpoints = sorted(rr._int(value, "dual failed endpoint") for value in declared)
    if endpoints != [0, 1]:
        _fail("Carbink dual plan must fail exactly endpoints 0 and 1")
    expected = (
        ("remote_endpoints", 8),
        ("server_count", 8),
        ("active_endpoints", 6),
        ("active_remote_endpoints", 6),
    )
    for key, value in expected:
        if rr._int(plan.get(key), key) != value:
            _fail(f"Carbink dual plan {key} is not {value}")
    standby = plan.get("standby_endpoints")
    if isinstance(standby, list):
        observed_standby = sorted(rr._int(value, "standby endpoint") for value in standby)
    else:
        observed_standby = sorted(
            rr._int(plan.get(key), key)
            for key in ("standby_endpoint", "standby_endpoint2")
        )
    if observed_standby != [6, 7]:
        _fail("Carbink dual plan must use standby endpoints 6 and 7")
    return endpoints


def _validate_dual_fault(
    rr, run_dir, plan, packing_ns, failure_ns, request_end_ns, endpoints
):
    fault = rr._read_json(run_dir / "fault-command.json")
    if fault.get("injected") is not True:
        _fail("Carbink dual fault receipt is not marked injected")
    declared = fault.get("endpoints")
    if not isinstance(declared, list) or sorted(
        rr._int(value, "fault endpoint") for value in declared
    ) != endpoints:
        _fail("Carbink dual fault receipt endpoints disagree with plan")
    begin = rr._nonnegative_int(
        fault.get("command_begin_monotonic_ns"), "dual fault command begin"
    )
    end = rr._nonnegative_int(
        fault.get("command_end_monotonic_ns"), "dual fault command end"
    )
    delay = rr._finite_number(
        plan.get("fault_delay_after_packing_ready_s"),
        "fault_delay_after_packing_ready_s",
    )
    if begin < packing_ns + int(delay * NANOSECONDS):
        _fail("Carbink dual fault command preceded packing-ready delay")
    if not begin <= failure_ns < request_end_ns or end < begin:
        _fail("Carbink dual fault timing is outside request boundary")
    controller = fault.get("controller")
    if not isinstance(controller, dict) or controller.get("returncode") != 0:
        _fail("Carbink dual fault controller did not return zero")
    try:
        stdout_receipt = json.loads(controller.get("stdout", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        _fail(f"invalid Carbink dual fault controller receipt: {exc}")
    remote_events = fault.get("remote_events")
    if not isinstance(remote_events, list) or len(remote_events) != 1:
        _fail("Carbink dual fault lacks exactly one remote fail-many receipt")
    remote_receipt = remote_events[0]
    if not isinstance(remote_receipt, dict):
        _fail("Carbink dual remote fault receipt is not an object")
    if stdout_receipt != remote_receipt:
        _fail("Carbink dual controller stdout disagrees with remote fault receipt")
    expected_ports = plan.get("server_ports")
    if not isinstance(expected_ports, list) or len(expected_ports) != 8:
        _fail("Carbink dual plan lacks eight server ports")
    expected_ports = {
        endpoint: rr._int(expected_ports[endpoint], f"server port {endpoint}")
        for endpoint in endpoints
    }
    root = plan.get("server_remote_root")
    binary = plan.get("server_binary")
    if not isinstance(root, str) or not isinstance(binary, str):
        _fail("Carbink dual plan lacks server binary identity")
    expected_binary = root.rstrip("/") + "/" + binary.lstrip("/")
    if remote_receipt.get("action") != "fail-many":
        _fail("Carbink dual fault action is not fail-many")
    if sorted(
        rr._int(value, "remote fault endpoint")
        for value in remote_receipt.get("endpoints", [])
    ) != endpoints:
        _fail("Carbink dual remote fault endpoints disagree with plan")
    for key in ("identity_verified", "process_stopped", "ports_cleared"):
        if remote_receipt.get(key) is not True:
            _fail(f"Carbink dual remote fault lacks {key}=true")
    if remote_receipt.get("signal") != "SIGKILL":
        _fail("Carbink dual fault did not record SIGKILL")
    processes = remote_receipt.get("processes")
    if not isinstance(processes, list) or len(processes) != len(endpoints):
        _fail("Carbink dual fault receipt lacks two process identities")
    seen = set()
    identities = []
    for process in processes:
        if not isinstance(process, dict):
            _fail("Carbink dual fault process identity is not an object")
        endpoint = rr._int(process.get("endpoint"), "fault process endpoint")
        if endpoint not in endpoints or endpoint in seen:
            _fail("Carbink dual fault process endpoints are not unique")
        seen.add(endpoint)
        port = rr._int(process.get("port"), "fault process port")
        if port != expected_ports[endpoint]:
            _fail("Carbink dual fault process port disagrees with plan")
        pid = rr._int(process.get("pid"), "fault process pid")
        ticks = rr._nonnegative_int(process.get("start_ticks"), "fault process start_ticks")
        exe = process.get("exe")
        if pid <= 0 or ticks <= 0 or exe != expected_binary:
            _fail("Carbink dual fault process identity disagrees with plan")
        if process.get("token") not in (None, plan.get("owner_token")):
            _fail("Carbink dual fault process owner token disagrees with plan")
        identities.append({
            "endpoint": endpoint,
            "port": port,
            "pid": pid,
            "start_ticks": ticks,
            "binary": exe,
            "token": process.get("token"),
        })
    if seen != set(endpoints):
        _fail("Carbink dual fault receipt does not cover both dead endpoints")
    return {
        "model": "owned logical service process",
        "scope": "logical service process",
        "declared_endpoints": endpoints,
        "receipt": remote_receipt,
        "process_identities": identities,
        "command_begin_monotonic_ns": begin,
        "command_end_monotonic_ns": end,
    }


def _validate_dual_background(rr, plan, grouped, event_details, aggregate, result):
    starts = [grouped[endpoint]["start"]["tokens"] for endpoint in sorted(grouped)]
    dones = [grouped[endpoint]["done"]["tokens"] for endpoint in sorted(grouped)]
    for label, records in (("start", starts), ("done", dones)):
        for key in (
            "expected_failures", "failed_mask", "read_policy", "read_accounting",
            "range_policy", "io_slots_per_job", "pipeline_depth", "scratch_bytes",
        ):
            values = [record.get(key) for record in records]
            if any(value is None for value in values) or len(set(values)) != 1:
                _fail(f"Carbink dual {label} {key} is not shared")
    for records in (starts, dones):
        if records[0].get("expected_failures") != "2":
            _fail("Carbink dual expected_failures is not 2")
        if records[0].get("failed_mask") != "3":
            _fail("Carbink dual failed_mask is not 3")
        if records[0].get("read_policy") != "four_survivors_shared":
            _fail("Carbink dual read policy is not four_survivors_shared")
        if records[0].get("read_accounting") != "first_missing_role":
            _fail("Carbink dual read accounting is not first_missing_role")
        if records[0].get("range_policy") != "live_groups":
            _fail("Carbink dual range policy is not live_groups")
        if records[0].get("io_slots_per_job") != "6":
            _fail("Carbink dual io_slots_per_job is not 6")
        if records[0].get("pipeline_depth") != "8":
            _fail("Carbink dual pipeline_depth is not 8")
        if records[0].get("scratch_bytes") != "50331648":
            _fail("Carbink dual scratch_bytes is not 50331648")
    start_times = [rr._token_int(grouped[endpoint]["start"], "monotonic_ns") for endpoint in sorted(grouped)]
    done_times = [rr._token_int(grouped[endpoint]["done"], "monotonic_ns") for endpoint in sorted(grouped)]
    if len(set(start_times)) != 1 or len(set(done_times)) != 1:
        _fail("Carbink dual start/done timestamps are not shared")
    traffic = aggregate.get("traffic")
    if not isinstance(traffic, dict):
        _fail("Carbink dual rebuild lacks shared traffic accounting")
    per_endpoint = {}
    selected_values = []
    skipped_values = []
    write_values = []
    range_values = []
    post_attempt_values = []
    for endpoint in sorted(grouped):
        done = grouped[endpoint]["done"]["tokens"]
        completed = rr._int(done.get("completed_stripes"), "completed_stripes")
        completed_ranges = rr._int(done.get("completed_ranges"), "completed_ranges")
        live_groups = rr._int(done.get("live_groups_at_copy"), "live_groups_at_copy")
        selected = rr._int(done.get("selected_bytes"), "selected_bytes")
        skipped = rr._int(done.get("skipped_bytes"), "skipped_bytes")
        empty = rr._int(done.get("empty_reactivated"), "empty_reactivated")
        scan_limit = rr._int(done.get("scan_limit"), "scan_limit")
        write_bytes = rr._int(done.get("write_bytes"), "write_bytes")
        post_attempts = rr._int(done.get("post_attempts"), "post_attempts")
        if selected != live_groups * 8192 or write_bytes != selected:
            _fail(f"Carbink dual endpoint {endpoint} selected/write bytes do not close")
        if skipped != completed * 262144 - selected or skipped < 0:
            _fail(f"Carbink dual endpoint {endpoint} skipped bytes do not close")
        if not completed <= completed_ranges <= live_groups:
            _fail(f"Carbink dual endpoint {endpoint} range count is out of bounds")
        # A scanned stripe need not contain this particular failed endpoint.
        if empty < 0 or completed + empty > scan_limit:
            _fail(f"Carbink dual endpoint {endpoint} scan accounting is out of bounds")
        selected_values.append(selected)
        skipped_values.append(skipped)
        write_values.append(write_bytes)
        range_values.append(completed_ranges)
        post_attempt_values.append(post_attempts)
        per_endpoint[str(endpoint)] = {
            "completed_stripes": completed,
            "completed_ranges": completed_ranges,
            "live_groups_at_copy": live_groups,
            "selected_bytes": selected,
            "skipped_bytes": skipped,
            "write_bytes": write_bytes,
            "read_bytes": rr._int(done.get("read_bytes"), "read_bytes"),
            "post_attempts": post_attempts,
        }
    # Each union range needs four reads, then one write per missing endpoint.
    # The union range count is at least the larger endpoint range count.
    if len(set(post_attempt_values)) != 1:
        _fail("Carbink dual post attempts are not shared")
    if post_attempt_values[0] < 4 * max(range_values) + sum(range_values):
        _fail("Carbink dual post attempts are below shared reads plus endpoint writes")
    common = {}
    for key in (
        "union_stripes", "union_selected_bytes", "union_skipped_bytes",
        "global_read_bytes", "global_write_bytes",
    ):
        values = [rr._int(done.get(key), f"dual done {key}") for done in dones]
        if len(set(values)) != 1:
            _fail(f"Carbink dual {key} is not shared")
        common[key] = values[0]
    if not max(selected_values) <= common["union_selected_bytes"] <= sum(selected_values):
        _fail("Carbink dual union_selected_bytes is outside endpoint bounds")
    if not max(skipped_values) <= common["union_skipped_bytes"] <= sum(skipped_values):
        _fail("Carbink dual union_skipped_bytes is outside endpoint bounds")
    if common["union_skipped_bytes"] != common["union_stripes"] * 262144 - common["union_selected_bytes"]:
        _fail("Carbink dual union skipped-byte accounting does not close")
    if common["global_read_bytes"] != 4 * common["union_selected_bytes"]:
        _fail("Carbink dual global read bytes do not equal four survivor reads")
    if common["global_write_bytes"] != sum(write_values):
        _fail("Carbink dual global write bytes do not equal endpoint writes")
    if traffic.get("global_read_bytes") != common["global_read_bytes"] or traffic.get("global_write_bytes") != common["global_write_bytes"]:
        _fail("Carbink dual aggregate traffic disagrees with done events")
    if traffic.get("read_formula") != "4*union_selected_bytes":
        _fail("Carbink dual aggregate read formula is not live-group four-way union")
    reported = result.get("background_rebuild_events")
    if not isinstance(reported, dict):
        _fail("Carbink dual result lacks background rebuild events")
    reported_done = reported.get("background_done")
    if not isinstance(reported_done, list) or len(reported_done) != 2:
        _fail("Carbink dual result lacks two background_done records")
    for record in reported_done:
        if not isinstance(record, dict):
            _fail("Carbink dual result background_done record is not an object")
        endpoint = rr._int(record.get("failed_endpoint"), "result background failed_endpoint")
        if str(endpoint) not in per_endpoint:
            _fail("Carbink dual result background_done has unknown endpoint")
        expected = per_endpoint[str(endpoint)]
        for key in ("completed_stripes", "completed_ranges", "selected_bytes", "skipped_bytes", "write_bytes"):
            if key in record and str(record[key]) != str(expected[key]):
                _fail(f"Carbink dual result background counter disagrees for {key}")
    return {
        "range_policy": "live_groups",
        "read_formula": "4*union_selected_bytes",
        "global_read_bytes": common["global_read_bytes"],
        "global_write_bytes": common["global_write_bytes"],
        "union_stripes": common["union_stripes"],
        "union_selected_bytes": common["union_selected_bytes"],
        "union_skipped_bytes": common["union_skipped_bytes"],
        "per_endpoint": per_endpoint,
    }


def _read_carbink_dual_run(result_path: Path, result: dict[str, Any],
                           plan: dict[str, Any], *, steady_before_s: float = 10.0):
    rr = _common()
    run_dir = result_path.parent
    endpoints = _dual_endpoints(rr, plan)
    run_id = result.get("run_id") or result.get("run") or plan.get("run_id") or run_dir.name
    if not isinstance(run_id, str) or not run_id:
        _fail("Carbink dual result lacks run id")
    client_path = run_dir / "client.log"
    client_text, evidence = rr._read_client_log(client_path)
    window_json = rr._read_json(run_dir / "kvs.completed-100ms.json")
    request_start_ns, request_end_ns = rr._native_request_times(evidence, window_json)
    expected_requests = rr._nonnegative_int(plan.get("requests"), "plan requests")
    if expected_requests <= 0:
        _fail("Carbink dual plan requests must be positive")
    windows, window_details = rr._read_windows(
        run_dir, request_start_ns, request_end_ns, expected_requests
    )
    _validate_runner(rr, run_dir, result, plan)
    receipt_details = _validate_receipt(rr, result, evidence, expected_requests)
    config_details = rr._validate_client_configuration(plan, evidence["configuration"])
    placement_details = rr._validate_placement(plan, evidence["worker_roles"])
    grouped = rr._group_rebuild_events(evidence, endpoints)
    aggregate, event_details = rr._validate_rebuild_events(
        grouped, request_start_ns, request_end_ns, system="carbink"
    )
    background_details = _validate_dual_background(
        rr, plan, grouped, event_details, aggregate, result
    )
    packing = rr._single(evidence["packing"], "runtime_packing_ready")
    packing_ns = rr._token_int(packing, "monotonic_ns")
    fault_details = _validate_dual_fault(
        rr, run_dir, plan, packing_ns, aggregate["failure_ns"], request_end_ns, endpoints
    )
    placement_result = result.get("placement")
    if not isinstance(placement_result, dict) or not isinstance(
        placement_result.get("endpoints"), list
    ) or len(placement_result["endpoints"]) != 8:
        _fail("Carbink dual placement evidence lacks eight endpoints")
    resources = result.get("server_resources")
    if not isinstance(resources, dict):
        _fail("Carbink dual result lacks server resource evidence")
    resource_servers = resources.get("servers")
    resource_count = len(resource_servers) if isinstance(resource_servers, list) else rr._int(
        resource_servers, "server_resources.servers"
    )
    if resource_count != 8:
        _fail("Carbink dual result lacks eight server resource records")
    stop = result.get("observed_fault_stop")
    if not isinstance(stop, dict) or (
        str(stop.get("completed")) != "6"
        or str(stop.get("skipped_dead")) != "2"
        or str(stop.get("failed")) != "0"
    ):
        _fail("Carbink dual shutdown STOP receipt is not 6/2/0")
    shadow = result.get("compaction_after_failure")
    if not isinstance(shadow, dict) or int(shadow.get("moved_spans_after_failure", 0)) <= 0:
        _fail("Carbink dual compaction-after-failure evidence is missing")
    workload_id, workload_payload, workload_canonical = rr._workload_identity(plan)
    failure_s = (aggregate["failure_ns"] - request_start_ns) / NANOSECONDS
    rebuild_start_s = (aggregate["start_ns"] - request_start_ns) / NANOSECONDS
    recovered_s = (aggregate["done_ns"] - request_start_ns) / NANOSECONDS
    packing_s = (packing_ns - request_start_ns) / NANOSECONDS
    steady_start_s = max(0.0, failure_s - steady_before_s, packing_s)
    if not steady_start_s < failure_s:
        _fail("no steady pre-failure interval remains after packing boundary")
    local_bytes = rr._finite_number(plan.get("local_bytes"), "local_bytes")
    logical_bytes = rr._finite_number(plan.get("logical_bytes"), "logical_bytes")
    ratio = local_bytes / logical_bytes * 100.0
    if ratio.is_integer():
        ratio = int(ratio)
    environment = f"{plan['compute']}->{plan['memory']}"
    details = {
        "adapter": "figure13.carbink_native",
        "source_type": "native_result_json",
        "result_path": str(result_path),
        "plan_path": str(run_dir / "plan.json"),
        "client_log_path": str(client_path),
        "client_log_sha256": evidence["sha256"],
        "raw_result_source": result_path.name,
        "recovery_definition": "background_start_to_done",
        "steady_before_s": steady_before_s,
        "packing_ready_ns": packing_ns,
        "packing_ready_elapsed_s": packing_s,
        "workload_id_payload": workload_payload,
        "workload_id_canonical_json": workload_canonical,
        "plan": plan,
        "selected_result": result,
        "fault": fault_details,
        "startup_configuration": config_details,
        "placement_validation": placement_details,
        "placement": placement_result,
        "server_resources": resources,
        "rebuild_events": event_details,
        "rebuild_aggregate": aggregate,
        "rebuild_traffic": {
            str(endpoint): event_details[str(endpoint)]["done"]
            for endpoint in endpoints
        },
        "background_rebuild": background_details,
        "native_rebuild_duration_s": aggregate["duration_ns"] / NANOSECONDS,
        "receipt_and_correctness": receipt_details,
        "windows": window_details,
        "source_files": {
            "result": result_path.name,
            "plan": "plan.json",
            "client_log": "client.log",
            "windows_csv": Path(window_details["csv_path"]).name,
            "windows_json": Path(window_details["json_path"]).name,
        },
    }
    run = {
        "run_id": run_id, "system": "carbink", "scenario": "2-node",
        "workload": "kv-b", "environment": environment,
        "workload_id": workload_id, "ratio": ratio,
        "app_workers": rr._int(plan.get("app_workers"), "app_workers"),
        "repeat": "unindexed", "phase": "work", "time_origin": "work_start",
        "steady_start_s": steady_start_s, "failure_elapsed_s": failure_s,
        "rebuild_start_elapsed_s": rebuild_start_s,
        "recovered_elapsed_s": recovered_s,
        "run_end_s": (request_end_ns - request_start_ns) / NANOSECONDS,
        "exit_status": 0, "correctness": "pass",
        "failure_confirmed": 1, "recovery_verified": 1, "panel": "both",
        "_source": str(result_path),
        "recovery_definition": "background_start_to_done",
    }
    return run, windows, details


def read_carbink_run(result_path: Path, result: dict[str, Any],
                     plan: dict[str, Any], *, steady_before_s: float = 10.0):
    rr = _common()
    declared_dual = plan.get("failed_endpoints")
    if not isinstance(declared_dual, list):
        declared_dual = plan.get("fault_endpoints")
    if plan.get("remote_endpoints") == 8 or (
        isinstance(declared_dual, list) and len(declared_dual) == 2
    ):
        return _read_carbink_dual_run(
            result_path, result, plan, steady_before_s=steady_before_s
        )
    # This adapter recognizes the validated six-active/one-spare, single
    # service runner only. Future two-node records need their own evidence.
    if (plan.get("scenario", "1-node") != "1-node" or
            plan.get("remote_endpoints") != 7 or
            plan.get("active_endpoints") != 6 or
            plan.get("standby_endpoint") != 6 or
            plan.get("fault_endpoint") != 0):
        _fail("unsupported Carbink runner topology; expected one failed service and one spare")
    run_dir = result_path.parent
    if (
        rr._int(plan.get("active_endpoints"), "active_endpoints") != 6
        or rr._int(plan.get("standby_endpoint"), "standby_endpoint") != 6
        or rr._int(plan.get("fault_endpoint"), "fault_endpoint") != 0
        or rr._int(plan.get("remote_endpoints"), "remote_endpoints") != 7
    ):
        _fail("Carbink adapter supports only the current 6-active/standby-6 schema")
    run_id = (
        result.get("run_id") or result.get("run") or plan.get("run_id")
        or run_dir.name
    )
    if not isinstance(run_id, str) or not run_id:
        _fail("Carbink result lacks run_id/run fallback")
    client_path = run_dir / "client.log"
    client_text, evidence = rr._read_client_log(client_path)
    window_json = rr._read_json(run_dir / "kvs.completed-100ms.json")
    request_start_ns, request_end_ns = rr._native_request_times(evidence, window_json)
    expected_requests = rr._nonnegative_int(plan.get("requests"), "plan requests")
    if expected_requests <= 0:
        _fail("Carbink plan requests must be positive")
    windows, window_details = rr._read_windows(
        run_dir, request_start_ns, request_end_ns, expected_requests
    )
    _validate_runner(rr, run_dir, result, plan)
    receipt_details = _validate_receipt(rr, result, evidence, expected_requests)
    config_details = rr._validate_client_configuration(plan, evidence["configuration"])
    placement_details = rr._validate_placement(plan, evidence["worker_roles"])
    endpoints = [rr._int(plan.get("fault_endpoint", 0), "fault_endpoint")]
    aggregate, event_details = rr._validate_rebuild_events(
        rr._group_rebuild_events(evidence, endpoints),
        request_start_ns, request_end_ns, system="carbink",
    )
    start_record = event_details[str(endpoints[0])]["start"]
    done_record = event_details[str(endpoints[0])]["done"]
    done_tokens = done_record["tokens"]
    completed = rr._int(done_tokens["completed_stripes"], "completed_stripes")
    write_bytes = rr._int(done_tokens["write_bytes"], "write_bytes")
    if rr._int(done_tokens["read_bytes"], "read_bytes") != 4 * write_bytes:
        _fail("Carbink background read/write bytes do not close")
    background_details = _validate_background(
        rr, plan, done_record, start_record, result, completed, write_bytes
    )
    packing = rr._single(evidence["packing"], "runtime_packing_ready")
    packing_ns = rr._token_int(packing, "monotonic_ns")
    fault_details = _validate_fault(
        rr, run_dir, plan, packing_ns, aggregate["failure_ns"], request_end_ns
    )
    placement_result = result.get("placement")
    if not isinstance(placement_result, dict):
        _fail("Carbink result lacks placement evidence")
    endpoints_result = placement_result.get("endpoints")
    if not isinstance(endpoints_result, list) or len(endpoints_result) != 7:
        _fail("Carbink placement evidence lacks seven endpoints")
    resources = result.get("server_resources")
    if not isinstance(resources, dict):
        _fail("Carbink result lacks seven server resource records")
    resource_servers = resources.get("servers")
    if isinstance(resource_servers, list):
        resource_count = len(resource_servers)
    else:
        resource_count = rr._int(resource_servers, "server_resources.servers")
    if resource_count != 7:
        _fail("Carbink result lacks seven server resource records")
    stop = result.get("observed_fault_stop")
    if not isinstance(stop, dict) or (
        str(stop.get("completed")) != "6"
        or str(stop.get("skipped_dead")) != "1"
        or str(stop.get("failed")) != "0"
    ):
        _fail("Carbink shutdown STOP receipt is not 6/1/0")
    shadow = result.get("compaction_after_failure")
    if not isinstance(shadow, dict) or int(shadow.get("moved_spans_after_failure", 0)) <= 0:
        _fail("Carbink compaction-after-failure evidence is missing")
    workload_id, workload_payload, workload_canonical = rr._workload_identity(plan)
    failure_s = (aggregate["failure_ns"] - request_start_ns) / NANOSECONDS
    rebuild_start_s = (aggregate["start_ns"] - request_start_ns) / NANOSECONDS
    recovered_s = (aggregate["done_ns"] - request_start_ns) / NANOSECONDS
    packing_s = (packing_ns - request_start_ns) / NANOSECONDS
    steady_start_s = max(0.0, failure_s - steady_before_s, packing_s)
    if not steady_start_s < failure_s:
        _fail("no steady pre-failure interval remains after packing boundary")
    local_bytes = rr._finite_number(plan.get("local_bytes"), "local_bytes")
    logical_bytes = rr._finite_number(plan.get("logical_bytes"), "logical_bytes")
    ratio = local_bytes / logical_bytes * 100.0
    if ratio.is_integer():
        ratio = int(ratio)
    environment = f"{plan['compute']}->{plan['memory']}"
    details = {
        "adapter": "figure13.carbink_native",
        "source_type": "native_result_json",
        "result_path": str(result_path),
        "plan_path": str(run_dir / "plan.json"),
        "client_log_path": str(client_path),
        "client_log_sha256": evidence["sha256"],
        "raw_result_source": result_path.name,
        "recovery_definition": "background_start_to_done",
        "steady_before_s": steady_before_s,
        "packing_ready_ns": packing_ns,
        "packing_ready_elapsed_s": packing_s,
        "workload_id_payload": workload_payload,
        "workload_id_canonical_json": workload_canonical,
        "plan": plan,
        "selected_result": result,
        "fault": fault_details,
        "startup_configuration": config_details,
        "placement_validation": placement_details,
        "placement": placement_result,
        "server_resources": resources,
        "rebuild_events": event_details,
        "rebuild_aggregate": aggregate,
        "rebuild_traffic": {str(endpoints[0]): done_tokens},
        "background_rebuild": background_details,
        "native_rebuild_duration_s": aggregate["duration_ns"] / NANOSECONDS,
        "receipt_and_correctness": receipt_details,
        "windows": window_details,
        "source_files": {
            "result": result_path.name,
            "plan": "plan.json",
            "client_log": "client.log",
            "windows_csv": Path(window_details["csv_path"]).name,
            "windows_json": Path(window_details["json_path"]).name,
        },
    }
    run = {
        "run_id": run_id, "system": "carbink", "scenario": "1-node",
        "workload": "kv-b", "environment": environment,
        "workload_id": workload_id, "ratio": ratio,
        "app_workers": rr._int(plan.get("app_workers"), "app_workers"),
        "repeat": "unindexed", "phase": "work", "time_origin": "work_start",
        "steady_start_s": steady_start_s, "failure_elapsed_s": failure_s,
        "rebuild_start_elapsed_s": rebuild_start_s,
        "recovered_elapsed_s": recovered_s,
        "run_end_s": (request_end_ns - request_start_ns) / NANOSECONDS,
        "exit_status": 0, "correctness": "pass", "source_type": "measured",
        "failure_confirmed": 1, "recovery_verified": 1, "panel": "both",
        "_source": str(result_path),
        "recovery_definition": "background_start_to_done",
    }
    return run, windows, details
