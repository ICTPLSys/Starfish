"""Figure 13 explicit result/sample protocol (schema_version=1 or 2).

Native recovery result directories are supported separately by --run-result;
they do not need manufactured figure13_result lines. For new explicit logs,
use schema_version=2 and add rebuild_start_elapsed_s. Version 2 duration is
background start to fully completed rebuild. Version 1 retains the legacy
failure-to-completion definition; never silently compare the two definitions.

One final result per run, after workload/recovery verification and process exit:
figure13_result schema_version=1 run_id=starfish-one-r1 system=starfish scenario=1-node workload=kv-b environment=pair-a workload_id=kvb-fixed ratio=25 app_workers=24 repeat=1 phase=work time_origin=work_start steady_start_s=0 failure_elapsed_s=20 recovered_elapsed_s=24 run_end_s=40 failure_confirmed=1 recovery_verified=1 exit_status=0 correctness=pass source_type=measured panel=both

Repeated, nonoverlapping throughput windows belonging to that run:
figure13_sample schema_version=1 run_id=starfish-one-r1 window_start_s=0 window_end_s=1 completed_ops=1000000

Numbers above are format examples, NOT measurements. No fault is injected by
these scripts. Log actual observed events, not a requested injection schedule.
For 2-node, failure_elapsed_s is when the entire failed-node set is unavailable.
recovered_elapsed_s means full lost-state reconstruction/protection completed
and verified, not the first successful read or merely detecting the failure.
All event/window times use one monotonic clock relative to this run's Work
start. run_end_s bounds the measurement interval; setup and cleanup are excluded.
steady_start_s excludes warmup when deriving the pre-failure reference.

completed_ops is the number of application requests completed during precisely
that window, not offered requests or a cumulative counter. A real zero is valid.
Every sample must have a final result with the same globally unique run_id.
All matched logs are checked for valid schema, unique final results and orphan
samples before ratio/repeat selection; narrow --logs-root/--pattern as needed.
Only Hydra, Carbink and Starfish; scenarios are 1-node and 2-node.
Context (workload/environment/workload_id/ratio/app_workers/repeat/phase)
must match across the selected comparison. workload_id identifies the same
data, request mix/scale and runtime experiment settings across systems.

panel is optional, default both: recovery supplies duration bars, trace supplies
the timeline, both supplies both. Separate runs may supply the two panels.
Each selected panel/scenario/system must have at most one final run.

Collector schema 1: recovery duration = recovered_elapsed_s - failure_elapsed_s.
Collector schema 2: recovery duration = recovered_elapsed_s - rebuild_start_elapsed_s.
Both keep the observed failure time for aligning the throughput plot. Do not
substitute a kill-command receipt time or the first successful degraded read.
Throughput = completed_ops / (window_end_s - window_start_s).
All curves divide by ONE Starfish pre-failure throughput: sum(completed_ops)
divided by sum(window durations), using complete contiguous windows within
[steady_start_s, failure_elapsed_s]. Observed windows must cover that entire
interval; boundary-crossing windows are excluded from the mean, not extrapolated.
Samples are plotted at their window end. Gaps are missing, not measured zeros.
Windows crossing failure/recovery are retained as measured window averages,
marked window_scope=crosses_event, not instantaneous or single-phase rates.
Their counts are never split or interpolated; they cannot supply the reference.

For display only, each run is shifted so its observed failure appears at
--failure-at-s (default 20). Original times and offsets remain in output.
This may yield negative display coordinates; original elapsed times stay
nonnegative. The display anchor is NOT used to calculate recovery durations.
"""

from __future__ import annotations

import math
import shlex

SYSTEMS = {"hydra": "Hydra", "carbink": "Carbink", "starfish": "Starfish"}
SCENARIOS = ("1-node", "2-node")
CONTEXT_FIELDS = ("workload", "environment", "workload_id", "ratio",
                  "app_workers", "repeat", "phase", "time_origin")
RESULT_TEXT = ("run_id", "system", "scenario", "workload", "environment",
               "workload_id", "phase", "time_origin", "correctness", "source_type")
RESULT_INTS = ("schema_version", "ratio", "app_workers", "repeat",
               "failure_confirmed", "recovery_verified", "exit_status")
RESULT_TIMES = ("steady_start_s", "failure_elapsed_s", "recovered_elapsed_s", "run_end_s")
SAMPLE_FIELDS = ("schema_version", "run_id", "window_start_s", "window_end_s", "completed_ops")


def integer(value, key):
    digits = value.removeprefix("-") if key == "exit_status" else value
    if not digits.isascii() or not digits.isdecimal():
        raise ValueError(f"{key} must be an integer")
    return int(value)


def seconds(value, key):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return value


def parse_line(line):
    parts = line.strip().split(maxsplit=1)
    if not parts or parts[0] not in ("figure13_result", "figure13_sample"):
        return None
    kind = parts[0].removeprefix("figure13_")
    record = {}
    for token in shlex.split(parts[1] if len(parts) > 1 else ""):
        key, separator, value = token.partition("=")
        if not separator or not key or not value.strip() or key in record:
            raise ValueError(f"invalid or duplicate key=value token: {token!r}")
        record[key] = value
    required = (RESULT_TEXT + RESULT_INTS + RESULT_TIMES
                if kind == "result" else SAMPLE_FIELDS)
    missing = set(required) - record.keys()
    if missing:
        raise ValueError("missing fields: " + ", ".join(sorted(missing)))
    int_keys = RESULT_INTS if kind == "result" else ("schema_version", "completed_ops")
    for key in int_keys:
        record[key] = integer(record[key], key)
    if record["schema_version"] not in (1, 2):
        raise ValueError("unsupported Figure 13 log schema")
    for key in (RESULT_TIMES if kind == "result" else ("window_start_s", "window_end_s")):
        record[key] = seconds(record[key], key)
    if kind == "result":
        if record["schema_version"] == 2:
            if "rebuild_start_elapsed_s" not in record:
                raise ValueError("schema 2 requires rebuild_start_elapsed_s")
            record["rebuild_start_elapsed_s"] = seconds(
                record["rebuild_start_elapsed_s"], "rebuild_start_elapsed_s")
            if not record["rebuild_start_elapsed_s"] < record["recovered_elapsed_s"]:
                raise ValueError("background rebuild must start before completion")
            record["recovery_definition"] = "background_start_to_done"
        else:
            record["recovery_definition"] = "failure_to_reconstruction"
        record["system"] = record["system"].lower()
        if record["system"] not in SYSTEMS or record["scenario"] not in SCENARIOS:
            raise ValueError("unknown system/scenario")
        if not 1 <= record["ratio"] <= 100 or record["app_workers"] < 1 or record["repeat"] < 1:
            raise ValueError("invalid ratio, workers or repeat")
        if record["phase"] != "work" or record["time_origin"] != "work_start":
            raise ValueError("event times must use the Work-start monotonic origin")
        if record["source_type"] != "measured":
            raise ValueError("only measured records can be collected")
        if not (record["steady_start_s"] < record["failure_elapsed_s"]
                < record["recovered_elapsed_s"] <= record["run_end_s"]):
            raise ValueError("expected steady_start < failure < recovered <= run_end")
        record.setdefault("panel", "both")
        if record["panel"] not in ("both", "recovery", "trace"):
            raise ValueError("panel must be both, recovery or trace")
    elif record["window_end_s"] <= record["window_start_s"]:
        raise ValueError("sample window must have positive duration")
    record["_kind"] = kind
    return record
