"""Final log record consumed by the Figure 12 collector.

Emit ONE key=value line per completed, healthy run:

figure12_result schema_version=1 workload=kv-b system=starfish run_id=kvb-starfish-r1 environment=pair-a workload_id=kvb-512b-zipf099-1b ratio=25 app_workers=24 repeat=1 phase=work elapsed_s=60 local_ec_cpu_seconds=12 metadata_peak_bytes=20000000 app_memory_bytes=1000000000 exit_status=0 correctness=pass source_type=measured

Example numbers describe the interface, not benchmark measurements.
Emit the final record only after workload verification and successful exit.
Only Carbink and Starfish are compared. Run_id must be unique in selected logs.
Workload_id identifies identical data, scale, requests and workload settings
across systems; environment identifies matched experimental hardware/site.

All values refer to the same complete Work interval, excluding initialization,
warmup and cleanup. local_ec_cpu_seconds sums compute-node CPU execution time
spent performing EC computation across workers (including coding/XOR work).
Do not count RDMA waits, scheduling/suspension or whole-worker elapsed time
as EC computation. elapsed_s is this run's Work wall-clock duration.
CPU normalization is:
  (local_ec_cpu_seconds / elapsed_s) /
  (Carbink.local_ec_cpu_seconds / Carbink.elapsed_s)
Thus different run durations are not mistaken for CPU-use differences.

metadata_peak_bytes is the peak simultaneous compute-side runtime metadata
allocation during Work: object references, region/allocation/group tracking,
backup/recompute mappings and related FT bookkeeping, each counted once.
Exclude application payload and EC data/parity buffers; do not substitute RSS
or sum unrelated per-category peaks.
app_memory_bytes is the application's full logical payload footprint at its
100-percent-memory setting, not its local cache, process RSS or remote parity.
Matched systems must report the same footprint. The displayed percentage is
100 * metadata_peak_bytes / app_memory_bytes (2.0 means 2 percent).

These are this collector's explicit reporting conventions. The script does
not instrument applications or claim the measurements already exist.
Missing fields are errors, not zero. Numeric zero must be measured.
"""
from __future__ import annotations

import math
import shlex

PREFIX = "figure12_result"
WORKLOADS = {
    "bfs": "BFS", "llama": "LLM", "mg": "MG", "wc": "WC",
    "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S", "nq": "NQ",
}
SYSTEMS = {"carbink": "Carbink", "starfish": "Starfish"}
ALIASES = {"llm": "llama", "wordcount": "wc",
           "kv_b": "kv-b", "kv_a": "kv-a", "kv_s": "kv-s"}
TEXT_FIELDS = ("workload", "system", "run_id", "environment", "workload_id",
               "phase", "correctness", "source_type")
INT_FIELDS = ("schema_version", "ratio", "app_workers", "repeat", "metadata_peak_bytes",
              "app_memory_bytes", "exit_status")
FLOAT_FIELDS = ("elapsed_s", "local_ec_cpu_seconds")
REQUIRED = frozenset(TEXT_FIELDS + INT_FIELDS + FLOAT_FIELDS)
CONTEXT_FIELDS = ("environment", "workload_id", "ratio", "app_workers",
                  "repeat", "phase", "app_memory_bytes")


def parse_line(line):
    """Return None for unrelated log output; malformed metric records fail."""
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
    missing = REQUIRED - record.keys()
    if missing:
        raise ValueError("missing log fields: " + ", ".join(sorted(missing)))
    for key in TEXT_FIELDS:
        if not record[key].strip():
            raise ValueError(f"{key} must not be blank")
    for key in INT_FIELDS:
        value = record[key]
        digits = value.removeprefix("-") if key == "exit_status" else value
        if not digits.isascii() or not digits.isdecimal():
            raise ValueError(f"{key} must be a nonnegative integer")
        record[key] = int(value)
    for key in FLOAT_FIELDS:
        record[key] = float(record[key])
        if not math.isfinite(record[key]) or record[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if record["schema_version"] != 1:
        raise ValueError("unsupported Figure 12 log schema")
    if not 1 <= record["ratio"] <= 100 or record["app_workers"] < 1 or record["repeat"] < 1:
        raise ValueError("invalid ratio, application worker count or repeat")
    if record["elapsed_s"] <= 0 or record["app_memory_bytes"] <= 0:
        raise ValueError("elapsed_s and app_memory_bytes must be positive")
    if record["phase"] != "work":
        raise ValueError("Figure 12 expects the complete work phase")
    if record["source_type"] != "measured":
        raise ValueError("log collector accepts measured records only")
    app = record["workload"].lower()
    record["workload"] = ALIASES.get(app, app)
    record["system"] = record["system"].lower()
    if record["workload"] not in WORKLOADS or record["system"] not in SYSTEMS:
        raise ValueError("unknown workload or system")
    return record
