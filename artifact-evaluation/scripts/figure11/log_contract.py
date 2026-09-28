"""Final log record consumed by the Figure 11 collector.

Emit ONE whitespace-separated key=value line per completed, healthy run:

figure11_result schema_version=1 workload=kv-b system=starfish run_id=kvb-starfish-r1 environment=pair-a workload_id=kvb-512b-zipf099-1b ratio=25 app_workers=24 repeat=1 phase=work elapsed_s=60 fetch_bytes=1000000 eviction_bytes=2000000 remote_cpu_seconds=90 remote_memory_peak_bytes=100000000 exit_status=0 correctness=pass source_type=measured

The numbers above illustrate the interface; they are not benchmark data.
The run wrapper should emit the final record only after workload verification,
process exit and collection of all owned memory-service counters.

All five metrics describe the SAME complete Work interval, excluding setup,
warmup and cleanup. Workload_id identifies the exact dataset/model, request
count/mix, repetitions and other workload settings, shared across systems.
Environment identifies the matched experimental hardware/site.
Run_id must be unique across the selected logs; duplicate summaries are errors.

fetch_bytes: total demand-fetch payload bytes, including any associated
protection reads. eviction_bytes: all eviction/protection-maintenance payload
bytes, including data, parity and read-modify-write reads/writes. Count each
transfer once, not at both sender and receiver. These are disjoint categories;
do not use elapsed-time bandwidth or hardware-port byte totals instead.

remote_cpu_seconds: SUM of user+system CPU seconds consumed by all owned
memory services during Work. Divide by elapsed_s to obtain mean occupied cores.
Exclude unrelated services; do not average per-node percentages.

remote_memory_peak_bytes: peak of the time-aligned SUM of memory occupied by
all owned memory services (data, parity, retained backups, metadata, allocated
alignment waste and invalid spans not yet reclaimed). Not the configured pool
capacity/RSS and not the sum of each service's independent peak.
These CPU and memory summaries are the Figure 11 log contract, not a claim
that the paper established an otherwise unspecified sampling convention.

Missing metrics are errors, not zeros. A numeric zero must be measured.
The script consumes logs only; producers and sampling are implemented elsewhere.
"""

from __future__ import annotations

import math
import shlex

PREFIX = "figure11_result"
WORKLOADS = {
    "bfs": "BFS", "llama": "LLM", "mg": "MG", "wc": "WC",
    "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S", "nq": "NQ",
}
SYSTEMS = {"hydra": "Hydra", "carbink": "Carbink",
           "starfish": "Starfish", "nonft": "Non-FT"}
ALIASES = {"llm": "llama", "wordcount": "wc",
           "kv_b": "kv-b", "kv_a": "kv-a", "kv_s": "kv-s"}
TEXT_FIELDS = ("workload", "system", "run_id", "environment", "workload_id",
               "phase", "correctness", "source_type")
INT_FIELDS = ("schema_version", "ratio", "app_workers", "repeat", "fetch_bytes",
              "eviction_bytes", "remote_memory_peak_bytes", "exit_status")
FLOAT_FIELDS = ("elapsed_s", "remote_cpu_seconds")
REQUIRED = frozenset(TEXT_FIELDS + INT_FIELDS + FLOAT_FIELDS)
CONTEXT_FIELDS = ("environment", "workload_id", "ratio", "app_workers",
                  "repeat", "phase")


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
        raise ValueError("unsupported Figure 11 log schema")
    if not 1 <= record["ratio"] <= 100 or record["app_workers"] < 1 or record["repeat"] < 1:
        raise ValueError("invalid ratio, application worker count or repeat")
    if record["elapsed_s"] <= 0:
        raise ValueError("elapsed_s must be positive")
    if record["phase"] != "work":
        raise ValueError("Figure 11 expects the complete work phase")
    if record["source_type"] != "measured":
        raise ValueError("log collector accepts measured records only")
    app = record["workload"].lower()
    record["workload"] = ALIASES.get(app, app)
    record["system"] = record["system"].lower().replace("non-ft", "nonft")
    if record["workload"] not in WORKLOADS or record["system"] not in SYSTEMS:
        raise ValueError("unknown workload or system")
    return record
