"""Figure 12 raw-runtime contract (successful AE run directories).

Two independent opt-ins are recorded in manifest.plan.client_env:
  FARLIB_RUNTIME_EC_CPU=1
  FARLIB_RUNTIME_METADATA=1

The legacy compute client emits one EC record for every complete profile Work:
  runtime_ec_cpu schema_version=1 system=starfish phase=work scope=compute_ec clock=tsc boundary_sequence=1 cycles=123 scopes=1
Schema 2 additionally emits one initialization record with boundary_sequence=0,
followed by the contiguous completed-Work records. Every schema-2 record must
declare boundary_semantics=admitted_scope_drain. The collector keeps
initialization and Work components separate and uses their sum for the Figure
12 value. cycles is a raw measured TSC/reference-cycle count for EC computation
scopes, not CPU seconds, PMU core cycles, whole-worker time, RDMA wait, or a
ratio to Carbink.
The sample number documents the syntax; it is not an experiment result.

The client also emits exactly one existing runtime_metadata schema_version=1
record at exit, saved from the last completed Work before application cleanup.
Its accounted_bytes = metadata_bytes (core) + measurement_aux_bytes.
Figure 12 includes the auxiliary structures: 100 * accounted_bytes / the
manifest's full application footprint, not local-cache bytes, a peak, or RSS.
CSV metadata_bytes is that inclusive total; core_metadata_bytes preserves the
raw core-only field. Do not add CSV metadata_bytes and accounted_bytes.

Missing instrumentation is an error, not a zero measurement. A recorded zero
is valid. Legacy figure12_result CPU-seconds/metadata-peak records and
local_ec_cpu_norm CSV values cannot be converted to this revised contract.
"""
from __future__ import annotations
import shlex

EC_PREFIX = "runtime_ec_cpu"
EC_SCHEMA_VERSIONS = (1, 2)
EC_BOUNDARY_SEMANTICS = "admitted_scope_drain"
WORKLOADS = {
    "bfs": "BFS", "llama": "LLM", "mg": "MG", "wordcount": "WC",
    "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S", "nq": "NQ",
}
SYSTEMS = {"carbink": "Carbink", "starfish": "Starfish"}
METRIC_UNITS = {"local_ec_cpu_cycles": "cycles",
                "metadata_space_pct": "% app memory"}
METRIC_SELECTIONS = {
    "all": tuple(METRIC_UNITS),
    "ec-cpu": ("local_ec_cpu_cycles",),
    "metadata": ("metadata_space_pct",),
}


def nonnegative_integer(value, name):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result >= 2**64:
        raise ValueError(f"{name} exceeds uint64")
    return result


def parse_line(line):
    tokens = shlex.split(line.strip())
    if not tokens or tokens[0] != EC_PREFIX:
        return None
    fields = {}
    for token in tokens[1:]:
        key, sep, value = token.partition("=")
        if not sep or not key or not value or key in fields:
            raise ValueError(f"invalid or duplicate EC field: {token!r}")
        fields[key] = value
    required = {"schema_version", "system", "phase", "scope", "clock",
                "boundary_sequence", "cycles", "scopes"}
    if missing := required - fields.keys():
        raise ValueError("missing EC fields: " + ", ".join(sorted(missing)))
    for key in ("schema_version", "boundary_sequence", "cycles", "scopes"):
        fields[key] = nonnegative_integer(fields[key], key)
    schema = fields["schema_version"]
    if schema not in EC_SCHEMA_VERSIONS:
        raise ValueError("unsupported EC schema")
    if schema == 1:
        if fields["phase"] != "work" or fields["boundary_sequence"] < 1:
            raise ValueError("schema 1 requires a completed Work boundary")
        if "boundary_semantics" in fields:
            raise ValueError("schema 1 must not declare boundary semantics")
    else:
        if fields.get("boundary_semantics") != EC_BOUNDARY_SEMANTICS:
            raise ValueError("unknown or missing schema 2 boundary semantics")
        if fields["phase"] == "initialization":
            if fields["boundary_sequence"] != 0:
                raise ValueError("schema 2 initialization must use boundary_sequence=0")
        elif fields["phase"] == "work":
            if fields["boundary_sequence"] < 1:
                raise ValueError("schema 2 Work must use boundary_sequence>=1")
        else:
            raise ValueError("schema 2 phase must be initialization or work")
    if fields["system"] not in SYSTEMS:
        raise ValueError("unsupported EC runtime")
    for key, expected in (("scope", "compute_ec"), ("clock", "tsc")):
        if fields[key] != expected:
            raise ValueError(f"EC {key} must be {expected}")
    if fields["cycles"] and not fields["scopes"]:
        raise ValueError("nonzero EC cycles require a measured computation scope")
    return fields
