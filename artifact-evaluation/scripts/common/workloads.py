"""Application-specific commands and log checks shared by AE figures."""

from __future__ import annotations

import math
from pathlib import Path
import re


# Application footprints used to render ratio-dependent capacity budgets.
FOOTPRINT_BYTES = {"llama": 26_954_711_068, "bfs": 17_483_494_400,
                   "mg": 28_520_742_912, "wordcount": 40 * 1024**3,
                   "nq": 32 * 1024**3,
                   **{app: 16 * 1024**3 for app in ("kv-b", "kv-a", "kv-s")}}
GENERATED_INPUTS = frozenset(("mg", "kv-b", "kv-a", "kv-s"))
BINARY_RELATIVE = {
    "llama": Path("benchmark/llama/run_chat_far"),
    "bfs": Path("benchmark/microbenchmarks/gapbs_bfs_chunked"),
    "mg": Path("benchmark/mg/mg"),
    "wordcount": Path("benchmark/mapreduce/wordcount_far"),
    "nq": Path("benchmark/microbenchmarks/nhop_graph"),
    **{app: Path("benchmark/kvs/kvs_throughput") for app in ("kv-b", "kv-a", "kv-s")},
}
WORKLOAD_LABEL = {"llama": "LLM", "bfs": "BFS", "mg": "MG",
                  "wordcount": "WC", "nq": "NQ",
                  "kv-b": "KV-B", "kv-a": "KV-A", "kv-s": "KV-S"}
SYSTEM_LABEL = {"nonft": "Non-FT", "starfish": "Starfish", "hydra": "Hydra",
                "carbink": "Carbink"}
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# These variables control the application/runtime placement or workload
# profile.  They must come from the recorded site profile rather than from a
# caller's shell, where a stale experiment can silently override the run.
CONTROLLED_ENV_PREFIXES = ("FARLIB_", "Fibre", "GAPBS_", "MG_", "NHOP_", "WORDCOUNT_")

NONFT_ENV = {
    "FARLIB_FIBRE_HEAP": "1",
    "FARLIB_ALLOC_SCOPE_CHECKPOINT": "1",
    "FARLIB_ASYNC_FULL_RETURNS": "1",
    "FARLIB_THREAD_HEAP_FULL_RETURN_BATCH": "1",
    "FARLIB_REGION_LIST_YIELD_LOCK": "1",
    "FARLIB_REMOTE_USAGE_SHARDS": "1",
    "FARLIB_PLANNER_BUDGET_EARLY_EXIT": "1",
    "FARLIB_RESIDENT_PROFILE_REQUIRE_WORK_PHASE": "1",
    "FARLIB_LEGACY_SCAN_CURSORS": "1",
}

# The healthy Design 2 profile uses measured remote heat and maps local bands
# to actual Resident/Streaming placement. It is not the legacy fixed-six mode.
DESIGN2_ENV = {
    "FARLIB_FIXED_SIX_GROUPS": "0",
    "FARLIB_FIXED_SIX_POOLS": "0",
    "FARLIB_LIST_ONLY_SIX": "1",
    "FARLIB_LIST_ONLY_SIX_GROUPS": "0",
    "FARLIB_SIMPLE_HOTCOLD": "1",
    "FARLIB_SIMPLE_HOTCOLD_RELINK": "1",
    "FARLIB_SIMPLE_LOCAL_ROUTING": "1",
    "FARLIB_SIMPLE_REMOTE_HOTCOLD": "1",
    "FARLIB_SIMPLE_LOCAL_RESIDENT": "1",
    "FARLIB_SIMPLE_SIX_GROUPS": "1",
    "FARLIB_SIMPLE_REGION_BUDGET": "adaptive",
    "FARLIB_SIMPLE_REGION_BUDGET_FAST": "1",
    "FARLIB_SIMPLE_REGION_BUDGET_INTERVAL_MS": "1000",
    "FARLIB_SIMPLE_REGION_HEAT": "1",
    "FARLIB_SIMPLE_REGION_HEAT_INTERVAL_MS": "5000",
    "FARLIB_SIMPLE_DIRTY_OBSERVE": "0",
    "FARLIB_LEGACY_SCAN_CURSORS": "1",
    "FARLIB_PROFILED_BACKUP_NONBLOCKING_ON_FULL": "1",
    "FARLIB_PLANNER_BUDGET_EARLY_EXIT": "1",
    "FARLIB_SCOPE_COUNTER_SHARDS": "1",
    "FARLIB_RESIDENT_PROFILE_REQUIRE_WORK_PHASE": "1",
}

# Hydra's page EC path has its own bounded write/reclaim controls.  Keep these
# in the recorded runtime profile instead of inheriting stale shell settings.
# The values match the source-side defaults/profile used by the Figure 9 Hydra
# builds: fixed 64-page reclaim/write batches, local polling, and registered
# zero-copy parity preparation.
HYDRA_ENV = {
    "FARLIB_FIBRE_HEAP": "1",
    "FARLIB_ALLOC_SCOPE_CHECKPOINT": "1",
    "FARLIB_ASYNC_FULL_RETURNS": "1",
    "FARLIB_THREAD_HEAP_FULL_RETURN_BATCH": "1",
    "FARLIB_REGION_LIST_YIELD_LOCK": "1",
    "FARLIB_REMOTE_USAGE_SHARDS": "1",
    "FARLIB_LEGACY_SCAN_CURSORS": "1",
    "FARLIB_SCOPE_COUNTER_SHARDS": "1",
    "FARLIB_LOCK_WAIT_SCOPE_YIELD": "1",
    "FARLIB_HYDRA_LOCAL_POLL": "1",
    "FARLIB_HYDRA_RECLAIM_BATCH": "64",
    "FARLIB_HYDRA_RECLAIM_NOTIFY": "0",
    "FARLIB_HYDRA_ZERO_COPY_WRITE": "1",
    "FARLIB_HYDRA_FAST_WRITE_PREPARE": "1",
    "FARLIB_HYDRA_WRITE_BATCH": "64",
}

CARBINK_ENV = {
    "FARLIB_OPT_READY_QUEUE": "0",
    "FARLIB_OPT_SERIALIZE_MARK_EVICT": "0",
    "FARLIB_OPT_FULL_PRIME_MARK": "1",
    "FARLIB_LEGACY_SCAN_CURSORS": "1",
    "FARLIB_CARBINK_COMPACTION_SCAN": "1",
    "FARLIB_CARBINK_PARITY_FENCE": "1",
    "FARLIB_HYDRA_LOCAL_POLL": "1",
}


def client_command(app: str, binary: Path, config: Path, input_path: Path | None,
                   ae_root: Path, tokenizer: Path | None = None,
                   local_bytes: int | None = None
                   ) -> tuple[list[str], Path | None]:
    if app == "llama":
        if tokenizer is None:
            raise ValueError("LLaMA requires an explicit tokenizer path")
        return [str(binary), str(config), str(input_path), "-m", "chat",
                "-z", str(tokenizer)], (
            ae_root / "apps/llama/llama_user_chat.txt"
        )
    if app == "bfs":
        return [str(binary), str(config), str(input_path)], None
    if app == "mg":
        return [str(binary), str(config)], None
    if app == "wordcount":
        return [str(binary), str(config), str(input_path), "24", "8053063696", "10"], None
    if app == "nq":
        return [str(binary), str(config), str(input_path)], None
    if app in ("kv-b", "kv-a", "kv-s"):
        if local_bytes is None or local_bytes <= 0:
            raise ValueError("KV requires its rendered local-memory capacity")
        return [str(binary), str(config), format(local_bytes / 1024**3, ".17g")], None
    raise ValueError(f"unsupported application: {app}")


def client_environment(app: str, site: dict, system: str = "nonft") -> dict[str, str]:
    if system not in ("nonft", "starfish", "hydra", "carbink"):
        raise ValueError(f"unsupported system: {system}")
    env = {
        "FibreWorkerCount": str(site.get("fibre_workers", 24)),
        "FARLIB_SEPARATE_BACKGROUND_CLUSTER": "1",
        "FARLIB_PIN_RDMA_THREAD": "0",
        "FARLIB_RDMA_DEVICE": str(site["ib_device"]),
        "FARLIB_RDMA_READ_BATCH": "1",
        "FARLIB_EXCLUSIVE_OWNED_BATCH": "1",
        "FARLIB_OPT_LEGACY_EXCLUSIVE_PIPELINE": "1",
    }
    # CPU placement is site-specific.  Keep the fields optional here so dry
    # runs and profile rendering can describe an incomplete site; the actual
    # runner validates that a complete placement is present before launch.
    if site.get("fibre_cpu_set") is not None:
        env["FibreCpuSet"] = str(site["fibre_cpu_set"])
    if site.get("background_cpu_base") is not None:
        env["FARLIB_BACKGROUND_CPU_BASE"] = str(site["background_cpu_base"])
    if app == "bfs":
        repetitions = site.get("bfs_repetitions", 1)
        if (isinstance(repetitions, bool) or not isinstance(repetitions, int)
                or repetitions < 1):
            raise ValueError("BFS repetitions must be a positive integer")
        env.update({
            "GAPBS_GRAPH_GENERATOR": "file",
            "GAPBS_BFS_SOURCE": "1",
            "GAPBS_BFS_WORKERS": str(site.get("bfs_workers", 48)),
            "GAPBS_BFS_REPETITIONS": str(site.get("bfs_repetitions", 1)),
            "GAPBS_BFS_VERIFY": "1",
            "GAPBS_BFS_OPTIMIZE": "1",
            "GAPBS_CHUNK_PREFETCH_DISTANCE": "1",
            "GAPBS_BFS_PHASE_REPLAN": "0",
            "GAPBS_BFS_ALPHA": "15",
            "GAPBS_BFS_BETA": "18",
            "HOTNESS_MAX": "3",
        })
    elif app == "mg":
        env.update({
            "MG_DESIGN1_PHASE_REPLAN": "1",
            "FARLIB_RESIDENT_PROFILE_PHASE_TRIGGERED": "1",
            "FARLIB_FIBRE_HEAP": "1",
            "FARLIB_ALLOC_SCOPE_CHECKPOINT": "1",
        })
        phase_metrics = site.get("mg_evac_phase_metrics", False)
        if not isinstance(phase_metrics, bool):
            raise ValueError("mg_evac_phase_metrics must be a boolean")
        if phase_metrics:
            env["FARLIB_EVAC_PHASE_METRICS"] = "1"
        if "mg_evict_breakdown" in site:
            breakdown = site["mg_evict_breakdown"]
            if not isinstance(breakdown, bool):
                raise ValueError("mg_evict_breakdown must be a boolean")
            env["FARLIB_EVAC_BREAKDOWN"] = "1" if breakdown else "0"
        # MG-only, explicit site overrides; no shared recipe defaults.
        if system == "starfish":
            env.update({
                "FARLIB_EC_SPLIT_DIRECT_READ": "1",
                "FARLIB_EC_SPLIT_DIRECT_WRITE": "1",
                "FARLIB_EC_SPLIT_WRITE_BATCH": "32",
                "FARLIB_EC_SPLIT_GROUP_BATCH": "8",
            })
        for suffix in ("DIRECT_READ", "DIRECT_WRITE", "GROUP_BATCH", "WRITE_BATCH"):
            key = "mg_split_" + suffix.lower()
            if key not in site:
                continue
            value = site[key]
            if system != "starfish":
                raise ValueError(f"{key} is only supported by Starfish")
            if suffix.startswith("DIRECT_"):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a boolean")
                value = int(value)
            else:
                maximum = 8 if suffix == "GROUP_BATCH" else 32
                if type(value) is not int or value not in (0, maximum):
                    raise ValueError(f"{key} must be 0 or {maximum}")
            env["FARLIB_EC_SPLIT_" + suffix] = str(value)
    elif app == "wordcount":
        env.update({
            "FARLIB_WORDCOUNT_GLOBAL_MAP_SHIFT": "25",
            "FARLIB_WORDCOUNT_LOCAL_MAP_SHIFT": "21",
            "FARLIB_WORDCOUNT_KEEP_LOCAL_FAR_MAPS": "1",
            "FARLIB_WORDCOUNT_MERGE_WORKERS": "24",
            "FARLIB_WORDCOUNT_REDUCE_READERS": "24",
            "FARLIB_WORDCOUNT_UTHREADS": "48",
        })
    elif app == "nq":
        env.update({"NHOP_RANDOM_NUM": "1600", "NHOP_UTHREAD_FACTOR": "2",
                    "NHOP_PRINT_SIZE_DISTRIBUTION": "0", "NHOP_WORK_PROGRESS": "1"})
    elif app in ("kv-b", "kv-a", "kv-s"):
        env.update({
            "FARLIB_KVS_INITIAL_DATA_COUNT": "33554432",
            "FARLIB_KVS_RANDOM_SEED": "20260917",
            "FARLIB_KVS_FIXED_REQUEST_COUNT": "1",
            "FARLIB_KVS_DRAIN_ALL_REQUESTS": "1",
            "FARLIB_KVS_DRAIN_TIMEOUT_MS": "60000",
            "FARLIB_KVS_MAX_RUNTIME_MS": "3600000",
            "FARLIB_KVS_MAX_SERVE_COUNT": "1000000000",
            "FARLIB_KVS_PUT_RATIO": {"kv-b": "0.05", "kv-a": "0.5", "kv-s": "0.95"}[app],
            "FARLIB_KVS_REMOVE_RATIO": "0",
            "FARLIB_KVS_ZIPFIAN_CONSTANT": "0.99",
            "FARLIB_KVS_MUTATING_VALUES": "1",
            "FARLIB_KVS_POST_VERIFY_SAMPLES": "4096",
            "FARLIB_KVS_RUN_ASYNC": "0",
            "FARLIB_KVS_SERVER_MODE": "sync",
            "FARLIB_KVS_EXECUTION_MODE": "direct",
            "FARLIB_KVS_DIRECT_FIBRES": "48",
            "FARLIB_KVS_DIRECT_MISS_YIELD": "1",
            "FARLIB_KVS_DEBUG_LOCKED_GET": "1",
            "FARLIB_KVS_VALIDATE_GET": "0",
            "FARLIB_KVS_HIST_SAMPLE_PERIOD": "1024",
            "FARLIB_HASHMAP_DEREF_STATS": "0",
        })
    elif app != "llama":
        raise ValueError(f"unsupported application: {app}")
    if system == "nonft":
        # The concurrent legacy pipeline requires phase-specific scan eligibility.
        env.update(NONFT_ENV)
        if app in ("kv-b", "kv-a", "kv-s"):
            env.update({"FARLIB_SCOPE_COUNTER_SHARDS": "1",
                        "FARLIB_LOCK_WAIT_SCOPE_YIELD": "1"})
    elif system == "starfish":
        env.update(DESIGN2_ENV)
        if app in ("mg", "wordcount", "nq", "kv-b", "kv-a", "kv-s"):
            env.update({
                "FARLIB_FIBRE_HEAP": "1",
                "FARLIB_ALLOC_SCOPE_CHECKPOINT": "1",
                "FARLIB_ASYNC_FULL_RETURNS": "1",
                "FARLIB_THREAD_HEAP_FULL_RETURN_BATCH": "1",
                "FARLIB_REGION_LIST_YIELD_LOCK": "1",
                "FARLIB_REMOTE_USAGE_SHARDS": "1",
                "FARLIB_LOCK_WAIT_SCOPE_YIELD": "1",
            })
        if app == "wordcount":
            env.update(FARLIB_WORDCOUNT_RECIPES="1",
                       FARLIB_WORDCOUNT_RECOMPUTABLE="0")
        if app in ("kv-b", "kv-a", "kv-s"):
            env["FARLIB_EC_BENCHMARK_PHASED"] = "1"
            policy = site.get("starfish_kv_backup_policy", {
                "backup_credits": True, "all_nonresident_backup": True,
            })
            if policy is not None:
                fields = {"backup_credits", "all_nonresident_backup"}
                if (not isinstance(policy, dict) or set(policy) != fields
                        or any(type(value) is not bool for value in policy.values())):
                    raise ValueError(
                        "starfish_kv_backup_policy requires backup_credits and "
                        "all_nonresident_backup booleans")
                env.update({
                    "FARLIB_EC_BEHAVIOR_ROUTING": "1",
                    "FARLIB_BACKUP_COUNTER_BYPASS": "0",
                    "FARLIB_BACKUP_CREDITS": str(int(policy["backup_credits"])),
                    "FARLIB_ALL_NONRESIDENT_BACKUP": str(int(policy["all_nonresident_backup"])),
                })
    elif system == "hydra":
        env.update(HYDRA_ENV)
        if app == "mg":
            env.update(MG_DESIGN1_PHASE_REPLAN="0",
                       FARLIB_RESIDENT_PROFILE_PHASE_TRIGGERED="0")
        elif app == "wordcount":
            env.update(FARLIB_WORDCOUNT_RECOMPUTABLE="0",
                       FARLIB_WORDCOUNT_RECIPES="0")
        elif app in ("kv-b", "kv-a", "kv-s"):
            env["FARLIB_HYDRA_RUNTIME_HOT_PACKING"] = "1"
    elif system == "carbink":
        env.update(CARBINK_ENV)
        if app == "mg":
            env.update(FARLIB_FIBRE_HEAP="0", FARLIB_ASYNC_FULL_RETURNS="0",
                       MG_DESIGN1_PHASE_REPLAN="0",
                       FARLIB_RESIDENT_PROFILE_PHASE_TRIGGERED="0")
        elif app in ("wordcount", "nq", "kv-b", "kv-a", "kv-s"):
            env.update({
                "FARLIB_FIBRE_HEAP": "1",
                "FARLIB_ALLOC_SCOPE_CHECKPOINT": "1",
                "FARLIB_ASYNC_FULL_RETURNS": "1",
                "FARLIB_THREAD_HEAP_FULL_RETURN_BATCH": "1",
                "FARLIB_REGION_LIST_YIELD_LOCK": "1",
                "FARLIB_REMOTE_USAGE_SHARDS": "1",
                "FARLIB_SCOPE_COUNTER_SHARDS": "1",
                "FARLIB_LOCK_WAIT_SCOPE_YIELD": "1",
            })
        if app == "wordcount":
            env.update(FARLIB_WORDCOUNT_RECOMPUTABLE="0",
                       FARLIB_WORDCOUNT_RECIPES="0")
    metadata = site.get("runtime_metadata", False)
    if not isinstance(metadata, bool):
        raise ValueError("runtime_metadata must be a boolean")
    if metadata:
        if system not in ("starfish", "carbink"):
            raise ValueError("runtime_metadata currently supports Starfish and Carbink")
        env["FARLIB_RUNTIME_METADATA"] = "1"
    ec_cpu = site.get("runtime_ec_cpu", False)
    if not isinstance(ec_cpu, bool):
        raise ValueError("runtime_ec_cpu must be a boolean")
    if ec_cpu:
        if system not in ("starfish", "carbink"):
            raise ValueError("runtime_ec_cpu currently supports Starfish and Carbink")
        env["FARLIB_RUNTIME_EC_CPU"] = "1"
    # One runtime timer applies to every benchmark, independent of request
    # counts and Work-phase boundaries. Reject the incompatible legacy opt-in.
    if site.get("kv_memory_samples", False):
        raise ValueError(
            "kv_memory_samples was replaced by remote_memory_samples "
            "and remote_memory_observer_cpu (benchmark time sampling)")
    memory_samples = site.get("remote_memory_samples", False)
    if not isinstance(memory_samples, bool):
        raise ValueError("remote_memory_samples must be a boolean")
    if memory_samples:
        observer_cpu = site.get("remote_memory_observer_cpu")
        if (isinstance(observer_cpu, bool) or
                not isinstance(observer_cpu, (int, str)) or
                not re.fullmatch(r"[0-9]+", str(observer_cpu))):
            raise ValueError(
                "remote_memory_observer_cpu must be a non-negative integer CPU index")
        env["FARLIB_REMOTE_MEMORY_SAMPLES"] = "1"
        env["FARLIB_REMOTE_MEMORY_OBSERVER_CPU"] = str(int(observer_cpu))
        env["FARLIB_REMOTE_MEMORY_SAMPLING_PROFILE"] = (
            "bfs_work_3s" if app == "bfs" else "work_10_20_30_40_50s")
        origin = site.get("remote_memory_work_origin",
                          "kvs_request_start" if app in ("kv-b", "kv-a", "kv-s")
                          else "profile_start_work")
        if origin not in ("profile_start_work", "kvs_request_start"):
            raise ValueError("unsupported remote-memory Work origin")
        if origin == "kvs_request_start" and app not in ("kv-b", "kv-a", "kv-s"):
            raise ValueError("kvs_request_start is only valid for KV workloads")
        env["FARLIB_REMOTE_MEMORY_WORK_ORIGIN"] = origin
    return env


def client_process_environment(app: str, site: dict, system: str,
                               inherited: dict[str, str]) -> dict[str, str]:
    # A parent shell's runtime, Fibre, or BFS flags must not leak into a run or
    # override the recorded profile. Keep unrelated host/library settings.
    env = {
        key: value for key, value in inherited.items()
        if not key.startswith(CONTROLLED_ENV_PREFIXES)
    }
    env.update(client_environment(app, site, system))
    return env


def validate_design2(log: str, *, expected_bfs_repetitions: int | None = None) -> dict:
    """Require runtime evidence of the selected grouping mode, not only flags."""
    required = (
        r"^simple_hotcold\.config\b[^\n]*\benabled=1\b[^\n]*\bmode=local_resident\b",
        r"^simple_remote_hotcold\.config\b[^\n]*\benabled=1\b",
        r"^simple_region_budget\.config\b[^\n]*\bsemantic_classes=6\b[^\n]*\blocal_resident=1\b",
    )
    if any(not re.search(pattern, log, re.MULTILINE) for pattern in required):
        raise ValueError("Design 2 resident-local six-group profile was not active")
    if expected_bfs_repetitions is not None:
        if (isinstance(expected_bfs_repetitions, bool)
                or not isinstance(expected_bfs_repetitions, int)
                or expected_bfs_repetitions < 1):
            raise ValueError("Design 2 BFS repetitions must be positive")
        starts = list(re.finditer(
            r"^gapbs_bfs_iteration_begin\b[^\n]*$", log, re.MULTILINE))
        ends = list(re.finditer(
            r"^gapbs_bfs_phase_stats\b[^\n]*\bphase=work\b[^\n]*$", log, re.MULTILINE))
        expected = [str(i) for i in range(expected_bfs_repetitions)]
        if ([_fields(m.group(0)).get("iteration") for m in starts] != expected
                or [_fields(m.group(0)).get("iteration") for m in ends] != expected):
            raise ValueError("Design 2 BFS Work boundaries differ from requested repetitions")
        spans = []
        for i, (start, end) in enumerate(zip(starts, ends)):
            if not start.end() < end.start():
                raise ValueError("Design 2 BFS Work boundaries are out of order")
            if i and starts[i].start() <= ends[i - 1].end():
                raise ValueError("Design 2 BFS Work profiles overlap")
            spans.append((start.end(), end.start()))
        ledger_lines = list(re.finditer(
            r"^simple_region_budget\.six_final\b[^\n]*$", log, re.MULTILINE))
        if any(not any(lo <= m.start() < hi for lo, hi in spans)
               for m in ledger_lines):
            raise ValueError("Design 2 group ledger falls outside a BFS Work profile")
        configs = "\n".join(re.search(pattern, log, re.MULTILINE).group(0)
                            for pattern in required)
        profiles = [validate_design2(configs + "\n" + log[lo:hi]) for lo, hi in spans]
        return {**profiles[-1], "work_profiles": len(profiles),
                "per_work_profile": profiles,
                "evidence_scope": "active_profile_and_complete_per_iteration_supply_ledgers"}
    # Validate one complete snapshot, never splice it with another interval.
    latest: dict[tuple[str, int, int], int] = {}
    for match in re.finditer(r"^simple_region_budget\.six_final\b[^\n]*$", log, re.MULTILINE):
        fields = _fields(match.group(0))
        try:
            domain, size_bin, cls = fields["domain"], int(fields["bin"]), int(fields["class"])
            actual = int(fields["actual"])
        except (KeyError, ValueError) as exc:
            raise ValueError("malformed Design 2 final group ledger") from exc
        if domain not in ("local", "remote") or not 0 <= cls < 6 or size_bin < 0 or actual < 0:
            raise ValueError("invalid Design 2 final group ledger")
        key = (domain, size_bin, cls)
        if key in latest:
            raise ValueError("Design 2 runner requires one complete Work-profile ledger")
        latest[key] = actual
    bins = {(domain, size_bin) for domain, size_bin, _ in latest}
    if not bins or "local" not in {domain for domain, _ in bins}:
        raise ValueError("Design 2 final local group ledger is missing")
    if any({cls for d, b, cls in latest if (d, b) == pair} != set(range(6)) for pair in bins):
        raise ValueError("Design 2 final group ledger is incomplete")
    local_regions = sum(value for (domain, _, _), value in latest.items() if domain == "local")
    if local_regions == 0:
        raise ValueError("Design 2 final ledger contains no local Regions")
    return {"profile": "resident_local_six", "remote_heat": "measured",
            "local_regions": local_regions,
            "remote_regions": sum(value for (domain, _, _), value in latest.items()
                                  if domain == "remote"),
            "evidence_scope": "active_profile_and_final_supply_ledger"}


def validate_hydra(log: str, *, endpoint_count: int,
                   expected_workers: dict[str, int] | None = None,
                   expected_resident_bytes: int | None = None) -> dict:
    """Validate Hydra's page-EC profile and cleanup evidence.

    Hydra is not the Starfish Design 2 profile and does not emit the Starfish
    recovery validator's ``ec_read_recovery`` contract.  The source-side
    markers below are the independent Hydra contract: page packing/layout,
    no retained-backup or behavior-group policy, fixed EC write buffers,
    application/background evacuation split, and final page/allocator cleanup.
    """
    if isinstance(endpoint_count, bool) or not isinstance(endpoint_count, int):
        raise ValueError("Hydra endpoint_count must be an integer")
    if endpoint_count < 6:
        raise ValueError("Hydra page EC requires at least six endpoints")

    runtime = _one(
        log,
        r"^FT runtime: ft_method=(hydra|ec_split) RS\(4,2\), "
        r"six single-sided segment writes, client-side recovery\s*$",
        "Hydra page-EC runtime",
    )
    layout = _one(
        log,
        r"^hydra\.layout page_bytes=(\d+) data_shards=(\d+) "
        r"parity_shards=(\d+) shard_bytes=(\d+) "
        r"max_packed_object_bytes=(\d+)\s*$",
        "Hydra page layout",
    )
    layout_values = tuple(int(value) for value in layout.groups())
    if layout_values != (8192, 4, 2, 2048, 8192):
        raise ValueError("Hydra page layout differs from the implemented 8KiB RS(4,2) path")

    policy = _one(
        log,
        r"^hydra\.policy backup=(\d+) behavior_groups=(\d+) "
        r"resident_budget_bytes=(\d+) codec=([^\s]+) rs=(\d+\+\d+) "
        r"encode_tables=([^\s]+)\s*$",
        "Hydra policy",
    )
    backup, groups, resident_budget, codec, rs, encode_tables = policy.groups()
    if (int(backup), int(groups), codec, rs, encode_tables) != (
            0, 0, "isa-l", "4+2", "process_once"):
        raise ValueError("Hydra policy is not backup-free ISA-L RS(4,2) page protection")
    if int(resident_budget) < 0:
        raise ValueError("Hydra resident budget must not be negative")

    if expected_resident_bytes is not None:
        if (isinstance(expected_resident_bytes, bool)
                or not isinstance(expected_resident_bytes, int)
                or expected_resident_bytes < 0):
            raise ValueError("Hydra expected_resident_bytes must be nonnegative")
        if int(resident_budget) != expected_resident_bytes:
            raise ValueError(
                "Hydra resident budget diagnostic does not match effective config")

    workers = None
    if expected_workers is not None:
        required_worker_keys = (
            "app_workers", "background_workers",
            "background_mark_workers", "background_evict_workers",
        )
        if (not isinstance(expected_workers, dict)
                or any(key not in expected_workers for key in required_worker_keys)):
            raise ValueError("Hydra expected_workers lacks a required role count")
        if any(isinstance(expected_workers[key], bool)
               or not isinstance(expected_workers[key], int)
               or expected_workers[key] <= 0 for key in required_worker_keys):
            raise ValueError("Hydra expected worker counts must be positive integers")
        try:
            workers = {key: int(expected_workers[key])
                       for key in required_worker_keys}
        except (TypeError, ValueError) as exc:
            raise ValueError("Hydra expected_workers contains a non-integer") from exc
        if (any(value < 0 for value in workers.values())
                or workers["background_mark_workers"]
                + workers["background_evict_workers"]
                != workers["background_workers"]):
            raise ValueError("Hydra expected worker role counts are inconsistent")

    buffers = _one(
        log,
        r"^hydra_evict_buffers\b[^\n]*$",
        "Hydra Evict buffers",
    )
    buffer_fields = _fields(buffers.group(0))
    try:
        owners = int(buffer_fields["owners"])
        pages_per_worker = int(buffer_fields["pages_per_worker"])
        write_batch = int(buffer_fields["write_batch"])
        parity_bytes = int(buffer_fields["parity_bytes"])
    except (KeyError, ValueError) as exc:
        raise ValueError("malformed Hydra Evict buffer diagnostic") from exc
    if pages_per_worker != 64 or write_batch != 64:
        raise ValueError("Hydra Evict buffer page/batch geometry differs")
    bank_contract = {"banks_per_worker": "2", "pages_per_bank": "32",
                     "reuse": "whole_bank", "slot_scan": "0",
                     "registrations_per_worker": "1", "bank_flush_pages": "32"}
    if any(buffer_fields.get(key) != value for key, value in bank_contract.items()):
        raise ValueError("Hydra Evict buffers differ from the retained A32/B32 layout")
    if workers is not None:
        if owners != workers["background_workers"]:
            raise ValueError("Hydra Evict owner count differs from effective workers")
        if parity_bytes != owners * 256 * 1024:
            raise ValueError("Hydra parity-buffer bytes differ from owner geometry")

    # Concurrent allocator startup can interrupt this multi-insertion line.
    evacuator_log = re.sub(
        r"allocator\.fibre_thread_heap_enabled=[01][ \t]*\r?\n", "", log)
    evacuator = _one(
        evacuator_log,
        r"\bruntime\.exclusive_owned_batch=(\d+)\b[^\n]*"
        r"\bmark_workers=(\d+)\b[^\n]*\bevict_workers=(\d+)\b[^\n]*$",
        "Hydra evacuator workers",
    )
    exclusive_owned, mark_workers, evict_workers = (
        int(value) for value in evacuator.groups())
    if exclusive_owned not in (0, 1):
        raise ValueError("Hydra exclusive-owned-batch marker is invalid")
    if workers is not None and (
            mark_workers != workers["background_mark_workers"]
            or evict_workers != workers["background_evict_workers"]):
        raise ValueError("Hydra evacuator role markers differ from effective workers")

    reclaim = _one(
        log,
        r"^hydra\.reclaim batch_regions=(\d+) early_notify=(\d+)\s*$",
        "Hydra reclaim profile",
    )
    if tuple(int(value) for value in reclaim.groups()) != (64, 0):
        raise ValueError("Hydra reclaim profile differs from the bounded page path")

    page_matches = list(re.finditer(
        r"^hydra\.pages created=(\d+) released=(\d+) "
        r"objects_created=(\d+) objects_released=(\d+) "
        r"page_bytes=(\d+) data_shard_bytes=(\d+)\s*$",
        log, re.MULTILINE))
    if len(page_matches) > 1:
        raise ValueError("Hydra emitted multiple page cleanup diagnostics")
    if not page_matches and re.search(r"^hydra\.pages\b", log, re.MULTILINE):
        raise ValueError("malformed Hydra page cleanup diagnostic")
    created = released = objects_created = objects_released = None
    if page_matches:
        page_values = tuple(int(value) for value in page_matches[0].groups())
        (created, released, objects_created, objects_released,
         page_bytes, shard_bytes) = page_values
        if (released != created or page_bytes != 8192 or shard_bytes != 2048
                or objects_created < 0 or objects_released < 0):
            raise ValueError(
                "Hydra page allocation was not fully released or has invalid layout")
    _one(log, r"^exact used bytes:\s*0\s*$", "Hydra allocator cleanup")
    return {
        "status": "passed",
        "profile": "page_ec_rs4+2",
        "ft_method": runtime.group(1),
        "endpoint_count": endpoint_count,
        "resident_budget_bytes": int(resident_budget),
        "workers": workers,
        "evacuator_workers": {"mark": mark_workers, "evict": evict_workers,
                              "exclusive_owned_batch": exclusive_owned},
        "pages_created": created,
        "pages_released": released,
        "objects_created": objects_created,
        "objects_released": objects_released,
        "evidence_scope": "hydra_runtime_policy_layout_workers_and_cleanup",
    }

def _positive(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _one(log: str, pattern: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, log, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"{label}: expected exactly one marker, found {len(matches)}")
    return matches[0]


def _fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def _count(fields: dict[str, str], name: str) -> int:
    value = int(fields[name])
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STARTUP_MARKER = "kvs_lock_wait_scope_yield"
_STARTUP_NORMAL = re.compile(
    r"^(?:1)?" + re.escape(_STARTUP_MARKER)
    + r"[ \t]+enabled=([0-9]+)"
    + r"(?:[ \t]+mark_workers=[0-9]+[ \t]+evict_workers=[0-9]+)?"
    + r"[ \t]*$")
_STARTUP_SPLIT = re.compile(
    r"^(?:1)?" + re.escape(_STARTUP_MARKER)
    + r"[ \t]+enabled=runtime\.exclusive_owned_batch=1[ \t]*$")
_STARTUP_SPLIT_CONTINUATION = re.compile(
    r"^1[ \t]+mark_workers=[0-9]+[ \t]+evict_workers=[0-9]+[ \t]*$")


def _localized_startup_line(line: str) -> str:
    """Remove only CSI escapes from one startup-diagnostic line."""
    return _ANSI_CSI.sub("", line).strip()


def _validate_nonft_kv_startup(log: str) -> None:
    """Require one unambiguous enabled=1 startup diagnostic.

    The runtime's stdout can prefix the marker with one numeric character or
    split the known exclusive-owned-batch record across two lines. Treat every
    line containing the marker as evidence: malformed, disabled, duplicate,
    or contradictory records must not be hidden by a later valid record.
    """
    evidence = 0
    disabled = 0
    malformed = 0
    lines = log.splitlines()
    index = 0
    while index < len(lines):
        line = _localized_startup_line(lines[index])
        if _STARTUP_MARKER not in line:
            index += 1
            continue
        normal = _STARTUP_NORMAL.fullmatch(line)
        if normal is not None:
            evidence += 1
            if normal.group(1) != "1":
                disabled += 1
            index += 1
            continue
        split = _STARTUP_SPLIT.fullmatch(line)
        if split is not None:
            continuation = (_localized_startup_line(lines[index + 1])
                            if index + 1 < len(lines) else "")
            if _STARTUP_SPLIT_CONTINUATION.fullmatch(continuation) is None:
                malformed += 1
                index += 1
                continue
            evidence += 1
            index += 2
            continue
        malformed += 1
        index += 1

    label = "Non-FT KV lock-wait scope handoff"
    if malformed:
        raise ValueError(f"{label}: malformed marker")
    if disabled:
        raise ValueError(f"{label}: enabled value must be exactly 1")
    if evidence != 1:
        raise ValueError(f"{label}: expected exactly one marker, found {evidence}")


def validate_nonft_kv(log: str, *, expected_workers: int) -> dict:
    # Only an enabled runtime prints the shutdown scope inventory. This also
    # tolerates startup messages interleaved with the background worker log.
    end = _one(log, r"^kvs_scope_shards_end registered=(\d+) v0=(\d+) v1=(\d+)\s*$",
               "Non-FT KV scope cleanup")
    registered, v0, v1 = map(int, end.groups())
    if registered != expected_workers or v0 != 0 or v1 != 0:
        raise ValueError("Non-FT KV scope inventory or cleanup differs")
    _validate_nonft_kv_startup(log)
    if re.search(r"\bkvs_scope_shards\b[^\n]*\bsupported=0\b", log):
        raise ValueError("Non-FT KV binary lacks scope-counter sharding")
    return {"status": "passed", "scope_counter_shards": True,
            "lock_wait_scope_yield": True, "registered_shards": registered,
            "scope_v0": v0, "scope_v1": v1}


def _parse_mg(log: str) -> dict:
    _one(log, r"^\s*Size:\s*1024x1024x1024 \(class_npb U\)\s*$", "MG size")
    _one(log, r"^\s*Iterations:\s*5\s*$", "MG iterations")
    first = _one(log, r"^\s*iter\s+1\s*$", "MG first iteration")
    last = _one(log, r"^\s*iter\s+5\s*$", "MG final iteration")
    perf = _one(log, r"^\s*perf result\s*$", "MG perf result")
    runtime = _one(log, r"^\s*runtime:\s*(" + NUMBER + r")\s+ms\s*$",
                   "MG native work time")
    completed = _one(log, r"^\s*Benchmark completed\s*$", "MG completion")
    _one(log, r"^\s*MG Benchmark Completed\s*$", "MG NPB completion")
    residual = _one(log, r"^FINAL_RESIDUAL rnm2=(" + NUMBER +
                    r") rnmu=(" + NUMBER + r")\s*$", "MG residual")
    if not (first.start() < last.start() < perf.start() < runtime.start()
            < completed.start() < residual.start()):
        raise ValueError("MG work/completion markers are out of order")
    # NPB class U reports 'NO VERIFICATION PERFORMED'; use the independently
    # recorded residual for this fixed 1024^3, five-iteration workload.
    oracle = (1.53587137116806717e-5, 9.88992103639334819e-2)
    for observed, expected in zip(residual.groups(), oracle):
        if not math.isclose(float(observed), expected, rel_tol=1e-7, abs_tol=0):
            raise ValueError("MG residual differs from the known workload")
    return {
        "elapsed_s": _positive(runtime.group(1), "MG native work time") / 1000,
        "measurement_phase": "mg_5_iterations_work",
        "measurement_method": "application native perf runtime (ms), excluding initialization",
        "correctness_scope": "fixed-workload final residual; NPB class U has no built-in verification",
        "correctness_evidence": residual.group(0),
    }


def _wc_result_with_known_interleave(log: str, result: re.Match[str]) -> tuple[int, str]:
    """Rejoin one known sampler-interleaved WC result without consuming logs."""
    prefix = result.group(0).rstrip("\r")
    fields = _fields(prefix)
    if all(name in fields for name in (
            "file", "bytes", "threads", "total_words", "unique_words",
            "hashmap_size", "checksum")):
        return result.start(), prefix

    lines = log.splitlines(keepends=True)
    line_index = log.count("\n", 0, result.start())
    if line_index >= len(lines):
        return result.start(), prefix
    if lines[line_index].rstrip("\r\n") != prefix:
        return result.start(), prefix

    def is_known_sampler(line: str) -> bool:
        tokens = line.split()
        if not tokens:
            return False
        marker = tokens[0]
        if marker not in (
                "runtime_remote_memory", "runtime_remote_memory_missing",
                "runtime_remote_memory_end"):
            return False
        fields = {}
        for token in tokens[1:]:
            key, separator, value = token.partition("=")
            if (not separator or not value
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                    or key in fields):
                return False
            fields[key] = value
        schema = fields.get("schema_version")
        required_fields = {
            ("runtime_remote_memory", "2"):
                "schema_version sample scheduled_elapsed_s observed_elapsed_s "
                "occupied_bytes metric source start_monotonic_ns window "
                "snapshot_start_monotonic_ns snapshot_ns endpoint_count "
                "endpoint_bytes consistency".split(),
            ("runtime_remote_memory", "3"):
                "schema_version sample work_window_id origin scheduled_elapsed_s "
                "observed_elapsed_s occupied_bytes metric source start_monotonic_ns "
                "window sampling snapshot_start_monotonic_ns snapshot_ns "
                "endpoint_count endpoint_bytes consistency".split(),
            ("runtime_remote_memory_missing", "2"):
                "schema_version sample scheduled_elapsed_s observed_elapsed_s "
                "status metric source start_monotonic_ns window".split(),
            ("runtime_remote_memory_missing", "3"):
                "schema_version sample work_window_id origin scheduled_elapsed_s "
                "observed_elapsed_s status metric source start_monotonic_ns "
                "window sampling".split(),
            ("runtime_remote_memory_end", "2"):
                "schema_version samples sampling metric source start_monotonic_ns "
                "expected_samples missed_notready missing".split(),
            ("runtime_remote_memory_end", "3"):
                "schema_version work_window_id origin end_monotonic_ns samples "
                "scheduled_samples missing sampling window metric source "
                "start_monotonic_ns expected_samples missed_notready status".split(),
        }
        required = required_fields.get((marker, schema))
        if required is None:
            return False
        # Unknown additive key=value fields remain allowed for forward
        # compatibility; required fields and duplicate keys stay strict.
        return set(required) <= fields.keys()
    end = line_index + 1
    sampler_records = 0
    while end < len(lines):
        line = lines[end].strip()
        if not line:
            end += 1
            continue
        if is_known_sampler(line):
            sampler_records += 1
            end += 1
            continue
        break
    if sampler_records == 0 or end >= len(lines):
        return result.start(), prefix

    prefix_complete = re.fullmatch(
        r"wordcount_far_result file=\S+ bytes=\d+ threads=\d+ "
        r"total_words=\d+ unique_words=\d+ hashmap_size=\d+",
        prefix)
    prefix_unique_split = re.fullmatch(
        r"wordcount_far_result file=\S+ bytes=\d+ threads=\d+ "
        r"total_words=\d+ unique_words=",
        prefix)
    suffix = lines[end].strip()
    suffix_checksum = re.compile(
        r"checksum=0x[0-9a-fA-F]+ too_long_words=\d+ "
        r"max_word_len=\d+ remote_used_bytes=\d+$")
    suffix_unique = re.compile(
        r"\d+ hashmap_size=\d+ checksum=0x[0-9a-fA-F]+ "
        r"too_long_words=\d+ max_word_len=\d+ remote_used_bytes=\d+$")
    suffix_candidates = []
    for line in lines:
        candidate = line.strip()
        if suffix_checksum.fullmatch(candidate):
            suffix_candidates.append(("checksum", candidate))
        if suffix_unique.fullmatch(candidate):
            suffix_candidates.append(("unique", candidate))
    if len(suffix_candidates) != 1:
        raise ValueError("WC result: expected exactly one complete suffix")
    suffix_kind, suffix_candidate = suffix_candidates[0]
    if suffix_candidate != suffix:
        return result.start(), prefix
    if prefix_complete and suffix_kind == "checksum":
        return result.start(), prefix + " " + suffix
    if prefix_unique_split and suffix_kind == "unique":
        return result.start(), prefix + suffix
    return result.start(), prefix


def _parse_wc(log: str, expected_checksum: str | None) -> dict:
    start = _one(log, r"^wordcount_far_work_phase_start phase=map_local\s*$",
                 "WC work start")
    end = _one(log, r"^wordcount_far_work_phase_end phase=reduce_checksum\s*$",
               "WC work end")
    result = _one(log, r"^wordcount_far_result\b[^\n]*$", "WC result")
    result_start, result_line = _wc_result_with_known_interleave(log, result)
    if not start.start() < end.start() < result_start:
        raise ValueError("WC work/result markers are out of order")
    phases = {}
    phase_positions = {}
    for name in ("map_local", "merge_global", "reduce_checksum"):
        match = _one(log, rf"^wordcount_far_phase phase={name}\b[^\n]*$",
                     f"WC {name}")
        if not start.start() < match.start() < end.start():
            raise ValueError(f"WC {name} is outside the work phase")
        phases[name] = _fields(match.group(0))
        phase_positions[name] = match.start()
    if not (phase_positions["map_local"] < phase_positions["merge_global"]
            < phase_positions["reduce_checksum"]):
        raise ValueError("WC work phases are out of order")
    result_keys = re.findall(r"(\w+)=", result_line)
    if len(result_keys) != len(set(result_keys)):
        raise ValueError("WC result: duplicate field marker")
    fields = _fields(result_line)
    size = _count(fields, "bytes")
    total = _count(fields, "total_words")
    unique = _count(fields, "unique_words")
    if (size == 0 or unique == 0 or total < unique
            or _count(fields, "hashmap_size") != unique
            or _count(phases["map_local"], "ops") == 0
            or phases["map_local"]["ops"] != phases["merge_global"]["ops"]
            or _count(phases["reduce_checksum"], "ops") != unique):
        raise ValueError("WC incomplete or inconsistent phase/result counts")
    # This oracle is for the *historical* enwiki-small corpus, not the
    # paper's larger Wikipedia input. Other inputs require an explicit oracle.
    known = {8_053_063_696: "0xb32c941b68eb5640"}
    oracle = expected_checksum or known.get(size)
    if oracle is None or fields["checksum"].lower() != oracle.lower():
        raise ValueError("WC needs a matching checksum for this input")
    return {
        "elapsed_s": sum(_positive(phases[name]["elapsed_s"], f"WC {name} time")
                         for name in ("map_local", "merge_global", "reduce_checksum")),
        "measurement_phase": "wc_map_merge_reduce_work",
        "measurement_method": "sum of application phase elapsed_s; excludes native_preagg",
        "correctness_scope": "word count and checksum for the declared input",
        "correctness_evidence": result_line,
        "input_bytes": size,
    }


def _parse_nq(log: str) -> dict:
    config = _one(log, r"^nhop_work_config\b[^\n]*$", "NQ work configuration")
    work = _one(log, r"^nhop_phase_stats phase=work\s+[^\n]*$", "NQ work phase")
    check = _one(log, r"^benchmark_lite_results:\s*(\d+)\s*$", "NQ checksum")
    result = _one(log, r"^nhop_graph_result\b[^\n]*$", "NQ graph result")
    if not config.start() < work.start() < check.start() < result.start():
        raise ValueError("NQ work/result markers are out of order")
    setup = _fields(config.group(0))
    phase = _fields(work.group(0))
    graph = _fields(result.group(0))
    queries = _count(setup, "query_count")
    if (queries != 1_968_240 or _count(phase, "ops") != queries
            or _count(graph, "queries") != queries
            or _count(setup, "queries_per_worker") * _count(setup, "worker_count") != queries
            or _count(graph, "vertices") != 65_608_366
            or _count(graph, "adjacency_lists") != 65_608_367
            or _count(graph, "worker_count") != _count(setup, "worker_count")
            or _count(graph, "random_num") != 1600
            or _count(graph, "result") != int(check.group(1))
            or int(check.group(1)) != 43_264_036_778):
        raise ValueError("NQ query count, graph, or checksum differs from the fixed workload")
    return {
        "elapsed_s": _positive(phase["elapsed_s"], "NQ work time"),
        "measurement_phase": "nhop_query_work",
        "measurement_method": "nhop_phase_stats phase=work elapsed_s (not rounded graph result)",
        "correctness_scope": "fixed Friendster workload query count and checksum",
        "correctness_evidence": result.group(0),
        "request_count": queries,
    }


def _parse_kv(app: str, log: str, expected_requests: int,
              expected_post_samples: int | None = None, *, require_cleanup: bool = True) -> dict:
    ratios = {"kv-b": 0.05, "kv-a": 0.5, "kv-s": 0.95}
    config = _one(log, r"^kvs_direct_config\b[^\n]*$", "KV direct configuration")
    start = _one(log, r"^kvs_phase name=direct event=request_start\b[^\n]*$",
                 "KV request start")
    generated = _one(log, r"^kvs_phase name=direct event=request_generation_end\b[^\n]*$",
                     "KV generation end")
    drained = _one(log, r"^kvs_phase name=direct event=request_drain_end\b[^\n]*$",
                   "KV drain end")
    receipt = _one(log, r"^kvs_receipt name=direct\b[^\n]*$", "KV receipt")
    cleanup = (_one(log, r"^exact used bytes:\s*0\s*$", "KV cleanup")
               if require_cleanup else None)
    if not require_cleanup and expected_post_samples is None:
        raise ValueError("measurement-only KV parsing requires an explicit postcheck count")
    if not config.start() < start.start() < generated.start() < drained.start() < receipt.start():
        raise ValueError("KV request/receipt markers are out of order")
    if cleanup is not None and cleanup.start() <= receipt.start():
        raise ValueError("KV cleanup precedes the request receipt")
    setup, begin, create, finish, counts = map(
        lambda m: _fields(m.group(0)), (config, start, generated, drained, receipt))
    put_ratio = _positive(setup["put_ratio"], "KV put ratio")
    if (not math.isclose(put_ratio, ratios[app], rel_tol=0, abs_tol=1e-9)
            or _count(setup, "object_bytes") != 512
            or _count(setup, "initial_count") == 0
            or len(re.findall(r"^kvs\.zipfian:\s*0\.99\s*$", log, re.MULTILINE)) != 1
            or len(re.findall(r"^kvs\.fixed_request_count:\s*1\s*$", log,
                              re.MULTILINE)) != 1):
        raise ValueError("KV object size, skew, mix, or fixed-count mode differs")
    if not isinstance(expected_requests, int) or expected_requests <= 0:
        raise ValueError("expected_requests must be a positive integer")
    total = _count(create, "generated")
    if (total != expected_requests or total != _count(create, "accepted")
            or total != _count(finish, "completed")
            or counts["accepted_equals_completed"] != "1"):
        raise ValueError("KV request count or completion receipt differs")
    for prefix in ("generated", "accepted", "completed"):
        operation_total = sum(_count(counts, f"{prefix}_{op}")
                              for op in ("get", "put", "remove"))
        if operation_total != total or _count(counts, f"{prefix}_remove") != 0:
            raise ValueError(f"KV {prefix} operation counts do not match")
    for op in ("get", "put", "remove"):
        if not (counts[f"generated_{op}"] == counts[f"accepted_{op}"]
                == counts[f"completed_{op}"]):
            raise ValueError(f"KV {op} request loss")
    if abs(_count(counts, "completed_put") / total - put_ratio) > 0.005:
        raise ValueError("KV completed request mix differs from its configuration")
    # The direct phase excludes KV initialization and later reporting.
    begin_ns = _count(begin, "monotonic_ns")
    generated_ns = _count(create, "monotonic_ns")
    end_ns = _count(finish, "monotonic_ns")
    if not begin_ns < generated_ns <= end_ns:
        raise ValueError("KV request clock markers are invalid")
    verify = re.findall(r"^kvs_post_verify\b[^\n]*$", log, re.MULTILINE)
    if len(verify) > 1:
        raise ValueError("KV has multiple post-verification results")
    if verify:
        evidence = _fields(verify[0])
        if _count(evidence, "checked") == 0 or _count(evidence, "failures") != 0:
            raise ValueError("KV sampled value check failed")
    validation = re.findall(r"^kvs_validation_config\b[^\n]*$", log, re.MULTILINE)
    if len(validation) > 1:
        raise ValueError("KV has multiple validation configurations")
    declared = _fields(validation[0]) if validation else {}
    # Current binaries report only get_content/real_put here; older logs may
    # also declare post_samples. The launcher supplies the required count.
    if "post_samples" in declared and _count(declared, "post_samples") > 0 and not verify:
        raise ValueError("KV declared post-verification has no result")
    if expected_post_samples is not None:
        if (isinstance(expected_post_samples, bool)
                or not isinstance(expected_post_samples, int)
                or expected_post_samples <= 0):
            raise ValueError("expected_post_samples must be a positive integer")
        if not verify or _count(_fields(verify[0]), "checked") != expected_post_samples:
            raise ValueError("KV post-verification count differs from the launcher")
        if (not receipt.start() < log.index(verify[0])
                or (cleanup is not None and log.index(verify[0]) >= cleanup.start())):
            raise ValueError("KV post-verification is outside the completed request/cleanup interval")
    return {
        "elapsed_s": (end_ns - begin_ns) / 1_000_000_000,
        "measurement_phase": "kv_fixed_count_direct_requests",
        "measurement_method": "request_start to request_drain_end monotonic_ns",
        "correctness_scope": ("request accounting and sampled value check" if verify
                              else "request accounting only; values not verified"),
        "correctness_evidence": receipt.group(0) + ("; " + verify[0] if verify else ""),
        "request_count": total,
        "paper_request_count_match": total == 1_000_000_000,
    }


def parse_result(app: str, log: str, *, expected_requests: int = 1_000_000_000,
                 expected_checksum: str | None = None,
                 expected_bfs_repetitions: int = 1,
                 expected_post_samples: int | None = None,
                 require_cleanup: bool = True) -> dict:
    """Extract one checked work phase; KV defaults to the paper's 1B requests.

    For historical non-paper KV logs, pass their actual expected_requests
    explicitly. WC inputs other than the archived small corpus require an
    independent expected_checksum. Parsing never certifies a system or setup.
    """
    app = app.lower().replace("_", "-")
    try:
        if app == "mg":
            return _parse_mg(log)
        if app in ("wc", "wordcount"):
            return _parse_wc(log, expected_checksum)
        if app == "nq":
            return _parse_nq(log)
        if app in ("kv-b", "kv-a", "kv-s"):
            return _parse_kv(app, log, expected_requests, expected_post_samples,
                             require_cleanup=require_cleanup)
    except KeyError as exc:
        raise ValueError(f"{app} log lacks required field {exc}") from exc
    if app == "llama":
        inline_times = re.findall(
            r"^wall time:\s*(" + NUMBER + r")\s+us\s*$",
            log, re.MULTILINE)
        # The remote-memory sampler may fwrite only its known records between
        # the benchmark's two wall-time output fragments. Keep this fallback
        # narrow so unrelated numeric output cannot become a wall-time value.
        split_times = re.findall(
            r"^wall time:[ \t]*\n"
            r"(?=(?:[ \t]*\n|runtime_remote_memory(?:_missing|_end)?\b[^\n]*\n)*"
            r"runtime_remote_memory(?:_missing|_end)?\b)"
            r"(?:[ \t]*\n|runtime_remote_memory(?:_missing|_end)?\b[^\n]*\n)*"
            r"[ \t]*(" + NUMBER + r")\s+us[ \t]*$",
            log, re.MULTILINE)
        times = inline_times + split_times
        throughputs = re.findall(r"^achieved tok/s:\s*(" + NUMBER + r")\s*$",
                                 log, re.MULTILINE)
        wall_markers = re.findall(r"^wall time:", log, re.MULTILINE)
        if (len(wall_markers) != 1 or len(times) != 1
                or not throughputs or "Assistant:" not in log):
            raise ValueError("LLaMA lacks a complete chat response or one wall-time marker")
        _positive(throughputs[-1], "LLaMA token throughput")
        return {
            "elapsed_s": _positive(times[0], "LLaMA wall time (us)") / 1_000_000,
            "measurement_phase": "chat_work",
            "measurement_method": "application cycles divided by fixed 2.8 GHz",
            "correctness_scope": "functional_completion; not semantic answer validation",
            "correctness_evidence": "Assistant response and positive achieved tok/s",
        }
    if app == "bfs":
        if (isinstance(expected_bfs_repetitions, bool)
                or not isinstance(expected_bfs_repetitions, int)
                or expected_bfs_repetitions < 1):
            raise ValueError("BFS repetitions must be a positive integer")
        results = re.findall(r"^gapbs_bfs_result\b[^\n]*$", log, re.MULTILINE)
        verifies = re.findall(r"^gapbs_bfs_verify\b[^\n]*$", log, re.MULTILINE)
        if len(results) != expected_bfs_repetitions or len(verifies) != 1:
            raise ValueError("BFS requires the expected measured repetitions and final verification")
        rows = [_fields(line) for line in results]
        verify = _fields(verifies[0])
        if any(not {"iteration", "visited", "elapsed_s"} <= row.keys() for row in rows):
            raise ValueError("BFS measured result lacks required fields")
        if [row.get("iteration") for row in rows] != [
                str(i) for i in range(expected_bfs_repetitions)]:
            raise ValueError("BFS iteration identifiers are incomplete or duplicated")
        if (verify.get("status") != "pass"
                or any(row.get("visited") != verify.get("visited") for row in rows)
                or any(_count(row, "visited") <= 0 for row in rows)
                or any(verify.get(key, "0") != "0"
                       for key in ("parent_errors", "edge_depth_errors"))):
            raise ValueError("BFS verification failed or did not match the measured run")
        if expected_bfs_repetitions > 1:
            if (verify.get("scope") != "after_measured_repetitions"
                    or verify.get("closed") != "1"):
                raise ValueError("BFS requires final full-tree verification")
            for key in ("vertices", "edges", "source", "levels", "max_depth"):
                if rows[0].get(key) is None or any(
                        row.get(key) != rows[0][key] for row in rows):
                    raise ValueError("BFS repetitions differ in workload or traversal summary")
            if verify.get("max_depth") != rows[-1]["max_depth"]:
                raise ValueError("BFS verification depth differs from the measured run")
        seconds = [_positive(row["elapsed_s"], "BFS work elapsed_s") for row in rows]
        phase_lines = re.findall(r"^gapbs_bfs_phase_stats\b[^\n]*$", log, re.MULTILINE)
        phases = [_fields(line) for line in phase_lines if _fields(line).get("phase") == "work"]
        method = "application steady_clock elapsed_s"
        if phases:
            if any(not {"iteration", "ops", "ops_s"} <= row.keys() for row in phases):
                raise ValueError("BFS work-phase marker lacks required fields")
            if ([row.get("iteration") for row in phases]
                    != [str(i) for i in range(expected_bfs_repetitions)]):
                raise ValueError("BFS work-phase markers differ from measured iterations")
            for result, phase in zip(rows, phases):
                if result.get("edges") is not None and phase.get("ops") != result["edges"]:
                    raise ValueError("BFS work-phase edge count differs from the measured graph")
            seconds = [_positive(row["ops"], "BFS work operations") /
                       _positive(row["ops_s"], "BFS work operations per second")
                       for row in phases]
            method = "application work operations divided by operations per second"
        return {
            "elapsed_s": sum(seconds) / expected_bfs_repetitions,
            "measurement_phase": ("bfs_work_iteration_0" if expected_bfs_repetitions == 1
                                  else f"bfs_work_mean_{expected_bfs_repetitions}_iterations"),
            "measurement_method": method,
            "correctness_scope": "final BFS tree verification and all traversal summaries",
            "correctness_evidence": verifies[0],
            "iteration_elapsed_s": seconds,
            "measured_repetitions": expected_bfs_repetitions,
        }
    raise ValueError(f"unsupported application: {app}")
