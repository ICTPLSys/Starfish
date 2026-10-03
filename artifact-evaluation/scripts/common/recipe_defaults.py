"""Shared AE policies for built-in recipes, before explicit recipe overrides.

Runtime config.def remains authoritative for unspecified fields. These are the
selected experiment policies that differ from those runtime defaults, not a
second copy of each application config. Deployment and figure-specific flags
are applied later by the renderer/runner and are never stored here.
"""

from pathlib import Path

from workloads import FOOTPRINT_BYTES


CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SYSTEMS = {"nonft", "starfish", "hydra", "carbink"}


def builtin_identity(template):
    path = Path(template).resolve()
    if path.parent.parent != CONFIGS.resolve() or path.parent.name not in FOOTPRINT_BYTES:
        return None
    name = path.stem
    if name not in SYSTEMS:
        return None
    return path.parent.name, name


def builtin_defaults(template):
    identity = builtin_identity(template)
    if identity is None:
        # Separate, explicit diagnostic recipe; no normal application policy.
        if Path(template).resolve() == CONFIGS / "wordcount/recomputable_ec.config":
            return {"evict_batch_size": "65536"}
        return {}
    app, system = identity
    values = {
        "max_thread_cnt": "24", "evict_batch_size": "65536",
    }
    if system in {"hydra", "carbink"}:
        # Selected policy: fixed Resident placement for every application;
        # retained backup and the adaptive planner remain disabled.
        values["enable_region_resident_placement"] = "1"
    if ((app, system) == ("mg", "starfish") or
            (app.startswith("kv-") and system in {"nonft", "starfish"})):
        values.update(mark_thread_cnt="2", evacuate_thread_cnt="10")
    if system in {"nonft", "starfish"}:
        values.update({
            "remote_backup_mode": "object_profiled",
            "enable_logical_object_profile": "1",
            "enable_resident_profile_planner": "1",
            "resident_profile_warmup_ms": "30000",
            "enable_region_resident_placement": "1",
            "enable_region_hotness_placement": "0" if app == "mg" else "1",
        })
        if system == "starfish":
            values.update(remote_backup_budget_pct="0",
                          resident_profile_require_work_phase="1")
            if app in {"llama", "bfs"}:
                values["evict_batch_size"] = "262144"
        elif app in {"nq", "wordcount"}:
            values["remote_backup_budget_pct"] = "0"
        if app in {"llama", "bfs", "wordcount", "nq"} or (
                system == "starfish" and app.startswith("kv-")):
            values["resident_profile_min_benefit_refs_per_replacement_object"] = "0"
        adaptive = app == "bfs" or (system == "starfish" and
                                    (app == "nq" or app.startswith("kv-")))
        if adaptive:
            values.update(resident_profile_warmup_ms="0",
                          resident_profile_initial_sample_windows="3",
                          resident_profile_migration_budget_pct="100",
                          enable_region_fetch_hotness_placement="1",
                          resident_profile_require_work_phase="1",
                          region_fetch_placement_policy="design1_gain")
            if app in {"bfs", "nq"}:
                values["resident_profile_group_bytes"] = str(64 * 1024**2)
            if not (system == "starfish" and app == "bfs"):
                values["region_fetch_resident_headroom_pct"] = "1"
        elif system == "starfish" and app == "llama":
            values["region_fetch_placement_policy"] = "design1_gain"
    elif system == "hydra":
        # Retained historical values, even for disabled planner components.
        values.update(resident_profile_warmup_ms="30000",
                      resident_profile_min_benefit_refs_per_replacement_object="0",
                      region_fetch_placement_policy="design1_gain",
                      region_fetch_resident_headroom_pct="1")
    else:
        # The selected Carbink protocol's queue/compaction capacities.
        values.update(cq_entries="65536", qp_recv_cap="1024",
                      qp_send_cap="1024", compaction_worker_count="4")
    return values


def fixed_resident_budget(template, local):
    """Preserve the selected fixed fraction, including original byte rounding.

Hydra and Carbink do not use Starfish's adaptive budget rule. Their original
fixed fraction was defined at the reference cache capacity; retain that exact
rational fraction rather than silently changing small rounding differences.
"""
    identity = builtin_identity(template)
    if identity is None:
        return None
    app, system = identity
    if system not in {"hydra", "carbink"}:
        return None
    anchor = FOOTPRINT_BYTES[app] // 4
    if system == "hydra" and app in {"llama", "bfs"}:
        anchor = (6 if app == "llama" else 8) * 1024**3
    anchor_resident = (anchor * 8 + 5) // 10
    return local * anchor_resident // anchor
