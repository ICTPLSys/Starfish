"""Offline checks for Design 2 and multi-endpoint recipe boundaries."""

from pathlib import Path
import sys
import tempfile
import unittest

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
from render_config import render
from workloads import (DESIGN2_ENV, client_environment,
                       client_process_environment, validate_design2)

AE = COMMON.parents[1]
SITE = {"ib_device": "mlx5_1"}


def values(config):
    return {line.split()[0]: line.split()[1] for line in config.splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def ledger():
    lines = [
        "simple_hotcold.config enabled=1 mode=local_resident measured_heat=remote_hint",
        "simple_remote_hotcold.config enabled=1",
        "simple_region_budget.config mode=adaptive semantic_classes=6 local_resident=1",
    ]
    for domain in ("local", "remote"):
        for cls in range(6):
            lines.append(f"simple_region_budget.six_final domain={domain} bin=31 "
                         f"class={cls} actual=1 target=1 applied_total=0")
    return "\n".join(lines)


class Design2Profile(unittest.TestCase):
    def test_starfish_profile_uses_simple_not_fixed_groups(self):
        for app in ("llama", "bfs"):
            env = client_environment(app, SITE, "starfish")
            self.assertEqual(env["FARLIB_FIXED_SIX_GROUPS"], "0")
            self.assertEqual(env["FARLIB_SIMPLE_SIX_GROUPS"], "1")
            self.assertEqual(env["FARLIB_SIMPLE_LOCAL_RESIDENT"], "1")
            self.assertEqual(env["FARLIB_LEGACY_SCAN_CURSORS"], "1")
            self.assertEqual(env["FARLIB_LIST_ONLY_SIX"], "1")

    def test_nonft_does_not_inherit_design2_profile(self):
        inherited = dict(DESIGN2_ENV, PATH="/usr/bin", LD_LIBRARY_PATH="/custom/lib")
        env = client_process_environment("llama", SITE, "nonft", inherited)
        self.assertEqual(set(DESIGN2_ENV) & set(env),
                         {"FARLIB_LEGACY_SCAN_CURSORS",
                          "FARLIB_PLANNER_BUDGET_EARLY_EXIT",
                          "FARLIB_RESIDENT_PROFILE_REQUIRE_WORK_PHASE"})
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/custom/lib")
        self.assertEqual(inherited["FARLIB_SIMPLE_LOCAL_RESIDENT"], "1")

    def test_starfish_profile_overrides_parent_switches(self):
        env = client_process_environment("llama", SITE, "starfish",
                                         {"FARLIB_FIXED_SIX_GROUPS": "1"})
        self.assertEqual(env["FARLIB_FIXED_SIX_GROUPS"], "0")
        for key, value in DESIGN2_ENV.items():
            self.assertEqual(env[key], value)

    def test_unknown_system_rejected(self):
        with self.assertRaises(ValueError):
            client_environment("llama", SITE, "typo")

    def test_bfs_prefetch_state_preserved(self):
        for system in ("nonft", "starfish"):
            env = client_environment("bfs", SITE, system)
            self.assertEqual(env["GAPBS_BFS_OPTIMIZE"], "1")
            self.assertEqual(env["GAPBS_CHUNK_PREFETCH_DISTANCE"], "1")
            self.assertEqual(env["GAPBS_BFS_VERIFY"], "1")

    def test_starfish_recipe_budgets(self):
        footprint = 10_000_000
        for app in ("llama", "bfs"):
            for ratio in (13, 25, 50, 75, 100):
                config = values(render(AE / "configs" / app / "starfish.config",
                                       system="starfish", ratio=ratio,
                                       footprint_bytes=footprint,
                                       server_addr="192.0.2.20", server_port=1998,
                                       ib_device="mlx5_0"))
                local = footprint * ratio // 100
                backup = footprint // 10
                resident = (local * 8 + 5) // 10 if ratio <= 50 else local - backup
                self.assertEqual(int(config["client_buffer_size"]), local)
                self.assertEqual(int(config["remote_backup_budget_bytes"]), backup)
                self.assertEqual(int(config["local_resident_budget_bytes"]), resident)
                self.assertEqual(config["ft_method"], "ec_batch")
                self.assertEqual(config["enable_region_resident_placement"], "1")
                self.assertEqual(config["remote_backup_mode"], "object_profiled")

    def test_nonft_keeps_backup_and_resident_without_ec(self):
        for app in ("llama", "mg", "bfs"):
            for ratio in (25, 50, 75):
                config = values(render(AE / "configs" / app / "nonft.config",
                                       system="nonft", ratio=ratio,
                                       footprint_bytes=10_000_000,
                                       server_addr="192.0.2.20", server_port=1998,
                                       ib_device="mlx5_0"))
                local = 10_000_000 * ratio // 100
                resident = (local * 8 + 5) // 10 if ratio <= 50 else local - 1_000_000
                self.assertEqual(config["ft_method"], "none")
                self.assertEqual(config["enable_selective_backup"], "1")
                self.assertEqual(config["remote_backup_mode"], "object_profiled")
                self.assertEqual(config["remote_backup_budget_bytes"], "1000000")
                self.assertEqual(int(config["local_resident_budget_bytes"]), resident)
                self.assertEqual(config["enable_region_resident_placement"], "1")
                self.assertEqual(config.get("region_placement_bind_groups", "0"), "0")

    def test_backup_defaults_follow_system(self):
        for system in ("nonft", "starfish", "hydra", "carbink"):
            template = AE / "configs/llama" / (system + ".config")
            config = values(render(template, system=system, ratio=25,
                                   footprint_bytes=10_000_000,
                                   server_addr="192.0.2.20", server_port=1998,
                                   ib_device="mlx5_0"))
            enabled = system in {"nonft", "starfish"}
            self.assertEqual(config["enable_selective_backup"],
                             "1" if enabled else "0")
            self.assertEqual(config["remote_backup_budget_bytes"],
                             "1000000" if enabled else "0")
            if not enabled:
                self.assertEqual(config["remote_backup_budget_pct"], "0")
            self.assertEqual(config["ft_method"],
                             {"starfish": "ec_batch", "nonft": "none"}.get(system, system))

    def test_ec_recipe_requires_explicit_endpoints_for_legacy_override(self):
        template = AE / "configs" / "wordcount" / "recomputable_ec.config"
        endpoints = [{"server_addr": "192.0.2.20", "server_port": 1998 + i}
                     for i in range(6)]
        config = values(render(template, system="starfish", ratio=25,
                               footprint_bytes=10_000_000, ib_device="mlx5_0",
                               server_endpoints=endpoints))
        self.assertEqual(config["ft_method"], "ec_batch")
        self.assertEqual(config["server_count"], "6")
        self.assertEqual(config["server_ports"], ",".join(str(1998 + i) for i in range(6)))
        with self.assertRaises(ValueError):
            render(template, system="starfish", ratio=25,
                   footprint_bytes=10_000_000, ib_device="mlx5_0")

    def test_ec_template_endpoints_are_preserved_without_override(self):
        # Legacy explicit external recipes still retain their endpoint lists;
        # shipped application recipes now intentionally have no deployment.
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "external.config"
            template.write_text("server_count 6\nserver_addrs " +
                                ",".join(["192.0.2.20"] * 6) +
                                "\nserver_ports 1400,1401,1402,1403,1404,1405\n")
            config = values(render(template, system="starfish", ratio=25,
                                   footprint_bytes=10_000_000, ib_device="mlx5_0"))
        self.assertEqual(config["server_count"], "6")
        self.assertEqual(config["server_ports"], "1400,1401,1402,1403,1404,1405")

    def test_active_profile_and_complete_ledger(self):
        evidence = validate_design2(ledger())
        self.assertEqual(evidence["local_regions"], 6)
        self.assertEqual(evidence["profile"], "resident_local_six")

    def test_missing_local_resident_config_rejected(self):
        with self.assertRaises(ValueError):
            validate_design2(ledger().replace("mode=local_resident", "mode=measured"))

    def test_incomplete_domain_or_class_rejected(self):
        for fragment in ("domain=local", "class=5"):
            log = "\n".join(line for line in ledger().splitlines() if fragment not in line)
            with self.assertRaises(ValueError):
                validate_design2(log)

    def test_malformed_counts_rejected(self):
        with self.assertRaises(ValueError):
            validate_design2(ledger().replace("actual=1", "actual=-1", 1))

    def test_repeated_or_truncated_snapshots_cannot_be_spliced(self):
        for final in (ledger(), "\n".join(ledger().splitlines()[:-1])):
            with self.assertRaises(ValueError):
                validate_design2(ledger() + "\n" + final)

    def test_local_only_workload_does_not_require_remote_allocations(self):
        log = "\n".join(line for line in ledger().splitlines() if "domain=remote" not in line)
        self.assertEqual(validate_design2(log)["remote_regions"], 0)

    def test_dynamic_target_is_not_mistaken_for_actual_supply(self):
        self.assertEqual(validate_design2(ledger().replace("target=1", "target=2"))["local_regions"], 6)

    def test_bfs_profiles_are_validated_independently(self):
        log = "\n".join(
            f"gapbs_bfs_iteration_begin iteration={i}\n" + ledger()
            + f"\ngapbs_bfs_phase_stats phase=work iteration={i}"
            for i in range(3))
        evidence = validate_design2(log, expected_bfs_repetitions=3)
        self.assertEqual(evidence["work_profiles"], 3)
        self.assertEqual(len(evidence["per_work_profile"]), 3)
        with self.assertRaises(ValueError):
            validate_design2(log, expected_bfs_repetitions=2)
        truncated = log.rsplit("simple_region_budget.six_final", 1)
        incomplete = truncated[0] + truncated[1].split("\n", 1)[1]
        with self.assertRaises(ValueError):
            validate_design2(incomplete, expected_bfs_repetitions=3)
        with self.assertRaises(ValueError):
            validate_design2(log + "\n" + ledger(), expected_bfs_repetitions=3)


if __name__ == "__main__":
    unittest.main()
