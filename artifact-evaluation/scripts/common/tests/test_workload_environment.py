"""Environment isolation and CPU-placement checks for client workloads."""

from pathlib import Path
import sys
import unittest


COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
from workloads import (CONTROLLED_ENV_PREFIXES, DESIGN2_ENV,
                       client_environment, client_process_environment)


SITE = {
    "ib_device": "mlx5_1",
    "fibre_cpu_set": "8-31",
    "background_cpu_base": 40,
}


def controlled(environment):
    return {
        key: value for key, value in environment.items()
        if key.startswith(CONTROLLED_ENV_PREFIXES)
    }


class WorkloadEnvironment(unittest.TestCase):
    def test_dirty_parent_has_same_controlled_profile_as_clean_parent(self):
        clean = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/custom/lib"}
        dirty = dict(clean, FARLIB_SIMPLE_LOCAL_ROUTING="bad",
                     FARLIB_BACKGROUND_CPU_BASE="999", FibreCpuSet="0",
                     GAPBS_BFS_WORKERS="1")
        clean_env = client_process_environment("bfs", SITE, "nonft", clean)
        dirty_env = client_process_environment("bfs", SITE, "nonft", dirty)
        self.assertEqual(controlled(clean_env), controlled(dirty_env))
        self.assertEqual(dirty_env["PATH"], clean["PATH"])
        self.assertEqual(dirty_env["LD_LIBRARY_PATH"], clean["LD_LIBRARY_PATH"])

    def test_llama_and_bfs_use_the_same_site_cpu_placement(self):
        llama = client_environment("llama", SITE, "nonft")
        bfs = client_environment("bfs", SITE, "nonft")
        for key in ("FibreCpuSet", "FARLIB_BACKGROUND_CPU_BASE"):
            self.assertEqual(llama[key], bfs[key])
        self.assertEqual(llama["FibreCpuSet"], "8-31")
        self.assertEqual(llama["FARLIB_BACKGROUND_CPU_BASE"], "40")

    def test_missing_cpu_fields_are_not_synthesized(self):
        env = client_environment("llama", {"ib_device": "mlx5_1"}, "nonft")
        self.assertNotIn("FibreCpuSet", env)
        self.assertNotIn("FARLIB_BACKGROUND_CPU_BASE", env)

    def test_nonft_does_not_inherit_starfish_flags(self):
        inherited = dict(DESIGN2_ENV, FARLIB_BACKGROUND_CPU_BASE="99",
                         FibreCpuSet="0", GAPBS_BFS_OPTIMIZE="0")
        env = client_process_environment("llama", SITE, "nonft", inherited)
        self.assertEqual(set(DESIGN2_ENV) & set(env),
                         {"FARLIB_LEGACY_SCAN_CURSORS",
                          "FARLIB_PLANNER_BUDGET_EARLY_EXIT",
                          "FARLIB_RESIDENT_PROFILE_REQUIRE_WORK_PHASE"})
        self.assertEqual(env["FibreCpuSet"], SITE["fibre_cpu_set"])
        self.assertEqual(env["FARLIB_BACKGROUND_CPU_BASE"], "40")
        self.assertEqual(env["FARLIB_SEPARATE_BACKGROUND_CLUSTER"], "1")
        self.assertEqual(env["FARLIB_PIN_RDMA_THREAD"], "0")

    def test_worker_defaults_remain_unchanged(self):
        llama = client_environment("llama", {"ib_device": "mlx5_1"}, "nonft")
        bfs = client_environment("bfs", {"ib_device": "mlx5_1"}, "nonft")
        self.assertEqual(llama["FibreWorkerCount"], "24")
        self.assertEqual(bfs["FibreWorkerCount"], "24")
        self.assertEqual(bfs["GAPBS_BFS_WORKERS"], "48")

    def test_nonft_concurrent_pipeline_requires_scan_cursors(self):
        for app in ("llama", "bfs", "mg"):
            env = client_process_environment(
                app, SITE, "nonft", {"FARLIB_LEGACY_SCAN_CURSORS": "0"})
            self.assertEqual(env["FARLIB_OPT_LEGACY_EXCLUSIVE_PIPELINE"], "1")
            self.assertEqual(env["FARLIB_LEGACY_SCAN_CURSORS"], "1")


if __name__ == "__main__":
    unittest.main()
