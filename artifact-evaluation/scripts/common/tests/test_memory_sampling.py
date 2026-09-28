"""The same fixed-time sampler environment is used by every benchmark."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import (FOOTPRINT_BYTES, SYSTEM_LABEL, client_environment,
                       client_process_environment)

SITE = {
    "ib_device": "mlx5_1",
    "fibre_cpu_set": "8-31",
    "background_cpu_base": 40,
    "remote_memory_samples": True,
    "remote_memory_observer_cpu": 6,
}


class MemorySampling(unittest.TestCase):
    def test_every_application_and_runtime_uses_the_same_switch(self):
        for app in FOOTPRINT_BYTES:
            for system in SYSTEM_LABEL:
                with self.subTest(app=app, system=system):
                    env = client_environment(app, SITE, system)
                    self.assertEqual(env["FARLIB_REMOTE_MEMORY_SAMPLES"], "1")
                    self.assertEqual(env["FARLIB_REMOTE_MEMORY_OBSERVER_CPU"], "6")
                    self.assertNotIn("FARLIB_KVS_MEMORY_SAMPLES", env)
                    self.assertNotIn("FARLIB_KVS_MEMORY_METRIC", env)

    def test_default_profiles_do_not_enable_sampling(self):
        for app in FOOTPRINT_BYTES:
            for system in SYSTEM_LABEL:
                env = client_environment(
                    app, dict(SITE, remote_memory_samples=False), system)
                self.assertNotIn("FARLIB_REMOTE_MEMORY_SAMPLES", env)

    def test_inherited_flags_and_time_origin_do_not_leak(self):
        stale = {
            "FARLIB_KVS_MEMORY_SAMPLES": "1",
            "FARLIB_REMOTE_MEMORY_SAMPLES": "0",
            "FARLIB_REMOTE_MEMORY_OBSERVER_CPU": "99",
            "FARLIB_BENCHMARK_START_MONOTONIC_NS": "1",
        }
        env = client_process_environment("bfs", SITE, "hydra", stale)
        self.assertEqual(env["FARLIB_REMOTE_MEMORY_SAMPLES"], "1")
        self.assertEqual(env["FARLIB_REMOTE_MEMORY_OBSERVER_CPU"], "6")
        self.assertNotIn("FARLIB_KVS_MEMORY_SAMPLES", env)
        self.assertNotIn("FARLIB_BENCHMARK_START_MONOTONIC_NS", env)

    def test_cpu_validation_is_strict(self):
        for bad in (None, True, -1, 1.5, "1.5", "not-a-cpu", ""):
            with self.subTest(cpu=bad), self.assertRaises(ValueError):
                client_environment(
                    "llama", dict(SITE, remote_memory_observer_cpu=bad), "nonft")
        self.assertEqual(client_environment(
            "mg", dict(SITE, remote_memory_observer_cpu="06"), "carbink")
            ["FARLIB_REMOTE_MEMORY_OBSERVER_CPU"], "6")

    def test_legacy_flag_fails_instead_of_silently_reusing_progress_sampling(self):
        with self.assertRaisesRegex(ValueError, "replaced"):
            client_environment("kv-b", dict(SITE, kv_memory_samples=True), "nonft")

    def test_enabled_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            client_environment("nq", dict(SITE, remote_memory_samples="false"), "starfish")


if __name__ == "__main__":
    unittest.main()
