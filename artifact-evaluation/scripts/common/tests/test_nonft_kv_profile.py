from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import client_environment, validate_nonft_kv


class NonftKvProfile(unittest.TestCase):
    def test_only_nonft_kv_enables_the_restored_mechanisms(self):
        site = {"ib_device": "mlx5_1"}
        for app in ("kv-b", "kv-a", "kv-s"):
            env = client_environment(app, site, "nonft")
            self.assertEqual(env["FARLIB_SCOPE_COUNTER_SHARDS"], "1")
            self.assertEqual(env["FARLIB_LOCK_WAIT_SCOPE_YIELD"], "1")
        for app in ("llama", "bfs", "mg", "nq", "wordcount"):
            env = client_environment(app, site, "nonft")
            self.assertNotIn("FARLIB_SCOPE_COUNTER_SHARDS", env)
            self.assertNotIn("FARLIB_LOCK_WAIT_SCOPE_YIELD", env)

    def test_runtime_must_prove_enabled_mechanisms_and_clean_scopes(self):
        log = ("kvs_lock_wait_scope_yield enabled=1\n"
               "kvs_scope_shards_end registered=24 v0=0 v1=0\n")
        self.assertEqual(validate_nonft_kv(log, expected_workers=24)["status"], "passed")
        for bad in (log.replace("enabled=1", "supported=0"),
                    log.replace("registered=24", "registered=1"),
                    log.replace("v0=0", "v0=1"),
                    log.replace("kvs_scope_shards_end", "missing_inventory")):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                validate_nonft_kv(bad, expected_workers=24)

    def test_observed_interleaved_startup_is_accepted(self):
        log = ("kvs_lock_wait_scope_yield enabled=runtime.exclusive_owned_batch=1\n"
               "1 mark_workers=2 evict_workers=4\n"
               "kvs_scope_shards_end registered=24 v0=0 v1=0\n")
        self.assertEqual(validate_nonft_kv(log, expected_workers=24)["status"], "passed")
