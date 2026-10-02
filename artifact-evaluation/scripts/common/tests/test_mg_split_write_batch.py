from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import client_environment, client_process_environment


class MgSplitWriteBatch(unittest.TestCase):
    def test_default_does_not_enable_experiment(self):
        env = client_environment("mg", {"ib_device": "mlx5_1"}, "starfish")
        self.assertNotIn("FARLIB_EC_SPLIT_WRITE_BATCH", env)

    def test_recorded_site_selects_control_and_candidate(self):
        for batch in (0, 32):
            with self.subTest(batch=batch):
                site = {"ib_device": "mlx5_1", "mg_split_write_batch": batch}
                env = client_process_environment("mg", site, "starfish",
                    {"FARLIB_EC_SPLIT_WRITE_BATCH": "999", "PATH": "/usr/bin"})
                self.assertEqual(env["FARLIB_EC_SPLIT_WRITE_BATCH"], str(batch))
                self.assertEqual(env["PATH"], "/usr/bin")

    def test_inherited_flag_cannot_silently_enable_default(self):
        env = client_process_environment("mg", {"ib_device": "mlx5_1"}, "starfish",
                                          {"FARLIB_EC_SPLIT_WRITE_BATCH": "32"})
        self.assertNotIn("FARLIB_EC_SPLIT_WRITE_BATCH", env)

    def test_rejects_unplanned_batch_sizes_and_types(self):
        for batch in (-1, 1, 31, 33, 64, True, "32", None):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                client_environment("mg", {"ib_device": "mlx5_1",
                                           "mg_split_write_batch": batch}, "starfish")

    def test_other_runtimes_cannot_claim_split_batch(self):
        for system in ("nonft", "hydra", "carbink"):
            with self.subTest(system=system), self.assertRaises(ValueError):
                client_environment("mg", {"ib_device": "mlx5_1",
                                           "mg_split_write_batch": 32}, system)

    def test_breakdown_is_explicit_boolean_and_recorded(self):
        for enabled, expected in ((False, "0"), (True, "1")):
            env = client_environment("mg", {"ib_device": "mlx5_1",
                                            "mg_evict_breakdown": enabled}, "starfish")
            self.assertEqual(env["FARLIB_EVAC_BREAKDOWN"], expected)
        for invalid in (0, 1, "1", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                client_environment("mg", {"ib_device": "mlx5_1",
                                           "mg_evict_breakdown": invalid}, "starfish")

    def test_inherited_breakdown_cannot_enable_default(self):
        env = client_process_environment("mg", {"ib_device": "mlx5_1"}, "starfish",
                                          {"FARLIB_EVAC_BREAKDOWN": "1"})
        self.assertNotIn("FARLIB_EVAC_BREAKDOWN", env)

    def test_direct_read_is_explicit_boolean_and_recorded(self):
        for enabled, expected in ((False, "0"), (True, "1")):
            env = client_process_environment("mg", {"ib_device": "mlx5_1",
                "mg_split_direct_read": enabled}, "starfish",
                {"FARLIB_EC_SPLIT_DIRECT_READ": "999", "PATH": "/usr/bin"})
            self.assertEqual(env["FARLIB_EC_SPLIT_DIRECT_READ"], expected)
            self.assertEqual(env["PATH"], "/usr/bin")
        for invalid in (0, 1, "1", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                client_environment("mg", {"ib_device": "mlx5_1",
                    "mg_split_direct_read": invalid}, "starfish")

    def test_direct_read_not_inherited_or_available_to_other_runtimes(self):
        env = client_process_environment("mg", {"ib_device": "mlx5_1"},
            "starfish", {"FARLIB_EC_SPLIT_DIRECT_READ": "1"})
        self.assertNotIn("FARLIB_EC_SPLIT_DIRECT_READ", env)
        for system in ("nonft", "hydra", "carbink"):
            with self.subTest(system=system), self.assertRaises(ValueError):
                client_environment("mg", {"ib_device": "mlx5_1",
                    "mg_split_direct_read": True}, system)


if __name__ == "__main__":
    unittest.main()
