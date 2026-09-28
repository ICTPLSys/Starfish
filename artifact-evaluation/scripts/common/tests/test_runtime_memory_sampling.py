"""Compile/run hardware-free observer fixtures, not benchmark workloads."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

AE = Path(__file__).resolve().parents[3]


class RuntimeMemorySampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "observer-test"
        subprocess.run([
            "g++", "-std=c++20", "-pthread", "-O0", "-Wall", "-Wextra",
            "-I", str(AE / "runtime/common"),
            str(Path(__file__).with_name("runtime_memory_sampling_test.cpp")),
            "-o", str(cls.binary),
        ], check=True, capture_output=True, text=True, timeout=90)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_fixture(self, mode):
        return subprocess.run([str(self.binary), mode], check=True,
                              capture_output=True, text=True, timeout=5)

    def test_schedule_and_delayed_readiness(self):
        self.assertIn("schedule passed", self.run_fixture("schedule").stdout)

    def test_disabled_sampler_does_not_run_or_log(self):
        self.assertEqual(self.run_fixture("disabled").stderr, "")

    def test_future_origin_is_rejected(self):
        self.assertIn("future rejected", self.run_fixture("future").stdout)

    def test_early_stop_returns_without_waiting_for_50_seconds(self):
        log = self.run_fixture("early").stderr
        self.assertIn("runtime_remote_memory_end schema_version=2 samples=0", log)
        self.assertNotIn("runtime_remote_memory schema_version=2", log)

    def test_one_real_callback_records_fractional_actual_time_and_sum(self):
        log = self.run_fixture("one").stderr
        lines = [line for line in log.splitlines()
                 if line.startswith("runtime_remote_memory ")]
        self.assertEqual(len(lines), 1)
        fields = dict(token.split("=", 1) for token in lines[0].split()[1:])
        self.assertEqual(fields["scheduled_elapsed_s"], "10")
        self.assertGreaterEqual(float(fields["observed_elapsed_s"]), 10)
        self.assertLess(float(fields["observed_elapsed_s"]), 11)
        self.assertIn(".", fields["observed_elapsed_s"])
        self.assertEqual(fields["occupied_bytes"], "300")
        self.assertEqual(fields["endpoint_bytes"], "0:100,1:200")
        self.assertIn("runtime_remote_memory_end schema_version=2 samples=1", log)

    def test_missed_notready_points_are_not_reconstructed(self):
        log = self.run_fixture("notready").stderr
        self.assertEqual(log.count("status=missed-notready"), 3)
        self.assertNotIn("runtime_remote_memory schema_version=2", log)
        self.assertIn("samples=0", log)
        self.assertIn("missed_notready=3", log)

    def test_all_runtime_caches_use_the_same_lifecycle_hook(self):
        for system in ("nonft", "starfish", "hydra", "carbink"):
            source = (AE / "runtime" / system /
                      "include/cache/concurrent_cache.hpp").read_text()
            self.assertIn('#include "../../../common/benchmark_memory.hpp"', source)
            self.assertIn("start_benchmark_memory_sampling();", source)
            # Figure 12 releases its saved snapshot reporter first; the sampler
            # must still stop before allocator/background teardown begins.
            destructor = source.split("~ConcurrentArrayCache() {", 1)[1][:400]
            self.assertRegex(destructor,
                             r"^\s*(?://[^\n]*\n\s*)*"
                             r"(?:runtime_metadata_reporter_\.reset\(\);\s*)?"
                             r"if \(benchmark_memory_sampler_\) "
                             r"benchmark_memory_sampler_->stop\(\);")


if __name__ == "__main__":
    unittest.main()
