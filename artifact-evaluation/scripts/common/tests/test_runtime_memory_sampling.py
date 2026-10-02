"""Compile/run hardware-free Work-boundary sampler fixtures."""
from pathlib import Path
import subprocess
import tempfile
import unittest

AE = Path(__file__).resolve().parents[3]


def fields(line):
    return dict(token.split("=", 1) for token in line.split()[1:])


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

    @staticmethod
    def sample_lines(log):
        return [line for line in log.splitlines()
                if line.startswith("runtime_remote_memory ")]

    @staticmethod
    def end_lines(log):
        return [line for line in log.splitlines()
                if line.startswith("runtime_remote_memory_end ")]

    def test_schedule_and_bfs_offsets(self):
        self.assertIn("schedule passed", self.run_fixture("schedule").stdout)

    def test_disabled_init_and_destroy_have_no_output(self):
        for mode in ("disabled", "init_delay", "destroy_before_work"):
            self.assertEqual(self.run_fixture(mode).stdout, "")

    def test_invalid_profile_is_fail_closed(self):
        self.assertIn("invalid profile rejected",
                      self.run_fixture("invalid_profile").stdout)

    def test_old_program_start_env_is_ignored(self):
        log = self.run_fixture("old_env_ignored").stdout
        self.assertEqual(len(self.sample_lines(log)), 1)
        self.assertIn("schema_version=3", log)
        self.assertIn("scheduled_samples=1", log)

    def test_backlog_marks_old_deadlines_missing_instead_of_reusing_latest(self):
        log = self.run_fixture("backlog").stdout
        samples = self.sample_lines(log)
        self.assertEqual(len(samples), 1)
        self.assertEqual(fields(samples[0])["scheduled_elapsed_s"], "30")
        missing = [line for line in log.splitlines()
                   if line.startswith("runtime_remote_memory_missing ")]
        self.assertEqual([fields(line)["scheduled_elapsed_s"] for line in missing],
                         ["10", "20"])
        end = fields(self.end_lines(log)[0])
        self.assertEqual(end["scheduled_samples"], "3")
        self.assertEqual(end["samples"], "1")
        self.assertEqual(end["missing"], "2")

    def test_missing_end_is_not_reported_as_a_complete_window(self):
        end = fields(self.end_lines(self.run_fixture("missing_end").stdout)[0])
        self.assertEqual(end["status"], "work-end-missing")

    def test_origin_mismatch_is_ignored(self):
        self.assertEqual(self.run_fixture("origin_mismatch").stdout, "")

    def test_normal_work_window_has_five_samples_and_exact_end(self):
        log = self.run_fixture("normal").stdout
        samples = self.sample_lines(log)
        self.assertEqual(len(samples), 5)
        self.assertEqual([fields(line)["scheduled_elapsed_s"] for line in samples],
                         ["10", "20", "30", "40", "50"])
        for line in samples:
            parsed = fields(line)
            self.assertEqual(parsed["schema_version"], "3")
            self.assertEqual(parsed["origin"], "profile_start_work")
            self.assertEqual(parsed["sampling"],
                             "work_start_10_20_30_40_50s")
            self.assertEqual(parsed["occupied_bytes"], "300")
        ends = self.end_lines(log)
        self.assertEqual(len(ends), 1)
        end = fields(ends[0])
        self.assertEqual(end["samples"], "5")
        self.assertEqual(end["scheduled_samples"], "5")
        self.assertEqual(end["missing"], "0")
        self.assertEqual(end["expected_samples"], "5")
        self.assertEqual(end["status"], "complete")

    def test_bfs_profile_has_only_three_second_point(self):
        log = self.run_fixture("bfs").stdout
        samples = self.sample_lines(log)
        self.assertEqual(len(samples), 1)
        parsed = fields(samples[0])
        self.assertEqual(parsed["scheduled_elapsed_s"], "3")
        self.assertEqual(parsed["window"], "work_start_3s")
        end = fields(self.end_lines(log)[0])
        self.assertEqual(end["expected_samples"], "1")
        self.assertEqual(end["scheduled_samples"], "1")
        self.assertEqual(end["status"], "complete")

    def test_short_work_does_not_fill_end_snapshot(self):
        log = self.run_fixture("short").stdout
        self.assertEqual(self.sample_lines(log), [])
        end = fields(self.end_lines(log)[0])
        self.assertEqual(end["samples"], "0")
        self.assertEqual(end["scheduled_samples"], "0")
        self.assertEqual(end["missing"], "0")
        self.assertEqual(end["status"], "short-work")

    def test_repeated_work_windows_are_independent(self):
        log = self.run_fixture("repeat").stdout
        samples = self.sample_lines(log)
        self.assertEqual([fields(line)["work_window_id"] for line in samples], ["1", "2"])
        ends = self.end_lines(log)
        self.assertEqual([fields(line)["work_window_id"] for line in ends], ["1", "2"])

    def test_kvs_explicit_origin_suppresses_generic_hook(self):
        log = self.run_fixture("kvs").stdout
        self.assertNotIn("origin=profile_start_work", log)
        self.assertIn("origin=kvs_request_start", log)
        self.assertEqual(len(self.end_lines(log)), 1)

    def test_end_hook_does_not_wait_for_snapshot(self):
        log = self.run_fixture("nonblocking").stdout
        end = fields(self.end_lines(log)[0])
        self.assertEqual(end["scheduled_samples"], "1")
        self.assertEqual(end["missing"], "1")
        self.assertNotIn("occupied_bytes=300", log)

    def test_record_repairs_unterminated_previous_writer(self):
        log = self.run_fixture("prefix").stdout
        lines = self.sample_lines(log)
        self.assertEqual(len(lines), 1)
        self.assertIn("sample=0", lines[0])
        self.assertNotIn("without newlineruntime_remote_memory", log)

    def test_concurrent_records_are_complete_and_unique(self):
        log = self.run_fixture("concurrent").stdout
        lines = [line for line in log.splitlines()
                 if line.startswith("runtime_remote_memory_test ")]
        self.assertEqual(len(lines), 8 * 32)
        pairs = set()
        for line in lines:
            parsed = fields(line)
            self.assertEqual(parsed["occupied_bytes"], "300")
            pairs.add((parsed["thread"], parsed["seq"]))
        self.assertEqual(len(pairs), 8 * 32)

    def test_all_runtime_caches_register_same_lifecycle(self):
        for system in ("nonft", "starfish", "hydra", "carbink"):
            source = (AE / "runtime" / system /
                      "include/cache/concurrent_cache.hpp").read_text()
            self.assertIn('#include "../../../common/benchmark_memory.hpp"', source)
            self.assertIn("benchmark_memory_sampler_->arm();", source)
            self.assertIn("benchmark_memory_sampler_->shutdown();", source)
            self.assertIn("benchmark_memory_begin_observer", source)
            self.assertIn("benchmark_memory_end_observer", source)

    def test_stats_have_nonblocking_callbacks(self):
        for system in ("nonft", "starfish", "hydra", "carbink"):
            source = (AE / "runtime" / system /
                      "include/utils/stats.hpp").read_text()
            self.assertIn("benchmark_memory_begin_observer", source)
            self.assertIn("benchmark_memory_end_observer", source)


if __name__ == "__main__":
    unittest.main()
