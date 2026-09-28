"""Hardware-free checks for the optional microbenchmark's renamed interface."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

COMMON = Path(__file__).resolve().parents[1]
ROOT = COMMON.parents[1]
sys.path.insert(0, str(COMMON))
import node_probe


class BandwidthMicrobenchmark(unittest.TestCase):
    def test_new_and_legacy_build_targets_select_same_option(self):
        for target in ("bandwidth_microbenchmark", "bandwidth_microbenchmark_512",
                       "object_size", "object_size_512"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "not-created"
                result = subprocess.run(
                    ["bash", str(COMMON / "build.sh"), "--system", "starfish",
                     "--targets", target, "--build-dir", str(build), "--dry-run"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("-DFARLIB_BUILD_BANDWIDTH_MICROBENCHMARK=ON",
                              result.stdout)
                self.assertIn("targets: " + target, result.stdout)
                self.assertFalse(build.exists())

    def test_other_runtimes_reject_optional_target_before_creating_build(self):
        for system in ("nonft", "hydra", "carbink"):
            for target in ("bandwidth_microbenchmark", "object_size_512"):
                with self.subTest(system=system, target=target), tempfile.TemporaryDirectory() as tmp:
                    build = Path(tmp) / "not-created"
                    result = subprocess.run(
                        ["bash", str(COMMON / "build.sh"), "--system", system,
                         "--targets", target, "--build-dir", str(build), "--dry-run"],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("require --system starfish", result.stderr)
                    self.assertFalse(build.exists())

    def test_default_targets_do_not_enable_microbenchmark(self):
        result = subprocess.run(
            ["bash", str(COMMON / "build.sh"), "--system", "starfish", "--dry-run"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("-DFARLIB_BUILD_BANDWIDTH_MICROBENCHMARK=ON", result.stdout)

    def test_node_probe_recognizes_new_and_old_binary_names(self):
        for name in ("bandwidth_microbenchmark", "bandwidth_microbenchmark_512",
                     "object_size", "object_size_512"):
            with self.subTest(name=name):
                record = node_probe._ProcRecord(
                    pid=10, ppid=1, comm=name[:15], argv=(name,),
                    exe="/tmp/" + name, user="test", rss_bytes=0,
                    utime_ticks=0, stime_ticks=0, starttime_ticks=1,
                )
                self.assertEqual(node_probe._classify(record), "workload")

    def test_only_new_source_directory_exists(self):
        self.assertTrue((ROOT / "apps/bandwidth_microbenchmark/bandwidth_microbenchmark.cpp").is_file())
        self.assertFalse((ROOT / "apps/object_size").exists())


if __name__ == "__main__":
    unittest.main()
