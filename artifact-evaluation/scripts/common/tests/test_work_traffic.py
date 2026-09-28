"""Compile/run synthetic client-byte accounting checks without RDMA services."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

AE = Path(__file__).resolve().parents[3]
SYSTEMS = ("starfish", "nonft", "hydra", "carbink")
SOURCE = Path(__file__).with_name("work_traffic_test.cpp")


@unittest.skipUnless(shutil.which("g++"), "requires a C++20 compiler")
class WorkTrafficTest(unittest.TestCase):
    def test_all_runtime_hooks_and_frozen_windows(self):
        with tempfile.TemporaryDirectory(prefix="work-traffic-test-") as directory:
            for system in SYSTEMS:
                with self.subTest(system=system):
                    binary = Path(directory) / system
                    compiled = subprocess.run(
                        ["g++", "-std=c++20", "-O1", "-pthread",
                         "-I" + str(AE / "runtime" / system / "include"),
                         str(SOURCE), "-o", str(binary)],
                        text=True, capture_output=True, timeout=120)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    result = subprocess.run([str(binary)], text=True,
                                            capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)
                    rows = [line for line in result.stderr.splitlines()
                            if line.startswith("runtime_traffic_work ")]
                    self.assertEqual(len(rows), 4)
                    records = []
                    for row in rows:
                        tokens = row.split()[1:]
                        record = dict(token.split("=", 1) for token in tokens)
                        self.assertEqual(len(tokens), len(record))
                        self.assertEqual(record["schema_version"], "1")
                        self.assertEqual(record["scope"], "client_rdma")
                        self.assertEqual(record["phase"], "work")
                        self.assertEqual(record["snapshot"], "rolling_per_worker")
                        self.assertEqual(record["accounting"], "existing_counter_sites")
                        self.assertEqual(int(record["elapsed_ns"]),
                                         int(record["end_monotonic_ns"]) -
                                         int(record["start_monotonic_ns"]))
                        self.assertEqual(int(record["fetch_bytes"]),
                                         int(record["read_bytes"]))
                        self.assertEqual(int(record["eviction_bytes"]),
                                         int(record["write_bytes"]) +
                                         int(record["rmw_read_bytes"]))
                        self.assertNotIn("rmw_write_bytes", record)
                        records.append(record)
                    self.assertEqual([int(r["window_id"]) for r in records],
                                     [1, 2, 3, 4])
                    self.assertEqual([int(r["fetch_bytes"]) for r in records],
                                     [42, 0, 24000, 0])
                    self.assertEqual([int(r["eviction_bytes"]) for r in records],
                                     [57, 0, 96000, 11])

    def test_identical_helpers_and_work_boundary_wiring(self):
        headers = [(AE / "runtime" / system / "include/utils/work_traffic.hpp")
                   .read_bytes() for system in SYSTEMS]
        self.assertTrue(all(header == headers[0] for header in headers))
        for system in SYSTEMS:
            source = (AE / "runtime" / system / "include/utils/stats.hpp").read_text()
            self.assertEqual(source.count("work_traffic::begin();"), 1)
            self.assertEqual(source.count("work_traffic::print(work_traffic::end());"), 1)
            self.assertEqual(source.count("work_traffic::count_read(bytes);"), 2)
            self.assertEqual(source.count("work_traffic::count_write(bytes);"), 1)
        rmw = (AE / "runtime/starfish/include/cache/core/rdma/ec_rmw_path.ipp").read_text()
        self.assertEqual(rmw.count("profile::work_traffic::count_rmw_read(bytes);"), 1)


if __name__ == "__main__":
    unittest.main()
