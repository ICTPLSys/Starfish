"""Focused checks for repeated full-graph BFS measurements."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import client_environment, parse_result


def log_for(count=3):
    lines = []
    for index in range(count):
        lines.append(f"gapbs_bfs_phase_stats phase=work iteration={index} "
                     "ops=1000 ops_s=200 elapsed_s=5.00")
        lines.append(f"gapbs_bfs_result iteration={index} vertices=101 edges=1000 "
                     "source=1 elapsed_s=5.01 levels=9 visited=100 max_depth=8")
    lines.append("gapbs_bfs_verify status=pass scope=after_measured_repetitions "
                 "closed=1 visited=100 max_depth=8 parent_errors=0 edge_depth_errors=0")
    return "\n".join(lines)


class BfsRepetitions(unittest.TestCase):
    def test_site_controls_repetitions(self):
        env = client_environment("bfs", {"ib_device": "mlx5_0", "bfs_repetitions": 3})
        self.assertEqual(env["GAPBS_BFS_REPETITIONS"], "3")
        for value in (True, 0, "3"):
            with self.assertRaises(ValueError):
                client_environment("bfs", {"ib_device": "mlx5_0", "bfs_repetitions": value})

    def test_three_work_phases_use_native_unrounded_mean(self):
        result = parse_result("bfs", log_for(), expected_bfs_repetitions=3)
        self.assertEqual(result["iteration_elapsed_s"], [5.0, 5.0, 5.0])
        self.assertEqual(result["elapsed_s"], 5.0)
        self.assertEqual(result["measured_repetitions"], 3)

    def test_single_iteration_default_remains_supported(self):
        self.assertEqual(parse_result("bfs", log_for(1))["elapsed_s"], 5.0)

    def test_incomplete_duplicate_and_mismatched_runs_rejected(self):
        for log in (log_for(1),
                    log_for().replace("iteration=2", "iteration=1"),
                    log_for().replace("visited=100", "visited=99", 1),
                    log_for().replace("vertices=101", "vertices=102", 1),
                    log_for().replace("status=pass", "status=fail"),
                    log_for().replace("ops_s=200", "", 1),
                    log_for().replace("elapsed_s=5.01", "", 1),
                    log_for().replace("ops=1000", "ops=999", 1)):
            with self.subTest(log=log), self.assertRaises(ValueError):
                parse_result("bfs", log, expected_bfs_repetitions=3)


if __name__ == "__main__":
    unittest.main()
