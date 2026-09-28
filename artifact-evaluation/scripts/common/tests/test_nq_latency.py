"""Focused synthetic parser checks, not benchmark results."""
import argparse
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nq_latency
from workloads import client_environment, client_process_environment


def spec():
    return nq_latency.specification(argparse.Namespace(
        app="nq", out=Path("/tmp/nq-fixture"), offered_load_ops=1000,
        latency_warmup_ms=1000, latency_measure_ms=1000, latency_drain_ms=1000,
        max_queue_delay_us=500))


def fixture():
    lines = [
        "nq_latency_config offered_load_ops=1000 fibres=48 os_workers=24 "
        "warmup_ns=1000000000 measurement_ns=1000000000 drain_timeout_ns=1000000000 "
        "max_queue_delay_ns=500000 deadline_action=drop_before_execution "
        "arrival=poisson_per_fibre queue_model=fifo_independent_lanes include_queue_wait=1 "
        "hist_sample_period=1 vertices=65608366 adjacency_lists=65608367 "
        "query=benchmark_lite vertex_distribution=uniform arrival_seed=20260917"]
    for i, name in enumerate(("warmup", "measurement")):
        start = 1000000000 + 3000000000 * i
        lines.extend([
            f"nq_latency_phase name={name} event=start monotonic_ns={start}",
            f"nq_latency_phase name={name} event=end monotonic_ns={start+1000001000}",
            f"nq_latency_receipt name={name} scheduled=1000 started=990 completed=990 "
            "completed_in_window=989 deadline_dropped=10 completed_after_deadline=2 "
            "skipped=0 request_fingerprint=42 result_checksum=12345",
            f"nq_latency_result name={name} status=passed arrival_window_ns=1000000000 "
            "drain_elapsed_ns=1000 drain_timeout=0 hist_count=990 hist_service_count=990 "
            "hist_dispatch_count=990 hist_dropped=0 merge_dropped=0 hist_write_errors=0 "
            "p99_ns=600000 p99_service_ns=550000 p99_dispatch_ns=100000"])
    lines.append("nq_latency_post_verify checked=4096 failures=0")
    lines.append("exact used bytes: 0")
    return "\n".join(lines) + "\n"


class NQLatencyChecks(unittest.TestCase):
    def test_nq_default_is_1500_milliseconds_and_zero_still_disables(self):
        args = argparse.Namespace(app="nq", out=Path("/tmp/nq-fixture"),
                                  offered_load_ops=60000)
        selected = nq_latency.specification(args)
        self.assertEqual(selected["max_queue_delay_us"], 1_500_000)
        self.assertEqual(nq_latency.environment({}, selected)["NHOP_MAX_QUEUE_DELAY_US"], "1500000")
        args.max_queue_delay_us = 0
        self.assertEqual(nq_latency.specification(args)["max_queue_delay_us"], 0)

    def test_valid_accounting_and_unclipped_slow_queries(self):
        result = nq_latency.parse_result(fixture(), spec(), expected_workers=24)
        measured = result["nq_latency"]["phases"]["measurement"]
        self.assertEqual(measured["completed"], 990)
        self.assertEqual(measured["drop_fraction"], 0.01)
        self.assertEqual(measured["completed_in_window_ops_s"], 989)
        self.assertEqual(measured["p99_ns"], 600000)

    def test_reject_corrupt_evidence(self):
        pairs = [
            ("vertices=65608366", "vertices=1000"),
            ("query=benchmark_lite", "query=fake"),
            ("deadline_dropped=10", "deadline_dropped=9"),
            ("skipped=0", "skipped=1"),
            ("hist_count=990", "hist_count=1000"),
            ("hist_dropped=0", "hist_dropped=1"),
            ("completed_after_deadline=2", "completed_after_deadline=991"),
            ("drain_timeout=0", "drain_timeout=1"),
            ("checked=4096", "checked=0"),
            ("failures=0", "failures=1"),
            ("p99_ns=600000", "p99_ns=500000"),
            ("exact used bytes: 0", "exact used bytes: 1"),
        ]
        for before, after in pairs:
            with self.subTest(before=before), self.assertRaises(ValueError):
                nq_latency.parse_result(fixture().replace(before, after, 1),
                                        spec(), expected_workers=24)
        with self.assertRaises(ValueError):
            nq_latency.parse_result(fixture(), spec(), expected_workers=23)
        with self.assertRaises(ValueError):
            nq_latency.parse_result(fixture(), spec(), expected_workers=24,
                                    histogram_dir=Path("/definitely/not/nq-histograms"))

    def test_figure9_default_unchanged_and_new_capability_required(self):
        args = argparse.Namespace(app="nq", out=Path("/tmp/nq-fixture"))
        self.assertIsNone(nq_latency.specification(args))
        base = client_environment("nq", {"ib_device": "mlx5_1"}, "nonft")
        self.assertEqual(nq_latency.environment(base, None), base)
        clean = client_process_environment("nq", {"ib_device": "mlx5_1"}, "nonft",
                                          {"NHOP_OFFERED_LOAD_OPS": "123"})
        self.assertNotIn("NHOP_OFFERED_LOAD_OPS", clean)
        self.assertEqual(nq_latency.environment(base, spec())["NHOP_MAX_QUEUE_DELAY_US"], "500")
        old = mock.Mock(returncode=1, stdout="usage", stderr="")
        with mock.patch.object(nq_latency.subprocess, "run", return_value=old):
            with self.assertRaises(ValueError):
                nq_latency.probe(Path("/unused"), {})


if __name__ == "__main__":
    unittest.main()
