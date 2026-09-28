"""Small synthetic log checks; these are not benchmark measurements."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
import kv_latency
from workloads import client_environment, client_process_environment


def arguments(root, rate=1000):
    return argparse.Namespace(app="kv-b", out=Path(root), offered_load_ops=rate,
                              latency_warmup_ms=1000, latency_measure_ms=1000,
                              latency_drain_ms=1000)


def fixture():
    lines = [
        "kvs_latency_config offered_load_ops=1000 fibres=48 os_workers=24 "
        "warmup_ns=1000000000 measurement_ns=1000000000 drain_timeout_ns=1000000000 "
        "arrival=poisson_per_fibre queue_model=fifo_independent_lanes include_queue_wait=1 "
        "hist_sample_period=1 object_bytes=512 initial_count=33554432 put_ratio=0.05 zipf=0.99 "
        "random_seed=20260917 arrival_seed=20260917 locked_get=1 real_put=1 get_content_validation=0"
    ]
    for index, phase in enumerate(("warmup", "measurement")):
        start = 1000000000 + index * 3000000000
        lines += [
            f"kvs_latency_phase name={phase} event=start monotonic_ns={start}",
            f"kvs_latency_phase name={phase} event=end monotonic_ns={start + 1000001000}",
            f"kvs_latency_receipt name={phase} scheduled=1000 started=1000 completed=1000 "
            "completed_in_window=999 skipped=0 generated_get=950 generated_put=50 generated_remove=0 "
            "completed_get=950 completed_put=50 completed_remove=0 request_fingerprint=12",
            f"kvs_latency_result name={phase} status=passed arrival_window_ns=1000000000 "
            "drain_elapsed_ns=1000 drain_timeout=0 hist_count=1000 hist_service_count=1000 "
            "hist_dispatch_count=1000 hist_dropped=0 merge_dropped=0 hist_write_errors=0 "
            "p99_ns=20000 p99_service_ns=10000 p99_dispatch_ns=15000",
        ]
    lines += ["kvs_post_verify checked=4096 failures=0",
              "kvs_lock_wait_scope_yield enabled=1",
              "kvs_scope_shards_end registered=24 v0=0 v1=0"]
    return "\n".join(lines) + "\n"


class KVLatencyChecks(unittest.TestCase):
    def test_queue_deadline_counts_and_slow_completed_samples(self):
        args = arguments("/tmp/unused-latency-fixture")
        args.max_queue_delay_us = 500
        spec = kv_latency.specification(args)
        log = fixture().replace(
            "get_content_validation=0",
            "get_content_validation=0 max_queue_delay_ns=500000 deadline_action=drop_before_execution")
        log = log.replace("started=1000 completed=1000", "started=990 completed=990")
        log = log.replace("completed_in_window=999", "completed_in_window=989")
        log = log.replace("completed_get=950", "completed_get=940")
        log = log.replace("request_fingerprint=12",
            "request_fingerprint=12 deadline_dropped=10 dropped_get=10 dropped_put=0 "
            "dropped_remove=0 completed_after_deadline=2")
        for key in ("hist_count", "hist_service_count", "hist_dispatch_count"):
            log = log.replace(key + "=1000", key + "=990")
        # Completed operations beyond 500 us remain valid histogram samples.
        log = log.replace("p99_ns=20000", "p99_ns=700000")
        result = kv_latency.parse_result(log, spec, expected_workers=24)["kv_latency"]
        self.assertEqual(result["phases"]["measurement"]["drop_fraction"], 0.01)
        self.assertEqual(result["p99_latency_ns"], 700000)
        bad = [("deadline_dropped=10", "deadline_dropped=9"),
               ("dropped_get=10", "dropped_get=9"),
               ("max_queue_delay_ns=500000", "max_queue_delay_ns=500001"),
               ("hist_count=990", "hist_count=1000"),
               ("completed_after_deadline=2", "completed_after_deadline=991")]
        for before, after in bad:
            with self.subTest(field=before), self.assertRaises(ValueError):
                kv_latency.parse_result(log.replace(before, after, 1), spec, expected_workers=24)
        disabled = dict(spec, max_queue_delay_us=0)
        with self.assertRaises(ValueError):
            kv_latency.parse_result(log, disabled, expected_workers=24)
        with self.assertRaises(ValueError):
            kv_latency.parse_result(log.replace("completed=990", "completed=0"),
                                    spec, expected_workers=24)

    def test_deadline_capability_and_limits(self):
        capability = mock.Mock(returncode=0,
            stdout="kvs_latency_capability schema=1 arrival=poisson_per_fibre "
                   "include_queue_wait=1 hist_sample_period=1\n", stderr="")
        with mock.patch.object(kv_latency.subprocess, "run", return_value=capability):
            with self.assertRaises(ValueError):
                kv_latency.probe(Path("/unused"), {"FARLIB_KVS_MAX_QUEUE_DELAY_US": "500"})
            capability.stdout = capability.stdout.rstrip() + " queue_deadline=drop_before_execution\n"
            kv_latency.probe(Path("/unused"), {"FARLIB_KVS_MAX_QUEUE_DELAY_US": "500"})
        args = arguments("/tmp/unused-latency-fixture")
        for value in (-1, True, 60_000_001):
            args.max_queue_delay_us = value
            with self.assertRaises(ValueError):
                kv_latency.specification(args)

    def test_valid_full_population_and_component_quantiles(self):
        spec = kv_latency.specification(arguments("/tmp/unused-latency-fixture"))
        result = kv_latency.parse_result(fixture(), spec, expected_workers=24)
        self.assertEqual(result["request_count"], 1000)
        self.assertEqual(result["kv_latency"]["p99_latency_ns"], 20000)
        self.assertEqual(result["kv_latency"]["phases"]["measurement"]
                         ["completed_in_window_ops_s"], 999)

    def test_reject_incomplete_loss_wrong_workload_and_bad_clock(self):
        spec = kv_latency.specification(arguments("/tmp/unused-latency-fixture"))
        cases = [
            ("status=passed", "status=invalid"),
            ("hist_dropped=0", "hist_dropped=1"),
            ("merge_dropped=0", "merge_dropped=1"),
            ("hist_write_errors=0", "hist_write_errors=1"),
            ("hist_service_count=1000", "hist_service_count=999"),
            ("skipped=0", "skipped=1"),
            ("arrival_seed=20260917", "arrival_seed=1"),
            ("object_bytes=512", "object_bytes=8192"),
            ("completed_put=50", "completed_put=49"),
            ("event=end monotonic_ns=2000001000", "event=end monotonic_ns=1999999999"),
        ]
        for before, after in cases:
            with self.subTest(field=before), self.assertRaises(ValueError):
                kv_latency.parse_result(fixture().replace(before, after, 1),
                                        spec, expected_workers=24)
        lines = fixture().splitlines()
        lines[2], lines[3] = lines[3], lines[2]
        with self.assertRaises(ValueError):
            kv_latency.parse_result("\n".join(lines), spec, expected_workers=24)

    def test_old_environment_unchanged_and_inherited_mode_is_removed(self):
        site = {"ib_device": "mlx5_1"}
        base = client_environment("kv-b", site, "nonft")
        args = arguments("/tmp/unused-latency-fixture", None)
        args.latency_warmup_ms = args.latency_measure_ms = args.latency_drain_ms = None
        self.assertIsNone(kv_latency.specification(args))
        self.assertEqual(kv_latency.environment(base, None), base)
        clean = client_process_environment("kv-b", site, "nonft",
                                          {"FARLIB_KVS_OFFERED_LOAD_OPS": "999"})
        self.assertNotIn("FARLIB_KVS_OFFERED_LOAD_OPS", clean)
        new = kv_latency.environment(base, kv_latency.specification(
            arguments("/tmp/unused-latency-fixture")))
        self.assertEqual(new["FARLIB_KVS_HIST_SAMPLE_PERIOD"], "1")
        self.assertNotIn("FARLIB_KVS_MAX_SERVE_COUNT", new)

    def test_preflight_rejects_old_binary_without_running_workload(self):
        old = mock.Mock(returncode=255, stdout="usage", stderr="")
        with mock.patch.object(kv_latency.subprocess, "run", return_value=old) as call:
            with self.assertRaises(ValueError):
                kv_latency.probe(Path("/unused/binary"), {})
            self.assertEqual(call.call_args.args[0][1], "--describe-latency-mode")
            self.assertEqual(call.call_args.kwargs["timeout"], 5)

    def test_collect_units_raw_reparse_and_failed_run_omission(self):
        location = COMMON.parent / "figure10/collect.py"
        module_spec = importlib.util.spec_from_file_location("collect_latency_test", location)
        collect = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(collect)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "kv-b-nonft-1000ops-r1"
            (run / "histograms").mkdir(parents=True)
            for phase in kv_latency.PHASES:
                for kind in kv_latency.HISTOGRAMS:
                    (run / "histograms" / f"{phase}.{kind}.hgrm").write_text("unit fixture")
            spec = kv_latency.specification(arguments(run))
            log = fixture()
            checked = kv_latency.parse_result(log, spec, expected_workers=24)
            analysis = {"status": "passed", "exit_status": 0, "correctness": "pass",
                        "application": "kv-b", "kv_latency": checked["kv_latency"],
                        "endpoints": [{"status": "stopped"}]}
            manifest = {"status": "passed", "plan": {
                "app": "kv-b", "system": "nonft", "kv_latency": spec,
                "worker_profile": {"app_workers": 24}, "effective_config": ""}}
            (run / "analysis.json").write_text(json.dumps(analysis))
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "client.log").write_text(log)
            point = collect.point(run)
            self.assertEqual(point["offered_load"], 0.001)
            self.assertEqual(point["p99_latency"], 20)
            analysis["status"] = "failed"
            (run / "analysis.json").write_text(json.dumps(analysis))
            self.assertIsNone(collect.point(run))
            other = Path(tmp) / "other-r1"
            other.mkdir()
            (other / "analysis.json").write_text("{}")
            rows = [dict(point, profile_id="one"), dict(point, offered_load_ops=2000,
                                                       profile_id="different")]
            with mock.patch.object(collect, "point", side_effect=rows):
                with self.assertRaisesRegex(ValueError, "mixed"):
                    collect.collect(Path(tmp), Path(tmp) / "must-not-exist.csv")
            self.assertFalse((Path(tmp) / "must-not-exist.csv").exists())


if __name__ == "__main__":
    unittest.main()
