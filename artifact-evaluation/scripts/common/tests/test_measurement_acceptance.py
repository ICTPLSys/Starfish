"""Verified measurement-only acceptance; no runtime or remote hosts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_status import classify_case
from measurement_acceptance import recover_measurement
from workloads import parse_result


def log():
    return """simple_hotcold.config enabled=1 mode=local_resident
simple_remote_hotcold.config enabled=1
simple_region_budget.config semantic_classes=6 local_resident=1
kvs.zipfian: 0.99
kvs.fixed_request_count: 1
kvs_validation_config get_content=0 real_put=1
kvs_direct_config fibres=48 object_bytes=512 initial_count=33554432 put_ratio=0.5
kvs_phase name=direct event=request_start monotonic_ns=1000000000
kvs_phase name=direct event=request_generation_end monotonic_ns=87000000000 generated=1000000000 accepted=1000000000
kvs_phase name=direct event=request_drain_end monotonic_ns=87000000001 completed=1000000000
kvs_receipt name=direct generated_get=500000000 generated_put=500000000 generated_remove=0 accepted_get=500000000 accepted_put=500000000 accepted_remove=0 completed_get=500000000 completed_put=500000000 completed_remove=0 accepted_equals_completed=1
kvs_post_verify checked=4096 failures=0
""" + "".join(f"simple_region_budget.six_final domain=local bin=0 class={c} actual=1\n"
              for c in range(6))


class MeasurementAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "client.log").write_text(log())
        endpoint = {"pid": 456, "status": "stopped"}
        self.analysis = {"application": "kv-a", "status": "failed", "correctness": "fail",
                         "exit_status": -15, "endpoints": [endpoint],
                         "error": "client exit status -15"}
        self.manifest = {"status": "failed", "plan": {"app": "kv-a", "system": "starfish"},
                         "endpoints": [endpoint]}

    def recover(self):
        return recover_measurement(self.root, self.analysis, self.manifest)

    def classify(self):
        (self.root / "analysis.json").write_text(json.dumps(self.analysis))
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        return classify_case(self.root, 1)

    def test_measurement_valid_but_lifecycle_still_failed(self):
        with self.assertRaises(ValueError):
            parse_result("kv-a", log(), expected_post_samples=4096)
        result = self.recover()
        self.assertEqual(result["request_count"], 1_000_000_000)
        self.assertAlmostEqual(result["elapsed_s"], 86.000000001)
        classified = self.classify()
        self.assertEqual(classified["status"], "warning")
        self.assertEqual(classified["client_exit_status"], -15)
        self.assertTrue(classified["measurement_usable"])
        self.assertTrue(classified["safe_to_continue"])
        self.assertEqual(json.loads((self.root / "analysis.json").read_text())["status"], "failed")

    def test_timeout_after_verified_work_is_usable(self):
        self.analysis.update(exit_status=124, timed_out=True)
        self.assertIsNotNone(self.recover())
        self.assertEqual(self.classify()["status"], "warning")
        self.analysis["timed_out"] = False
        self.assertIsNone(self.recover())

    def test_crashes_and_other_profiles_are_not_relaxed(self):
        for code in (-11, -6, 1, 0):
            self.analysis["exit_status"] = code
            self.assertIsNone(self.recover())
        self.analysis["exit_status"] = -15
        for system in ("nonft", "hydra", "carbink"):
            self.manifest["plan"]["system"] = system
            self.assertIsNone(self.recover())

    def test_partial_or_corrupt_work_is_rejected(self):
        for old, new in (
                ("completed=1000000000", "completed=999999999"),
                ("checked=4096", "checked=4095"),
                ("failures=0", "failures=1"),
                ("kvs_post_verify checked=4096 failures=0\n", ""),
                ("semantic_classes=6", "semantic_classes=5")):
            (self.root / "client.log").write_text(log().replace(old, new))
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.recover()

    def test_live_service_stops_batch_without_discarding_verified_work(self):
        self.analysis["endpoints"][0]["status"] = "cleanup_failed"
        classified = self.classify()
        self.assertEqual(classified["status"], "error")
        self.assertFalse(classified["safe_to_continue"])
        self.assertTrue(classified["measurement_usable"])


if __name__ == "__main__":
    unittest.main()
