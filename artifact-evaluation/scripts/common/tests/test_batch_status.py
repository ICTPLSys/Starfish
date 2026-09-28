"""Synthetic batch outcomes; no RDMA, datasets, or remote services."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_status import classify_case


class BatchStatus(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def record(self, **changes):
        value = {"status": "passed", "correctness": "pass", "exit_status": 0,
                 "error": "", "timed_out": False,
                 "client_cleanup": {"pid": 123, "reaped": True, "exit_status": 0},
                 "endpoints": [{"pid": 456, "status": "stopped"}]}
        value.update(changes)
        (self.root / "analysis.json").write_text(json.dumps(value))
        return value

    def test_success_requires_verified_record(self):
        self.assertEqual(classify_case(self.root, 0)["status"], "error")
        self.record()
        result = classify_case(self.root, 0)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["safe_to_continue"])
        self.record(endpoints=[{"status": "preflight_pass", "pid": None}])
        self.assertEqual(classify_case(self.root, 0)["status"], "error")

    def test_timeout_is_warning_only_after_cleanup(self):
        self.record(status="failed", correctness="fail", exit_status=124,
                    timed_out=True, error="client timed out after 1800 s")
        result = classify_case(self.root, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["safe_to_continue"])
        self.record(status="failed", exit_status=124, timed_out=True,
                    client_cleanup={"pid": 123, "reaped": False})
        result = classify_case(self.root, 1)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["safe_to_continue"])

    def test_program_exit_124_and_crash_are_errors(self):
        for code in (124, -11, -6):
            self.record(status="failed", correctness="fail", exit_status=code,
                        error=f"client exit status {code}")
            result = classify_case(self.root, 1)
            self.assertEqual(result["status"], "error")
            self.assertTrue(result["safe_to_continue"])

    def test_verification_and_preflight_errors_continue(self):
        result = classify_case(self.root, 2)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["safe_to_continue"])
        self.record(status="failed", correctness="fail", error="checksum mismatch")
        result = classify_case(self.root, 1)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["safe_to_continue"])

    def test_incomplete_service_cleanup_stops_batch(self):
        for state in ({"status": "cleanup_failed", "pid": 456},
                      {"status": "running", "pid": 456},
                      {"status": "start_failed", "pid": None, "remote_dir_created": True}):
            self.record(status="failed", exit_status=124, timed_out=True, endpoints=[state])
            self.assertFalse(classify_case(self.root, 1)["safe_to_continue"])

    def test_incomplete_or_corrupt_runner_record_stops_batch(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"status": "running"}))
        self.assertFalse(classify_case(self.root, 2)["safe_to_continue"])
        path.write_text("{")
        self.assertFalse(classify_case(self.root, 2)["safe_to_continue"])

    def test_preflight_endpoint_without_process_is_safe(self):
        endpoint = self.root / "endpoints/endpoint-00"
        endpoint.mkdir(parents=True)
        (endpoint / "endpoint-manifest.json").write_text(json.dumps(
            {"status": "preflight_pass", "pid": None, "remote_dir_created": False}))
        self.assertTrue(classify_case(self.root, 2)["safe_to_continue"])

    def test_interrupted_runner_is_not_a_skippable_case_error(self):
        self.record()
        for code in (-15, -9, 130, 143):
            self.assertFalse(classify_case(self.root, code)["safe_to_continue"])

    def test_legacy_timeout_and_already_exited_services(self):
        value = self.record(status="failed", correctness="fail", exit_status=124,
                            endpoints=[{"pid": 456, "status": "already_exited"}])
        value.pop("timed_out")
        value.pop("client_cleanup")
        (self.root / "analysis.json").write_text(json.dumps(value))
        result = classify_case(self.root, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["safe_to_continue"])


if __name__ == "__main__":
    unittest.main()
