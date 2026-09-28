import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import parse_result


def native_log():
    return """kvs.zipfian: 0.99
kvs.fixed_request_count: 1
kvs_validation_config get_content=0 real_put=1
kvs_direct_config fibres=48 object_bytes=512 initial_count=33554432 put_ratio=0.05
kvs_phase name=direct event=request_start monotonic_ns=1000000000
kvs_phase name=direct event=request_generation_end monotonic_ns=2000000000 generated=20 accepted=20
kvs_phase name=direct event=request_drain_end monotonic_ns=2000000001 completed=20
kvs_receipt name=direct generated_get=19 generated_put=1 generated_remove=0 accepted_get=19 accepted_put=1 accepted_remove=0 completed_get=19 completed_put=1 completed_remove=0 accepted_equals_completed=1
kvs_post_verify checked=4096 failures=0
exact used bytes: 0
"""


class KvResult(unittest.TestCase):
    def parse(self, log):
        return parse_result("kv-b", log, expected_requests=20, expected_post_samples=4096)

    def test_current_binary_validation_fields(self):
        result = self.parse(native_log())
        self.assertEqual(result["request_count"], 20)
        self.assertIn("sampled value check", result["correctness_scope"])

    def test_missing_short_or_failed_postcheck_is_rejected(self):
        for replacement in ("", "kvs_post_verify checked=1 failures=0",
                            "kvs_post_verify checked=4096 failures=1"):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.parse(native_log().replace("kvs_post_verify checked=4096 failures=0",
                                                replacement))

    def test_postcheck_must_follow_requests(self):
        line = "kvs_post_verify checked=4096 failures=0\n"
        with self.assertRaises(ValueError):
            self.parse(line + native_log().replace(line, ""))

    def test_legacy_declaration_still_requires_postcheck(self):
        log = native_log().replace("real_put=1", "real_put=1 post_samples=4096")
        log = log.replace("kvs_post_verify checked=4096 failures=0\n", "")
        with self.assertRaises(ValueError):
            parse_result("kv-b", log, expected_requests=20)
