"""Recovery orchestration tests: real local children, no RDMA or host faults."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recovery_check
import run_case


PROOF = {"endpoint": 0, "owned_identity_verified": True, "process_stopped": True}
DIAG = ("ec_read_recovery diag [cache_dtor_begin]: dead_endpoints=1/7 "
        "completions=3 verify_pass=3 verify_fail=0 rebuild_failures=0 "
        "tokens_in_use=0 scratch_in_use=0 split_buffers_in_use=0")
LOG = "INFO: ec_recovery endpoint_dead endpoint=0 monotonic_ns=123\n" + DIAG


class RecoveryCheck(unittest.TestCase):
    def test_four_read_recovery_requires_independent_byte_oracle(self):
        log = LOG.replace("verify_pass=3", "verify_pass=0")
        log = log.replace("INFO: ec_recovery endpoint_dead endpoint=0 monotonic_ns=123",
                          "INFO: ec_recovery endpoint_dead endpoint=ERROR: interleaved\n"
                          "INFO: ec_recovery degraded_read missing_shard=2 endpoint_dead=0")
        with self.assertRaises(ValueError):
            recovery_check.validate_log(log, PROOF)
        result = recovery_check.validate_log(log, PROOF, answer_matches_reference=True)
        self.assertEqual(result["completions"], 3)
        self.assertEqual(result["verify_pass"], 0)
        with self.assertRaises(ValueError):
            recovery_check.validate_log(log.replace("verify_fail=0", "verify_fail=1"),
                                        PROOF, answer_matches_reference=True)
        with tempfile.TemporaryDirectory() as root:
            capture, reference = Path(root) / "chat", Path(root) / "nonft"
            reference.write_bytes(b"full answer")
            capture.write_bytes(b"full answer")
            self.assertTrue(recovery_check.compare_chat(capture, reference)["matched"])
            capture.write_bytes(b"full answeX")
            with self.assertRaises(ValueError):
                recovery_check.compare_chat(capture, reference)

    def test_verified_recovery_and_healthy_control(self):
        self.assertEqual(recovery_check.validate_log(LOG, PROOF)["completions"], 3)
        healthy = DIAG.replace("dead_endpoints=1/7", "dead_endpoints=0/7")
        self.assertEqual(recovery_check.validate_log(healthy)["dead_endpoints"], 0)
        with self.assertRaises(ValueError):
            recovery_check.validate_log(LOG)

    def test_no_false_pass_on_absent_fault_or_invalid_counters(self):
        for bad in ("", LOG + "\n" + DIAG, LOG.replace("endpoint=0", "endpoint=1"),
                    LOG.replace("completions=3", "completions=0"),
                    LOG.replace("verify_pass=3", "verify_pass=0"),
                    LOG.replace("verify_fail=0", "verify_fail=1"),
                    LOG.replace("rebuild_failures=0", "rebuild_failures=1"),
                    LOG.replace("tokens_in_use=0", "tokens_in_use=1")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                recovery_check.validate_log(bad, PROOF)
        with self.assertRaises(ValueError):
            recovery_check.validate_log(LOG, {"endpoint": 0})

    def test_invalid_requests_fail_before_side_effects(self):
        recovery_check.validate_request("llama", "starfish", 0, 7, "ec_batch", "6")
        for values in (("llama", "nonft", 0, 7, "ec_batch", "6"),
                       ("llama", "starfish", 0, 7, "none", "6"),
                       ("llama", "starfish", 0, 6, "ec_batch", "6"),
                       ("llama", "starfish", 6, 7, "ec_batch", "6")):
            with self.assertRaises(ValueError):
                recovery_check.validate_request(*values)

    def test_environment_cannot_inherit_fault_observer_overrides(self):
        original = {key: "bad" for key in recovery_check.CONTROL_ENV}
        original["PATH"] = "/usr/bin"
        env = recovery_check.client_env(original, None, False)
        self.assertEqual(env, {"PATH": "/usr/bin"})
        env = recovery_check.client_env(original, Path("/tmp/chat"), True)
        self.assertEqual(env["FARLIB_EC_RECOVERY_VERIFY"], "1")
        self.assertEqual(env["FARLIB_CAPTURE_CHAT_OUTPUT"], "/tmp/chat")
        self.assertEqual(original["FARLIB_EC_RECOVERY_VERIFY"], "bad")

    def test_benchmark_origin_is_injected_for_memory_or_cpu_opt_in(self):
        command = [
            sys.executable, "-c",
            "import os; print(os.environ.get("
            "'FARLIB_BENCHMARK_START_MONOTONIC_NS', ''))",
        ]
        with tempfile.TemporaryDirectory() as root:
            for env, cpu_opt_in in (
                    ({"FARLIB_REMOTE_MEMORY_SAMPLES": "1"}, False),
                    ({}, True),
                    ({}, False)):
                execution = {}
                output_path = Path(root) / ("origin-" + str(len(list(Path(root).iterdir()))))
                with output_path.open("w", encoding="utf-8") as output:
                    status = recovery_check.run_client(
                        command, stdin=None, stdout=output, env=env, timeout=3,
                        capture=None, inject=None, evidence={},
                        execution_state=execution,
                        benchmark_start_env=cpu_opt_in)
                self.assertEqual(status, 0)
                origin = output_path.read_text(encoding="utf-8").strip()
                if env or cpu_opt_in:
                    self.assertTrue(origin.isdigit())
                    self.assertEqual(
                        int(origin), execution["benchmark_start_monotonic_ns"])
                else:
                    self.assertEqual(origin, "")
                    self.assertNotIn("benchmark_start_monotonic_ns", execution)


    def test_first_answer_triggers_exactly_one_injection(self):
        with tempfile.TemporaryDirectory() as root, open(os.devnull, "w") as output:
            capture = Path(root) / "chat"
            evidence = {}
            command = [sys.executable, "-c",
                       "import pathlib,time; pathlib.Path(" + repr(str(capture)) +
                       ").write_text('hello'); time.sleep(.2)"]
            with patch("run_case.inject_owned_failure", return_value=PROOF) as inject:
                status = recovery_check.run_client(
                    command, stdin=None, stdout=output, env=os.environ, timeout=3,
                    capture=capture, inject=lambda: inject({}), evidence=evidence)
            self.assertEqual(status, 0)
            self.assertEqual(inject.call_count, 1)
            self.assertEqual(evidence, PROOF)

    def test_early_exit_timeout_and_callback_error_reap_child(self):
        with tempfile.TemporaryDirectory() as root, open(os.devnull, "w") as output:
            for source, timeout, expected in (
                    ("pass", 2, RuntimeError),
                    ("import time; time.sleep(20)", .1, subprocess.TimeoutExpired),
                    ("import pathlib,time; pathlib.Path(" + repr(root + "/chat") +
                     ").write_text('hi'); time.sleep(20)", 2, RuntimeError)):
                capture = Path(root) / "chat"
                if capture.exists():
                    capture.unlink()
                children = []
                real_popen = subprocess.Popen
                def spawn(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    children.append(child)
                    return child
                def inject():
                    raise RuntimeError("ownership mismatch")
                with patch("recovery_check.subprocess.Popen", side_effect=spawn):
                    execution = {}
                    with self.assertRaises(expected):
                        recovery_check.run_client(
                            [sys.executable, "-c", source], stdin=None, stdout=output,
                            env=os.environ, timeout=timeout, capture=capture,
                            inject=inject, evidence={}, execution_state=execution)
                self.assertTrue(execution["reaped"])
                self.assertIsNotNone(execution["exit_status"])
                self.assertTrue(all(child.poll() is not None for child in children))

    def test_identity_is_checked_before_signal_and_no_name_based_kill(self):
        state = dict(pid=123, server_bin="/tmp/owned-server", remote_dir="/tmp/owned",
                     starttime="234", index=0, memory_host="memory")
        def reject(host, script, check=False):
            self.assertIn("uid=", script)
            self.assertLess(script.index("exe="), script.index("kill -KILL"))
            self.assertIn('if [ "$start" != 234 ]', script)
            self.assertNotIn("pkill", script)
            self.assertNotIn("killall", script)
            return subprocess.CompletedProcess([], 11, "", "")
        with patch.object(run_case, "ssh", side_effect=reject):
            with self.assertRaises(RuntimeError):
                run_case.inject_owned_failure(state)


if __name__ == "__main__":
    unittest.main()
