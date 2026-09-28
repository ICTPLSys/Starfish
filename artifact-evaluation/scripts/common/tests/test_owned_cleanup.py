"""Local subprocess regressions for memory-server ownership and exit races."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_case


@unittest.skipUnless(sys.platform == "linux", "requires Linux procfs")
class OwnedCleanup(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ae-owned-cleanup-")
        self.config = Path(self.directory.name) / "server.config"
        self.config.write_text("while :; do sleep 0.1; done\n")
        self.process = subprocess.Popen(
            ["/bin/bash", "server.config"], cwd=self.directory.name)
        fields = Path(f"/proc/{self.process.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        self.state = dict(pid=self.process.pid, server_bin="/bin/bash",
                          remote_dir=self.directory.name, starttime=fields[19])
        self.prefix = ""
        self.patched_ssh = mock.patch.object(run_case, "ssh", self.local_ssh)
        self.patched_ssh.start()

    def tearDown(self):
        self.patched_ssh.stop()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=3)
        self.directory.cleanup()

    def local_ssh(self, host, script, check=True, timeout=60):
        return subprocess.run(
            ["bash", "-c", self.prefix + script], text=True,
            capture_output=True, check=False, timeout=20)

    def test_identity_and_live_mismatches(self):
        valid = ("/bin/bash", str(self.config), self.state["starttime"])
        script = run_case.process_identity_script(self.process.pid, *valid)
        self.assertEqual(self.local_ssh("", script).returncode, 0)
        variants = {
            "start": (valid[0], valid[1], "0"),
            "exe": ("/bin/sleep", valid[1], valid[2]),
            "argv": (valid[0], str(self.config.with_name("wrong.config")), valid[2]),
            "cwd": (valid[0], "/tmp/server.config", valid[2]),
        }
        for name, args in variants.items():
            with self.subTest(name=name):
                result = self.local_ssh(
                    "", run_case.process_identity_script(self.process.pid, *args))
                self.assertEqual(result.returncode, 11, result.stderr)
                self.assertIsNone(self.process.poll())

    def test_wrong_uid_cleanup_never_signals(self):
        self.prefix = f"stat() {{ echo {os.getuid() + 1}; }}; "
        with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
            run_case.stop_server_process("", self.state)
        self.assertIsNone(self.process.poll())

    def test_owned_term_and_absent_process(self):
        self.assertEqual(run_case.stop_server_process("", self.state), "stopped")
        self.process.wait(timeout=3)
        self.assertEqual(run_case.stop_server_process("", self.state), "already_exited")

    def test_exit_between_stat_and_exe_read(self):
        self.prefix = (
            'readlink() { case "$*" in *"/proc/$pid/exe"*) '
            'command kill -TERM "$pid"; sleep 0.15;; esac; '
            'command readlink "$@"; }; ')
        self.assertEqual(run_case.stop_server_process("", self.state), "already_exited")

    def test_exit_between_identity_and_term(self):
        self.prefix = (
            'kill() { if [ "$1" = -TERM ]; then command kill -TERM "$2"; '
            'sleep 0.15; return 1; fi; command kill "$@"; }; ')
        self.assertEqual(run_case.stop_server_process("", self.state), "stopped")


if __name__ == "__main__":
    unittest.main()
