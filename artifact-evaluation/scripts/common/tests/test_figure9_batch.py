"""Synthetic Figure 9 batch orchestration tests.

The fixture copies only the shell runner and classifier into a temporary
artifact-evaluation tree. Every runtime-facing helper is a local Python stub;
these tests never contact SSH hosts or launch a benchmark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


AE_ROOT = Path(__file__).resolve().parents[3]
RUN_SOURCE = AE_ROOT / "scripts" / "figure9" / "run.sh"
STATUS_SOURCE = AE_ROOT / "scripts" / "common" / "batch_status.py"


RUN_CASE_STUB = r'''#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--app", required=True)
parser.add_argument("--system", required=True)
parser.add_argument("--ratio", type=int, required=True)
parser.add_argument("--site", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--build-root")
parser.add_argument("--timeout", required=True)
parser.add_argument("--baseline-variant")
parser.add_argument("--collect-remote-cpu", action="store_true")
parser.add_argument("--collect-remote-memory", action="store_true")
parser.add_argument("--repeat", type=int)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

run_id = args.out.name
if args.dry_run:
    fail = os.environ.get("FIGURE9_STUB_PLAN_FAIL", "")
    if fail == f"{args.system}/{args.app}/{args.ratio}":
        print("synthetic plan failure", flush=True)
        raise SystemExit(2)
    print(json.dumps({"app": args.app, "system": args.system,
                      "ratio": args.ratio, "run_id": run_id}))
    raise SystemExit(0)

timeout_ratio = int(os.environ.get("FIGURE9_STUB_TIMEOUT_RATIO", "-1"))
error_ratio = int(os.environ.get("FIGURE9_STUB_ERROR_RATIO", "-1"))
unsafe_ratio = int(os.environ.get("FIGURE9_STUB_UNSAFE_RATIO", "-1"))
if args.ratio == timeout_ratio:
    behavior = "timeout"
elif args.ratio == error_ratio:
    behavior = "error"
elif args.ratio == unsafe_ratio:
    behavior = "unsafe"
else:
    behavior = "pass"

args.out.mkdir(parents=True)
endpoint_status = "running" if behavior == "unsafe" else "stopped"
endpoint = {"status": endpoint_status, "pid": 1}
if behavior == "pass":
    analysis = {
        "status": "passed", "correctness": "pass", "exit_status": 0,
        "endpoints": [endpoint], "client_cleanup": {"reaped": True},
    }
    manifest_status = "passed"
    rc = 0
elif behavior == "timeout":
    analysis = {
        "status": "failed", "correctness": "fail", "exit_status": 124,
        "timed_out": True, "error": "synthetic client timeout",
        "endpoints": [endpoint], "client_cleanup": {"reaped": True},
    }
    manifest_status = "failed"
    rc = 1
else:
    analysis = {
        "status": "failed", "correctness": "fail", "exit_status": 1,
        "error": "synthetic runner crash",
        "endpoints": [endpoint], "client_cleanup": {"reaped": True},
    }
    manifest_status = "failed"
    rc = 1
manifest = {"status": manifest_status, "endpoints": [endpoint]}
(args.out / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
(args.out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
print(f"synthetic runner detail {run_id}", flush=True)
raise SystemExit(rc)
'''


SUPPORT_STUB = r'''#!/usr/bin/env python3
raise SystemExit(0)
'''


COLLECT_STUB = r'''#!/usr/bin/env python3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--runs-dir", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.write_text(
    "workload,system,ratio,elapsed_s\nLLM,Non-FT,25,1.0\n",
    encoding="utf-8",
)
print("synthetic collector detail", flush=True)
'''


PLOT_STUB = r'''#!/usr/bin/env python3
print("synthetic plot detail", flush=True)
'''


class Figure9Batch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts" / "figure9").mkdir(parents=True)
        (self.root / "scripts" / "common").mkdir(parents=True)
        shutil.copy2(RUN_SOURCE, self.root / "scripts" / "figure9" / "run.sh")
        shutil.copy2(STATUS_SOURCE, self.root / "scripts" / "common" / "batch_status.py")
        self._write("scripts/common/figure9_support.py", SUPPORT_STUB)
        self._write("scripts/common/run_case.py", RUN_CASE_STUB)
        self._write("scripts/figure9/collect.py", COLLECT_STUB)
        self._write("scripts/figure9/plot.py", PLOT_STUB)
        self.site = self.root / "site.json"
        self.site.write_text("{}", encoding="utf-8")
        deps = self.root / "python-deps"
        deps.mkdir()
        (deps / "matplotlib.py").write_text("", encoding="utf-8")
        (deps / "numpy.py").write_text("", encoding="utf-8")
        self.env = os.environ.copy()
        self.env["PYTHON"] = sys.executable
        self.env["PYTHONPATH"] = str(deps)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, relative, content):
        path = self.root / relative
        path.write_text(content, encoding="utf-8")

    def run_batch(self, *args, **extra_env):
        command = ["bash", str(self.root / "scripts" / "figure9" / "run.sh"), *args]
        env = self.env.copy()
        env.update({key: str(value) for key, value in extra_env.items()})
        return subprocess.run(
            command, cwd=self.root, env=env, text=True,
            capture_output=True, timeout=60, check=False,
        )

    def test_default_dry_run_is_full_system_major_matrix(self):
        output_dir = self.root / "default-batch"
        result = self.run_batch(
            "--site", str(self.site), "--out", str(output_dir), "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "This script will run NonFT, NonFT-backup-off, Starfish, Hydra, Carbink sequentially.",
            result.stdout,
        )
        plan_lines = [line for line in result.stdout.splitlines()
                      if line.startswith("PLAN ")]
        systems = ("NONFT", "NONFT-BACKUP-OFF", "STARFISH", "HYDRA", "CARBINK")
        apps = ("LLAMA", "BFS", "MG", "WORDCOUNT", "KV-B", "KV-A", "KV-S", "NQ")
        ratios = (13, 25, 50, 75, 100)
        expected = [
            (system, app, ratio)
            for system in systems for app in apps for ratio in ratios
        ]
        self.assertEqual(len(plan_lines), len(expected))
        observed = []
        for line in plan_lines:
            fields = line.split()
            observed.append((fields[1], fields[2], int(fields[3][:-1])))
            self.assertEqual(fields[-2:], ["r1", "OK"])
        self.assertEqual(observed, expected)
        self.assertFalse(output_dir.exists(), "dry-run must not create batch files")

    def test_dry_run_plan_failure_is_nonzero_and_read_only(self):
        output_dir = self.root / "dry-failure"
        result = self.run_batch(
            "--systems", "starfish", "--apps", "llama", "--ratios", "13,25",
            "--site", str(self.site), "--out", str(output_dir), "--dry-run",
            FIGURE9_STUB_PLAN_FAIL="starfish/llama/13",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("PLAN STARFISH LLAMA 13% r1 ERROR", result.stdout)
        self.assertIn("PLAN STARFISH LLAMA 25% r1 OK", result.stdout)
        self.assertIn("plan_errors=1", result.stdout)
        self.assertFalse(output_dir.exists(), "failed dry-run must not write output")

    def test_default_batch_publishes_within_clone_root(self):
        output_dir = self.root / "local-batch"
        result = self.run_batch(
            "--systems", "nonft", "--apps", "llama", "--ratios", "25",
            "--site", str(self.site), "--out", str(output_dir),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((output_dir / "figure9.csv").is_file())
        self.assertTrue((self.root / "data" / "figure9.csv").is_file())
        self.assertNotIn("shared publication disabled", result.stdout)

    def test_timeout_and_error_are_warnings_or_errors_then_continue(self):
        output_dir = self.root / "continue-batch"
        result = self.run_batch(
            "--systems", "nonft", "--apps", "llama", "--ratios", "13,25,50,75",
            "--site", str(self.site), "--out", str(output_dir),
            FIGURE9_STUB_TIMEOUT_RATIO=25, FIGURE9_STUB_ERROR_RATIO=50,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("NonFT Start...", result.stdout)
        self.assertIn("NONFT LLAMA 13% finish", result.stdout)
        self.assertIn("NONFT LLAMA 25% WARNING:", result.stdout)
        self.assertIn("NONFT LLAMA 50% ERROR:", result.stdout)
        self.assertIn("NONFT LLAMA 75% finish", result.stdout)
        self.assertIn("NONFT App LLAMA FINISH! (failures=2)", result.stdout)
        self.assertIn("Figure 9 summary: passed=2 timeouts=1 errors=1", result.stdout)
        self.assertTrue(
            (output_dir / "runs" / "llama-nonft-75-r1").is_dir(),
            "safe failures must not skip later cases",
        )
        batch_log = (output_dir / "batch.log").read_text(encoding="utf-8")
        self.assertIn("synthetic runner detail llama-nonft-50-r1", batch_log)
        self.assertNotIn("synthetic runner detail", result.stdout)

    def test_unsafe_cleanup_stops_later_cases(self):
        output_dir = self.root / "unsafe-batch"
        result = self.run_batch(
            "--systems", "nonft", "--apps", "llama", "--ratios", "13,25,50",
            "--site", str(self.site), "--out", str(output_dir),
            FIGURE9_STUB_UNSAFE_RATIO=25,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("NONFT LLAMA 13% finish", result.stdout)
        self.assertIn("NONFT LLAMA 25% ERROR:", result.stdout)
        self.assertIn("NONFT App LLAMA ABORTED! (failures=1)", result.stdout)
        self.assertIn("NONFT ABORTED! (failures=1)", result.stdout)
        self.assertIn("unsafe_cleanup=1", result.stdout)
        self.assertNotIn("NONFT LLAMA 50%", result.stdout)
        self.assertFalse(
            (output_dir / "runs" / "llama-nonft-50-r1").exists(),
            "unsafe cleanup must stop before later cases",
        )


if __name__ == "__main__":
    unittest.main()
