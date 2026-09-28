"""Hardware-free runtime EC TSC producer contract tests."""
import subprocess
import tempfile
import unittest
from pathlib import Path

AE = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("runtime_ec_cpu_test.cpp")


class RuntimeEcCpuTest(unittest.TestCase):
    def test_disabled_invalid_and_enabled_windows(self):
        with tempfile.TemporaryDirectory(prefix="runtime-ec-cpu-test-") as directory:
            binary = Path(directory) / "runtime-ec-cpu"
            subprocess.run(
                ["g++", "-std=c++20", "-O1", "-pthread", "-Wall", "-Wextra",
                 "-Werror", "-I", str(AE / "runtime/common"), str(SOURCE),
                 "-o", str(binary)],
                check=True, capture_output=True, text=True, timeout=90)
            disabled = subprocess.run([str(binary), "disabled"], check=True,
                                       capture_output=True, text=True, timeout=10)
            self.assertNotIn("runtime_ec_cpu ", disabled.stdout)
            invalid = subprocess.run([str(binary), "invalid"], check=True,
                                     capture_output=True, text=True, timeout=10)
            self.assertIn("invalid rejected", invalid.stdout)
            enabled = subprocess.run([str(binary), "enabled"], check=True,
                                     capture_output=True, text=True, timeout=10)
            rows = [line for line in enabled.stdout.splitlines()
                    if line.startswith("runtime_ec_cpu ")]
            self.assertEqual(len(rows), 2, enabled.stdout)
            records = []
            for row in rows:
                fields = dict(token.split("=", 1) for token in row.split()[1:])
                self.assertEqual(fields["schema_version"], "1")
                self.assertEqual(fields["phase"], "work")
                self.assertEqual(fields["scope"], "compute_ec")
                self.assertEqual(fields["clock"], "tsc")
                self.assertGreater(int(fields["cycles"]), 0)
                self.assertGreater(int(fields["scopes"]), 0)
                records.append(fields)
            self.assertEqual([int(record["boundary_sequence"]) for record in records], [1, 2])
            self.assertEqual([record["system"] for record in records], ["starfish", "carbink"])
            self.assertEqual(int(records[0]["scopes"]), 2)
            self.assertEqual(int(records[1]["scopes"]), 1)
            metadata = [line for line in enabled.stdout.splitlines()
                        if line.startswith("runtime_metadata ")]
            self.assertEqual(len(metadata), 1)
            self.assertIn("boundary_sequence=2", metadata[0])
            gate = subprocess.run([str(binary), "gate"], check=True,
                                  capture_output=True, text=True, timeout=10)
            gate_rows = [dict(token.split("=", 1) for token in line.split()[1:])
                         for line in gate.stdout.splitlines()
                         if line.startswith("runtime_ec_cpu ")]
            self.assertEqual([int(row["scopes"]) for row in gate_rows], [0, 1])
            stress = subprocess.run([str(binary), "stress"], check=True,
                                    capture_output=True, text=True, timeout=20)
            stress_rows = [dict(token.split("=", 1) for token in line.split()[1:])
                           for line in stress.stdout.splitlines()
                           if line.startswith("runtime_ec_cpu ")]
            self.assertEqual([int(row["boundary_sequence"]) for row in stress_rows],
                             list(range(1, 101)))


if __name__ == "__main__":
    unittest.main()
