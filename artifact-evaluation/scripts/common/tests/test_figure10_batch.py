"""Mocked Figure 10 batch planning, status, and collection checks."""
import argparse
import contextlib
import csv
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve()
COMMON = HERE.parents[1]
FIGURE10 = COMMON.parent / "figure10"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(FIGURE10))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load_module("figure10_batch_run", FIGURE10 / "run.py")
collect_module = load_module("figure10_batch_collect", FIGURE10 / "collect.py")
from paper_loads import OFFERED_LOAD_OPS


def plan_args(root, *, apps="kv-b,nq", systems="nonft,starfish,hydra,carbink",
              loads_ops=None, recipe=None, repeats=1):
    root = Path(root)
    site = root / "site.json"
    site.write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        apps=apps, systems=systems, site=site, ratio=25,
        build_root=root / "build", config_root=root / "configs", recipe=recipe,
        loads_ops=loads_ops, warmup_ms=10000, measure_ms=10000, drain_ms=60000,
        max_queue_delay_us=None, repeats=repeats, timeout=3600,
        out=root / "batch", dry_run=False)


def parsed_args(root, **changes):
    args = plan_args(root, **{key: value for key, value in changes.items()
                              if key in {"apps", "systems", "loads_ops", "recipe", "repeats"}})
    for key, value in changes.items():
        setattr(args, key, value)
    return args


class Figure10Planning(unittest.TestCase):
    def test_default_matrix_is_system_major_kv_then_nq_with_existing_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, repeats=2)
            matrix = list(run.plans(args))
            self.assertEqual([(p["system"], p["app"]) for p in matrix[:4]],
                             [("nonft", "kv-b"), ("nonft", "kv-b"),
                              ("nonft", "kv-b"), ("nonft", "kv-b")])
            curves = []
            for plan in matrix:
                curve = (plan["system"], plan["app"])
                if not curves or curves[-1] != curve:
                    curves.append(curve)
            self.assertEqual(curves, [(system, app) for system in run.SYSTEMS
                                      for app in run.APPS])
            for system in run.SYSTEMS:
                for app in run.APPS:
                    expected = len(OFFERED_LOAD_OPS[app][system]) * 2
                    self.assertEqual(sum(1 for p in matrix
                                         if p["system"] == system and p["app"] == app),
                                     expected)
            kv = next(p for p in matrix if p["app"] == "kv-b")
            nq = next(p for p in matrix if p["app"] == "nq")
            self.assertEqual(kv["command"][-1], "0")
            self.assertEqual(nq["command"][-1], "1500000")

    def test_explicit_rate_and_recipe_reject_ambiguous_multiapp_or_multisystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "exactly one selected application"):
                list(run.plans(plan_args(temporary, loads_ops="1000")))
            recipe = Path(temporary) / "recipe.config"
            with self.assertRaisesRegex(ValueError, "one selected application"):
                list(run.plans(plan_args(temporary, recipe=recipe)))
            with self.assertRaisesRegex(ValueError, "one selected application"):
                list(run.plans(plan_args(temporary, apps="kv-b", recipe=recipe,
                                         systems="nonft,starfish")))
            args = plan_args(temporary, apps="nq", systems="nonft",
                             loads_ops="1000,2000", recipe=recipe)
            matrix = list(run.plans(args))
            self.assertEqual([p["offered_load_ops"] for p in matrix], [1000, 2000])


class Figure10Execution(unittest.TestCase):
    def test_usable_teardown_measurement_is_reported_as_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, apps="kv-b", systems="starfish", loads_ops="1000")
            outcome = {"status": "warning", "safe_to_continue": True,
                       "reason": "verified measurement retained; shutdown timed out",
                       "measurement_usable": True, "exit_status": 1, "client_exit_status": 124}
            fake = mock.Mock(returncode=1)
            with mock.patch.object(run.subprocess, "run", return_value=fake), \
                 mock.patch.object(run, "classify_case", return_value=outcome), \
                 mock.patch.object(run, "collect", return_value=[]):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(run.execute(argparse.ArgumentParser(), args), 1)
            self.assertIn("STARFISH KV-B 1000 QPS WARNING", stderr.getvalue())
            self.assertIn("warnings=1 errors=0", stdout.getvalue())
            self.assertNotIn("1000 QPS finish", stdout.getvalue())
            record = json.loads((args.out / "batch-status.json").read_text())[0]
            self.assertTrue(record["measurement_usable"])
            self.assertEqual(record["client_exit_status"], 124)

    def test_setup_failures_continue_with_real_classifier_and_empty_collector(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, apps="kv-b", systems="nonft", loads_ops="1000,2000")
            fake = mock.Mock(returncode=2, stdout="", stderr="")
            with mock.patch.object(run.subprocess, "run", return_value=fake) as invoke:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run.execute(argparse.ArgumentParser(), args), 1)
            self.assertEqual(invoke.call_count, 2)
            status = json.loads((args.out / "batch-status.json").read_text())
            self.assertEqual([record["status"] for record in status], ["error", "error"])
            self.assertTrue(all(record["safe_to_continue"] for record in status))
            with (args.out / "figure10.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_dry_run_prints_only_json_and_does_not_create_batch_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, apps="kv-b", systems="nonft", loads_ops="1000")
            args.dry_run = True
            parser = argparse.ArgumentParser()
            fake = mock.Mock(returncode=0, stdout="{}\n", stderr="")
            with mock.patch.object(run.subprocess, "run", return_value=fake) as invoke:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(run.execute(parser, args), 0)
            lines = stream.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIsInstance(json.loads(lines[0]), dict)
            self.assertEqual(invoke.call_count, 1)
            self.assertFalse(args.out.exists())

    def test_safe_failures_continue_but_final_status_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, apps="kv-b", systems="nonft,starfish",
                             loads_ops="1000,2000")
            parser = argparse.ArgumentParser()
            outcomes = iter((
                {"status": "timeout", "safe_to_continue": True, "reason": "deadline",
                 "exit_status": 1, "client_exit_status": 124},
                {"status": "error", "safe_to_continue": True, "reason": "checksum",
                 "exit_status": 1, "client_exit_status": 1},
                {"status": "passed", "safe_to_continue": True, "reason": "ok",
                 "exit_status": 0, "client_exit_status": 0},
                {"status": "passed", "safe_to_continue": True, "reason": "ok",
                 "exit_status": 0, "client_exit_status": 0},
            ))
            fake = mock.Mock(returncode=1, stdout="", stderr="")
            with mock.patch.object(run.subprocess, "run", return_value=fake), \
                 mock.patch.object(run, "classify_case", side_effect=lambda *_: next(outcomes)), \
                 mock.patch.object(run, "collect", return_value=[]):
                output = io.StringIO()
                errors = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    self.assertEqual(run.execute(parser, args), 1)
            status = json.loads((args.out / "batch-status.json").read_text())
            self.assertEqual(len(status), 4)
            self.assertEqual([record["status"] for record in status],
                             ["timeout", "error", "passed", "passed"])
            self.assertIn("NONFT KV-B 1000 QPS WARNING", errors.getvalue())
            self.assertIn("NONFT KV-B 2000 QPS ERROR", errors.getvalue())
            self.assertIn("STARFISH KV-B 1000 QPS finish", output.getvalue())

    def test_unsafe_cleanup_aborts_without_success_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, apps="kv-b", systems="nonft", loads_ops="1000,2000")
            parser = argparse.ArgumentParser()
            unsafe = {"status": "error", "safe_to_continue": False,
                      "reason": "cleanup is incomplete", "exit_status": 1,
                      "client_exit_status": 1}
            fake = mock.Mock(returncode=1, stdout="", stderr="")
            with mock.patch.object(run.subprocess, "run", return_value=fake), \
                 mock.patch.object(run, "classify_case", return_value=unsafe), \
                 mock.patch.object(run, "collect", return_value=[]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run.execute(parser, args), 1)
            status = json.loads((args.out / "batch-status.json").read_text())
            self.assertEqual(len(status), 1)
            self.assertNotIn("finish", output.getvalue())
            self.assertIn("ABORTED", output.getvalue())

    def test_setup_error_without_run_directory_continues_after_safe_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = plan_args(temporary, apps="kv-b", systems="nonft", loads_ops="1000,2000")
            parser = argparse.ArgumentParser()
            outcomes = iter((
                {"status": "error", "safe_to_continue": True, "reason": "recipe missing",
                 "exit_status": 2, "client_exit_status": None},
                {"status": "passed", "safe_to_continue": True, "reason": "ok",
                 "exit_status": 0, "client_exit_status": 0},
            ))
            fake = mock.Mock(returncode=2, stdout="", stderr="")
            with mock.patch.object(run.subprocess, "run", return_value=fake) as invoke, \
                 mock.patch.object(run, "classify_case", side_effect=lambda *_: next(outcomes)), \
                 mock.patch.object(run, "collect", return_value=[]):
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run.execute(parser, args), 1)
            self.assertEqual(invoke.call_count, 2)
            status = json.loads((args.out / "batch-status.json").read_text())
            self.assertEqual([record["status"] for record in status], ["error", "passed"])


def synthetic_row(workload, system, load, profile):
    return {"workload": workload, "system": system, "offered_load_ops": load,
            "offered_load": load / (1000 if workload == "NQ" else 1_000_000),
            "p99_latency": 2.5, "load_unit": "Kops" if workload == "NQ" else "Mops",
            "latency_unit": "ms" if workload == "NQ" else "us", "source_type": "measured",
            "source": "fixture", "exit_status": 0, "correctness": "pass",
            "profile_id": profile}


class Figure10Collection(unittest.TestCase):
    def test_same_system_kv_and_nq_are_separate_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kv = root / "kv-b-nonft-1000000ops-r1"
            nq = root / "nq-nonft-1000ops-r1"
            for directory in (kv, nq):
                directory.mkdir()
                (directory / "analysis.json").write_text("{}")
            rows = {
                kv: synthetic_row("KV-B", "Non-FT", 1000000, "kv-profile"),
                nq: synthetic_row("NQ", "Non-FT", 1000, "nq-profile"),
            }
            with mock.patch.object(collect_module, "point", side_effect=lambda path: rows[path]):
                result = collect_module.collect(root, root / "figure10.csv")
            self.assertEqual(len(result), 2)
            self.assertEqual({r["workload"]: r["latency_unit"] for r in result},
                             {"KV-B": "us", "NQ": "ms"})

    def test_nq_primary_units_and_already_exited_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "nq-nonft-1000ops-r1"
            directory.mkdir()
            (directory / "client.log").write_text("exact used bytes: 0\n", encoding="utf-8")
            spec = {"offered_load_ops": 1000, "max_queue_delay_us": 1500000,
                    "histogram_dir": str(directory / "histograms")}
            analysis_result = {"p99_latency_ns": 2500000,
                              "phases": {"measurement": {
                                  "scheduled": 1000, "completed_in_window_ops_s": 999,
                                  "p99_service_ns": 3000, "p99_dispatch_ns": 4000,
                                  "realized_offered_ops_s": 1000, "completed": 1000,
                                  "deadline_dropped": 0, "drop_fraction": 0,
                                  "completed_after_deadline": 0}}}
            analysis = {"status": "passed", "exit_status": 0, "correctness": "pass",
                        "application": "nq", "nq_latency": analysis_result,
                        "endpoints": [{"status": "already_exited"}]}
            plan = {"app": "nq", "system": "nonft", "nq_latency": spec,
                    "worker_profile": {"app_workers": 24}, "effective_config": "",
                    "client_env": {}, "ratio": 25}
            manifest = {"status": "passed", "plan": plan, "client_sha256": "client"}
            (directory / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(collect_module.nq_latency, "parse_result",
                                  return_value={"nq_latency": analysis_result}):
                row = collect_module.point(directory)
            self.assertEqual(row["p99_latency"], 2.5)
            self.assertEqual(row["latency_unit"], "ms")
            self.assertEqual(row["p99_service_us"], 3.0)
            self.assertEqual(row["p99_dispatch_us"], 4.0)

    def test_collection_errors_are_saved_and_other_rows_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "kv-b-nonft-1000ops-r1"
            bad = root / "kv-b-nonft-2000ops-r1"
            good.mkdir()
            bad.mkdir()
            for directory in (good, bad):
                (directory / "analysis.json").write_text("{}", encoding="utf-8")
            rows = {
                good: synthetic_row("KV-B", "Non-FT", 1000, "profile-a"),
            }
            def point(path):
                if path == good:
                    return rows[path]
                raise ValueError("bad histogram")
            output = root / "figure10.csv"
            status = root / "collection-status.json"
            with mock.patch.object(collect_module, "point", side_effect=point):
                collected = collect_module.collect(root, output, status_output=status)
            self.assertEqual(len(collected), 1)
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            record = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["errors"][0]["run_dir"], str(bad))

    def test_mixed_profiles_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "kv-b-nonft-1000ops-r1"
            second = root / "kv-b-nonft-2000ops-r1"
            first.mkdir()
            second.mkdir()
            for directory in (first, second):
                (directory / "analysis.json").write_text("{}", encoding="utf-8")
            rows = iter((synthetic_row("KV-B", "Non-FT", 1000, "a"),
                         synthetic_row("KV-B", "Non-FT", 2000, "b")))
            with mock.patch.object(collect_module, "point", side_effect=lambda _: next(rows)):
                with self.assertRaisesRegex(ValueError, "mixed runtime"):
                    collect_module.collect(root, root / "figure10.csv")


if __name__ == "__main__":
    unittest.main()
