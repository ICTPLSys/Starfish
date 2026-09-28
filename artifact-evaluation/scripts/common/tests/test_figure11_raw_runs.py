import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


AE = Path(__file__).resolve().parents[3]
RAW_PATH = AE / "scripts/figure11/raw_runs.py"
spec = importlib.util.spec_from_file_location("figure11_raw_runs", RAW_PATH)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)
contract = raw.log_contract


def _line(kind, **fields):
    return kind + " " + " ".join(f"{key}={value}" for key, value in fields.items())


def _memory_lines(system="carbink", count=3, metric=None, observed=None):
    metric = metric or contract.MEMORY_METRICS[system]
    observed = observed or [10.2 + index * 10 for index in range(count)]
    values = []
    for index in range(count):
        point = (10, 20, 30, 40, 50)[index]
        occupied = 100 + index * 10
        values.append(_line(
            "runtime_remote_memory", schema_version=2, sample=index,
            scheduled_elapsed_s=point, observed_elapsed_s=observed[index],
            occupied_bytes=occupied, metric=metric, source="remote_global_heap",
            window="benchmark_start_10_20_30_40_50s", start_monotonic_ns=1000,
            snapshot_start_monotonic_ns=1000 + int(observed[index] * 1_000_000_000), snapshot_ns=10,
            endpoint_count=1, endpoint_bytes=f"0:{occupied}",
            consistency="rolling_allocator_snapshot"))
    values.append(_line(
        "runtime_remote_memory_end", schema_version=2, samples=count,
        sampling="benchmark_start_10_20_30_40_50s", metric=metric,
        start_monotonic_ns=1000))
    return values


def _traffic(eviction=100, windows=1):
    lines = []
    for index in range(windows):
        start = 100 + index * 20
        end = start + 10
        write, rmw = (70, 30)
        lines.append(_line(
            "runtime_traffic_work", schema_version=1, scope="client_rdma",
            phase="work", window_id=index, start_monotonic_ns=start,
            end_monotonic_ns=end, read_bytes=100, write_bytes=write,
            rmw_read_bytes=rmw, fetch_bytes=100, eviction_bytes=eviction))
    return lines


def _metadata(system="carbink", run_id="carbink-r2", *, status="passed",
              correctness="pass", exit_status=0):
    analysis = {
        "schema_version": 1, "application": "kv-b", "system": system,
        "ratio": 25, "run_id": run_id, "environment": "test-env",
        "workload_id": "kvb-test", "app_workers": 4, "repeat": 1,
        "elapsed_s": 1.0, "exit_status": exit_status, "status": status,
        "correctness": correctness, "correctness_evidence": "verified",
    }
    manifest = {
        "schema_version": 1, "status": status, "exit_status": exit_status,
        "plan": {"app": "kv-b", "system": system, "ratio": 25,
                  "run_id": run_id, "site": "test-env",
                  "configured_cpu_profile": {"app_workers": 4}},
    }
    return analysis, manifest


def _write_run(root, name="carbink-r2", *, memory_count=3, metric=None,
               end_count=None, legacy=False, eviction=100, status="passed",
               correctness="pass", exit_status=0, windows=1):
    run = root / name
    run.mkdir()
    analysis, manifest = _metadata(
        run_id=name, status=status, correctness=correctness,
        exit_status=exit_status)
    (run / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if legacy:
        memory = [_line("kvs_remote_memory", sample=0, target_completed=10,
                        occupied_bytes=100)]
        memory.append(_line("kvs_remote_memory_end", samples=1))
    else:
        memory = _memory_lines(count=memory_count, metric=metric)
        if end_count is not None:
            memory[-1] = _line(
                "runtime_remote_memory_end", schema_version=2,
                samples=end_count, sampling="benchmark_start_10_20_30_40_50s",
                metric=metric or contract.MEMORY_METRICS["carbink"], start_monotonic_ns=1000)
    log = "\n".join(_traffic(eviction=eviction, windows=windows) + memory) + "\n"
    (run / "client.log").write_text(log, encoding="utf-8")
    return run


class Figure11RawRuns(unittest.TestCase):
    def test_accepts_early_timed_prefix_and_preserves_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, memory_count=3)
            records = raw.collect_records(
                root, components=("fetch_traffic", "eviction_traffic", "remote_memory"))
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["remote_memory_samples"], 3)
            self.assertEqual(record["remote_memory_scheduled_times_s"], [10, 20, 30])
            self.assertEqual(record["remote_memory_sample_times_s"], [10.2, 20.2, 30.2])
            self.assertEqual(record["remote_memory_mean_bytes"], 110.0)
            self.assertEqual(record["eviction_bytes"], 100)

    def test_zero_or_missing_timed_samples_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=1, end_count=0)
            with self.assertRaisesRegex(ValueError, "sample count"):
                raw.collect_records(root, components="remote_memory")
            run.joinpath("client.log").write_text(
                "runtime_remote_memory_end schema_version=2 samples=0 "
                "sampling=benchmark_start_10_20_30_40_50s "
                "metric=live_stripe_allocated_bytes start_monotonic_ns=1000\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                raw.collect_records(root, components="remote_memory")

    def test_wrong_schedule_metric_and_count_fail(self):
        for kwargs, message in (
                ({"metric": "allocator_occupied_bytes"}, "metric"),
                ({"end_count": 2}, "sample count")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _write_run(root, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    raw.collect_records(root, components="remote_memory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=1)
            text = run.joinpath("client.log").read_text(encoding="utf-8")
            text = text.replace("scheduled_elapsed_s=10", "scheduled_elapsed_s=20")
            run.joinpath("client.log").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schedule"):
                raw.collect_records(root, components="remote_memory")

    def test_legacy_request_progress_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, legacy=True)
            with self.assertRaisesRegex(ValueError, "legacy"):
                raw.collect_records(root, components="remote_memory")

    def test_real_timed_line_and_missed_point_do_not_insert_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=2)
            text = run.joinpath("client.log").read_text(encoding="utf-8")
            text = text.replace(
                "sample=1 scheduled_elapsed_s=20 observed_elapsed_s=20.2",
                "sample=2 scheduled_elapsed_s=30 observed_elapsed_s=30.2")
            text = text.replace("snapshot_start_monotonic_ns=20200001000", "snapshot_start_monotonic_ns=30200001000")
            text = text.replace(
                "runtime_remote_memory_end schema_version=2",
                "runtime_remote_memory_missing schema_version=2 sample=1 "
                "scheduled_elapsed_s=20 observed_elapsed_s=20.1 "
                "status=missed-notready\n"
                "runtime_remote_memory_end schema_version=2")
            run.joinpath("client.log").write_text(text, encoding="utf-8")
            record = raw.collect_records(root, components="remote_memory")[0]
            self.assertEqual(record["remote_memory_samples"], 2)
            self.assertEqual(record["remote_memory_scheduled_times_s"], [10, 30])
            self.assertEqual(record["remote_memory_mean_bytes"], 105.0)

    def test_failed_correctness_is_skipped_but_only_failure_is_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "carbink-r1", status="failed", correctness="fail",
                       exit_status=1)
            _write_run(root, "carbink-r2")
            records = raw.collect_records(root, components="remote_memory")
            self.assertEqual([record["run_id"] for record in records], ["carbink-r2"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "carbink-r1", status="failed", correctness="fail",
                       exit_status=1)
            with self.assertRaisesRegex(ValueError, "no complete"):
                raw.collect_records(root, components="remote_memory")

    def test_traffic_windows_sum_without_rmw_double_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, eviction=100, windows=2)
            record = raw.collect_records(
                root, components=("fetch_traffic", "eviction_traffic"))[0]
            self.assertEqual(record["fetch_bytes"], 200)
            self.assertEqual(record["eviction_bytes"], 200)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, eviction=130)
            with self.assertRaisesRegex(ValueError, "double-count"):
                raw.collect_records(root, components="eviction_traffic")

    def test_timed_schema_origin_and_endpoint_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=1)
            text = run.joinpath("client.log").read_text(encoding="utf-8")
            text = text.replace("runtime_remote_memory schema_version=2",
                                "runtime_remote_memory schema_version=3", 1)
            run.joinpath("client.log").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                raw.collect_records(root, components="remote_memory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=1)
            text = run.joinpath("client.log").read_text(encoding="utf-8")
            text = text.replace("endpoint_bytes=0:100", "endpoint_bytes=0:100,0:0")
            run.joinpath("client.log").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate endpoint"):
                raw.collect_records(root, components="remote_memory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=1)
            text = run.joinpath("client.log").read_text(encoding="utf-8")
            text = text.replace("start_monotonic_ns=1000", "start_monotonic_ns=2000", 1)
            run.joinpath("client.log").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "origins"):
                raw.collect_records(root, components="remote_memory")

    def test_cpu_sidecar_must_match_manifest_endpoint_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root, memory_count=1)
            manifest = json.loads(run.joinpath("manifest.json").read_text())
            manifest["endpoints"] = [{"memory_host": "mem-a", "pid": 12,
                                      "starttime": "34"}]
            run.joinpath("manifest.json").write_text(json.dumps(manifest))
            sidecar = {"schema_version": 1, "status": "passed",
                       "remote_cpu_seconds": 2.0, "remote_cpu_elapsed_s": 1.0,
                       "remote_cpu_window": "initialization_and_requests",
                       "endpoints": [{"host": "mem-b", "pid": 12, "starttime": "34"}],
                       "provenance": {"source": "same-run"}}
            run.joinpath("remote-cpu.json").write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ValueError, "do not match"):
                raw.collect_records(root)

    def test_fallback_identity_includes_input_or_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _write_run(root, "carbink-a", memory_count=1)
            second = _write_run(root, "carbink-b", memory_count=1)
            for run, digest in ((first, "input-a"), (second, "input-b")):
                analysis = json.loads(run.joinpath("analysis.json").read_text())
                analysis.pop("workload_id")
                analysis["input_bytes"] = 100
                analysis["prompt_sha256"] = digest
                run.joinpath("analysis.json").write_text(json.dumps(analysis))
            records = raw.collect_records(root, components="remote_memory")
            self.assertEqual({record["workload_id"] for record in records},
                             {json.dumps({"input_bytes": 100, "prompt_sha256": "input-a", "workload": "kv-b", "ratio": 25, "app_workers": 4, "repeat": 1}, sort_keys=True, separators=(",", ":")),
                              json.dumps({"input_bytes": 100, "prompt_sha256": "input-b", "workload": "kv-b", "ratio": 25, "app_workers": 4, "repeat": 1}, sort_keys=True, separators=(",", ":"))})

    def test_missing_cpu_is_not_estimated_or_spliced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root)
            with self.assertRaisesRegex(ValueError, "CPU sidecar"):
                raw.collect_records(root)
            records = raw.collect_records(root, components="remote_memory")
            self.assertNotIn("remote_cpu_seconds", records[0])

    def test_malformed_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root)
            run.joinpath("analysis.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                raw.collect_records(root, components="remote_memory")

    def test_missing_endpoints_and_observer_failure_are_not_partial_success(self):
        for change in ("no-endpoints", "observer-error"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = _write_run(root, memory_count=1)
                path = run / "client.log"
                text = path.read_text()
                if change == "no-endpoints":
                    text = text.replace(" endpoint_bytes=0:100", "")
                else:
                    text += ("runtime_remote_memory_missing schema_version=2 "
                             "sample=-1 status=observer-error detail=test\n")
                path.write_text(text)
                with self.assertRaises(ValueError):
                    raw.collect_records(root, components="remote_memory")

    def test_all_applications_and_runtimes_replay_through_the_same_adapter(self):
        import sys
        sys.path.insert(0, str(AE / "scripts"))
        from figure11 import collect
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for workload in contract.WORKLOADS:
                for system in contract.SYSTEMS:
                    name = workload + "-" + system
                    run = _write_run(root, name, memory_count=3)
                    analysis, manifest = _metadata(system, name)
                    analysis.update(application=workload, workload_id=workload + "-fixture")
                    manifest["plan"].update(app=workload)
                    if system == "nonft":
                        analysis.update(baseline_variant="nonft-backup-off", backup_enabled=False)
                        manifest["plan"].update(baseline_variant="nonft-backup-off", backup_enabled=False)
                        (run / "effective.config").write_text(
                            "enable_selective_backup 0\nremote_backup_budget_bytes 0\n"
                            "remote_backup_budget_pct 0\n")
                    (run / "analysis.json").write_text(json.dumps(analysis))
                    (run / "manifest.json").write_text(json.dumps(manifest))
                    (run / "client.log").write_text(
                        "\n".join(_traffic() + _memory_lines(system=system, count=3)) + "\n")
            rows = collect.collect_rows(
                root, components="remote_memory", raw_collector=raw.collect_records)
            self.assertEqual(len(rows), 32)
            self.assertEqual({row["samples"] for row in rows}, {"3"})
            self.assertEqual({row["window"] for row in rows}, {"benchmark"})

    def test_nonft_on_is_skipped_and_off_requires_actual_disabled_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variant, enabled in (("nonft", True), ("nonft-backup-off", False)):
                run = _write_run(root, variant, memory_count=1)
                analysis, manifest = _metadata("nonft", variant)
                analysis.update(baseline_variant=variant, backup_enabled=enabled)
                manifest["plan"].update(baseline_variant=variant, backup_enabled=enabled)
                (run / "analysis.json").write_text(json.dumps(analysis))
                (run / "manifest.json").write_text(json.dumps(manifest))
                (run / "client.log").write_text(
                    "\n".join(_memory_lines("nonft", count=1)) + "\n"
                    if not enabled else "no resource metrics in the regular run\n")
                (run / "effective.config").write_text(
                    "enable_selective_backup 0\nremote_backup_budget_bytes 0\n"
                    "remote_backup_budget_pct 0\n")
            records = raw.collect_records(root, components="remote_memory")
            self.assertEqual([r["run_id"] for r in records], ["nonft-backup-off"])
            self.assertFalse(records[0]["backup_enabled"])
            (root / "nonft-backup-off/effective.config").write_text(
                "enable_selective_backup 1\nremote_backup_budget_bytes 100\n"
                "remote_backup_budget_pct 0\n")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                raw.collect_records(root, components="remote_memory")


if __name__ == "__main__":
    unittest.main()
