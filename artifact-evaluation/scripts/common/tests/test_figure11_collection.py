import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AE = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "figure11_collect", AE / "scripts/figure11/collect.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
contract = module.log_contract


def record(system, run_id, *, environment="site-a", components=None,
           elapsed_s=10.0):
    components = tuple(components or contract.ALL_COMPONENTS)
    result = {
        "schema_version": 3,
        "workload": "kv-b",
        "system": system,
        "run_id": run_id,
        "environment": environment,
        "workload_id": "kvb-test",
        "ratio": 25,
        "app_workers": 24,
        "repeat": 1,
        "phase": "work",
        "components": ",".join(components),
        "elapsed_s": elapsed_s,
        "exit_status": 0,
        "correctness": "pass",
        "source_type": "measured",
    }
    if "fetch_traffic" in components:
        result["fetch_bytes"] = 100
    if system == "nonft":
        result.update(baseline_variant="nonft-backup-off", backup_enabled=False)
    if "eviction_traffic" in components:
        result["eviction_bytes"] = 200
    if "remote_cpu_cores" in components:
        result.update(
            remote_cpu_seconds=30.0,
            remote_cpu_elapsed_s=5.0,
            remote_cpu_window=contract.REMOTE_CPU_WINDOW,
        )
    if "remote_memory" in components:
        result.update(
            remote_memory_mean_bytes=1000.5,
            remote_memory_samples=5,
            remote_memory_sampling=contract.REMOTE_MEMORY_SAMPLING,
            remote_memory_metric=contract.MEMORY_METRICS[system],
            remote_memory_window=contract.REMOTE_MEMORY_WINDOW,
            remote_memory_scheduled_times_s="10,20,30,40,50",
            remote_memory_sample_times_s="10.01,20.01,30.01,40.01,50.01",
        )
    return result


def write_records(root, records, name="records.log"):
    path = root / name
    lines = []
    for item in records:
        fields = []
        for key, value in item.items():
            fields.append(f"{key}={value}")
        lines.append("figure11_result " + " ".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class Figure11Collection(unittest.TestCase):
    def test_schema_v1_peak_is_rejected(self):
        line = (
            "figure11_result schema_version=1 workload=kv-b system=carbink "
            "run_id=r environment=e workload_id=w ratio=25 app_workers=24 "
            "repeat=1 phase=work elapsed_s=1 fetch_bytes=1 eviction_bytes=1 "
            "remote_cpu_seconds=1 remote_memory_peak_bytes=1 exit_status=0 "
            "correctness=pass source_type=measured"
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            contract.parse_line(line)

    def test_missing_selected_field_fails_closed(self):
        item = record("carbink", "r", components=("remote_memory",))
        item.pop("remote_memory_mean_bytes")
        with self.assertRaisesRegex(ValueError, "missing log fields"):
            contract.validate_record(item)

    def test_timed_memory_accepts_partial_sample_counts(self):
        for count in range(1, 6):
            item = record("nonft", "r", components=("remote_memory",))
            item["remote_memory_samples"] = count
            item["remote_memory_scheduled_times_s"] = list(range(10, count * 10 + 1, 10))
            item["remote_memory_sample_times_s"] = [t + .01 for t in
                item["remote_memory_scheduled_times_s"]]
            self.assertEqual(contract.validate_record(item)["remote_memory_samples"], count)
        for count in (0, 6):
            item["remote_memory_samples"] = count
            with self.assertRaisesRegex(ValueError, "1..5"):
                contract.validate_record(item)

    def test_request_progress_v2_is_not_timed_memory(self):
        item = record("nonft", "r", components=("remote_memory",))
        item["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema"):
            contract.validate_record(item)
        item["schema_version"] = 3
        item["remote_memory_sampling"] = "request_progress_10_30_50_70_90"
        with self.assertRaisesRegex(ValueError, "remote_memory_sampling"):
            contract.validate_record(item)

    def test_all_components_and_cpu_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                record("nonft", "nonft-r"),
                record("starfish", "starfish-r"),
                record("hydra", "hydra-r"),
                record("carbink", "carbink-r"),
            ]
            write_records(root, records)
            rows = module.collect_rows(root)
            self.assertEqual(len(rows), 16)
            cpu = next(row for row in rows
                       if row["system"] == "Carbink"
                       and row["component"] == "remote_cpu_cores")
            self.assertEqual(cpu["value"], "6")
            self.assertEqual(cpu["window"], contract.REMOTE_CPU_WINDOW)
            memory = next(row for row in rows
                          if row["system"] == "Carbink"
                          and row["component"] == "remote_memory")
            self.assertEqual(memory["metric"], "live_stripe_allocated_bytes")
            self.assertEqual(memory["aggregation"], "mean")

    def test_explicit_partial_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_records(root, [
                record("nonft", "nonft-r", components=("remote_memory",)),
                record("carbink", "carbink-r", components=("remote_memory",)),
            ])
            rows = module.collect_rows(root, components="remote_memory")
            self.assertEqual(len(rows), 2)
            with self.assertRaisesRegex(ValueError, "missing selected component"):
                module.collect_rows(root)

    def test_timed_memory_collection_is_application_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for workload in contract.WORKLOADS:
                for system in contract.SYSTEMS:
                    item = record(system, workload + "-" + system,
                                  components=("remote_memory",))
                    item["workload"] = workload
                    item["workload_id"] = workload + "-test"
                    records.append(item)
            write_records(root, records)
            rows = module.collect_rows(root, components="remote_memory")
            self.assertEqual(len(rows), 32)
            self.assertEqual({r["window"] for r in rows}, {"benchmark"})

    def test_early_finish_mean_keeps_actual_and_baseline_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = record("nonft", "n", components=("remote_memory",))
            shorter = record("carbink", "c", components=("remote_memory",))
            shorter.update(remote_memory_samples=3,
                           remote_memory_scheduled_times_s="10,20,30",
                           remote_memory_sample_times_s="10.01,20.01,30.01")
            write_records(root, [baseline, shorter])
            row = next(r for r in module.collect_rows(root, components="remote_memory")
                       if r["system"] == "Carbink")
            self.assertEqual(row["samples"], "3")
            self.assertEqual(row["baseline_samples"], "5")
            self.assertIn("sample counts differ", row["warning"])

    def test_environment_mismatch_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_records(root, [
                record("nonft", "nonft-r", environment="82-84"),
                record("carbink", "carbink-r", environment="82-54-84"),
            ])
            with self.assertRaisesRegex(ValueError, "environment"):
                module.collect_rows(root)
            rows = module.collect_rows(root, allow_unmatched_environments=True)
            row = next(item for item in rows
                       if item["system"] == "Carbink")
            self.assertEqual(row["environment_match"], "unmatched_allowed")
            self.assertEqual(row["baseline_environment"], "82-84")
            self.assertIn("unmatched environment", row["warning"])

    def test_raw_fallback_is_used_only_without_final_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def raw_collector(logs_root, **kwargs):
                calls.append((Path(logs_root), kwargs))
                return [
                    record("nonft", "nonft-r"),
                    record("carbink", "carbink-r"),
                ]

            rows = module.collect_rows(
                root, components="remote_memory",
                raw_collector=raw_collector)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["components"], ("remote_memory",))

    def test_final_records_do_not_call_raw_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_records(root, [
                record("nonft", "nonft-r"),
                record("carbink", "carbink-r"),
            ])

            def should_not_run(*_args, **_kwargs):
                raise AssertionError("raw adapter must not run")

            rows = module.collect_rows(
                root, components="remote_memory",
                raw_collector=should_not_run)
            self.assertEqual(len(rows), 2)

    def test_regular_nonft_is_not_selected_as_figure11_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = record("nonft", "nonft-on", components=("remote_memory",))
            regular.update(baseline_variant="nonft", backup_enabled=True,
                           remote_memory_mean_bytes=123456)
            off = record("nonft", "nonft-off", components=("remote_memory",))
            write_records(root, [regular, off, record(
                "starfish", "starfish", components=("remote_memory",))])
            rows = module.collect_rows(root, components="remote_memory")
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["baseline_run_id"] for r in rows}, {"nonft-off"})
            self.assertEqual({r["normalizer_variant"] for r in rows}, {"nonft-backup-off"})
            self.assertEqual({r["value"] for r in rows}, {"1"})

    def test_nonft_off_name_cannot_hide_enabled_backup(self):
        item = record("nonft", "bad", components=("remote_memory",))
        item["backup_enabled"] = True
        with self.assertRaisesRegex(ValueError, "has backup enabled"):
            contract.validate_record(item)


if __name__ == "__main__":
    unittest.main()
