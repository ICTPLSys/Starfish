# Figure 10: tail latency

The two-panel axes, line order, markers and log-latency scale follow the
submitted figure. Data comes from the supplied CSV; this script does not run
experiments or embed latency measurements.

Required CSV columns:

`workload,system,offered_load,p99_latency,load_unit,latency_unit,source_type,source`

- Workloads: `KV-B` (`KVS YCSB-B`) uses `Mops` and `us`; `NQ` (`Nhop`) uses
  `Kops` and `ms`.
- Systems: `Hydra`, `Carbink`, `Starfish`, `Non-FT`.
- One row represents one load point. `p99_latency` blank means no valid
  measurement at that offered load; it leaves a gap in the line. Entirely
  missing series are blank. A duplicate workload/system/load is rejected,
  not averaged or silently overwritten.
- `source_type` defaults to `measured`; `paper_reference` and `synthetic`
  require explicit `--source-type` and are exported under distinct names.
  Nonblank values require a source identifier. Optional `exit_status` and
  `correctness` fields reject incomplete or failed runs.
- The plotter checks the CSV contract; it does not verify that the underlying
  runs have identical workloads, hardware, memory ratios and timing windows.

Run with the shared environment's dependencies (Matplotlib and NumPy):

```bash
python3 scripts/figure10/plot.py --input data/figure10.csv
python3 scripts/figure10/plot.py --input data/figure10.csv --validate-only
```

The input file and PNG/PDF/JSON output are ignored by Git. A header-only CSV
is valid and renders empty panels; no paper data is filled automatically.
