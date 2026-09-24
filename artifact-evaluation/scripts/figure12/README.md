# Figure 12: compute-node overhead

This CSV-fed plot follows the submitted figure's two rows, workload order,
colors and legend. It reads measurements from the supplied CSV.

Required CSV columns:

`workload,system,metric,value,unit,source_type,source`

One row is one workload/system/metric measurement. Workloads: BFS, LLM,
MG, WC, KV-B, KV-A, KV-S, NQ. Systems: Carbink and Starfish. Metrics:

- `local_ec_cpu_norm` with `unit=x`: CPU use normalized to Carbink under
  the same workload and timing window.
- `metadata_space_pct` with `unit=% app memory`: the numeric value is the
  percentage-point quantity displayed on the paper axis (e.g. `0.25`
  means 0.25% of application memory, not a fraction `0.0025`).

A missing or blank `value` leaves its bar position empty. Numeric zero is a
valid measured value. Duplicate workload/system/metric rows and mixed source
types are errors. `measured` is the default; `paper_reference` and
`synthetic` require an explicit flag and are exported under separate names.
Optional `exit_status` and `correctness` columns reject failed runs. If a
measurement exceeds the original fixed y-axis, the scale expands so it is
never silently clipped.

```bash
python3 scripts/figure12/plot.py --input data/figure12.csv
python3 scripts/figure12/plot.py --input data/figure12.csv --validate-only
```

The figure compares complete runs, not isolated best sub-stages. The
plotter checks the CSV schema but cannot establish workload comparability.
