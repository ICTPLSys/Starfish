# Figure 11: fault-tolerance tax

The plot has eight workload columns and three metric rows. Its bar ordering,
stacked traffic design, colors and per-panel scaling follow the submitted
figure; values come from the supplied CSV.

Required columns:

`workload,system,component,value,unit,source_type,source`

Workload labels: BFS, LLM/LLaMA, MG, WC, KV-B, KV-A, KV-S, NQ. Systems:
Hydra, Carbink, Starfish, Non-FT. One row is one component per workload and
system. Components/units:

- `fetch_traffic`, `eviction_traffic`: `x`, the paper's normalized traffic
  components. Both must be present for a stacked traffic bar to appear.
- `remote_cpu_cores`: `cores` (raw core count).
- `remote_memory`: `x` (normalized remote memory use).

Rows with a blank `value` are missing, not zero. Numeric zero is permitted
only when it is a real measurement. Duplicate conditions and mixed
`source_type` values are errors. The default is `measured`; reference and
synthetic output require an explicit flag and use separate filenames.
Optional `exit_status` and `correctness` fields reject failed runs.

```bash
python3 scripts/figure11/plot.py --input data/figure11.csv
python3 scripts/figure11/plot.py --input data/figure11.csv --validate-only
```

This input contract does not make measurements comparable automatically;
the experiment collector must use the same workload and metric definitions
for all systems. No paper-reference values are filled into missing cells.
