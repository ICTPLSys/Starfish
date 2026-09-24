# Figure 13: failure recovery

The left panel shows normalized foreground throughput over time; the right
compares recovery durations for one- and two-node failures. Throughput and
recovery events must come from observed CSV data; this plotter does not
synthesize them.

Two CSVs are required:

- `--recovery-csv`: `scenario,system,recovery_s,source_type,source`.
  Scenarios are `1-node`, `2-node`; systems are Hydra, Carbink, Starfish.
  One row per condition; a blank `recovery_s` leaves its bar empty.
- `--throughput-csv`:
  `system,elapsed_s,throughput_ops_per_s,event,normalization,reference_ops_per_s,source_type,source`.
  `elapsed_s` is wall time since this experiment's time origin. A numeric
  `throughput_ops_per_s` is a real observation, including a real zero. The
  plot computes normalized throughput as
  `throughput_ops_per_s / reference_ops_per_s`. A blank raw throughput
  breaks the line at that timestamp. Set
  `normalization=starfish_pre_failure` for every row; the experiment collector
  must record one positive `reference_ops_per_s` denominator for all systems
  under the same measurement boundary. The plotter rejects mixed denominators.
  An event-only row with `system=all,event=failure` gives the failure time;
  a row with `event=recovered` gives that system's observed recovery time.
  Without those rows, the plot has no fabricated failure marker or shaded
  recovery window.

Duplicate conditions/timestamps, invalid units/values and mixed source types
are rejected. Optional `exit_status` and `correctness` columns reject failed
runs. `measured` is the default source type; `paper_reference` and `synthetic`
require explicit flags and are exported under separate filenames. A blank
input does not get replaced with the paper's generated trace.

```bash
python3 scripts/figure13/plot.py --recovery-csv data/figure13-recovery.csv \
  --throughput-csv data/figure13-throughput.csv
```

Input CSVs must use consistent time origins and observed trace samples. The
two panels may come from separate runs, so a summary recovery duration need
not equal an event interval in the throughput trace.
