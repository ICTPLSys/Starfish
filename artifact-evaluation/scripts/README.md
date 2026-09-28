# Artifact-evaluation scripts

Run these commands from `artifact-evaluation/` on a Linux x86-64 host. The
long-form setup and build instructions are in [docs/INSTALL.md](../docs/INSTALL.md);
experiment commands and result files are described in
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).

## Command index

| Purpose | Command |
|---|---|
| Inspect dependency/build plan | `bash scripts/common/setup_environment.sh --dry-run` |
| Prepare dependencies and plotting | `bash scripts/common/setup_environment.sh` |
| Check a build host | `bash scripts/common/check_environment.sh` |
| Build the default runtime | `bash scripts/common/build.sh` |
| Check LLaMA 25% with Non-FT, Starfish and recovery | `bash scripts/run_fast_check.sh` |
| Build MG explicitly | `bash scripts/common/build.sh --system nonft --targets server,mg --jobs 4` |
| Plan a Figure 9 batch | `bash scripts/figure9/run.sh --dry-run --apps llama,bfs --systems nonft` |
| Plan a Figure 12 batch | `bash scripts/figure12/run.sh --dry-run` |
| Plot a CSV | `bash scripts/plot.sh figure9 --input data/figure9.csv` |

Select the runtime with `build.sh --system` and set the server IPs in a site
file. See [multi-server configuration](../docs/MULTI-SERVER.md)
for the fields and examples.

## Run outputs

`bash scripts/figure9/run.sh` creates a new
`results/figure9/batch-*/` directory. Each case keeps its manifest, effective
configuration, client/server logs, and `analysis.json`. The collector writes
only successful, correctness-checked measured rows to that batch's
`figure9.csv`; a nonempty successful batch is also copied to
`data/figure9.csv`. A failed or unrun case is absent, not filled from a
reference value.

The plot wrappers read CSV input and do not run applications:

```bash
bash scripts/plot.sh figure9 --input results/figure9/<batch>/figure9.csv
bash scripts/plot.sh figure10 --input data/figure10.csv
bash scripts/plot.sh figure11 --input data/figure11.csv
bash scripts/plot.sh figure12 --input data/figure12.csv
bash scripts/plot.sh figure13 --recovery-csv data/figure13-recovery.csv \
  --throughput-csv data/figure13-throughput.csv
```

Measured output defaults to `results/figures/<figure>/`. Pass
`--source-type paper_reference` or `--source-type synthetic` explicitly for a
non-measured preview; those outputs use separate names.

## Figure-specific contracts

- [Figure 9](figure9/README.md): measured two-server LLaMA/BFS rows, or an
  explicit paper-reference/synthetic preview; default output is
  `results/figures/figure9/figure9.{png,pdf,json}`.
- [Figure 10](figure10/README.md): KV-B and NQ tail-latency CSV.
- [Figure 11](figure11/README.md): traffic, remote CPU, and remote-memory CSV.
- [Figure 12](figure12/README.md): compute-node EC reference-cycle counts and
  final Work metadata snapshot. Use its instrumented runner; ordinary
  Figure 9 logs without these records cannot supply both metrics.
- [Figure 13](figure13/README.md): recovery summary plus observed throughput
  trace CSVs.

Each figure-specific guide describes its input columns, units, plotting
options and output paths.
