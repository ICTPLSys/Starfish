# Figure 9: performance across applications

Draw the paper's eight-panel grouped-bar comparison: (a) BFS, (b) LLM,
(c) MG, (d) WC, (e) KV-B, (f) KV-A, (g) KV-S, (h) NQ. At each local-memory
ratio (13/25/50/75/100%), bars appear in Hydra, Carbink, Starfish, Non-FT
order. A Native Linux All local measurement, when present, is a dashed
horizontal reference for that workload. Read a CSV and generate PNG/PDF;
no experiments, raw-log collection, or reference filling are performed.

The panel order, bar grouping, palette, hatching, axis guides and legend
follow the submitted figure. Panel spacing expands slightly for CSV-derived
y-axis ranges. Values come only from the CSV, and the Native baseline requires
an explicit `native` row.

The supplied figure has a seconds axis labelled **Elapsed Time (s)** while
its caption says "normalized to Non-FT". This plotter follows the displayed
seconds axis; it does not silently normalize or rescale the CSV values.

## Run the experiment

Build Non-FT (see [the scripts guide](../README.md)), then inspect the
LLaMA/BFS run plan:

~~~bash
bash scripts/figure9/run.sh --dry-run --apps llama,bfs --systems nonft
~~~

For a two-server run, copy `scripts/common/site.example.json` to
`data/site.json`, fill in the SSH memory host, RDMA address, per-system
remote server binaries and input paths (including the LLaMA tokenizer), and run from
`artifact-evaluation/`:

Use `ib_device` for the compute server and `memory_ib_device` if the
memory server has a different RDMA device name.
The compute runtime requires 2 MiB HugePages for its local RDMA buffer;
the preflight checks enough free pages for the largest requested ratio.
Reserve these pages on the compute server before running and restore the
original reservation afterward. Build the memory-service binary on the
memory server if its glibc differs from the compute server's.

~~~bash
mkdir -p data
cp -n scripts/common/site.example.json data/site.json
# Edit data/site.json for this pair of servers and local workload inputs.
bash scripts/figure9/run.sh --site data/site.json --apps llama \
  --systems nonft --ratios 25 --repeats 1
~~~

The common runner saves each case under a fresh `results/figure9/batch-*/runs/`
directory. The batch collector writes **only successful measured rows** to
that batch's `figure9.csv`, then this plotter renders the unchanged eight-panel
layout. After plotting, the runner also copies the CSV to `data/figure9.csv`.
This is the latest batch with at least one successful measurement's convenient
plotting input; a wholly failed batch leaves it unchanged. The original
CSV and run records remain in the batch directory. A later completed batch
replaces only the copy in `data/`, not any earlier batch results.
An unrun or failed case has no CSV value and leaves its bar slot empty. A
batch with a failed case exits nonzero even if it produced a partial figure.

The initial LLaMA parser records the application's work-phase `wall time`,
which that application computes from cycles using a fixed 2.8 GHz divisor.
BFS uses one `gapbs_bfs_result elapsed_s` work iteration and requires the
subsequent `gapbs_bfs_verify status=pass` line. These are the current
measurement boundaries; do not compare them with results measured over a
different phase or protocol.

## Plot an existing CSV

From artifact-evaluation/:

~~~bash
python3 -m venv .venv-plot
.venv-plot/bin/python -m pip install -r scripts/common/requirements-plot.txt
bash scripts/plot.sh figure9
~~~

The default input is the latest completed batch's `data/figure9.csv`, copied
there by `run.sh`. To replot an earlier batch, pass
`--input results/figure9/<batch>/figure9.csv` explicitly. The default output
is results/figures/figure9/figure9.png, .pdf and .json.
The JSON records the input hash, selected data, repetitions, source paths
and plotting-code/library versions. The wrapper uses .venv-plot/bin/python
if present, or the PYTHON environment variable if explicitly set.

## CSV

template.csv contains a header only. Data values are never embedded in the
plotting code. Required fields:

| Field | Meaning |
| --- | --- |
| workload | BFS, LLM/LLaMA, MG, WC, KV-B, KV-A, KV-S or NQ; KVS YCSB-B/A and KVS Synthetic also accepted |
| system | Hydra, Carbink, Starfish, Non-FT or Native Linux All local (aliases `NonFT`, `native`) |
| ratio | Local memory capacity / application footprint, integer percent |
| elapsed_s | Positive, finite elapsed time in seconds; blank means no measurement |
| source_type | measured, paper_reference or synthetic (run aliases measured) |
| source | Raw-result path relative to `artifact-evaluation/` (absolute if the batch is outside it), or reference citation |
| run_id | Optional for one observation per condition; required and unique within a repeated condition |

Default selection: all eight panels; Hydra, Carbink, Starfish and Non-FT;
13%, 25%, 50%, 75%, 100%. For each workload, an optional `system=native`,
`ratio=100` row supplies the all-local reference line. The CSV may contain
one or multiple valid native measurements per workload; repeats are averaged
under the same rules as bars. Native is not a fifth group of bars.
Conditions with no CSV row, or with a blank elapsed_s, remain empty in the
plot. A missing bar leaves its slot empty; a missing native row leaves its
reference line absent. No measurement is synthesized.
To focus on some systems or ratios:

~~~bash
bash scripts/plot.sh figure9 --input data/figure9.csv --systems nonft
bash scripts/plot.sh figure9 --input data/figure9.csv \
  --workloads llama --systems nonft starfish --ratios 25 50
~~~

Selections preserve workload/system order; ratios are numerically ordered.
Unselected conditions and row counts are recorded in the JSON.
Unknown labels and malformed values are errors, never silently ignored.
Ratio values must be mathematically integral; 13.0 and 13 both mean 13%.
The run alias applies to CSV rows; use --source-type measured on the command line.

One run gives one point. Multiple rows with distinct run_id values give the
arithmetic mean with sample-standard-deviation bars (ddof=1), not a confidence
interval. For n=1, no uncertainty bar is drawn and stddev_s is null.
Reference points must be unique. No interpolation, best-run selection or
zero filling is used. Each workload has its own seconds axis starting at zero.
The horizontal native line represents the mean Native Linux All local
measurement in seconds, not a normalization of the bars.

Compared rows must use the same workload/input, environment, correctness
criteria and timing boundaries. elapsed_s means the chosen application
measurement phase; do not mix process wall time and work-phase time.
The plotter validates the table, not the underlying experiment's correctness.
If supplied, exit_status must be 0 and correctness must be pass.
Optional experiment_id, environment and measurement_phase must be nonempty
and consistent within each selected workload.

## Reference data

Default invocation accepts measured rows only. An entirely reference or
synthetic CSV requires an explicit flag; mixed data kinds are rejected:

~~~bash
bash scripts/plot.sh figure9 --input data/figure9-reference.csv \
  --source-type paper_reference --output-dir results/reference/figure9
~~~

Non-measured output filenames include -paper_reference or -synthetic.
These previews are not AE reproduction results. Sources, counts and caveats
remain in the companion JSON/caption rather than as paragraphs inside plots.
data/ and results/ remain ignored by Git.

Concurrent exports to the same output prefix are rejected. JSON is written
last as the completion marker; after an interrupted publication it may be
absent. Rerun before using incomplete outputs. Its hashes identify the matching
PNG/PDF pair. A hidden lock file in the output directory is normal.
JSON records missing_conditions, blank_rows, native_baselines and each point's
n; an unmeasured point has n=0 and mean_s/stddev_s=null. Header-only CSV is
valid and produces an empty eight-panel chart. A blank elapsed_s row may also
leave source_type and source blank, because there is no observation to
attribute.

## Validation

~~~bash
python3 scripts/figure9/plot.py --input data/figure9.csv --validate-only
~~~
