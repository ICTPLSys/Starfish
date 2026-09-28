# Figure 12: compute-node EC and metadata overhead

## Scope and entry points

Systems: **Starfish, Carbink**. Applications: **BFS, LLM, MG, WC, KV-B,
KV-A, KV-S, NQ**. All commands below run from the artifact-evaluation root.
The runner delegates deployment, correctness verification and owned-process
cleanup to the same common runner as Figures 9/10. It does not change workload
scale, CPU/NUMA placement, worker counts or enable remote CPU collection.

The default executes the full matrix in system-major order: Starfish's eight
apps, then Carbink's eight apps, ratio 25, one repetition. **Use --dry-run for
planning only.** A valid site with the full endpoint inventory is required;
the example site is not a deployment configuration.

```bash
# No experiment: validate and print all 16 effective plans.
bash scripts/figure12/run.sh --site /path/to/site.json --dry-run

# Actual experiments, only when authorized. Use a fresh output directory.
bash scripts/figure12/run.sh --site /path/to/site.json \
  --build-root build --out results/figure12-new --build

# Optional explicit subsets; no hidden reduction in workload size.
bash scripts/figure12/run.sh --site /path/to/site.json \
  --apps kv-b,nq --systems starfish,carbink --metrics metadata \
  --out results/figure12-metadata-new
```

Use --site-map for app/system-specific site files or overlays; see run.py --help.
For example, a map may contain
`{"sites":{"kv-b/starfish":"/path/to/site82.json","nq/carbink":"/path/to/site56.json"}}`.
The runner materializes effective sites with runtime_metadata and runtime_ec_cpu
set according to --metrics all|metadata|ec-cpu (--metric is an alias). Other
site settings are preserved. The common runner records the corresponding
FARLIB_RUNTIME_METADATA=1 and FARLIB_RUNTIME_EC_CPU=1 environment switches.
Rebuild instrumented runtimes before collecting EC records; the switches
cannot add instrumentation to an old binary.

Terminal output has system/App START/FINISH and per-case status. Timeouts are
WARNING; crashes/correctness failures are ERROR. Verified cleanup permits the
next case; unverified cleanup causes ERROR + ABORTED. Any case failure returns
a nonzero batch exit. After successful application verification the runner also
checks the selected Figure 12 records: missing/invalid producer output cannot
be marked passed. A retained teardown warning keeps its warning level and
separately records whether Figure 12 metrics are usable.
Raw logs and failures remain in place. batch-plan.json,
batch-status.json and batch.json record the plan, case status and provenance.

## Measurement definitions

**EC:** local_ec_cpu_cycles, unit cycles. Compute client only; sum raw cycle
counts across all completed profile Work intervals (including all BFS
traversals), not a ratio to Carbink. Only pure codec encode/update/XOR/decode
scopes are measured, including background computation; nested scopes count
once. Work end closes admission and drains already admitted short codec scopes;
a scope crossing that boundary is counted wholly in its admitting Work.
Initialization, application computation, network posting/waiting and
server processes are excluded.

The clock is explicitly **TSC/reference cycles**, matching the existing
profile clock. These are elapsed reference ticks around EC scopes, not PMU
active core cycles or CPU seconds; preemption within a measured scope can be
included. Do not convert old CPU-seconds logs into this metric or claim PMU
semantics. The plot displays raw values divided by 1e9 for readability only.

**Metadata:** metadata_space_pct = 100 * accounted_bytes /
manifest.plan.workload_footprint_bytes. The numerator is compute-side
region/group/stripe/span descriptors, bitmaps and mappings, including retained
container capacity and identified measurement auxiliary state. Application
object handles/hash tables, payload/parity buffers, RDMA staging payload,
malloc bookkeeping and RSS are excluded.

The runtime saves metadata after each completed Work timer and before
application cleanup, then prints only the last snapshot once at exit.
This is not a peak/mean. Raw runtime schema v1 metadata_bytes is core-only;
accounted_bytes = metadata_bytes + measurement_aux_bytes. CSV metadata_bytes
is the inclusive total and core_metadata_bytes preserves the raw core field.
Never add CSV metadata_bytes to accounted_bytes: they are aliases.
The enabled EC observer's allocated counters/registry are included in the
auxiliary breakdown; disabled observers must not allocate it.

Sizes mean compiled sizeof plus actual allocated capacities. Component locks
protect mutable structures; this is not a globally atomic census.
The optional fixed-six registry's additional policy tables are not covered;
the metadata collector rejects that mode. Current recipes use
FARLIB_FIXED_SIX_GROUPS=0. The profile timer ends before
the metadata snapshot, but enclosing wall timers may include snapshot_us.
Instrumented measurements do not replace throughput/latency baselines.

## Collect and plot

```bash
bash scripts/figure12/collect.sh --logs-root results/figure12-new \
  --repeat 1 --metrics all --output data/figure12.csv \
  --summary results/figure12-new/figure12-summary.json
bash scripts/figure12/plot.sh --input data/figure12.csv \
  --output-dir results/figures/figure12
# Or collect and plot directly from the indexed batch:
bash scripts/figure12/plot.sh --logs-root results/figure12-new --repeat 1
```

The plot wrapper uses PYTHON, then the local .venv-plot, then python3.
Plot dependencies are in scripts/common/requirements-plot.txt. CSV output
is never overwritten; choose a fresh path. Figure export writes PNG, PDF and
a JSON provenance manifest with exact plotted values, input hash and warnings.

The collector validates analysis/manifest identity, correctness, phase,
instrumentation switches, system, EC sequence continuity and metadata sums.
With both metrics, their final Work boundaries must agree. Paired systems must
use the same application footprint. The plot requires complete coverage of
the explicitly selected apps/metrics. Missing is an error, never zero; an
explicitly recorded zero is valid. Legacy CPU-normalized/seconds and
metadata-peak data are rejected.

Repetitions come from batch-plan.json, never from directory name suffixes.
Explicit older run directories are labelled repeat=unindexed; --repeat cannot
relabel them as another repetition. An explicitly selected wrong ratio/runtime
is an error, not silently omitted.
Without a batch index, duplicate app/system/metric conditions are errors.
Select retries or older mixed result trees using repeated --run-dir:

```bash
bash scripts/figure12/collect.sh --metrics metadata \
  --run-dir results/selected-kv-b-starfish \
  --run-dir results/selected-kv-b-carbink \
  --output data/figure12-kv-metadata.csv
bash scripts/figure12/plot.sh --input data/figure12-kv-metadata.csv \
  --apps kv-b --metrics metadata --output-dir results/figures/figure12-kv
```

Existing KV-B/NQ metadata logs remain usable with --metrics metadata.
They have no EC-cycle records: new instrumented runs are necessary for EC.
collect_metadata.py remains a strict metadata-only compatibility entry point.

The common user-approved exception can retain completed, reverified Starfish
KV work after a postcheck shutdown timeout/termination. The main collector
preserves the nonzero exit, teardown_failed state and warning; it still requires
the actual selected Figure 12 record. In particular, missing exit metadata is
not reconstructed. Other crashes/partial runs remain errors.

For layout tests only, --source-type synthetic must be explicit and produces
separately named, visibly labelled output; never mix synthetic and measured
rows. --validate-only performs all parsing/completeness checks without drawing.

## Hardware-free regression checks

```bash
python3 -B -m unittest discover -s scripts/common/tests -p 'test_figure12_*.py' -v
python3 -B -m unittest discover -s scripts/common/tests -p 'test_runtime_ec_cpu.py' -v
python3 -B -m unittest discover -s scripts/common/tests -p 'test_runtime_metadata.py' -v
```

Use a Python with plotting dependencies for the first command. These tests
cover the full app/system plan and parser matrix, not actual experimental
coverage of every application.
