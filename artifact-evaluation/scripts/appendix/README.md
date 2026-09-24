# Appendix Figures 14 and 15: benchmark suites

Both appendix plots read CSV files from an explicitly supplied directory.
Absent access-amplification rows remain blank; they do not become a false
1.0x value.

```bash
python3 scripts/appendix/plot.py access --data-dir data/benchmark_suites
python3 scripts/appendix/plot.py cdf --data-dir data/benchmark_suites
```

For a *paper-reference preview* only, pass an explicit directory of reference
CSVs with `--source-type paper_reference`. Measured and reference outputs go
to separate result directories; CSV files are never merged or auto-filled.

The access figure expects `access_amplification.csv` with columns
`suite,application,amplification_rate`. The object-size CDF (cumulative
distribution of memory bytes by object size) expects
`dcperf_object_size_cdf_combined_fine.csv` and
`tailbench_object_size_cdf_raw.csv` with their original headers.
The object-size plot is written as `object_size_memory_cdf.pdf`.
