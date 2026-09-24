# NonFT Runtime

This directory contains the NonFT runtime and builds the shared LLaMA and BFS
applications from `../../apps/`.

The Non-FT recipe disables selective backup, the resident-placement planner,
and recovery by design. Diagnostic counters remain available in raw logs.

## Build on Linux

The build requires a C++20 compiler, CMake, libibverbs, libfibre, Boost
program_options, OpenSSL, and hdr_histogram. PAPI and ISA-L are detected when
available.

From `artifact-evaluation/`:

```bash
bash scripts/common/setup_environment.sh --jobs 4
bash scripts/common/build.sh --system nonft --jobs 4 \
  --targets server,run_chat_far,gapbs_bfs_chunked
```

The setup fetches exact public revisions into ignored `deps/` and never
modifies system packages unless `--install-apt` is explicitly requested.
See [the scripts guide](../../scripts/README.md) to use existing dependencies.

The binaries are written to `build/nonft/server`,
`build/nonft/benchmark/llama/`, and
`build/nonft/benchmark/microbenchmarks/`. The `benchmark` name here is only an
output path, not a source dependency.

Model weights, graph data, and site-specific run configurations are supplied
separately.
