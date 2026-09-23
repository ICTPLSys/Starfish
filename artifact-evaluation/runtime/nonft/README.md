# NonFT Runtime

This directory contains the NonFT runtime and builds the shared LLaMA and BFS
applications from `../../apps/`. The legacy `benchmark/` source directory is
not needed for these targets and is not included in this source drop.

The NonFT configuration keeps selective backup and recovery disabled. The
source tree includes the common placement and cache code used by the baseline,
but does not include Design 2 or Recovery implementations.

## Build on Linux

The build requires a C++20 compiler, CMake, libibverbs, libfibre, Boost
program_options, OpenSSL, and hdr_histogram. PAPI and ISA-L are detected when
available.

From the repository root:

```bash
export LIBFIBRE_DIR=/path/to/libfibre
cmake -S artifact-evaluation/runtime/nonft -B build/nonft \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/nonft \
  --target server run_chat_far run_chat_far_prof gapbs_bfs_chunked -j 4
```

The binaries are written to `build/nonft/server`,
`build/nonft/benchmark/llama/`, and
`build/nonft/benchmark/microbenchmarks/`. The `benchmark` name here is only an
output path, not a source dependency.

Model weights, graph data, and site-specific run configurations are not part
of this source-only drop.
