# Bandwidth microbenchmark

This optional synthetic workload measures memory-access throughput and latency
with fixed 4000-byte or 512-byte objects. It is not an application object-size
profiler and is not part of the default Figure 9–12 application matrix.
The rename from `object_size` does not change the workload or its arguments.

Build both sizes from the artifact-evaluation root:

```bash
bash scripts/common/build.sh --system starfish \
  --targets bandwidth_microbenchmark,bandwidth_microbenchmark_512
```

Executables are written to
`build/starfish/benchmark/microbenchmarks/bandwidth_microbenchmark` and
`bandwidth_microbenchmark_512`. Direct CMake users can enable
`FARLIB_BUILD_BANDWIDTH_MICROBENCHMARK=ON`; it is off by default.

Historical `object_size` / `object_size_512` build targets and executable
symlinks remain compatibility aliases to the same binaries.
`FARLIB_BUILD_OBJECT_SIZE` is a deprecated CMake option alias.
These optional targets are supported only by the Starfish runtime.

Building does not run the workload. Reuse a reviewed configuration and record
the original workload scale, object size, cache/backup settings and CPU
placement before launching; this rename is not a new performance baseline.
