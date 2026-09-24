# Linux environment and build scripts

Run from `artifact-evaluation/` on a Linux x86-64 compute server.

## Dependencies

- CMake 3.16 or newer, a C++20 compiler, and Ninja or Make.
- RDMA verbs development headers and libraries.
- Boost program_options, OpenSSL, and HdrHistogram (including its CMake package).
- ISA-L is used if installed. PAPI is optional: hardware counter events may
  be unavailable even when its library is installed, so the build script
  disables it by default. Use `--enable-papi` only on a verified host.
- The pinned libfibre fork and HdrHistogram_c 0.11.8. The setup script builds
  both under ignored `deps/`; existing installations can be supplied instead.
- Python 3 for the environment checker and experiment helpers.

Building needs the RDMA libraries, but does not require a live memory server.
Running an application needs a compute server and a separate memory server,
working RDMA connectivity, site configuration, and the model or graph input.
WireGuard provides remote access or Internet connectivity; it is not the
RDMA experiment data path.

## Prepare and build

```bash
bash scripts/common/setup_environment.sh --dry-run
bash scripts/common/setup_environment.sh --jobs 4
bash scripts/common/check_environment.sh --system nonft --mode build
bash scripts/common/build.sh --system nonft --jobs 4
bash scripts/common/collect_environment.sh \
  --system nonft --build-dir build/nonft \
  --output build/nonft/environment.txt
```

On a fresh Ubuntu/Debian host with permission to modify system packages,
explicitly add `--install-apt`. To install plot dependencies in the ignored
`.venv-plot`, add `--with-plot`. Neither happens by default. The script clones
the [public libfibre fork](https://github.com/Crazylqx/libfibre) at
`885d74dfb6746966a911ff6c821cc4a88fda3b91` (GPL-3.0) and
[HdrHistogram_c](https://github.com/HdrHistogram/HdrHistogram_c) at
`8dcce8f68512fca460b171bccc3a5afce0048779` (0.11.8); it checks the
exact commits on every rerun and never resets an existing checkout. Its
`--libfibre-dir` and `--hdr-prefix` options accept already prepared copies,
including on a host without GitHub access. When using external paths, also
set `LIBFIBRE_DIR` and `HDR_HISTOGRAM_PREFIX` for `build.sh`.

Run setup separately on each physical server that builds AE code. It does
not configure kernel modules, HugePages, RDMA interfaces, WireGuard, server
addresses, or datasets. Those remain site-specific preflight steps.

### Separate reviewer accounts on the provided servers

Each reviewer should work in their own checkout and results directory. The
provided servers already have HdrHistogram 0.11.8 at `/usr/local`, so reuse
it rather than reinstalling it. Let setup build libfibre under this checkout's
ignored `deps/` so each reviewer account has access to its own copy. On a
compute server:

```bash
bash scripts/common/setup_environment.sh --hdr-prefix /usr/local \
  --with-plot --jobs 2
bash scripts/common/build.sh --system nonft --jobs 2
```

Omit `--with-plot` on a memory server that only builds/runs `server`. Do not
use `--install-apt` in a reviewer account on the shared servers. An
administrator must supply readable application inputs, an accessible memory
server binary (built for that host), site-specific SSH/RDMA access, and
compute-side HugePages before any real experiment. The preparation command
does not grant access to another user's private files.

`build.sh` builds the memory `server` plus the released LLaMA and BFS
applications. Raw runtime diagnostics remain in each case's `client.log` and
`server.log`; the terminal prints one short result per case.

Sources are selected from `runtime/<system>/`; outputs go to
`build/<system>/`. `--runtime-dir` and `--build-dir` allow explicit
overrides; `--dry-run` prints the build commands.

The binary paths currently remain `build/nonft/server`,
`build/nonft/benchmark/llama/run_chat_far`, and
`build/nonft/benchmark/microbenchmarks/gapbs_bfs_chunked`. Here
`benchmark` is a build-output path, not a dependency on the omitted legacy
source directory.

`check_environment.sh` diagnoses prerequisites but does not install them.
`setup_environment.sh` installs into ignored `deps/` and only touches system
packages with the explicit `--install-apt` flag. Copy
`scripts/common/site.example.json` to ignored `data/site.json` and edit the
server addresses and workload input paths for a run.
Environment snapshots are stored in the ignored build directory; review them
before sharing because they contain hostnames, addresses, and local paths.

## Figure 9 experiment path

`scripts/figure9/run.sh` enumerates cases and delegates each run to
`scripts/common/run_case.py`. Application-specific commands and log checks
are in `scripts/common/workloads.py`; `scripts/figure9/collect.py` accepts
only completed, verified run records. It never fills gaps from reference
data. Each batch keeps its own `figure9.csv`; after plotting, the runner copies
it to ignored `data/figure9.csv` for independent plotting if it contains at
least one successful measurement. A wholly failed batch leaves the previous
copy untouched. See [the Figure 9 guide](figure9/README.md) for the site configuration,
read-only plan, initial Non-FT smoke run, and measurement boundaries.

## Plotting

Plotting is independent of builds and experiment execution. From the
artifact directory, run `bash scripts/plot.sh figure9` to plot the latest
completed batch, or pass `--input results/figure9/<batch>/figure9.csv` to
replot an earlier one.
The wrapper also dispatches `figure10`, `figure11`, `figure12`, `figure13`,
and `appendix`; each directory's README specifies its CSV
contract. Install plotting-only Python dependencies from
`scripts/common/requirements-plot.txt` in a separate virtual environment.

Each figure reads a CSV input (Figure 9 defaults to `data/figure9.csv`). Missing measurements stay
blank. Plots labelled `paper_reference` or `synthetic` are not AE experimental
results. Input `data/` and generated `results/` remain ignored by Git.
