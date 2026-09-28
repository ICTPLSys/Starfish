# Installation

Follow these steps on each server that builds the artifact. The reference
hardware, operating-system requirements and resource estimates are in the
[README](../../README.md#environment-and-inputs). Configure the compute and
memory servers using [MULTI-SERVER.md](MULTI-SERVER.md), then follow
[EXPERIMENTS.md](EXPERIMENTS.md). Prepare workload inputs using
[DATA.md](DATA.md).

Use a case-sensitive filesystem: libfibre contains both `Fibre.h` and `fibre.h`.

## Runtime builds

The available runtime source trees are:

- `nonft` — Non-FT runtime;
- `starfish` — Starfish runtime;
- `hydra` — Hydra runtime, which requires ISA-L headers and library.

## Prerequisites

Use a Linux x86-64 host with:

- CMake 3.16 or newer, a C++20 compiler, GNU Make or Ninja, Git, Bash, and
  Python 3 with venv support, curl, gzip, and GNU coreutils;
- RDMA verbs development headers/libraries, Boost `program_options`, OpenSSL,
  zlib, and HdrHistogram C 0.11.8;
- ISA-L (`libisal-dev` or an equivalent readable installation) for Hydra;
- SSH/SCP, `ibv_devinfo`, and working RDMA configuration for a two-server run.

A live memory server is not required for a build. A run additionally needs a
separate memory server, RDMA connectivity, compute-side HugePages, the site
configuration, and the declared workload inputs. WireGuard is an access path;
it is not the RDMA data path.

## 1. Inspect and prepare dependencies

From `artifact-evaluation/`, inspect the plan first:

```bash
bash scripts/common/setup_environment.sh --dry-run
```

The dry run prints the dependency plan without changing files or packages.

The normal setup verifies the vendored pinned libfibre source under
`third_party/libfibre` (including its required `src/errnoname` content), copies
it to the ignored `deps/` build location, and builds HdrHistogram there. It does
not clone libfibre from the network. In a root shell on a fresh Ubuntu/Debian
host, it installs missing system prerequisites automatically. In an ordinary
reviewer account it prepares account-local dependencies without implicit sudo:

```bash
bash scripts/common/setup_environment.sh
```

An administrator can also request system-package installation explicitly.
Setup creates the ignored `.venv-plot` environment used by the plotting
wrappers by default:

```bash
bash scripts/common/setup_environment.sh --install-apt
```

`--install-apt` may require `sudo`; it installs the declared Ubuntu/Debian
packages, including `libisal-dev`, and changes system packages. If HdrHistogram
0.11.8 is already installed at a readable prefix, reuse it instead of building
another copy:

```bash
bash scripts/common/setup_environment.sh --hdr-prefix /usr/local
```

For a supplied libfibre checkout at the pinned revision, pass
`--libfibre-dir DIR`; `setup_environment.sh --dry-run` shows the selected
paths. The script refuses to silently change an existing dependency checkout.

## 2. Check a host

A check diagnoses prerequisites; it does not install or repair them:

```bash
bash scripts/common/check_environment.sh
bash scripts/common/check_environment.sh --system starfish --mode build
bash scripts/common/check_environment.sh --system hydra --mode build
```

For a run-phase check, add the site-specific variables and use:

```bash
bash scripts/common/check_environment.sh --system nonft --mode run
bash scripts/common/check_environment.sh --system nonft --mode full \
  --require-workloads
```

A successful check ends with `summary: failures=0`; warnings remain visible.
PAPI hardware counters are optional and disabled by default.

## 3. Build a runtime

With no arguments, the build selects Non-FT and four parallel jobs. The
default targets are the memory `server`, LLaMA and BFS:

```bash
bash scripts/common/build.sh
bash scripts/common/build.sh --system starfish
bash scripts/common/build.sh --system hydra
```

Build MG explicitly (the target list is comma-separated):

```bash
bash scripts/common/build.sh --system nonft --targets server,mg --jobs 4
```

Use the corresponding `--system starfish` or `--system hydra` when that
runtime is the intended binary. Starfish's optional tests are selected with
`--with-tests`.

Expected output locations are:

```text
build/<system>/server
build/<system>/benchmark/llama/run_chat_far
build/<system>/benchmark/microbenchmarks/gapbs_bfs_chunked
build/<system>/benchmark/mg/mg       # after an explicit MG target
```

`build.sh --dry-run` prints the selected runtime, build directory, targets, and
CMake commands without configuring or compiling. A completed real build prints
`build: PASS` and records an environment snapshot under the build directory.

## 4. Continue to a run

Select the server IPs in `data/site.json` as described in
[MULTI-SERVER.md](MULTI-SERVER.md). Device, NUMA and input-path defaults are
resolved automatically; set overrides in the same file as needed. Then use the
commands in [EXPERIMENTS.md](EXPERIMENTS.md).
