# Starfish Artifact Evaluation

Companion artifact for **Starfish: Fault-Tolerant Far Memory with Low Resource
and Performance Overhead** (ATC 2026; paper supplied with the AE submission).
By Yanwen Xia, Benyong Deng, Hong Huang, Yuzheng Wang, Quanxi Li, Xingda Wei,
Xiaobing Feng, Huimin Cui, and Chenxi Wang.
[Paper and author information](https://ssk015.github.io/publications/).

Our code uses [Apache-2.0](LICENSE); third-party code and inputs retain their
own licenses, including GPL-3.0 for libfibre.

**Archive:** GitHub is the development copy. A permanent versioned archive
and DOI are not yet published; this Available requirement remains pending.

Reproducing this system requires multiple machines connected by 100 Gbps InfiniBand (up to nine machines in the paper, including recovery standbys).

🚧 **Notice! This repo is still under construction!**

The code is still being organized and will be filled in ASAP before the Kick
the Tires deadline. Running the complete experiment requires up to nine physical
servers. Several are currently occupied by other lab members, and we will
coordinate their availability by September 29. Before then, AE reviewers can
prepare the environment and run Kick the Tires on the servers that are already
available; see [Kick the Tires](#kick-the-tires) below.

**We apologize for the delay.** *The deadline is very tight (four days from
notification).* And we are already trying our best. We hope the AE reviewers
will understand.

AE reviewers should provide an SSH public key. Server access for AE reviewers
will be provided through WireGuard.

## (1) Current artifact status

- ✅ Runtime code
- ✅ Starfish system code
- ✅ Non-FT system code
- ✅ Design functionality
- ✅ Recovery functionality
- ✅ Prepared application code: LLM, BFS, WC, and MG
- 🔄 Applications being organized: KV, NQ
- ✅ One-command run and plotting script
- 🔄 Carbink — being organized
- ✅ Hydra

Paths below are relative to `artifact-evaluation/`:

| Component | Paper relationship |
| --- | --- |
| `runtime/starfish/`, `runtime/nonft/`, `runtime/hydra/` | Starfish, its non-FT baseline and our Hydra-like implementation (not upstream Hydra) |
| `apps/`, `configs/` | Evaluation applications and configurations |
| `scripts/figure9/` | Application-performance runs, collection and plots; current runner supports LLaMA/BFS |
| `scripts/figure10/`–`figure13/`, `scripts/appendix/` | Plots for latency, overhead, recovery and appendix results; existing CSV inputs required |

## (2) First functional check

Reviewers can currently run LLaMA to confirm the functional path. This check
requires only two servers:

- one compute server;
- one memory server.

## (3) Ongoing updates

The remaining code and scripts will continue to be organized and updated
before the Kick the Tires deadline.

## Kick the Tires

Run the following commands from `artifact-evaluation/` on Linux x86-64.

### Environment and inputs

Hardware reference and operating-system requirements:

| Role | CPU / RAM | OS |
| --- | --- | --- |
| Compute | 2× Xeon Gold 6342, 48 cores / 256 GiB | Ubuntu 22.04 |
| Memory | 2× Xeon Silver 4316, 40 cores / 256 GiB | Ubuntu 22.04 |

No specific Linux kernel version is required; compatible RDMA drivers are needed.

Compute tools: GCC 13.1, CMake 3.22.1, ConnectX-5 Ex and RDMA userspace
`2410mlnx54-1.2410068`. Build the memory server locally if its libc differs.
Dependencies and pinned libfibre/HdrHistogram builds are in the
[scripts guide](artifact-evaluation/scripts/README.md). On a fresh Ubuntu/Debian
host, setup's `--install-apt` requires administrator approval. RDMA/SSH access
and 2 MiB HugePages must already be configured; no VM/container recipe is
currently supplied.

| Application | Dataset / model |
| --- | --- |
| LLaMA | LLaMA 2 7B Chat (FP32) |
| BFS | Friendster social graph |
| MG | NAS Parallel Benchmarks, Class D |
| WordCount | English Wikipedia |
| KV-B | YCSB Workload B: 95% reads, 5% updates; 1 billion operations |
| KV-A | YCSB Workload A: 50% reads, 50% updates; 1 billion operations |
| KV-S | Synthetic: 5% reads, 95% updates; 1 billion operations |
| NQ | Friendster social graph; 2-hop neighborhood queries, 2 million queries |

KV uses 512-byte records and Zipfian skew 0.99. This table describes the paper
workloads, not a claim that every workload is already connected to the runner.

### Estimated resources and time (three repetitions)

Paper-scale runs use 24 application cores and 256 GB RAM per node with
100 Gbps InfiniBand: two nodes for the Non-FT example, seven for steady-state
RS(4,2) experiments, and up to nine including recovery standbys. Reserve about
200 GiB of compute-side workspace for inputs, builds and logs as a planning
allowance, not a measured disk requirement.

The following are **serial work-phase estimates**, not measured AE completion
times. They exclude installation, input loading, warm-up and service resets.
Figure 9 uses `3 × sum(all plotted elapsed times)`; native-reference runs are
not included. Figures 11–12 use the corresponding Figure 9 times as proxies.

| Experiment | Scope with three repetitions | Estimated work time |
| --- | --- | --- |
| Application performance (Fig. 9) | 8 workloads × 4 systems × 5 ratios × 3 = 480 runs | 11.8 hours |
| Tail latency (Fig. 10) | 102 load points across both workloads and four systems × 3 = 306 runs | 5.1 hours, assuming a 60-second measurement window per point |
| FT resource cost (Fig. 11) | 8 workloads × 4 systems at 25% × 3 = 96 runs | 3.2 hours if collected separately |
| Compute-node overhead (Fig. 12) | 8 workloads × 2 systems at 25% × 3 = 48 runs | 1.6 hours if collected separately |
| Failure recovery (Fig. 13) | 3 systems × 2 failure scenarios × 3 = 18 runs | 9–10 minutes, assuming 30–32-second observation windows |
| Backup-budget sweep (§5.5) | MG and LLM at 50%, three runs each per budget value | About 5.5 minutes per budget value; sweep count is additional |
| Coding / optional replication (§5.5) | Three runs per coding or replication configuration | About 0.6 minutes per BFS setting, or 1.5 minutes per recovery setting, using default-setting / 30-second-window proxies |
| Appendix profiling (Figs. 14–15) | 14 suite workloads × 3 = 42 profiling runs, if both metrics are collected together | Not inferable from the plotted ratios/CDFs; needs a pilot estimate |

Figure 9's per-workload totals are approximately: BFS 12 min, LLM 75 min,
MG 48 min, WC 68 min, KV-B 99 min, KV-A 107 min, KV-S 116 min and NQ 184 min.
For scheduling, allow roughly 2–3 days for Figure 9 if each independent run
adds 3–5 minutes of preparation. This overhead is a planning assumption, not
reported by the paper. Figures 11–12 need not add runs if their metrics are
captured during the corresponding Figure 9 runs.

### 1. Check the environment and build

On each server that needs a build, prepare dependencies and build Non-FT:

```bash
bash scripts/common/setup_environment.sh --with-plot --jobs 4
bash scripts/common/check_environment.sh --system nonft --mode build
bash scripts/common/build.sh --system nonft --jobs 4
```

### 2. Run

Use a compute server and a memory server. `data/site.json` is a local site
file (not an application or system config): it specifies the memory server's
SSH host, RDMA address and binary, RDMA devices, and workload input paths.
It is ignored by Git. Copy the template and edit it for your servers.
For a quick functional check, run one LLaMA/Non-FT case at 25% local memory:

```bash
mkdir -p data
cp -n scripts/common/site.example.json data/site.json
# Edit data/site.json for your servers and inputs.
bash scripts/figure9/run.sh --site data/site.json --apps llama \
  --systems nonft --ratios 25 --repeats 1
```

To run LLaMA and BFS with Non-FT at all five local-memory ratios:

```bash
bash scripts/figure9/run.sh --site data/site.json --apps llama,bfs \
  --systems nonft --ratios 13,25,50,75,100 --repeats 3
```

Success requires exit status 0 and a measured row in the batch CSV. LLaMA checks
chat completion, not answer quality; BFS requires `gapbs_bfs_verify status=pass`.
Effective configs and logs are under `results/figure9/batch-*/runs/`. Extend the
example with `--apps`, `--ratios` and `--repeats`, without editing generated
configs. Missing dependencies, insufficient HugePages, timeouts and failed
verification are errors, not expected warnings.

Runs start and stop memory services: use unused ports and dedicated result
directories on authorized hosts. Failure injection must target isolated
services, never production endpoints.

### 3. Plot

The runner copies successful measurements to `data/figure9.csv`. Plot them
with:

```bash
bash scripts/plot.sh figure9 --input data/figure9.csv
```

See [the scripts guide](artifact-evaluation/scripts/README.md) for detailed
server setup, experiment options, and CSV formats.

**Remaining release requirements:** permanent archive; exact input/provenance
instructions, calibrated end-to-end budgets and examples for each experiment type; reproducible
OS-image recipe or explicit AE guidance for bare-metal RDMA. The two-server
example does not certify unfinished components or reproduce other figures.
