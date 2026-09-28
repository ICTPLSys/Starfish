# Starfish Artifact Evaluation

Companion artifact for **Starfish: Fault-Tolerant Far Memory with Low Resource
and Performance Overhead** (ATC 2026; paper supplied with the AE submission).
By Yanwen Xia, Benyong Deng, Hong Huang, Yuzheng Wang, Quanxi Li, Xingda Wei,
Xiaobing Feng, Huimin Cui, and Chenxi Wang.

Our code uses [Apache-2.0](LICENSE); third-party code and inputs retain their
own licenses, including GPL-3.0 for libfibre.

Reproducing this system requires multiple machines connected by 100 Gbps InfiniBand (up to nine machines in the paper, including recovery standbys).

The Starfish runtime, the Non-FT, Hydra-like and Carbink-like baselines,
and the evaluation environment are ready. See [Quick Start](#quick-start)
for setup and a minimal experiment.

AE reviewers should post their SSH public key in standard OpenSSH format in
the AE discussion. Once we authorize the key, connect to the compute server:

```bash
ssh -J atc26ae@8.140.50.79 atc26ae@10.208.130.56
```

No reviewer-side WireGuard installation or configuration is required. Use the
matching private key on your own machine; never post or upload it. If the key
is not a default SSH identity, configure `IdentityFile` for both the jump host
and compute host in your local SSH config. The authors will provide the
prepared artifact directory and site configuration in the AE discussion.

## Overview

Paths below are relative to `artifact-evaluation/`:

| Component | Paper relationship |
| --- | --- |
| `runtime/starfish/`, `runtime/nonft/`, `runtime/hydra/`, `runtime/carbink/` | Starfish, Non-FT, and our Hydra-like and Carbink-like implementations |
| `apps/`, `configs/` | Evaluation applications and configurations |
| `scripts/figure9/` | Application-performance experiments, result collection and Figure 9 |
| `scripts/figure10/`–`figure13/` | Latency, resource cost, compute overhead and recovery figures |
| `third_party/libfibre/` | Pinned libfibre source and its errnoname dependency; upstream licenses and source information are retained |

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](artifact-evaluation/docs/INSTALL.md) | Dependencies, installation and build commands |
| [Data and models](artifact-evaluation/docs/DATA.md) | Download sources, preparation and input paths |
| [Multi-server configuration](artifact-evaluation/docs/MULTI-SERVER.md) | Compute/memory roles, SSH/TCP addresses and ports |
| [Configuration layout](artifact-evaluation/configs/README.md) | Recipes, overrides, generated files and deployment checks |
| [Recovery integration status](artifact-evaluation/docs/RECOVERY-STATUS.md) | Starfish v13 source, supported mode and historical evidence |
| [Server table](artifact-evaluation/docs/MACHINES.md) | Server IPs, IB devices and NUMA placement |
| [Experiments](artifact-evaluation/docs/EXPERIMENTS.md) | Experiment commands, result files and plotting |
| [Script index](artifact-evaluation/scripts/README.md) | Figure-specific instructions and CSV formats |

## Quick Start

From the repository root:

~~~bash
cd artifact-evaluation
bash scripts/common/setup_environment.sh
bash scripts/common/check_environment.sh
bash scripts/common/build.sh
~~~

Use the prepared data provided under `/data/starfish-ae/`:

~~~bash
bash scripts/common/use_prepared_data.sh
~~~

The script records input paths in `data/site.json`. Set the server addresses
once using [multi-server configuration](artifact-evaluation/docs/MULTI-SERVER.md).
To download and prepare inputs from their original sources instead, follow
[Data and models](artifact-evaluation/docs/DATA.md).

Before running benchmarks, we recommend that reviewers run `bash scripts/check_nodes.sh`
and confirm that the intended compute and memory nodes are idle.

### Fast check

Run LLaMA at 25% local memory once each with Non-FT, Starfish, and Starfish
with recovery from one memory-service failure:

~~~bash
bash scripts/run_fast_check.sh
~~~

The script builds the required targets and checks that all three runs finish
with the same chat output. See [fast-check details](artifact-evaluation/docs/EXPERIMENTS.md#fast-check)
for configuration and result files.

Start Figure 9 with:

~~~bash
bash scripts/figure9/run.sh
~~~

## Kick the Tires

The minimal working example uses one compute server and one memory server.

### Environment and inputs

Hardware reference and operating-system requirements:

| Role | CPU / RAM | OS requirement |
| --- | --- | --- |
| Compute | 2× Xeon Gold 6342, 48 cores / 256 GiB | Ubuntu 22.04 |
| Memory | 2× Xeon Silver 4316, 40 cores / 256 GiB | Ubuntu 22.04 |

No specific Linux kernel version is required; compatible RDMA drivers are needed.

Compute tools: GCC 13.1, CMake 3.22.1, ConnectX-5 Ex and RDMA userspace
`2410mlnx54-1.2410068`. Build the memory server locally if its libc differs.
Dependencies and pinned libfibre/HdrHistogram builds are in the
[installation guide](artifact-evaluation/docs/INSTALL.md). RDMA/SSH access
and 2 MiB HugePages must be configured before running.

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

KV uses 512-byte records and Zipfian skew 0.99. Download sources and preparation
instructions are in [Data and models](artifact-evaluation/docs/DATA.md).

KV records and requests, NQ queries, and MG grids are generated at runtime
using the workload parameters specified in the paper; NQ uses the prepared
Friendster graph.

### Estimated resources and time (three repetitions)

Paper-scale runs use 24 application cores and 256 GB RAM per node with
100 Gbps InfiniBand: two nodes for the Non-FT example, seven for steady-state
RS(4,2) experiments, and up to nine including recovery standbys. Reserve about
200 GiB of compute-side workspace for inputs, builds and logs.

The following estimates cover serial application work for three repetitions.
Installation, input loading, warm-up and service resets are additional.

| Experiment | Estimated work time |
| --- | --- |
| Application performance (Fig. 9) | 11.8 hours |
| Tail latency (Fig. 10) | 5.1 hours |
| FT resource cost (Fig. 11) | 3.2 hours |
| Compute-node overhead (Fig. 12) | 1.6 hours |
| Failure recovery (Fig. 13) | 10 minutes |

Please allow 2–3 days for Figure 9.

## Run and plot

From `artifact-evaluation/`:

| Figure | Run / collect | Plot |
| --- | --- | --- |
| [Figure 9: application performance](artifact-evaluation/scripts/figure9/README.md) | `bash scripts/figure9/run.sh` | `bash scripts/plot.sh figure9` |
| [Figure 10: tail latency](artifact-evaluation/scripts/figure10/README.md) | `bash scripts/figure10/run.sh` | `bash scripts/plot.sh figure10 --input data/figure10.csv` |
| [Figure 11: FT resource cost](artifact-evaluation/scripts/figure11/README.md) | `bash scripts/figure11/collect.sh --logs-root results/figure9` | `bash scripts/plot.sh figure11 --input data/figure11.csv` |
| [Figure 12: compute-node overhead](artifact-evaluation/scripts/figure12/README.md) | `bash scripts/figure12/run.sh --build --metrics all`, then follow its collection guide | `bash scripts/plot.sh figure12 --input data/figure12.csv` |
| [Figure 13: failure recovery](artifact-evaluation/scripts/figure13/README.md) | `bash scripts/figure13/collect.sh --logs-root results/figure13` | `bash scripts/plot.sh figure13 --logs-root results/figure13` |

The Figure 9 runner writes `data/figure9.csv`. Input formats and output files
are described in the linked figure guides.

Runs start and stop memory services: use unused ports and dedicated result
directories on authorized hosts. Failure injection must target isolated
services, never production endpoints.

See [the scripts guide](artifact-evaluation/scripts/README.md) for detailed
server setup, experiment options, and CSV formats.
