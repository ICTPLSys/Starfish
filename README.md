# Starfish Artifact Evaluation

Reproducing this system requires multiple machines connected by 100 Gbps InfiniBand (eight machines in the paper).

🚧 **Notice! This repo is still under construction!**

The code is still being organized and will be filled in ASAP before the Kick
the Tires deadline. Running the complete experiment requires eight physical
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
- ✅ Prepared application code: LLM, BFS, and WC
- 🔄 Applications being organized: KV, MG, NQ
- ✅ One-command run and plotting script
- 🔄 Carbink — being organized
- ✅ Hydra

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
  --systems nonft --ratios 13,25,50,75,100 --repeats 1
```

### 3. Plot

The runner copies successful measurements to `data/figure9.csv`. Plot them
with:

```bash
bash scripts/plot.sh figure9 --input data/figure9.csv
```

See [the scripts guide](artifact-evaluation/scripts/README.md) for detailed
server setup, experiment options, and CSV formats.
