# Running the experiments

Install dependencies and build the binaries using [INSTALL.md](INSTALL.md).
Configure the compute and memory servers once in a site file using
[MULTI-SERVER.md](MULTI-SERVER.md). Commands below are run from
`artifact-evaluation/`.

The [README](../../README.md#environment-and-inputs) lists the paper's
workloads and inputs; [DATA.md](DATA.md) describes where to obtain and how to
prepare them. Its [resource and time estimates](../../README.md#estimated-resources-and-time-three-repetitions)
cover three repetitions, server requirements and workspace capacity.

## Fast check

~~~bash
bash scripts/run_fast_check.sh
~~~

This runs the full LLaMA 2 7B Chat FP32 `hello` conversation at 25% local
memory, once per condition: Non-FT, Starfish, and Starfish with recovery.
All three cases use the same prompt, model, tokenizer and sampling settings.
Required Non-FT and Starfish targets are built automatically.

Both Starfish runs use the same RS(4,2) EC configuration with a standby
endpoint. With a single memory host in `data/site.json`, the script starts
seven service processes on that host, using `server_port` through
`server_port + 6`; these ports must be unused. Each service reserves 12 GiB.
With seven or more explicit memory endpoints, the first seven are used
instead. Non-FT uses the first endpoint.

The recovery run stops only its own endpoint-0 process after chat output
begins. The check tests a service-process failure, not loss of
the entire memory host. It requires observed EC reconstruction, successful
runtime verification, normal application completion, and byte-identical
chat output across all three runs.

Results are under `results/fast-check/batch-*/`: `summary.json`, build
logs, and per-case `runs/*/analysis.json`, `client.log`, `chat-output.txt`,
effective configuration and memory-service logs. A failure exits nonzero.
Fast-check results are stored separately from Figure 9.

## Figure 9: application performance

### 1. Configure the servers and inputs

~~~bash
mkdir -p data
cp -n scripts/common/site.example.json data/site.json
# Set the compute and memory server IPs.
~~~

Use one compute server and a separate memory server for the Non-FT example.
For the provided eight-server layout, use
`scripts/common/site.eight-server.example.json`.

LLaMA uses the LLaMA 2 7B Chat FP32 model, its tokenizer and
`apps/llama/llama_user_chat.txt`. BFS uses the Friendster graph prefix with
shards `.0` through `.31`. Inputs default to `/data/starfish-ae/`;
override individual paths in `data/site.json` if needed.

Inspect the run plan before starting:

~~~bash
bash scripts/figure9/run.sh --site data/site.json --dry-run \
  --apps llama,bfs --systems nonft --ratios 25
~~~

### 2. Run the minimal example

~~~bash
bash scripts/figure9/run.sh --site data/site.json \
  --apps llama --systems nonft --ratios 25 --repeats 1
~~~

The command prints a result for each case and writes a new
`results/figure9/batch-*/` directory. Each run contains:

| File | Contents |
| --- | --- |
| `effective.config` | Application and runtime configuration used for the run |
| `client.command.txt` | Client invocation |
| `client.log` | Application output and timing |
| `endpoints/endpoint-*/` | Memory-service configurations, manifests and logs |
| `manifest.json` | Source, binary, input and environment records |
| `analysis.json` | Exit status, correctness checks and measured result |
| `figure9.csv` (batch directory) | Collected successful measurements |

The top-level `server.config` and `server.log` also contain the first
endpoint's configuration and output.

Success requires exit status zero and `correctness: "pass"` in
`analysis.json`. LLaMA checks chat completion, an `Assistant:` response and
positive token throughput. BFS requires `gapbs_bfs_verify status=pass` and
matching visited counts. A failed case causes a nonzero batch exit status;
inspect that case's logs before rerunning.

### 3. Extend the example

Run LLaMA and BFS at all five local-memory ratios with three repetitions:

~~~bash
bash scripts/figure9/run.sh --site data/site.json \
  --apps llama,bfs --systems nonft \
  --ratios 13,25,50,75,100 --repeats 3
~~~

Select a different system with `--systems`, for example:

~~~bash
bash scripts/figure9/run.sh --site data/site.json \
  --apps bfs --systems starfish --ratios 25 --repeats 3
~~~

The launcher renders effective configurations from the application templates.
Change the site file and command-line selections rather than editing
generated configurations. Keep workload inputs and timing boundaries the
same when comparing systems.

LLaMA's `elapsed_s` is the chat-work time reported by the application
(cycles divided by 2.8 GHz). BFS's `elapsed_s` is one traversal's reported
work time. Loading and service startup are outside these timing boundaries.

### 4. Plot the results

The collector writes `results/figure9/<batch>/figure9.csv`. A successful
batch also publishes a convenient copy at `data/figure9.csv`:

~~~bash
bash scripts/plot.sh figure9 --input data/figure9.csv
~~~

To plot an earlier batch, use its CSV path with `--input`. Outputs are
`results/figures/figure9/figure9.png`, `.pdf` and `.json`. The JSON keeps
the input and plotting metadata.

See the [Figure 9 guide](../scripts/figure9/README.md) for the CSV columns,
repetition aggregation and plot options.

## Figure and result map

| Paper result | Plot command | Instructions |
| --- | --- | --- |
| Figure 9: application performance | `bash scripts/plot.sh figure9 --input data/figure9.csv` | [Figure 9](../scripts/figure9/README.md) |
| Figure 10: tail latency | `bash scripts/plot.sh figure10 --input data/figure10.csv` | [Figure 10](../scripts/figure10/README.md) |
| Figure 11: fault-tolerance resource cost | `bash scripts/plot.sh figure11 --input data/figure11.csv` | [Figure 11](../scripts/figure11/README.md) |
| Figure 12: compute-node overhead | `bash scripts/plot.sh figure12 --input data/figure12.csv` | [Figure 12](../scripts/figure12/README.md) |
| Figure 13: recovery | `bash scripts/plot.sh figure13 --recovery-csv data/figure13-recovery.csv --throughput-csv data/figure13-throughput.csv` | [Figure 13](../scripts/figure13/README.md) |

Each linked guide describes the input columns, units and output files.
Figure outputs are stored under `results/figures/`; preserve the input CSV
and raw logs alongside each experiment.

## Operational notes

- Use unused service ports and separate result directories on authorized
  compute and memory servers. Runs start and stop memory services.
- Reserve enough 2 MiB HugePages for the selected local-memory capacity
  before starting; restore the original reservation afterward.
- Missing dependencies, insufficient HugePages, timeouts and failed
  correctness checks require attention before retrying.
- Failure-injection experiments must target isolated memory services,
  never production endpoints.
