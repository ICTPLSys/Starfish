# Configuration layout

Configuration is resolved per run; the files in this directory are templates,
not ready-to-launch configurations for an arbitrary machine.

| Input | Role |
| --- | --- |
| `data/site.json` (account-local, not tracked) | Compute host, memory endpoints, input paths and explicit overrides |
| `configs/machines.json` | Defaults by machine IP: IB device/port, NUMA node and any recorded CPU placement |
| `configs/<app>/<system>.config` | Only application/runtime differences; no endpoint or NIC placeholders |
| `scripts/common/recipe_defaults.py` | Shared selected AE policies, resolved before explicit recipe overrides |
| `scripts/common/render_config.py` | Resolves endpoints, memory ratio and supported feature overrides |
| Run directory's `effective.config` and `endpoints/*/server.config` | Exact generated configuration used by that run |

The eight application directories are `bfs`, `llama`, `mg`, `wordcount`,
`kv-b`, `kv-a`, `kv-s` and `nq`. Each has `nonft`, `starfish`,
`hydra` and `carbink` recipes. `llama/starfish_ec.config` is the separate
fast-check recovery recipe; `wordcount/recomputable_ec.config` is an additional
special-purpose recipe. Neither replaces a default Figure 9 recipe.

The runtime's `include/rdma/config.def` defines fields and defaults, expanded
by macros in `config.hpp` at compile time. `config.hpp` also contains parser
and validation logic; it is not an offline-generated recipe. Rebuild after
changing these definitions and keep validation aligned with new fields.
Figure 12 instrumentation switches are recorded launch-environment settings
(`runtime_metadata` / `runtime_ec_cpu` in the site), not recipe parameters.

## Overrides and budgets

Explicit site device/NUMA values override machine defaults. CPU placement
applies machine defaults first, then site settings. Within each layer the order
is general fields, `placement_by_system`, `placement_by_app`, then
`placement_by_case` (keys such as `nq/starfish`; last wins). Explicit site
settings therefore override even case-specific machine defaults.

Application recipes contain no addresses, ports, server counts or NIC names.
Those come exclusively from the selected site and machine defaults. Local
capacity, backup capacity and Resident budgets are generated for the selected
ratio, not copied into each recipe. Common runtime-default values are omitted.
The shared AE policies retain the existing application's non-default planner,
batch and protocol choices; they do not change runtime `config.def` defaults.
Explicit application differences override those policies, then figure/site
overrides apply. The generated `effective.config` records the resolved policy
and overrides; omitted fields retain their compiled runtime defaults.

Use the common launcher or renderer, not a raw application recipe. Shared
policies are selected for built-in `configs/<app>/<system>.config` paths (plus
the two named special recipes). External `--recipe` files remain explicit,
self-contained configs: copying a compact recipe elsewhere does not copy its
shared policy. Do not edit a generated configuration to change a later run.

For NonFT and Starfish, retained backup is on by default with a budget of
10% of the full application footprint T, not 10% of local cache L. The Resident
budget is 80% of L at ratios up to and including 50%, otherwise L - 10% of T.
Hydra and Carbink disable retained backup and preserve their recipe's fixed
Resident/cache fraction. Explicit supported feature overrides apply afterward.
The `nonft-backup-off` variant forces backup off; it does not by itself mean
Resident is off.

## Validate a deployment separately

New `data/site.json` files created by either data-registration entry point use
`scripts/common/site.eight-server.example.json`: compute56 and seven distinct
physical memory servers (22, 54, 58, 74, 76, 82, 84). Existing site files are
preserved rather than silently migrated.

Machine56 defaults keep 24 application workers and the recipe's original
background worker count. Background workers remain on local-data/NIC NUMA0;
application spillover to NUMA1 remains explicitly allowed:

| Recipe | Application CPUs | Background CPUs |
| --- | --- | --- |
| Six-background-worker recipes | 0–17,24–29 | 18–23 |
| Hydra BFS/LLM | 0–15,24–31 | 16–23 |
| Starfish NQ | 0–14,24–32 | 15–23 |
| Carbink except BFS | 0–12,24–34 | 13–23 |
| Carbink BFS | 0–10,24–36 | 11–23 |

The site keeps the seven-node inventory. Normal Starfish/Hydra runs use six
active endpoints plus endpoint 6 as standby. The current Carbink protocol
selects active inventory indices 0–5 through
`memory_endpoint_indices_by_system`; server84 is reserved, not an implemented
Carbink recovery standby. Plans record both the full inventory and selected
endpoints. Active endpoint numbering follows the selected list's order.
A single-endpoint site remains supported for explicit functional checks.

Use the selected runner's `--dry-run` to inspect effective plans, then the
common launcher's `--check-local` for local prerequisites and CPU/NUMA checks.
A dry run is not evidence of hardware availability, credentials, or a
successful application run. See [MULTI-SERVER.md](../docs/MULTI-SERVER.md)
for host fields and [Figure 12](../scripts/figure12/README.md) for measurement
switches. Changes to site files do not affect already-running processes.
