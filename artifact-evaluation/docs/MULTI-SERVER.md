# Compute and memory server configuration

Run the launcher on the compute server. It reads data/site.json and starts
memory services over SSH. The [server table](MACHINES.md) lists the IP, IB device
and NIC-local NUMA node for each provided machine.

## Eight physical servers

From artifact-evaluation/:

~~~bash
mkdir -p data
cp scripts/common/site.eight-server.example.json data/site.json
~~~

The file contains:

~~~json
{
  "name": "eight-server",
  "compute_ip": "10.208.130.56",
  "memory_endpoints": [
    {"ip": "10.208.130.22"},
    {"ip": "10.208.130.54"},
    {"ip": "10.208.130.58"},
    {"ip": "10.208.130.74"},
    {"ip": "10.208.130.76"},
    {"ip": "10.208.130.82"},
    {"ip": "10.208.130.84"}
  ]
}
~~~

Execute the experiment on server 56. Each memory endpoint is one separate
physical server; the fast check uses six active EC endpoints and endpoint 6
(server 84) as standby. The recovery case terminates only the owned service
process on endpoint 0 (server 22), not the machine or unrelated processes.
Arrange access to all machines before running.

For a two-machine check, use this instead:

~~~json
{
  "name": "two-server",
  "compute_ip": "10.208.130.56",
  "memory_ip": "10.208.130.76"
}
~~~

For fast check, a single memory endpoint expands into seven processes on
that host, on ports 1893 through 1899. This is a service-process check, not
seven independent physical failure domains.

## Defaults and overrides

| Setting | Default | Override |
| --- | --- | --- |
| Compute server | Machine running the script; compute_ip checks its identity | compute_ip |
| SSH target / TCP setup address | The endpoint IP | memory_host / memory_addr |
| TCP service port | 1893 | server_port, globally or per endpoint |
| IB port | 1 | ib_port / memory_ib_port |
| Compute IB device / memory NUMA | machines.json for compute_ip; otherwise mlx5_1 / 0 | ib_device / numa_node |
| Memory IB device / memory NUMA | machines.json for endpoint IP; otherwise mlx5_1 / 0 | memory_ib_device / memory_numa_node |
| Memory-side repository root | Same absolute project root as the compute checkout | memory_project_root |
| Memory service binary | Root + artifact-evaluation/build/SYSTEM/server | memory_server_bins |
| Prepared inputs | /data/starfish-ae/ | data_dir or individual inputs paths |

SSH uses the current account and its normal SSH configuration. memory_host
may be a plain IP, an SSH alias, or user@IP when a different login is needed.
memory_addr is the TCP setup address, not a second mandatory RDMA IP.
For an SSH alias, provide memory_addr explicitly.

The memory-side checkout and built server binaries must already exist.
If its repository location differs, set memory_project_root once, globally
or in an endpoint; individual binary paths are not otherwise needed.
Use memory_server_bins or inputs to override individual paths.

Data paths are resolved automatically:

~~~text
/data/starfish-ae/llama/llama2_7b_chat.bin
/data/starfish-ae/llama/tokenizer.bin
/data/starfish-ae/bfs/graph.0 ... graph.31
/data/starfish-ae/wordcount/enwiki-small-8g.txt
~~~

The launcher does not require an inputs section for this layout.
use_prepared_data.sh provides a separate prepared-data validation step.

## How configuration reaches a run

The files have separate roles:

- data/site.json selects the compute and memory servers and any overrides.
- configs/machines.json supplies each IP's default IB device, IB port and
  NUMA node. Explicit values in data/site.json take precedence.
- configs/APP/SYSTEM.config supplies the application's runtime settings.
- MACHINES.md describes the server layout; editing the Markdown table does
  not change execution.

Both scripts/run_fast_check.sh and scripts/figure9/run.sh read data/site.json
by default. Their common launcher, scripts/common/run_case.py, resolves the
site through site_defaults.py, then passes the application template and
resolved addresses/devices to render_config.py.

The selected IB device and port are written into the generated runtime
configuration. NUMA placement is applied to the launch command: the compute
process uses numactl --membind=N; each memory service also uses --cpunodebind=N.
The selected model/tokenizer paths become application command arguments.

TCP server_port defaults to 1893 and is overridden in data/site.json.
In machines.json, the CPU, hostname, link-speed and verification-date fields
describe the hardware; they do not select application worker counts or CPU
affinity.

Edit data/site.json for the next invocation. Editing an example JSON does
not update an existing site file, and editing configuration does not change
an already-running process. Per-run files retain the resolved configuration
used for that run.

## Generated runtime files

The launcher writes effective.config for the compute process and a
server.config for each memory endpoint under the run's endpoints/ directory.
It transfers each server config, starts the selected binary, runs the workload
and collects the logs. Do not edit these generated files.

Startup checks selected devices and NUMA nodes and rejects occupied service
ports. Cleanup verifies the recorded PID, process start time, executable,
account, working directory and command before signaling an owned process.
Each run retains its resolved configuration and per-endpoint records.

Use --dry-run to inspect the plan without contacting hosts. --check-local
checks compute-side inputs, binaries, device, NUMA binding and HugePages
without starting memory services.
