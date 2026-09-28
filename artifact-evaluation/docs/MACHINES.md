# Server configuration

The eight-server layout uses one compute server and seven separate memory
servers. [site.eight-server.example.json](../scripts/common/site.eight-server.example.json)
selects server 56 for compute and the remaining seven servers for memory.
The machine-readable device defaults are in [machines.json](../configs/machines.json).

| Server | SSH / TCP setup IP | CPU | IB device | IB port | NIC-local memory NUMA |
| --- | --- | --- | --- | --- | --- |
| 22 | 10.208.130.22 | Xeon Gold 6342 | mlx5_1 | 1 | 1 |
| 54 | 10.208.130.54 | Xeon Silver 4316 | mlx5_0 | 1 | 0 |
| 56 (compute) | 10.208.130.56 | Xeon Gold 6342 | mlx5_1 | 1 | 0 |
| 58 | 10.208.130.58 | Xeon Gold 6342 | mlx5_1 | 1 | 0 |
| 74 | 10.208.130.74 | Xeon Gold 6342 | mlx5_1 | 1 | 0 |
| 76 | 10.208.130.76 | Xeon Silver 4316 | mlx5_1 | 1 | 0 |
| 82 | 10.208.130.82 | Xeon Gold 6342 | mlx5_0 | 1 | 1 |
| 84 | 10.208.130.84 | Xeon Silver 4316 | mlx5_0 | 1 | 0 |

Use the listed 100-Gb/s InfiniBand ports.
The default TCP service port is 1893. Separate physical servers can all use
the same port.

For multiple logical memory servers on one physical host, set
`"memory_ip": "10.208.130.76", "memory_server_count": 7` in `data/site.json`.
The services use ports 1893–1899, separate logs and independent processes.
Starfish uses six active EC services plus one standby, each reserving 12 GiB
(84 GiB total). See [the explicit endpoint example](../scripts/common/site.multi-endpoint.example.json)
to assign ports individually.

Run the scripts from the compute server's checkout. The launcher copies its
built memory-server executable to each service's run directory; the memory
hosts do not need an identical project path. Explicit `memory_server_bins`
or `memory_project_root` settings select preinstalled binaries instead.
If the hosts have incompatible system-library versions, the launcher builds
the same server source on the memory host; this requires `g++` and
`libibverbs-dev` there.

The IP is used for SSH and the initial TCP exchange of RDMA connection
information. Bulk memory traffic uses the listed IB device. An IPoIB network
interface may have no IP or appear DOWN while its underlying verbs port is
ACTIVE; the launcher checks the verbs port, not just the network interface.
On server 74, mlx5_0 is an Ethernet port and must not be confused with the
InfiniBand port mlx5_1.

The launcher binds compute memory with numactl --membind and each memory
service's memory/CPU node with numactl --membind/--cpunodebind. It checks
the configured device, ACTIVE InfiniBand port at 100 Gb/s or above, NUMA node
and binding support before starting services.
