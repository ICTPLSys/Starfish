"""Carbink's client and legacy memory-service protocol adapter."""

from pathlib import Path
import re

SERVER_BINARY = Path("legacy/carbink_server/carbink_server")
SERVER_KEYS = (
    "server_count", "server_addr", "server_port", "server_addrs", "server_ports",
    "server_buffer_size", "client_buffer_size", "cq_entries", "qp_recv_cap",
    "qp_send_cap", "qp_count", "evict_batch_size", "max_thread_cnt",
    "evacuate_thread_cnt", "compaction_worker_count", "enable_eager_evict",
    "exclusive_cache", "remote_mapping",
)


def endpoint_ports(specs):
    """Give the legacy port-indexed protocol an unambiguous endpoint list."""
    result = [dict(spec) for spec in specs]
    if not result:
        raise ValueError("Carbink requires memory endpoints")
    requested = [int(spec["server_port"]) for spec in result]
    # The legacy parser identifies itself by SERVER_PORT, not (address, port).
    # Preserve distinct explicit ports; derive a sequence for repeated defaults.
    derived = len(set(requested)) != len(requested)
    ports = [requested[0] + i for i in range(len(result))] if derived else requested
    occupied = {}
    for spec, original, port in zip(result, requested, ports):
        if not 1 <= port <= 55535:
            raise ValueError("Carbink service port must leave room for its +10000 peer port")
        peer = port + 10000
        used = occupied.setdefault(spec["memory_addr"], set())
        if port in used or peer in used:
            raise ValueError("Carbink service/peer port collision on " + spec["memory_addr"])
        used.update((port, peer))
        spec.update(server_port=port, peer_server_port=peer,
                    requested_server_port=original,
                    port_reason="logical_index_derived" if derived else "site_requested")
    return result


def memory_lower_bound(effective, workers, *, endpoint_count=None, figure13=False):
    """Known fixed reservations, excluding verbs and other process memory."""
    values = {}
    for line in effective.splitlines():
        fields = line.split()
        if len(fields) >= 2 and not fields[0].startswith("#"):
            values[fields[0]] = fields[1]
    eager = values.get("enable_eager_evict", "1") in ("1", "true")
    clients = workers["rdma_clients"] if eager else workers["app_workers"]
    qp_count = int(values.get("qp_count", "1"))
    receive_cap = int(values.get("qp_recv_cap", "128"))
    send_cap = int(values.get("qp_send_cap", "128"))
    data_pool = int(values["server_buffer_size"])
    if figure13 and endpoint_count not in (7, 8):
        raise ValueError("Figure13 Carbink reservation needs seven or eight endpoints")
    if min(qp_count, receive_cap, send_cap, data_pool) <= 0:
        raise ValueError("Carbink memory reservation requires positive capacities")
    data_qps = (clients + 1) * qp_count
    depth = min(1024, receive_cap, send_cap)
    # x86-64 sizeof values from the shipped legacy server's nested structures.
    # init_rpc_resources() allocates two batch buffers for every receive slot.
    recv_slot, send_slot, batch_buffer = 8448, 8384, 197376
    rpc = data_qps * depth * (recv_slot + send_slot)
    batch = data_qps * depth * 2 * batch_buffer
    endpoints = int(values["server_count"])
    # The legacy vectors reserve slots for every endpoint, including self.
    # These two server-side counts match server_config() below.
    rpc_workers, payload_qps_per_peer = 2, 2
    payload_queues = endpoints * payload_qps_per_peer
    peer_depth = 4096
    lane_depth = max(1, (min(send_cap, receive_cap, peer_depth)
                         + rpc_workers - 1) // rpc_workers)
    lane_pending_depth = max(16384, data_qps * send_cap)
    batch_send_depth = max(128, min(send_cap, 1024))
    batch_pending_depth = min(512, max(128, batch_send_depth * 2))
    ack_depth = max(64, min(send_cap, 1024))
    peer = {
        "recv_bytes": endpoints * receive_cap * 65792,
        "payload_recv_bytes": payload_queues * receive_cap * 65792,
        "send_bytes": endpoints * peer_depth * 8384,
        "free_index_bytes": endpoints * peer_depth * 2,
        "lane_send_bytes": endpoints * rpc_workers * lane_depth * 8384,
        "batch_send_bytes": payload_queues * batch_send_depth * 65856,
        "ack_send_bytes": endpoints * ack_depth * 8384,
        "lane_pending_bytes": endpoints * rpc_workers * lane_pending_depth * 8320,
        "batch_pending_bytes": payload_queues * batch_pending_depth * 65792,
    }
    worker_storage = rpc_workers * 64 * 131520
    return {
        "schema_version": 1, "kind": "lower_bound_not_total",
        "abi": "x86_64", "source": "legacy/carbink_server/include/rdma/server.hpp",
        "source_functions": ["init_rpc_resources", "init_peer_rpc_resources",
                             "init_rpc_workers"],
        "data_qps": data_qps, "rpc_queue_depth": depth,
        "rpc_recv_slot_bytes": recv_slot, "rpc_send_slot_bytes": send_slot,
        "batch_message_buffer_bytes": batch_buffer,
        "data_pool_bytes": data_pool, "rpc_buffer_bytes": rpc,
        "batch_buffer_bytes": batch,
        "peer_buffer_bytes": peer,
        "worker_pair_storage_bytes": worker_storage,
        "peer_counts": {
            "endpoints": endpoints, "rpc_workers": rpc_workers,
            "payload_qps_per_peer": payload_qps_per_peer,
            "rpc_send_depth": peer_depth, "lane_depth": lane_depth,
            "lane_pending_depth": lane_pending_depth,
            "batch_send_depth": batch_send_depth,
            "batch_pending_depth": batch_pending_depth, "ack_depth": ack_depth,
        },
        "minimum_bytes_per_service": (
            data_pool + rpc + batch + sum(peer.values()) + worker_storage),
        "excluded": ["verbs QP/CQ resources", "small containers and worker queues",
                     "other process mappings and stacks", "other host memory"],
    }


def server_config(effective, spec, site, *, figure13=False):
    values = {}
    for line in effective.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in SERVER_KEYS:
            values[fields[0]] = fields[1]
    count = int(values["server_count"])
    capacity = int(values["server_buffer_size"])
    if figure13:
        if count not in (7, 8):
            raise ValueError("Figure13 Carbink requires seven or eight endpoints")
    elif count != 6:
        raise ValueError("Carbink requires exactly six endpoints")
    if capacity <= 0 or capacity % (512 * 1024):
        raise ValueError("Carbink requires 512KiB-aligned per-endpoint capacity")
    # The frozen server divides TOTAL capacity before SERVER_PORT selects its
    # index. Retain all peers for its server-to-server compaction connections.
    values["server_buffer_size"] = str(capacity * count)
    values["ft_method"] = "ec_span"
    values["rdma_device_name"] = spec["memory_ib_device"]
    # The frozen parser reads uint8_t as a character. Its numeric default is
    # port 1; explicitly emitting "1" would instead select ASCII port 49.
    if spec["memory_ib_port"] != 1:
        raise ValueError("Carbink's legacy memory server supports only its default IB port 1")
    values["server_cq_count"] = "4"
    values["rpc_worker_count"] = "2"
    pin = site.get("carbink_server_pin_cores")
    if pin is not None:
        rows = pin.split(",") if isinstance(pin, str) else []
        valid_pin = len(rows) == count and all(
            re.fullmatch(r"[0-9]+(?::[0-9]+){3}", row) for row in rows)
        if not valid_pin:
            raise ValueError(
                "carbink_server_pin_cores needs one four-CPU row per endpoint")
        else:
            values["server_pin_cores"] = pin
    return "".join(f"{key} {value}\n" for key, value in values.items())


def validate_log(log, workers, *, expected_resident_bytes=0):
    required = (
        r"^ft_method: carbink$",
        r"^carbink.layout span_bytes=8192 data_spans=4 parity_spans=2 evict=full_stripe compaction=background$",
        r"^carbink.policy backup=0 behavior_groups=0 resident_budget_bytes="
        + str(expected_resident_bytes)
        + r" codec=isa-l rs=4\+2 encode_tables=process_once$",
    )
    if any(not re.search(pattern, log, re.MULTILINE) for pattern in required):
        raise ValueError("Carbink full-stripe EC and backup-free policy were not active")
    # Worker startup records can interrupt the multi-insertion config line.
    # Remove only complete known records, then validate the original fields.
    worker_log = re.sub(
        r"worker_role role=(?:app|background) index=\d+ client_id=\d+ "
        r"tid=\d+ cpu=\d+ allowed=\S+ allowed_count=\d+[ \t]*\r?\n",
        "", log)
    # An intact record can follow another thread's unfinished prefix.
    # Require exactly one complete record and retain exact count checks.
    worker_records = list(re.finditer(
        r"\bcarbink\.compaction workers=(\d+) scanner=1 [^\n]* os_workers=(\d+) rdma_clients=(\d+)$",
        worker_log, re.MULTILINE))
    worker = worker_records[0] if len(worker_records) == 1 else None
    expected = (workers["background_compaction_workers"],
                workers["app_workers"] + workers["background_workers"],
                workers["rdma_clients"])
    if worker is None or tuple(map(int, worker.groups())) != expected:
        raise ValueError("Carbink compaction worker/client counts do not match the recipe")
    compaction = re.search(r"^carbink.compaction_total .* queue_remaining=(\d+)$", log, re.MULTILINE)
    if compaction is None or int(compaction[1]) != 0:
        raise ValueError("Carbink compaction queue was not drained")
    evict = re.search(
        r"^carbink.evict_total .* ring_in_use=(\d+) fences_in_use=(\d+)$",
        log, re.MULTILINE)
    if evict is None or tuple(map(int, evict.groups())) != (0, 0):
        raise ValueError("Carbink eviction ring or parity fences were not drained")
    pages = re.search(r"^carbink.pages created=(\d+) released=(\d+) ", log, re.MULTILINE)
    if pages is None or pages[1] != pages[2]:
        raise ValueError("Carbink page ownership was not released")
    if not re.search(r"^exact used bytes: 0\s*$", log, re.MULTILINE):
        raise ValueError("Carbink exact remote allocation count did not return to zero")
    return {"full_stripe_ec": True, "compaction_workers": expected[0],
            "os_workers": expected[1], "rdma_clients": expected[2],
            "compaction_queue_remaining": 0, "exact_used_bytes": 0,
            "ring_in_use": 0, "fences_in_use": 0, "pages_released": int(pages[2])}
