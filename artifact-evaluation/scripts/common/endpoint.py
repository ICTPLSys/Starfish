"""Shared validation helpers for multi-endpoint AE configurations."""

from __future__ import annotations

from ipaddress import ip_address
import re
from typing import Any, List, Mapping, Optional, Tuple, Dict


HOST = re.compile(r"^[A-Za-z0-9_.@-]+$")
REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")


def validate_address(value: Any, *, label: str = "server address") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an IP address")
    value = value.strip()
    try:
        return str(ip_address(value))
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid IP address: {value!r}") from exc


def validate_port(value: Any, *, label: str = "server port") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer TCP port")
    if not 1 <= value <= 65535:
        raise ValueError(f"{label} must be in [1, 65535]")
    return value


def _render_record(raw: Any, index: int, label: str) -> Tuple[Any, Any]:
    if isinstance(raw, Mapping):
        return (raw.get("server_addr", raw.get("memory_addr", raw.get("addr"))),
                raw.get("server_port", raw.get("memory_port", raw.get("port"))))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return raw[0], raw[1]
    raise ValueError(f"{label}[{index}] must be an object or [address, port]")


def validate_render_endpoints(records: Any, *,
                              label: str = "server_endpoints") -> List[Dict[str, Any]]:
    """Validate client-config endpoint records.

    Repeated addresses on different ports are valid; an address/port pair is
    the endpoint identity and must be unique.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError(f"{label} must be a non-empty list")
    result: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(records):
        address, port = _render_record(raw, index, label)
        address = validate_address(address, label=f"{label}[{index}].address")
        port = validate_port(port, label=f"{label}[{index}].port")
        key = (address, port)
        if key in seen:
            raise ValueError(f"{label} contains duplicate address/port pair: {address}:{port}")
        seen.add(key)
        result.append({"server_addr": address, "server_port": port})
    return result


def validate_site_endpoints(records: Any, *, system: Optional[str] = None,
                            dry_run: bool = False) -> List[Dict[str, Any]]:
    """Validate richer site.memory_endpoints records before SSH side effects."""
    if not isinstance(records, list) or not records:
        raise ValueError("memory_endpoints must be a non-empty list")
    result: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(records):
        label = f"memory_endpoints[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be an object")
        host = raw.get("memory_host")
        if (not isinstance(host, str) or host.startswith("-")
                or not HOST.fullmatch(host)):
            raise ValueError(f"{label}.memory_host must be a simple SSH host or user@host")
        address = validate_address(raw.get("memory_addr"), label=f"{label}.memory_addr")
        port = validate_port(raw.get("server_port"), label=f"{label}.server_port")
        key = (address, port)
        if key in seen:
            raise ValueError(
                f"memory_endpoints contains duplicate address/port pair: {address}:{port}"
            )
        seen.add(key)

        bins = raw.get("memory_server_bins")
        if not isinstance(bins, Mapping) or not bins:
            raise ValueError(f"{label}.memory_server_bins must be a non-empty object")
        normalized_bins: Dict[str, str] = {}
        for name, path in bins.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{label}.memory_server_bins has an invalid system name")
            if (not isinstance(path, str) or not REMOTE_PATH.fullmatch(path)
                    or ".." in path.split("/")):
                raise ValueError(
                    f"{label}.memory_server_bins[{name!r}] must be an absolute path "
                    "without spaces or '..'"
                )
            normalized_bins[name] = path
        if system is not None and not dry_run and system not in normalized_bins:
            raise ValueError(f"{label}.memory_server_bins has no binary for {system}")

        memory_ib_device = raw.get("memory_ib_device")
        if memory_ib_device is not None and (
                not isinstance(memory_ib_device, str) or not memory_ib_device.strip()):
            raise ValueError(f"{label}.memory_ib_device must be a non-empty string")
        memory_ld_library_path = raw.get("memory_ld_library_path")
        if memory_ld_library_path is not None and not isinstance(memory_ld_library_path, str):
            raise ValueError(f"{label}.memory_ld_library_path must be a string")
        memory_numa_node = raw.get("memory_numa_node", 0)
        if (isinstance(memory_numa_node, bool) or not isinstance(memory_numa_node, int)
                or memory_numa_node < 0):
            raise ValueError(f"{label}.memory_numa_node must be a nonnegative integer")
        ib_port = raw.get("memory_ib_port", 1)
        if isinstance(ib_port, bool) or not isinstance(ib_port, int) or not 1 <= ib_port <= 255:
            raise ValueError(f"{label}.memory_ib_port must be in 1..255")
        result.append({
            "index": index,
            "memory_host": host,
            "memory_addr": address,
            "server_port": port,
            "memory_server_bins": normalized_bins,
            "memory_ib_device": memory_ib_device,
            "memory_ld_library_path": memory_ld_library_path,
            "memory_numa_node": memory_numa_node,
            "memory_ib_port": ib_port,
            "stage_memory_server": raw.get("stage_memory_server", False),
        })
    return result


def endpoint_name(index: int) -> str:
    if not isinstance(index, int) or index < 0:
        raise ValueError("endpoint index must be a non-negative integer")
    return f"endpoint-{index:02d}"
