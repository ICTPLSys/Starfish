#!/usr/bin/env python3
"""Render one application recipe, preserving or overriding endpoint lists."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from endpoint import validate_port, validate_render_endpoints


KEY = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s+(\S+)")


def replace(lines: List[str], key: str, value: str, comment: str = "") -> None:
    matches = [i for i, line in enumerate(lines)
               if (match := KEY.match(line)) and match.group(1) == key]
    if len(matches) > 1:
        raise ValueError(f"duplicate configuration key: {key}")
    line = f"{key} {value}" + (f" # {comment}" if comment else "") + "\n"
    if matches:
        lines[matches[0]] = line
    else:
        lines.append(line)


def capacity(value: int) -> str:
    return f"{value / 1e9:.2f} GB ({value / 1024**3:.2f} GiB)"


def _template_values(lines: List[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in lines:
        match = KEY.match(line)
        if not match:
            continue
        key, value = match.groups()
        if key in values:
            raise ValueError(f"duplicate configuration key: {key}")
        values[key] = value
    return values


def _split_list(value: str, label: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{label} must be a non-empty comma-separated list")
    return values


def _template_endpoints(lines: List[str]) -> Optional[List[Dict[str, Any]]]:
    values = _template_values(lines)
    count_text = values.get("server_count")
    if count_text is None:
        if "server_addrs" not in values and "server_ports" not in values:
            return None
        raise ValueError("template has server_addrs/server_ports but no server_count")
    try:
        count = int(count_text)
    except ValueError as exc:
        raise ValueError("template server_count must be an integer") from exc
    if count < 1:
        raise ValueError("template server_count must be positive")
    addrs = values.get("server_addrs")
    ports = values.get("server_ports")
    if addrs is None or ports is None:
        if count == 1 and "server_addr" in values and "server_port" in values:
            return validate_render_endpoints([{
                "server_addr": values["server_addr"],
                "server_port": validate_port(int(values["server_port"]),
                                               label="template server port"),
            }], label="template endpoints")
        if count > 1:
            raise ValueError("multi-endpoint template must define server_addrs and server_ports")
        return None
    addr_list = _split_list(addrs, "template server_addrs")
    port_list = _split_list(ports, "template server_ports")
    if len(addr_list) != count or len(port_list) != count:
        raise ValueError("template server_addrs/server_ports count must match server_count")
    return validate_render_endpoints([
        {"server_addr": address,
         "server_port": validate_port(int(port), label="template server port")}
        for address, port in zip(addr_list, port_list)
    ], label="template endpoints")


def _endpoint_values(records: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    addresses = ",".join(str(record["server_addr"]) for record in records)
    ports = ",".join(str(record["server_port"]) for record in records)
    return addresses, ports, str(records[0]["server_addr"])


def render(template: Path, *, system: str, ratio: int, footprint_bytes: int,
           server_addr: Optional[str] = None, server_port: Optional[int] = None,
           ib_device: str, ib_port: Optional[int] = None,
           server_endpoints: Optional[List[Dict[str, Any]]] = None) -> str:
    if not 1 <= ratio <= 100 or footprint_bytes <= 0:
        raise ValueError("ratio must be 1..100 and footprint_bytes must be positive")
    if system not in {"nonft", "starfish", "hydra"}:
        raise ValueError(f"unsupported system: {system}")
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    template_values = _template_values(lines)
    selected_port = (ib_port if ib_port is not None
                     else int(_template_values(lines).get("ib_port", "1")))
    if isinstance(selected_port, bool) or not isinstance(selected_port, int) or not 1 <= selected_port <= 255:
        raise ValueError("IB port must be an integer in 1..255")
    replace(lines, "ib_port", str(selected_port))
    existing = _template_endpoints(lines)
    if server_endpoints is not None:
        endpoints = validate_render_endpoints(server_endpoints)
    elif existing is not None:
        if server_addr is not None or server_port is not None:
            if len(existing) != 1:
                raise ValueError(
                    "multi-endpoint template requires server_endpoints; "
                    "server_addr/server_port cannot collapse it to one endpoint"
                )
            if server_addr is None or server_port is None:
                raise ValueError("server_addr and server_port must be provided together")
            endpoints = validate_render_endpoints([{
                "server_addr": server_addr,
                "server_port": validate_port(server_port),
            }])
        else:
            endpoints = existing
    else:
        if server_addr is None or server_port is None:
            raise ValueError("server_addr and server_port are required for a single-endpoint template")
        endpoints = validate_render_endpoints([{
            "server_addr": server_addr, "server_port": validate_port(server_port)
        }])
    local = footprint_bytes * ratio // 100
    backup = footprint_bytes // 10
    addresses, ports, first_address = _endpoint_values(endpoints)
    replace(lines, "server_count", str(len(endpoints)))
    replace(lines, "server_addr", first_address)
    replace(lines, "server_port", ports.split(",")[0])
    replace(lines, "server_addrs", addresses)
    replace(lines, "server_ports", ports)
    replace(lines, "ib_device_name", ib_device)
    replace(lines, "client_buffer_size", str(local), capacity(local))
    if system in {"nonft", "starfish"}:
        resident = ((local * 8 + 5) // 10 if ratio <= 50 else max(0, local - backup))
        replace(lines, "local_resident_budget_bytes", str(resident), capacity(resident))
    if system == "nonft":
        replace(lines, "ft_method", "none")
        replace(lines, "enable_selective_backup", "0")
        replace(lines, "remote_backup_budget_bytes", "0", "disabled for Non-FT")
    elif system == "starfish":
        replace(lines, "enable_selective_backup", "1")
        replace(lines, "remote_backup_budget_bytes", str(backup),
                capacity(backup) + ", 10% of application footprint")
    elif system == "hydra":
        # Preserve the recipe's fixed Resident/cache fraction. Hydra has no
        # retained backup or behavior-group planner; do not borrow Starfish's
        # footprint-minus-backup rule for ratios above 50 percent.
        template_local = int(template_values.get("client_buffer_size", "0"))
        template_resident = int(template_values.get("local_resident_budget_bytes", "0"))
        if template_local <= 0 or not 0 <= template_resident <= template_local:
            raise ValueError("Hydra recipe requires valid cache and fixed Resident budgets")
        resident = local * template_resident // template_local
        replace(lines, "local_resident_budget_bytes", str(resident), capacity(resident))
        replace(lines, "enable_selective_backup", "0")
        replace(lines, "remote_backup_budget_bytes", "0", "Hydra has no retained backup")
        for key in ("enable_logical_object_profile", "enable_resident_profile_planner",
                    "enable_region_hotness_placement", "enable_region_fetch_hotness_placement",
                    "region_placement_bind_groups"):
            replace(lines, key, "0")
    result = "".join(lines)
    if "@" in result:
        raise ValueError("unresolved configuration placeholder")
    return result


def _parse_endpoint_token(token: str) -> Dict[str, Any]:
    text = token.strip()
    if text.startswith("["):
        close = text.find("]:")
        if close < 0:
            raise ValueError(f"invalid --server-endpoint: {token!r}")
        address, port_text = text[1:close], text[close + 2:]
    else:
        try:
            address, port_text = text.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(f"invalid --server-endpoint: {token!r}") from exc
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"invalid endpoint port: {token!r}") from exc
    return {"server_addr": address, "server_port": port}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--system", choices=("nonft", "starfish", "hydra"), required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--footprint-bytes", type=int, required=True)
    parser.add_argument("--server-addr")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--server-endpoint", action="append", default=[],
                        metavar="ADDR:PORT",
                        help="explicit endpoint; repeat for multi-endpoint recipes")
    parser.add_argument("--ib-device", required=True)
    args = parser.parse_args()
    if args.server_endpoint and (args.server_addr is not None or args.server_port is not None):
        parser.error("use --server-endpoint or --server-addr/--server-port, not both")
    endpoints = None
    if args.server_endpoint:
        endpoints = [_parse_endpoint_token(token) for token in args.server_endpoint]
    result = render(args.template, system=args.system, ratio=args.ratio,
                    footprint_bytes=args.footprint_bytes,
                    server_addr=args.server_addr, server_port=args.server_port,
                    ib_device=args.ib_device, server_endpoints=endpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
