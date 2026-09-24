#!/usr/bin/env python3
"""Render a short application recipe for one system and memory ratio."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


KEY = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s+(\S+)")


def replace(lines: list[str], key: str, value: str, comment: str = "") -> None:
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


def render(template: Path, *, system: str, ratio: int, footprint_bytes: int,
           server_addr: str, server_port: int, ib_device: str) -> str:
    if not 1 <= ratio <= 100 or footprint_bytes <= 0:
        raise ValueError("ratio must be 1..100 and footprint_bytes must be positive")
    if system not in {"nonft", "starfish"}:
        raise ValueError(f"unsupported recipe system: {system}")
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    local = footprint_bytes * ratio // 100
    backup = footprint_bytes // 10
    replace(lines, "server_count", "1")
    replace(lines, "server_addr", server_addr)
    replace(lines, "server_port", str(server_port))
    replace(lines, "ib_device_name", ib_device)
    replace(lines, "client_buffer_size", str(local), capacity(local))
    if system == "nonft":
        replace(lines, "ft_method", "none")
        replace(lines, "enable_selective_backup", "0")
        replace(lines, "remote_backup_budget_bytes", "0", "disabled for Non-FT")
    elif system == "starfish":
        resident = ((local * 8 + 5) // 10 if ratio <= 50
                    else max(0, local - backup))
        replace(lines, "local_resident_budget_bytes", str(resident),
                capacity(resident))
        replace(lines, "enable_selective_backup", "1")
        replace(lines, "remote_backup_budget_bytes", str(backup),
                capacity(backup) + ", 10% of application footprint")
    result = "".join(lines)
    if "@" in result:
        raise ValueError("unresolved configuration placeholder")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--system", choices=("nonft", "starfish"), required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--footprint-bytes", type=int, required=True)
    parser.add_argument("--server-addr", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--ib-device", required=True)
    args = parser.parse_args()
    result = render(args.template, system=args.system, ratio=args.ratio,
                    footprint_bytes=args.footprint_bytes,
                    server_addr=args.server_addr, server_port=args.server_port,
                    ib_device=args.ib_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
