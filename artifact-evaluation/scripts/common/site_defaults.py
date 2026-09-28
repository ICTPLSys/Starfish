"""Resolve site-local paths and per-machine defaults without contacting hosts."""

import copy
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re

from endpoint import REMOTE_PATH


SYSTEMS = ("nonft", "starfish", "hydra", "carbink")
INPUTS = {
    "llama": "llama/llama2_7b_chat.bin",
    "llama_tokenizer": "llama/tokenizer.bin",
    "bfs": "bfs/graph",
    "wordcount": "wordcount/enwiki-small-8g.txt",
    "nq": "bfs/graph",
}
PLACEMENT_KEYS = frozenset((
    "fibre_workers",
    "fibre_cpu_set",
    "background_cpu_base",
    "allow_cross_numa_app",
))
_KNOWN_APPS = frozenset(INPUTS) | frozenset(("mg", "kv-a", "kv-b", "kv-s"))


def machine_defaults(ae_root, address):
    inventory = ae_root / "configs/machines.json"
    if not inventory.is_file():
        return {}
    records = json.loads(inventory.read_text())["servers"]
    return next((item for item in records if item["ip"] == address), {})


def address_from_host(host):
    value = host.rsplit("@", 1)[-1]
    try:
        return str(ip_address(value))
    except ValueError as exc:
        raise ValueError("use an IP for memory_ip, or supply memory_addr "
                         "when memory_host is an SSH alias") from exc


def validate_numa(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative NUMA node number")
    return value


def validate_device(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("IB device must be a device name, for example mlx5_1")
    return value


def validate_ib_port(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
        raise ValueError("IB port must be an integer in 1..255")
    return value


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _placement_override(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - PLACEMENT_KEYS)
    if unknown:
        raise ValueError(
            f"{label} contains unsupported placement field(s): "
            + ", ".join(unknown))
    return copy.deepcopy(value)


def _placement_named_map(value, label, selector, allowed_names):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    selected = {}
    for name, override in value.items():
        if name not in allowed_names:
            raise ValueError(f"{label} contains unknown placement selector {name!r}")
        layer = _placement_override(override, f"{label}.{name}")
        if name == selector:
            selected.update(layer)
    return selected


def _apply_placement(site, compute, *, system, app):
    placement = {}
    # All machine defaults precede all explicit site settings. Within either
    # layer, the more specific selector wins; preserve existing site precedence.
    cases = {f"{name}/{runtime}" for name in _KNOWN_APPS for runtime in SYSTEMS}
    case = f"{app}/{system}" if app is not None and system is not None else None
    for layer in (compute, site):
        placement.update({
            key: copy.deepcopy(layer[key])
            for key in PLACEMENT_KEYS
            if key in layer
        })
        for field, selector, allowed in (
                ("placement_by_system", system, SYSTEMS),
                ("placement_by_app", app, _KNOWN_APPS),
                ("placement_by_case", case, cases)):
            if field in layer:
                placement.update(_placement_named_map(
                    layer[field], field, selector, allowed))
    site.update(placement)


def _validate_placement(site):
    if "fibre_workers" in site:
        _positive_int(site["fibre_workers"], "fibre_workers")
    if "background_cpu_base" in site:
        base = site["background_cpu_base"]
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise ValueError("background_cpu_base must be a nonnegative integer")
    if "allow_cross_numa_app" in site and not isinstance(
            site["allow_cross_numa_app"], bool):
        raise ValueError("allow_cross_numa_app must be a boolean")


def resolve(raw, ae_root, *, system=None, app=None):
    if not isinstance(raw, dict):
        raise ValueError("site configuration must be an object")
    if system is not None and system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    if app is not None and app not in _KNOWN_APPS:
        raise ValueError(f"unknown application {app!r}")
    site = copy.deepcopy(raw)
    compute = machine_defaults(ae_root, site.get("compute_ip"))
    _apply_placement(site, compute, system=system, app=app)
    site.setdefault("name", "ae-site")
    site.setdefault("server_port", 1893)
    site.setdefault("ib_device", compute.get("ib_device", "mlx5_1"))
    site.setdefault("ib_port", compute.get("ib_port", 1))
    site.setdefault("numa_node", compute.get("numa_node", 0))
    _validate_placement(site)
    validate_device(site["ib_device"])
    validate_ib_port(site["ib_port"])
    validate_numa(site["numa_node"], "numa_node")
    if "compute_ip" in site:
        site["compute_ip"] = str(ip_address(site["compute_ip"]))
    site.setdefault("remote_run_root", "/tmp/starfish-ae-runs")
    root = site.setdefault("memory_project_root", str(ae_root.parent))
    if not isinstance(root, str) or not REMOTE_PATH.fullmatch(root) or ".." in Path(root).parts:
        raise ValueError("memory_project_root must be an absolute project root")
    directory = site.get("data_dir", os.environ.get("AE_PREPARED_DATA_DIR", "/data/starfish-ae"))
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise ValueError("data_dir must be an absolute directory")
    site["data_dir"] = directory
    supplied = site.get("inputs", {})
    if not isinstance(supplied, dict):
        raise ValueError("inputs must be an object")
    site["inputs"] = {key: str(Path(directory) / relative) for key, relative in INPUTS.items()}
    site["inputs"].update(supplied)
    if any(not isinstance(path, str) or not Path(path).is_absolute()
           for path in site["inputs"].values()):
        raise ValueError("input overrides must be absolute file paths")

    def endpoint(record):
        result = copy.deepcopy(record)
        if not isinstance(result, dict):
            raise ValueError("each memory endpoint must be an object")
        result.setdefault("stage_memory_server", site.get("stage_memory_server",
                          not any(key in raw or key in record
                                  for key in ("memory_project_root", "memory_server_bins"))))
        if not isinstance(result["stage_memory_server"], bool):
            raise ValueError("stage_memory_server must be a boolean")
        ip = result.pop("ip", result.pop("memory_ip", None))
        if ip is not None:
            ip = str(ip_address(ip))
            result.setdefault("memory_host", ip)
            result.setdefault("memory_addr", ip)
        host = result.get("memory_host")
        if not isinstance(host, str):
            raise ValueError("set memory_ip, or memory_host and memory_addr")
        if "memory_addr" not in result:
            result["memory_addr"] = address_from_host(host)
        known = machine_defaults(ae_root, result["memory_addr"])
        result.setdefault("server_port", site["server_port"])
        result.setdefault("memory_ib_device",
                          site.get("memory_ib_device", known.get("ib_device", "mlx5_1")))
        result.setdefault("memory_numa_node",
                          site.get("memory_numa_node", known.get("numa_node", 0)))
        result.setdefault("memory_ib_port", site.get("memory_ib_port", known.get("ib_port", 1)))
        # Normalized legacy sites may contain explicit null fallback values.
        if result["memory_ib_device"] is None:
            result["memory_ib_device"] = site.get("memory_ib_device", known.get("ib_device", "mlx5_1"))
        validate_device(result["memory_ib_device"])
        validate_numa(result["memory_numa_node"], "memory_numa_node")
        validate_ib_port(result["memory_ib_port"])
        project = result.get("memory_project_root", root)
        if (not isinstance(project, str) or not REMOTE_PATH.fullmatch(project)
                or ".." in Path(project).parts):
            raise ValueError("endpoint memory_project_root must be an absolute project root")
        bins = {system: str(Path(project) / "artifact-evaluation/build" / system / "server")
                for system in SYSTEMS}
        for overrides in (site.get("memory_server_bins", {}), result.get("memory_server_bins", {})):
            if not isinstance(overrides, dict):
                raise ValueError("memory_server_bins must be an object")
            bins.update(overrides)
        result["memory_server_bins"] = bins
        return result

    if "memory_endpoints" in site:
        if not isinstance(site["memory_endpoints"], list) or not site["memory_endpoints"]:
            raise ValueError("memory_endpoints must be a nonempty list")
        site["memory_endpoints"] = [endpoint(item) for item in site["memory_endpoints"]]
    else:
        keys = ("memory_host", "memory_addr", "memory_ip", "server_port",
                "memory_server_bins", "memory_ib_device", "memory_numa_node", "memory_ib_port",
                "memory_ld_library_path", "stage_memory_server")
        site.update(endpoint({key: site[key] for key in keys if key in site}))
        count = site.get("memory_server_count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
            raise ValueError("memory_server_count must be an integer in 1..64")
        if count > 1:
            site["memory_endpoints"] = []
            for index in range(count):
                item = {key: copy.deepcopy(site[key]) for key in keys if key in site}
                item["server_port"] = site["server_port"] + index
                site["memory_endpoints"].append(endpoint(item))
            site.pop("memory_server_count", None)
    if "memory_server_count" in raw and "memory_endpoints" in raw:
        raise ValueError("use memory_server_count or memory_endpoints, not both")
    return site
