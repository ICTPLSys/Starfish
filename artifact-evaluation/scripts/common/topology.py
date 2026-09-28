"""Read-only checks and launch prefixes for explicit NIC/NUMA placement."""

import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

from site_defaults import validate_device, validate_numa, validate_ib_port


# Hydra/Starfish config.def defaults. These are used only when an effective
# recipe omits a key whose runtime config.def supplies the default.
BACKGROUND_MARK_WORKERS = 2
BACKGROUND_WORKERS = 6
BACKGROUND_EVICT_WORKERS = BACKGROUND_WORKERS - BACKGROUND_MARK_WORKERS
DEFAULT_APP_WORKERS = 24
MAX_APP_WORKERS = 24
_CPU_TOKEN = re.compile(r"[0-9]+(?:-[0-9]+)?$")
_RUNTIME_WORKER_KEYS = (
    "max_thread_cnt",
    "evacuate_thread_cnt",
    "mark_thread_cnt",
)
_RUNTIME_CONFIG_KEYS = _RUNTIME_WORKER_KEYS + ("optimized_evacuator",)


def _parse_cpu_list(value, label, *, expected=None):
    """Expand a strict Linux cpulist and reject duplicates or negatives."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an explicit nonempty CPU list")
    cpus = []
    seen = set()
    for token in value.split(","):
        if not _CPU_TOKEN.fullmatch(token):
            raise ValueError(f"{label} contains malformed CPU token {token!r}")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"{label} contains descending range {token!r}")
            values = range(start, end + 1)
        else:
            values = (int(token),)
        for cpu in values:
            if cpu in seen:
                raise ValueError(f"{label} contains duplicate CPU {cpu}")
            seen.add(cpu)
            cpus.append(cpu)
            if expected is not None and len(cpus) > expected:
                raise ValueError(f"{label} must contain exactly {expected} CPUs")
    if expected is not None and len(cpus) != expected:
        raise ValueError(f"{label} must contain exactly {expected} CPUs; found {len(cpus)}")
    return cpus


def parse_lscpu_records(output):
    """Parse lscpu -p=CPU,CORE,SOCKET,NODE output for test injection."""
    records = []
    for line_number, line in enumerate(output.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"malformed lscpu record on line {line_number}")
        try:
            cpu, core, socket, node = (int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"malformed lscpu record on line {line_number}") from exc
        records.append({"cpu": cpu, "core": core, "socket": socket, "node": node})
    if not records:
        raise ValueError("lscpu returned no online CPU records")
    return records


def _read_cpu_records():
    try:
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("unable to query online CPU topology with lscpu") from exc
    return parse_lscpu_records(output)


def _normalise_cpu_records(records):
    if isinstance(records, str):
        records = parse_lscpu_records(records)
    elif isinstance(records, Mapping):
        records = records.values()
    result = {}
    for record in records:
        if isinstance(record, Mapping):
            try:
                cpu = int(record["cpu"])
                core = int(record["core"])
                socket = int(record["socket"])
                node = int(record["node"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "CPU topology records require cpu/core/socket/node") from exc
        else:
            try:
                cpu, core, socket, node = (int(value) for value in record)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "CPU topology records require four integer fields") from exc
        if cpu < 0:
            raise ValueError(f"CPU topology contains negative CPU {cpu}")
        if cpu in result:
            raise ValueError(f"CPU topology contains duplicate CPU {cpu}")
        result[cpu] = {"cpu": cpu, "core": core, "socket": socket, "node": node}
    if not result:
        raise ValueError("CPU topology contains no records")
    return result


def _config_integer(value, label, *, allow_zero=False):
    """Parse a size_t-like worker value without accepting bools or floats."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        number = int(value.strip())
    else:
        raise ValueError(f"{label} must be an integer")
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return number


def _config_bool(value, label):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in (
            "0", "1", "false", "true"):
        return value.strip().lower() in ("1", "true")
    raise ValueError(f"{label} must be a boolean")


def _parse_runtime_config(runtime_config):
    """Return worker fields from effective config text or a mapping.

    The renderer emits "key value" lines, while callers that already parsed
    the recipe may pass a mapping. Unknown runtime options are intentionally
    ignored; worker fields are parsed strictly and duplicate text entries are
    rejected.
    """
    if isinstance(runtime_config, Mapping):
        values = {}
        for key in _RUNTIME_CONFIG_KEYS:
            if key in runtime_config:
                if key == "optimized_evacuator":
                    values[key] = _config_bool(
                        runtime_config[key], f"runtime config {key}")
                else:
                    values[key] = _config_integer(
                        runtime_config[key], f"runtime config {key}",
                        allow_zero=(key == "mark_thread_cnt"))
        return values
    if isinstance(runtime_config, os.PathLike):
        try:
            text = Path(runtime_config).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("unable to read runtime worker configuration") from exc
    elif isinstance(runtime_config, str):
        text = runtime_config
    else:
        raise ValueError("runtime_config must be rendered text or a mapping")

    values = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        fields = content.split()
        key = fields[0]
        if key not in _RUNTIME_CONFIG_KEYS:
            continue
        if len(fields) != 2:
            raise ValueError(
                f"runtime config {key} must have one value on line {line_number}")
        if key in values:
            raise ValueError(f"runtime config contains duplicate {key}")
        if key == "optimized_evacuator":
            values[key] = _config_bool(
                fields[1], f"runtime config {key}")
        else:
            values[key] = _config_integer(
                fields[1], f"runtime config {key}",
                allow_zero=(key == "mark_thread_cnt"))
    return values


def runtime_workers(site, runtime_config=None):
    """Resolve application/background worker roles for an effective recipe.

    With no runtime config this preserves the legacy site API: the site worker
    count is used (default 24) and the six-worker background profile is used.
    With a rendered config, max_thread_cnt is required and must match the
    site fibre_workers (or the legacy site default of 24). evacuate_thread_cnt
    and mark_thread_cnt
    follow the Hydra/Starfish config.def defaults when a recipe leaves them
    implicit; the remaining evacuation workers are Evict workers.
    """
    configured = runtime_config is not None
    values = _parse_runtime_config(runtime_config) if configured else {}

    if configured:
        if "max_thread_cnt" not in values:
            raise ValueError("runtime config is missing max_thread_cnt")
        app_workers = values["max_thread_cnt"]
    else:
        app_workers = site.get("fibre_workers", DEFAULT_APP_WORKERS)
        app_workers = _config_integer(app_workers, "fibre_workers")

    if app_workers > MAX_APP_WORKERS:
        raise ValueError(
            f"fibre_workers must be an integer in 1..{MAX_APP_WORKERS}")
    site_workers = _config_integer(
        site.get("fibre_workers", DEFAULT_APP_WORKERS), "fibre_workers")
    if site_workers > MAX_APP_WORKERS:
        raise ValueError(
            f"fibre_workers must be an integer in 1..{MAX_APP_WORKERS}")
    if site_workers != app_workers:
        raise ValueError(
            f"site fibre_workers={site_workers} does not match "
            f"runtime max_thread_cnt={app_workers}")

    evacuate_workers = _config_integer(
        values.get("evacuate_thread_cnt", BACKGROUND_WORKERS),
        "evacuate_thread_cnt")
    mark_workers = _config_integer(
        values.get("mark_thread_cnt", BACKGROUND_MARK_WORKERS),
        "mark_thread_cnt", allow_zero=True)
    if mark_workers > evacuate_workers:
        raise ValueError(
            "mark_thread_cnt must not exceed evacuate_thread_cnt")
    if values.get("optimized_evacuator", True) and not (
            0 < mark_workers < evacuate_workers):
        raise ValueError(
            "optimized_evacuator requires 0 < mark_thread_cnt "
            "< evacuate_thread_cnt")
    evict_workers = evacuate_workers - mark_workers
    return {
        "app_workers": app_workers,
        "background_workers": evacuate_workers,
        "background_mark_workers": mark_workers,
        "background_evict_workers": evict_workers,
    }


def check_cpu_profile(site, runtime_config=None):
    """Validate a CPU profile without probing lscpu or process affinity."""
    workers = runtime_workers(site, runtime_config)
    if "fibre_cpu_set" not in site or site["fibre_cpu_set"] is None:
        raise ValueError("CPU placement requires explicit site field fibre_cpu_set")
    if "background_cpu_base" not in site or site["background_cpu_base"] is None:
        raise ValueError(
            "CPU placement requires explicit site field background_cpu_base")

    app_cpus = _parse_cpu_list(
        site["fibre_cpu_set"], "fibre_cpu_set",
        expected=workers["app_workers"])
    background_base = site["background_cpu_base"]
    if (isinstance(background_base, bool)
            or not isinstance(background_base, int)
            or background_base < 0):
        raise ValueError("background_cpu_base must be a nonnegative integer")
    background_cpus = list(range(
        background_base, background_base + workers["background_workers"]))
    if set(app_cpus).intersection(background_cpus):
        raise ValueError("application and background CPU pools overlap")

    numa_node = validate_numa(site["numa_node"], "numa_node")
    allow_cross_numa = site.get("allow_cross_numa_app", False)
    if not isinstance(allow_cross_numa, bool):
        raise ValueError("allow_cross_numa_app must be a boolean")
    return {
        "app_cpus": tuple(app_cpus),
        "background_cpus": tuple(background_cpus),
        "app_workers": workers["app_workers"],
        "background_workers": workers["background_workers"],
        "background_mark_workers": workers["background_mark_workers"],
        "background_evict_workers": workers["background_evict_workers"],
        "numa_node": numa_node,
        "allow_cross_numa_app": allow_cross_numa,
    }


def check_cpu_placement(site, runtime_config=None, *,
                        cpu_records=None, allowed_cpus=None):
    """Validate explicit application/background CPU pools against this host.

    cpu_records and allowed_cpus are injectable so offline tests can validate
    placement without depending on the host running the test.
    """
    profile = check_cpu_profile(site, runtime_config)
    app_cpus = list(profile["app_cpus"])
    background_cpus = list(profile["background_cpus"])
    numa_node = profile["numa_node"]
    allow_cross_numa = profile["allow_cross_numa_app"]

    records = _normalise_cpu_records(
        _read_cpu_records() if cpu_records is None else cpu_records)
    try:
        allowed = (set(os.sched_getaffinity(0))
                   if allowed_cpus is None else set(allowed_cpus))
    except (AttributeError, OSError) as exc:
        raise ValueError("unable to query this process CPU affinity") from exc

    selected = app_cpus + background_cpus
    for cpu in selected:
        if cpu not in records:
            raise ValueError(
                f"selected CPU {cpu} is not online according to lscpu")
        if cpu not in allowed:
            raise ValueError(f"selected CPU {cpu} is outside sched_getaffinity")

    physical_cores = {}
    for pool_name, pool in (("application", app_cpus),
                            ("background", background_cpus)):
        for cpu in pool:
            record = records[cpu]
            physical = (record["socket"], record["core"])
            previous = physical_cores.get(physical)
            if previous is not None:
                raise ValueError(
                    f"{pool_name} CPU {cpu} shares physical core {physical} "
                    f"with CPU {previous}")
            physical_cores[physical] = cpu
            if pool_name == "background" and record["node"] != numa_node:
                raise ValueError(
                    f"background CPU {cpu} is on NUMA {record['node']}, "
                    f"expected site NUMA {numa_node}")
            if (pool_name == "application" and not allow_cross_numa
                    and record["node"] != numa_node):
                raise ValueError(
                    f"application CPU {cpu} is on NUMA {record['node']}, "
                    f"expected site NUMA {numa_node}; set "
                    "allow_cross_numa_app=true only for an explicit exception")

    return {
        "app_cpus": tuple(app_cpus),
        "background_cpus": tuple(background_cpus),
        "app_workers": profile["app_workers"],
        "background_workers": profile["background_workers"],
        "background_mark_workers": profile["background_mark_workers"],
        "background_evict_workers": profile["background_evict_workers"],
        "numa_node": numa_node,
        "allow_cross_numa_app": allow_cross_numa,
    }


def memory_binding(node, *, server=False):
    validate_numa(node, "numa_node")
    prefix = ["numactl", f"--membind={node}"]
    if server:
        prefix.append(f"--cpunodebind={node}")
    return prefix


def preflight_script(device, node, port=1):
    validate_device(device)
    validate_numa(node, "numa_node")
    validate_ib_port(port)
    directory = shlex.quote("/sys/class/infiniband/" + device)
    return (
        "command -v numactl >/dev/null || { echo 'numactl is required' >&2; exit 2; }; "
        f"d={directory}; "
        f"test -d \"$d\" || {{ echo 'missing IB device {device}' >&2; exit 2; }}; "
        f"test \"$(cut -d: -f1 < \"$d/ports/{port}/state\")\" = 4 || "
        f"{{ echo '{device} port {port} is not ACTIVE' >&2; exit 2; }}; "
        f"test \"$(cat \"$d/ports/{port}/link_layer\")\" = InfiniBand || "
        f"{{ echo '{device} port {port} is not InfiniBand' >&2; exit 2; }}; "
        f"awk '$1 >= 100 {{ok=1}} END {{exit !ok}}' \"$d/ports/{port}/rate\" || "
        f"{{ echo '{device} port {port} is below 100 Gb/s' >&2; exit 2; }}; "
        f"test \"$(cat \"$d/device/numa_node\")\" = {node} || "
        f"{{ echo '{device} is not on configured NUMA {node}' >&2; exit 2; }}; "
        f"test -d /sys/devices/system/node/node{node} || "
        f"{{ echo 'missing NUMA {node}' >&2; exit 2; }}; "
        f"numactl --membind={node} --cpunodebind={node} true; "
        f"printf 'RDMA preflight PASS: device=%s port={port} numa={node}\\n' " + shlex.quote(device)
    )


def check_compute(site, runtime_config=None):
    if site.get("compute_ip"):
        interfaces = json.loads(subprocess.check_output(["ip", "-j", "addr", "show"], text=True))
        addresses = {info["local"] for interface in interfaces
                     for info in interface.get("addr_info", []) if "local" in info}
        if site["compute_ip"] not in addresses:
            raise ValueError(f"run this site on compute_ip={site['compute_ip']}; "
                             "the configured address is not local")
    checked = subprocess.run(
        ["bash", "-ec", preflight_script(site["ib_device"], site["numa_node"], site["ib_port"])],
        text=True, capture_output=True, check=False)
    if checked.returncode:
        raise ValueError(f"compute NIC/NUMA preflight failed: {site['ib_device']} must "
                         f"have ACTIVE >=100-Gb/s InfiniBand port {site['ib_port']} "
                         f"on NUMA {site['numa_node']}, "
                         "and numactl must be available with permission to bind memory. "
                         + checked.stderr.strip())
    print(checked.stdout.strip())
    return check_cpu_placement(site, runtime_config)
