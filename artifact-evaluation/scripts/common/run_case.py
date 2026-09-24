#!/usr/bin/env python3
"""Run one two-server AE case and preserve its inputs, logs, and outcome."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from render_config import render
from workloads import BINARY_RELATIVE, FOOTPRINT_BYTES, WORKLOAD_LABEL
from workloads import client_command, client_environment, parse_result

AE_ROOT = Path(__file__).resolve().parents[2]
HOST = re.compile(r"^[A-Za-z0-9_.@-]+$")
REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
SERVER_KEYS = ("server_count", "server_addr", "server_port",
               "server_buffer_size", "ib_device_name")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(AE_ROOT), *args],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def server_config(rendered: str) -> str:
    """The memory service does not accept client-only FT/cache settings."""
    values: dict[str, str] = {}
    for line in rendered.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in SERVER_KEYS:
            values[fields[0]] = fields[1]
    if len(values) != len(SERVER_KEYS):
        raise ValueError("rendered config lacks a required memory-server setting")
    return "".join(f"{key} {values[key]}\n" for key in SERVER_KEYS)


def check_hugepages(local_bytes: int) -> None:
    """The RDMA client mmaps its entire local buffer with 2 MiB HugePages."""
    fields = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(":")
        if key in ("HugePages_Free", "Hugepagesize"):
            fields[key] = int(value.split()[0])
    page_bytes = fields["Hugepagesize"] * 1024
    free_pages = fields["HugePages_Free"]
    required = (local_bytes + page_bytes - 1) // page_bytes
    if free_pages < required:
        raise ValueError(
            f"not enough HugePages for client_buffer_size={local_bytes}: "
            f"need {required} free pages of {page_bytes} bytes, have {free_pages}; "
            "reserve them on the compute server before running"
        )


def site_config(path: Path, *, dry_run: bool) -> dict:
    site = json.loads(path.read_text(encoding="utf-8"))
    required = ("name", "memory_host", "memory_addr", "memory_server_bins",
                "remote_run_root", "server_port", "ib_device", "inputs")
    absent = [key for key in required if not site.get(key)]
    if absent:
        raise ValueError(f"site configuration is missing: {', '.join(absent)}")
    if (not isinstance(site["inputs"], dict)
            or not isinstance(site["memory_server_bins"], dict)
            or not isinstance(site["name"], str)):
        raise ValueError("site name and inputs have invalid types")
    if not HOST.fullmatch(site["memory_host"]):
        raise ValueError("memory_host must be a simple SSH host or user@host")
    for key in ("remote_run_root",):
        if not REMOTE_PATH.fullmatch(site[key]) or ".." in Path(site[key]).parts:
            raise ValueError(f"{key} must be an absolute path without spaces or '..'")
    if not isinstance(site["server_port"], int) or not 1 <= site["server_port"] <= 65535:
        raise ValueError("server_port must be a TCP port")
    if not dry_run:
        ipaddress.ip_address(site["memory_addr"])
        if site["memory_host"].startswith("CHANGE_"):
            raise ValueError("replace the example SSH host before executing")
    return site


def ssh(host: str, script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Quote one complete script for the SSH remote shell. Never interpolate a
    # command supplied by a run-plan CSV or a plot file.
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         "bash -lc " + shlex.quote("set -e; " + script)],
        text=True, capture_output=True, check=False, timeout=30,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"memory host {host}: {detail}")
    return result


def scp(source: str, destination: str) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", source, destination],
                   check=True, timeout=60)


def run(args: argparse.Namespace) -> int:
    if args.app not in FOOTPRINT_BYTES:
        raise ValueError(f"unsupported application: {args.app}")
    if args.system not in ("nonft", "starfish"):
        raise ValueError(f"unsupported system: {args.system}")
    if not 1 <= args.ratio <= 100 or args.timeout <= 0:
        raise ValueError("ratio must be 1..100 and timeout must be positive")
    site = site_config(args.site, dry_run=args.dry_run)
    server_bin = site["memory_server_bins"].get(args.system)
    if not server_bin and not args.dry_run:
        raise ValueError(f"memory server binary not configured for {args.system}")
    if server_bin and (not REMOTE_PATH.fullmatch(server_bin)
                       or ".." in Path(server_bin).parts):
        raise ValueError("memory server binary must be an absolute path without spaces or '..'")
    config = AE_ROOT / "configs" / args.app / f"{args.system}.config"
    build_dir = AE_ROOT / "build" / args.system
    client_bin = build_dir / BINARY_RELATIVE[args.app]
    input_path = Path(site["inputs"].get(args.app, ""))
    tokenizer = (Path(site["inputs"].get("llama_tokenizer", ""))
                 if args.app == "llama" else None)
    command, stdin_path = client_command(
        args.app, client_bin, args.out / "effective.config", input_path, AE_ROOT,
        tokenizer)
    batch_id = args.out.parent.parent.name if args.out.parent.name == "runs" else "single"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id + args.out.name):
        raise ValueError("batch and run directory names must be simple identifiers")
    remote_dir = (site["remote_run_root"].rstrip("/") + "/"
                  + batch_id + "/" + args.out.name)
    if not REMOTE_PATH.fullmatch(remote_dir) or ".." in Path(remote_dir).parts:
        raise ValueError("invalid remote run directory")
    plan = {
        "app": args.app, "system": args.system, "ratio": args.ratio,
        "run_id": args.out.name, "site": site["name"],
        "recipe": str(config), "client_bin": str(client_bin),
        "memory_host": site["memory_host"], "memory_addr": site["memory_addr"],
        "memory_server_bin": server_bin,
        "remote_run_dir": remote_dir, "input": str(input_path),
        "client_command": command, "client_env": client_environment(args.app, site),
    }
    if args.dry_run:
        plan["available"] = {
            "recipe": config.is_file(),
            "runtime": (AE_ROOT / "runtime" / args.system / "CMakeLists.txt").is_file(),
            "client_binary": client_bin.is_file(),
            "memory_server_path": bool(server_bin),
        }
        print(json.dumps(plan, sort_keys=True))
        return 0

    if args.out.exists():
        raise ValueError(f"refusing to overwrite existing run directory: {args.out}")
    if not config.is_file() or not (AE_ROOT / "runtime" / args.system / "CMakeLists.txt").is_file():
        raise ValueError(f"runtime or recipe missing for {args.app}/{args.system}")
    if not os.access(client_bin, os.X_OK):
        raise ValueError(f"client binary missing or not executable: {client_bin}")
    # Friendster is passed as a prefix; the BFS loader opens .0 through .31.
    input_files = ([Path(f"{input_path}.{i}") for i in range(32)]
                   if args.app == "bfs" else [input_path])
    missing = [path for path in input_files if not path.is_file()]
    if missing:
        raise ValueError(f"workload input missing: {missing[0]}")
    input_stats = [path.stat() for path in input_files]
    if any(stat.st_size == 0 for stat in input_stats):
        raise ValueError("workload input contains an empty file")
    if stdin_path is not None and not stdin_path.is_file():
        raise ValueError(f"prompt file not found: {stdin_path}")
    if tokenizer is not None and not tokenizer.is_file():
        raise ValueError(f"LLaMA tokenizer not found: {tokenizer}")
    effective = render(
        config, system=args.system, ratio=args.ratio,
        footprint_bytes=FOOTPRINT_BYTES[args.app],
        server_addr=site["memory_addr"], server_port=site["server_port"],
        ib_device=site["ib_device"])
    check_hugepages(FOOTPRINT_BYTES[args.app] * args.ratio // 100)
    if args.check_local:
        print(f"local preflight PASS: {args.app}/{args.system}")
        return 0
    server_effective = server_config(render(
        config, system=args.system, ratio=args.ratio,
        footprint_bytes=FOOTPRINT_BYTES[args.app],
        server_addr=site["memory_addr"], server_port=site["server_port"],
        ib_device=site.get("memory_ib_device", site["ib_device"])))
    host = site["memory_host"]
    quoted_dir = shlex.quote(remote_dir)
    quoted_bin = shlex.quote(server_bin)
    ssh(host, "test -x " + quoted_bin + "; test ! -e " + quoted_dir)
    server_sha256 = ssh(host, "sha256sum " + quoted_bin).stdout.split()[0]
    args.out.mkdir(parents=True)
    (args.out / "effective.config").write_text(effective, encoding="utf-8")
    (args.out / "server.config").write_text(server_effective, encoding="utf-8")
    (args.out / "client.command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "start_time": now(), "status": "running",
        "plan": plan,
        "client_sha256": sha256(client_bin),
        "server_sha256": server_sha256,
        "effective_config_sha256": sha256(args.out / "effective.config"),
        "server_config_sha256": sha256(args.out / "server.config"),
        "input_bytes": sum(stat.st_size for stat in input_stats),
        "input_mtime_ns": max(stat.st_mtime_ns for stat in input_stats),
        "input_files": [str(path) for path in input_files],
        "declared_input_sha256": site.get("input_sha256", {}).get(args.app),
        "prompt_sha256": sha256(stdin_path) if stdin_path else None,
        "tokenizer_sha256": sha256(tokenizer) if tokenizer else None,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
    }
    write_json(args.out / "manifest.json", manifest)
    pid: int | None = None
    exit_status = 1
    check: dict = {}
    error = ""
    try:
        ssh(host, "mkdir -m 700 -p " + quoted_dir)
        scp(str(args.out / "server.config"), f"{host}:{remote_dir}/server.config")
        remote_env = ("SERVER_PORT=" + str(site["server_port"]) + " FARLIB_RDMA_DEVICE="
                      + shlex.quote(site.get("memory_ib_device", site["ib_device"])))
        if site.get("memory_ld_library_path"):
            remote_env += " LD_LIBRARY_PATH=" + shlex.quote(site["memory_ld_library_path"])
        start = ("cd " + quoted_dir + "; nohup env " + remote_env + " "
                 + quoted_bin + " server.config > server.log 2>&1 < /dev/null & echo $!")
        response = ssh(host, start).stdout.strip()
        if not response.isdigit():
            raise RuntimeError(f"cannot identify owned server PID: {response!r}")
        pid = int(response)
        time.sleep(float(site.get("server_start_wait_s", 5)))
        ssh(host, f"kill -0 {pid}")
        environment = os.environ.copy()
        environment.update(client_environment(args.app, site))
        if site.get("compute_ld_library_path"):
            environment["LD_LIBRARY_PATH"] = site["compute_ld_library_path"]
        with (args.out / "client.log").open("w", encoding="utf-8") as output:
            with (stdin_path.open("r", encoding="utf-8") if stdin_path else open(os.devnull)) as input_stream:
                try:
                    completed = subprocess.run(
                        command, stdin=input_stream, stdout=output,
                        stderr=subprocess.STDOUT, env=environment,
                        timeout=args.timeout, check=False,
                    )
                    exit_status = completed.returncode
                except subprocess.TimeoutExpired:
                    exit_status = 124
        if exit_status != 0:
            raise RuntimeError(f"client exit status {exit_status}")
        check = parse_result(args.app, (args.out / "client.log").read_text(
            encoding="utf-8", errors="replace"))
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        error = str(exc)
    finally:
        if pid is not None:
            # A PID is owned only if it still resolves to this exact server binary.
            stop = (f"if kill -0 {pid} 2>/dev/null; then "
                    f"exe=$(readlink -f /proc/{pid}/exe); "
                    f"expected=$(readlink -f {quoted_bin}); "
                    f"if [ \"$exe\" = \"$expected\" ]; then "
                    f"kill -TERM {pid}; else exit 3; fi; fi")
            try:
                ssh(host, stop)
            except (OSError, subprocess.SubprocessError) as exc:
                error += f"; server cleanup check: {exc}"
        try:
            scp(f"{host}:{remote_dir}/server.log", str(args.out / "server.log"))
        except (OSError, subprocess.SubprocessError) as exc:
            error += f"; server log unavailable: {exc}"
        passed = exit_status == 0 and bool(check) and not error
        analysis = {
            "schema_version": 1, "application": args.app,
            "workload": WORKLOAD_LABEL[args.app],
            "system": "Non-FT" if args.system == "nonft" else "Starfish",
            "ratio": args.ratio, "run_id": args.out.name,
            "environment": site["name"],
            "exit_status": exit_status, "correctness": "pass" if passed else "fail",
            "status": "passed" if passed else "failed",
            "elapsed_s": check.get("elapsed_s") if passed else None,
            "measurement_phase": check.get("measurement_phase"),
            "measurement_method": check.get("measurement_method"),
            "correctness_scope": check.get("correctness_scope"),
            "correctness_evidence": check.get("correctness_evidence"),
            "error": error,
        }
        write_json(args.out / "analysis.json", analysis)
        manifest.update({"end_time": now(), "status": analysis["status"],
                         "exit_status": exit_status, "error": error,
                         "memory_server_pid": pid})
        write_json(args.out / "manifest.json", manifest)
    if analysis["status"] == "passed":
        print(f"PASS {analysis['run_id']}: work {analysis['elapsed_s']:.6f} s "
              f"(details: {args.out / 'analysis.json'})")
    else:
        print(f"FAIL {analysis['run_id']}: {error or 'incomplete run'} "
              f"(details: {args.out / 'analysis.json'})", file=sys.stderr)
    return 0 if analysis["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=sorted(FOOTPRINT_BYTES), required=True)
    parser.add_argument("--system", choices=("nonft", "starfish"), required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-local", action="store_true",
                        help="validate local inputs and recipe without writing or using SSH")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
