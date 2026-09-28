#!/usr/bin/env python3
"""Run one complete LLaMA chat per Non-FT/Starfish/recovery condition."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from endpoint import validate_port
import run_case


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("llama-nonft-25", "nonft", None),
    ("llama-starfish-25", "starfish", None),
    ("llama-starfish-recovery-25", "starfish", 0),
)


def case_sites(site_path: Path, dry_run: bool = False) -> dict:
    """Reuse site addresses; colocated service processes are not extra nodes."""
    site = run_case.site_config(site_path, dry_run=dry_run)
    endpoints = site["memory_endpoints"]
    if len(endpoints) == 1:
        base = endpoints[0]
        ec_endpoints = []
        for index in range(7):
            endpoint = copy.deepcopy(base)
            endpoint["index"] = index
            endpoint["server_port"] = validate_port(base["server_port"] + index)
            ec_endpoints.append(endpoint)
    elif len(endpoints) >= 7:
        ec_endpoints = copy.deepcopy(endpoints[:7])
    else:
        raise ValueError("fast check requires one memory host or at least seven endpoints")
    result = {}
    for system, selected in (("nonft", endpoints[:1]), ("starfish", ec_endpoints)):
        derived = copy.deepcopy(site)
        derived["memory_endpoints"] = selected
        # Normalized sites always use explicit endpoints, not the legacy fields.
        derived.pop("legacy_memory_endpoint", None)
        derived.pop("memory_server_count", None)
        for endpoint in selected:
            if system not in endpoint["memory_server_bins"]:
                raise ValueError(f"site has no {system} memory server binary")
        result[system] = derived
    return result


def case_args(run_id: str, system: str, recovery: int | None,
              site: Path, out: Path, timeout: int, *, dry_run: bool = False,
              check_local: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        app="llama", system=system, ratio=25, site=site,
        out=out / "runs" / run_id, timeout=timeout, dry_run=dry_run,
        check_local=check_local, capture_chat=True, recover_endpoint=recovery,
        reference_chat=(out / "runs/llama-nonft-25/chat-output.txt"
                        if system == "starfish" else None),
        recipe=(ROOT / "configs/llama/starfish_ec.config"
                if system == "starfish" else None),
    )


def verify_answers(out: Path) -> dict:
    outputs = [(out / "runs" / name / "chat-output.txt").read_bytes()
               for name, _, _ in CASES]
    if not outputs[0] or any(value != outputs[0] for value in outputs[1:]):
        raise ValueError("complete chat outputs differ across the three cases or are empty")
    return {"matching_chat_outputs": True, "answer_bytes": len(outputs[0])}


def execute(args: argparse.Namespace) -> int:
    sites = case_sites(args.site, args.dry_run)
    if args.dry_run:
        # Planning does not build, contact hosts, write files or reserve ports.
        print(json.dumps({
            "workload": "LLaMA 2 7B Chat FP32, hello/chat, default seed 1",
            "ratio": 25, "repeats": 1, "build_systems": ["nonft", "starfish"],
            "cases": [
                {"run_id": name, "system": system,
                 "recover_endpoint": recovery, "site": sites[system],
                 "recipe": str(case_args(name, system, recovery, args.site,
                                        args.out, args.timeout).recipe
                               or ROOT / "configs/llama/nonft.config")}
                for name, system, recovery in CASES],
            "output": str(args.out),
        }, indent=2))
        return 0
    if args.out.exists():
        raise ValueError(f"refusing to overwrite fast check: {args.out}")
    args.out.mkdir(parents=True)
    summary = {"status": "running", "ratio": 25, "repeats": 1, "cases": []}
    try:
        for system, site in sites.items():
            run_case.write_json(args.out / f"site-{system}.json", site)
        # The Quick Start default build is Non-FT. Build both required targets
        # incrementally here, so reviewers need no extra system-selection flags.
        for system in ("nonft", "starfish"):
            print(f"Fast check: building {system} (log: {args.out / f'build-{system}.log'})",
                  flush=True)
            with (args.out / f"build-{system}.log").open("w") as output:
                subprocess.run(
                    ["bash", str(ROOT / "scripts/common/build.sh"),
                     "--system", system, "--targets", "server,run_chat_far"],
                    stdout=output, stderr=subprocess.STDOUT, check=True)
        for name, system, recovery in CASES:
            run_case.run(case_args(
                name, system, recovery, args.out / f"site-{system}.json",
                args.out, args.timeout, check_local=True))
        for name, system, recovery in CASES:
            print(f"Fast check: {name}", flush=True)
            status = run_case.run(case_args(
                name, system, recovery, args.out / f"site-{system}.json",
                args.out, args.timeout))
            analysis_path = args.out / "runs" / name / "analysis.json"
            analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else {}
            summary["cases"].append({
                "run_id": name, "status": analysis.get("status", "failed"),
                "elapsed_s": analysis.get("elapsed_s"),
                "analysis": str(analysis_path), "recovery": analysis.get("recovery"),
            })
            if status != 0 or analysis.get("status") != "passed":
                raise RuntimeError(f"{name} failed; see {analysis_path}")
            if recovery is not None and not (
                    analysis.get("recovery", {}).get("status") == "passed"
                    and analysis.get("recovery", {}).get("injection", {}).get("process_stopped")):
                raise RuntimeError(f"{name} lacks verified fault/recovery evidence")
        summary.update(verify_answers(args.out))
        summary["status"] = "passed"
        print(f"Fast check PASS: all three runs and matching chat outputs ({args.out})")
        return 0
    except KeyboardInterrupt:
        summary.update(status="interrupted", error="interrupted by user")
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        summary.update(status="failed", error=str(exc))
        print(f"Fast check FAIL: {exc} (logs: {args.out})", file=sys.stderr)
        return 1
    finally:
        run_case.write_json(args.out / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "data/site.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/fast-check" /
                        ("batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.site = args.site.resolve()
    args.out = args.out.resolve()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.dry_run and not args.site.exists() and args.site == ROOT / "data/site.json":
        args.site = ROOT / "scripts/common/site.eight-server.example.json"
    try:
        return execute(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Fast check error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
