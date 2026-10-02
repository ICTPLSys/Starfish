#!/usr/bin/env python3
"""Explicitly revalidate complete real runs rejected only by a log parser."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from site_runs import read_site_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    analysis = json.loads((directory / "analysis.json").read_text())
    figure = manifest["plan"]["figure13"]
    result = dict(schema="figure13-site-result-v1", passed=True, status="passed",
                  manifest="manifest.json", analysis="analysis.json",
                  revalidation=dict(mode="revalidated_current_parser",
                                    original_error=analysis.get("error")))
    result.update({key: figure[key] for key in ("system", "scenario", "repeat", "run_id")})
    destination = directory / "figure13-revalidated-result.json"
    pending = directory / "figure13-revalidation.pending.json"
    if destination.exists():
        parser.error("revalidation result already exists; original evidence is never overwritten")
    created = False
    try:
        with pending.open("x") as stream:
            created = True
            json.dump(result, stream, indent=2)
        run, windows, details = read_site_run(pending)
        result["revalidation"].update(
            completed_requests=details["windows"]["completed"],
            window_count=len(windows),
            client_log_sha256=details["client_log_sha256"],
            recovery_s=details["native_rebuild_duration_s"])
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.rename(destination)
        print(destination)
    finally:
        if created:
            pending.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
