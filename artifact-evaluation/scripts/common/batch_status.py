"""Keep measurement usability separate from execution and cleanup outcomes."""

import argparse
import json
from pathlib import Path
import sys


def data_completeness(run_dir, analysis, manifest, measurement_usable):
    """Validate absolute raw metrics, without requiring a normalization pair."""
    if not measurement_usable:
        return False, "verified performance measurement is unavailable"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from figure11.raw_runs import collect_records
        plan = manifest.get("plan", {})
        ratio = int(analysis.get("ratio", plan.get("ratio", 0)))
        repeat = int(analysis.get("repeat", plan.get("repeat", 1)))
        records = collect_records(run_dir, ratio=ratio, repeat=repeat,
                                  include_regular_nonft=True)
        own_records = [record for record in records
                       if Path(record.get("_raw_run_dir", "")).resolve()
                       == Path(run_dir).resolve()]
        if len(own_records) != 1:
            raise ValueError("exactly one complete record for this case is required")
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        return False, str(error)
    record = own_records[0]
    return True, (
        "performance, correctness, Work traffic, remote CPU and Work memory usable; "
        f"memory samples={record['remote_memory_samples']}/"
        f"{record['remote_memory_expected_samples']}, "
        f"scheduled={record['remote_memory_scheduled_samples']}, "
        f"missing={record['remote_memory_missing_samples']}, "
        f"end_status={record['remote_memory_end_status']}; "
        "mean uses successful samples only")


def classify_case(run_dir, returncode):
    """Use saved runner evidence; never contact or signal an experiment host."""
    run_dir = Path(run_dir)
    result = {"status": "error", "safe_to_continue": True, "reason": "",
              "exit_status": returncode, "client_exit_status": None,
              "measurement_usable": False, "data_complete": False,
              "data_complete_reason": "case has not reached verified completion"}
    try:
        def read(path):
            if not path.exists():
                return {}
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"invalid case record: {path}")
            return value

        analysis = read(run_dir / "analysis.json")
        manifest = read(run_dir / "manifest.json")
        endpoints = []
        for record in (manifest, analysis):
            states = record.get("endpoints", [])
            if not isinstance(states, list) or any(not isinstance(s, dict) for s in states):
                raise ValueError("invalid endpoint records")
            endpoints.extend(states)
        endpoints.extend(read(path) for path in
                         (run_dir / "endpoints").glob("*/endpoint-manifest.json"))
        for state in endpoints:
            status = state.get("status")
            if status in ("stopped", "already_exited"):
                continue
            if (status in ("planned", "preflight_pass")
                    and state.get("pid") is None
                    and state.get("launch_attempted") is not True
                    and not state.get("remote_dir_created")):
                continue
            if (status == "not_started" and state.get("pid") is None
                    and state.get("launch_attempted") is False):
                continue
            result.update(safe_to_continue=False,
                          reason="memory-service cleanup is incomplete or unverified")
        client = analysis.get("client_cleanup")
        if client is not None and (not isinstance(client, dict) or not client.get("reaped")):
            result.update(safe_to_continue=False, reason="client exit is not confirmed")
        if manifest and not analysis:
            result.update(safe_to_continue=False, reason="runner did not finalize its analysis")
        if returncode < 0 or returncode in (130, 143):
            result.update(safe_to_continue=False, reason="batch runner was interrupted")
        recovered = None
        if manifest.get("plan", {}).get("system") == "starfish":
            from measurement_acceptance import recover_measurement
            try:
                recovered = recover_measurement(run_dir, analysis, manifest)
            except (OSError, KeyError, TypeError, ValueError) as error:
                result["measurement_error"] = str(error)
        result["measurement_usable"] = bool(recovered)
        if not result["safe_to_continue"]:
            return result
        result["client_exit_status"] = analysis.get("exit_status")
        if (returncode == 0 and analysis.get("status") == "passed"
                and analysis.get("correctness") == "pass"
                and analysis.get("exit_status") == 0
                and endpoints and all(s.get("status") in ("stopped", "already_exited")
                                      for s in endpoints)
                and not analysis.get("error")
                and (not manifest or manifest.get("status") == "passed")):
            result.update(status="passed", reason="verified case completed",
                          measurement_usable=True)
        elif recovered:
            result.update(status="warning", reason=recovered["measurement_warning"])
        elif (analysis.get("timed_out") is True or
              ("timed_out" not in analysis and analysis.get("exit_status") == 124)):
            # Legacy records identify client timeouts by code 124. New records
            # distinguish an actual deadline from a program returning 124.
            result.update(status="timeout", reason=analysis.get("error") or "client timeout")
        else:
            result["reason"] = analysis.get("error") or (
                f"runner exited {returncode} without a verified successful result")
        result["data_complete"], result["data_complete_reason"] = data_completeness(
            run_dir, analysis, manifest, result["measurement_usable"])
    except (OSError, ValueError, TypeError) as error:
        result.update(safe_to_continue=False, reason=f"invalid cleanup evidence: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    result = classify_case(args.run_dir, args.exit_status)
    if args.format == "shell":
        print(result["status"], int(result["safe_to_continue"]))
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
