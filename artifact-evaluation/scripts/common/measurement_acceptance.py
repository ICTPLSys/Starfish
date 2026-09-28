"""User-approved reuse of verified KV work before an unsuccessful shutdown."""

from pathlib import Path

import kv_latency
from workloads import parse_result, validate_design2


def recover_measurement(run_dir, analysis, manifest):
    """Reparse raw work; do not promote lifecycle failure or alter old records.

    This exception covers Starfish KV, including the retained KV-A shutdown
    hang. Other systems, unverified workloads, crashes and partial measurements
    retain their existing strict acceptance rules.
    """
    plan = manifest.get("plan", {})
    app = plan.get("app")
    code = analysis.get("exit_status")
    timeout = (code == 124 and (analysis.get("timed_out") is True
                               or "timed_out" not in analysis))
    if (plan.get("system") != "starfish" or app not in ("kv-b", "kv-a", "kv-s")
            or analysis.get("application") != app
            or (not timeout and code not in (-15, -9))
            or plan.get("failure_injection_endpoint") is not None
            or analysis.get("failure_injection")):
        return None
    root = Path(run_dir)
    log = (root / "client.log").read_text(encoding="utf-8", errors="replace")
    spec = plan.get("kv_latency")
    if spec is not None:
        if app != "kv-b":
            raise ValueError("only KV-B has a Figure 10 measurement contract")
        check = kv_latency.parse_result(
            log, spec, expected_workers=plan["worker_profile"]["app_workers"],
            histogram_dir=root / "histograms")
    else:
        check = parse_result(app, log, expected_requests=1_000_000_000,
                             expected_post_samples=4096, require_cleanup=False)
    # Keep runtime-profile verification: relaxing shutdown does not waive it.
    check["design2"] = validate_design2(log)
    check["measurement_warning"] = (
        "verified measurement retained; client shutdown timed out"
        if timeout else
        "verified measurement retained; client was terminated after postcheck")
    return check
