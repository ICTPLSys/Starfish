"""Bounded client execution and evidence checks for the recovery fast check."""

from pathlib import Path
import re
import subprocess
import time


CONTROL_ENV = (
    "FARLIB_CAPTURE_CHAT_OUTPUT", "FARLIB_EC_RECOVERY_PROFILE",
    "FARLIB_EC_RECOVERY_VERIFY", "FARLIB_EC_RECOVERY_DIAG_QUIET",
)


def client_env(environment, capture, ec):
    result = {key: value for key, value in environment.items() if key not in CONTROL_ENV}
    if capture is not None:
        result["FARLIB_CAPTURE_CHAT_OUTPUT"] = str(capture.resolve())
    if ec:
        result.update(FARLIB_EC_RECOVERY_PROFILE="0", FARLIB_EC_RECOVERY_VERIFY="1",
                      FARLIB_EC_RECOVERY_DIAG_QUIET="0")
    return result


def validate_request(app, system, endpoint, endpoint_count, method, standby):
    if endpoint is None:
        return
    if (app != "llama" or system != "starfish" or method != "ec_batch"
            or endpoint_count != 7 or standby != "6"
            or isinstance(endpoint, bool) or not 0 <= endpoint < 6):
        raise ValueError("recovery requires LLaMA/Starfish ec_batch, seven endpoints, "
                         "standby 6, and a failed endpoint in 0..5")


def run_client(command, *, stdin, stdout, env, timeout, capture, inject, evidence):
    """A captured answer byte, not initialization or a prompt, arms injection."""
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(command, stdin=stdin, stdout=stdout,
                               stderr=subprocess.STDOUT, env=env)
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            if (inject is not None and not evidence and capture.is_file()
                    and capture.stat().st_size):
                if process.poll() is not None:
                    break
                evidence.update(inject())
            time.sleep(0.05)
        if inject is not None and not evidence:
            raise RuntimeError("client exited before recovery was injected "
                               f"(exit status {process.returncode})")
        return process.returncode
    finally:
        # Exceptions and Ctrl-C must not leave the workload using services that
        # the owning runner is about to stop.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        else:
            process.wait()


def compare_chat(capture, reference):
    actual, expected = Path(capture).read_bytes(), Path(reference).read_bytes()
    if not expected or actual != expected:
        raise ValueError("complete chat output differs from the Non-FT reference or is empty")
    return {"matched": True, "answer_bytes": len(actual), "reference": str(reference)}


def validate_log(log, injection=None, endpoint_count=7, *, answer_matches_reference=False):
    matches = re.findall(
        r"^ec_read_recovery diag \[cache_dtor_begin\]: ([^\n]+)$",
        log, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("EC run needs exactly one final recovery diagnostic")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", matches[0]))
    required = ("completions", "verify_pass", "verify_fail", "rebuild_failures",
                "tokens_in_use", "scratch_in_use", "split_buffers_in_use")
    try:
        counts = {key: int(fields[key]) for key in required}
        dead, total = (int(value) for value in fields["dead_endpoints"].split("/"))
    except (KeyError, ValueError) as exc:
        raise ValueError("malformed final recovery counters") from exc
    if any(value < 0 for value in counts.values()) or total != endpoint_count:
        raise ValueError("invalid final recovery counters")
    if any(counts[key] for key in ("verify_fail", "rebuild_failures",
                                    "tokens_in_use", "scratch_in_use",
                                    "split_buffers_in_use")):
        raise ValueError("EC verification/rebuild failed or recovery resources remain live")
    if injection is None:
        if dead != 0:
            raise ValueError("healthy EC control unexpectedly lost an endpoint")
    else:
        endpoint = injection.get("endpoint")
        if (not injection.get("owned_identity_verified")
                or not injection.get("process_stopped")
                or dead != 1 or counts["completions"] == 0
                # The four-survivor decoder intentionally skips the optional
                # five-survivor re-encode observer. Require an independent
                # complete-answer byte oracle if that counter is zero.
                or (counts["verify_pass"] == 0 and not answer_matches_reference)
                # Concurrent logging can interleave the one-off death line.
                # A completed degraded read also records the failed endpoint.
                or not re.search(
                    r"INFO: ec_recovery (?:endpoint_dead endpoint=|"
                    r"degraded_read [^\n]*\bendpoint_dead=)" + str(endpoint) + r"\b", log)):
            raise ValueError("missing owned fault injection or successful EC recovery evidence")
    return {"status": "passed", "injection": injection,
            "answer_matches_reference": answer_matches_reference,
            "dead_endpoints": dead, "endpoint_count": total, **counts}
