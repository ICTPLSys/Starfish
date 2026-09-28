"""Figure 9 source/adapter support, not a claim of runtime validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


AE_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_LABEL = {"nonft": "Non-FT", "starfish": "Starfish",
                "hydra": "Hydra", "carbink": "Carbink"}
APPLICATIONS = ("llama", "bfs", "mg", "wordcount", "kv-b", "kv-a", "kv-s", "nq")
ADAPTERS = frozenset(("llama", "bfs"))
SYSTEM_ADAPTERS = frozenset(("nonft", "starfish", "hydra"))


def support(app, system, *, recipe=None, root=AE_ROOT):
    if app not in APPLICATIONS:
        raise ValueError(f"unknown application {app!r}; choose {','.join(APPLICATIONS)}")
    if system not in SYSTEM_LABEL:
        raise ValueError(f"unknown system {system!r}; choose {','.join(SYSTEM_LABEL)}")
    selected = Path(recipe) if recipe is not None else root / "configs" / app / f"{system}.config"
    gaps = []
    if app not in ADAPTERS:
        gaps.append("workload/footprint/correctness/measurement adapter not integrated")
    if system not in SYSTEM_ADAPTERS:
        gaps.append("runtime configuration/environment/evidence adapter not integrated")
    if not (root / "runtime" / system / "CMakeLists.txt").is_file():
        gaps.append("runtime source not integrated")
    if not selected.is_file():
        gaps.append(f"recipe missing: {selected}")
    return {"app": app, "system": system, "script_supported": not gaps,
            "runtime_verified": False, "gaps": gaps}


def require_supported(app, system, *, recipe=None):
    record = support(app, system, recipe=recipe)
    if record["gaps"]:
        raise ValueError(f"unsupported Figure 9 case {app}/{system}: " + "; ".join(record["gaps"]))
    return record


def selections(value, allowed, label):
    values = value.split(",")
    if not values or any(not item or item not in allowed for item in values):
        raise ValueError(f"{label} must be a comma-separated selection from {','.join(allowed)}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} would reuse output directories")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-supported", action="store_true")
    parser.add_argument("--apps", default=",".join(APPLICATIONS))
    parser.add_argument("--systems", default=",".join(SYSTEM_LABEL))
    args = parser.parse_args()
    try:
        apps = selections(args.apps, APPLICATIONS, "applications")
        systems = selections(args.systems, SYSTEM_LABEL, "systems")
        if args.list_supported:
            print("Script support only; binaries, site readiness and end-to-end correctness are NOT verified.")
            for app in apps:
                for system in systems:
                    row = support(app, system)
                    state = "IMPLEMENTED / UNTESTED" if row["script_supported"] else "NOT INTEGRATED"
                    print(f"{app:10} {system:9} {state}: " + "; ".join(row["gaps"]))
        else:
            failures = []
            for app in apps:
                for system in systems:
                    row = support(app, system)
                    if row["gaps"]:
                        failures.append(f"{app}/{system}: " + "; ".join(row["gaps"]))
            if failures:
                raise ValueError("requested matrix is not integrated:\n" + "\n".join(failures))
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
