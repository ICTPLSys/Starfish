#!/usr/bin/env python3
"""Native manifest/batch entry, with the existing CSV plot CLI preserved."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
args = []
for arg in sys.argv[1:]:
    if arg.startswith("--") and "=" in arg:
        args.extend(arg.split("=", 1))
    else:
        args.append(arg)
native = "--manifest" in args
if "--logs-root" in args and args.index("--logs-root") + 1 < len(args):
    i = args.index("--logs-root")
    index = Path(args[i + 1]) / "runs.json"
    if index.is_file():
        args[i:i + 2] = ["--manifest", str(index)]
        native = True
if native:
    if "--panel" not in args:
        args += ["--panel", "both"]
    if not any(flag in args for flag in ("--display-points", "--bin-windows", "--average-windows")):
        args += ["--display-points", "41"]
    if "--output-dir" not in args:
        root = Path(__file__).resolve().parents[2]
        args += ["--output-dir", str(root / "results/figures/figure13")]
    sys.argv = [sys.argv[0]] + args
    from render import main
else:
    if not any(flag in args for flag in ("--display-points", "--bin-windows", "--average-windows")):
        args += ["--display-points", "41"]
    sys.argv = [sys.argv[0]] + args
    from plot import main
raise SystemExit(main())
