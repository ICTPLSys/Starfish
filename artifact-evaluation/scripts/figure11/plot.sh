#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT/.venv-plot/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv-plot/bin/python"
else
  PYTHON_BIN=python3
fi
exec "$PYTHON_BIN" "$HERE/plot.py" "$@"
