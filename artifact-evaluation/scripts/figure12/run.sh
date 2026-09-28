#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" "$ROOT/scripts/figure12/run.py" "$@"
