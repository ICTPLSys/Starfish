#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=$(printenv PYTHON || true)
test -n "$PYTHON_BIN" || PYTHON_BIN=python3
exec "$PYTHON_BIN" "$HERE/run.py" "$@"
