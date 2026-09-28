#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$HERE/run.py" "$@"
