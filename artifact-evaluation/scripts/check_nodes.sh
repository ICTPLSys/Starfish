#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python3}" -B "$ROOT/scripts/common/check_nodes.py" "$@"
