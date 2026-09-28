#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/use_prepared_data.py" "$@"
