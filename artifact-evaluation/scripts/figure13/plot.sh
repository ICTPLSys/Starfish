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
explicit=0
for arg in "$@"; do
  case "$arg" in
    -h|--help|--manifest|--manifest=*|--logs-root|--logs-root=*|--recovery-csv|--recovery-csv=*|--throughput-csv|--throughput-csv=*|--run-result|--run-result=*) explicit=1 ;;
  esac
done
if ((explicit)); then
  printf 'plot source: explicit manifest/logs/CSV argument\n'
else
  source_path=$("$PYTHON_BIN" "$ROOT/scripts/common/plot_inputs.py" figure13)
  printf 'plot source: latest Figure 13 runs manifest %s\n' "$source_path"
  set -- --manifest "$source_path" "$@"
fi
exec "$PYTHON_BIN" "$HERE/plot_entry.py" "$@"
