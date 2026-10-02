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
    -h|--help|--input|--input=*|--logs-root|--logs-root=*) explicit=1 ;;
  esac
done
if ((explicit)); then
  printf 'plot source: explicit input/logs-root argument\n'
else
  source_path=$("$PYTHON_BIN" "$ROOT/scripts/common/plot_inputs.py" figure11)
  printf 'plot source: latest Figure 9 raw logs %s\n' "$source_path"
  set -- --logs-root "$source_path" "$@"
fi
exec "$PYTHON_BIN" "$HERE/plot.py" "$@"
