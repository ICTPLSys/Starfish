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
has_input=0
for arg in "$@"; do
  case "$arg" in
    -h|--help|--input|--input=*) has_input=1 ;;
  esac
done
if ((has_input)); then
  printf 'plot source: explicit input argument\n'
else
  source_path=$("$PYTHON_BIN" "$ROOT/scripts/common/plot_inputs.py" figure9)
  printf 'plot source: %s\n' "$source_path"
  set -- --input "$source_path" "$@"
fi
exec "$PYTHON_BIN" "$HERE/plot.py" "$@"
