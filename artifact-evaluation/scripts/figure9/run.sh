#!/usr/bin/env bash
set -euo pipefail

# Figure 9 chooses a matrix; common/run_case.py owns every two-server run.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON:-python3}"
PLOT_PYTHON="$PYTHON_BIN"
[[ -x "$ROOT/.venv-plot/bin/python" ]] && PLOT_PYTHON="$ROOT/.venv-plot/bin/python"
APPS=llama,bfs
SYSTEMS=nonft
RATIOS=13,25,50,75,100
REPEATS=1
TIMEOUT=1800
SITE="$ROOT/data/site.json"
OUT="$ROOT/results/figure9/batch-$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/figure9/run.sh [options]
  --apps llama,bfs                  applications (default: llama,bfs)
  --systems nonft,starfish          released runtime variants (default: nonft)
  --ratios 13,25,50,75,100          local memory percentages
  --repeats N                       independent runs per condition (default: 1)
  --site FILE                       local, ignored server/input configuration
  --out DIR                         new batch directory under results/
  --timeout SEC                     client timeout (default: 1800)
  --dry-run                         show plans; do not write or contact servers

Only successful, verified cases enter the measured-only CSV. Missing cases
stay absent. Install plot dependencies before running an actual batch.
EOF
}

while (($#)); do
  case "$1" in
    --apps) APPS=$2; shift 2 ;;
    --systems) SYSTEMS=$2; shift 2 ;;
    --ratios) RATIOS=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --site) SITE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$REPEATS" =~ ^[1-9][0-9]*$ && "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "--repeats and --timeout must be positive integers" >&2; exit 2;
}
IFS=',' read -r -a app_list <<< "$APPS"
IFS=',' read -r -a system_list <<< "$SYSTEMS"
IFS=',' read -r -a ratio_list <<< "$RATIOS"
((${#app_list[@]} && ${#system_list[@]} && ${#ratio_list[@]})) || {
  echo "applications, systems, and ratios must be nonempty" >&2; exit 2;
}
for app in "${app_list[@]}"; do
  case "$app" in llama|bfs) ;; *) echo "unsupported application: $app" >&2; exit 2 ;; esac
done
for system in "${system_list[@]}"; do
  case "$system" in nonft|starfish) ;;
    *) echo "unsupported system: $system" >&2; exit 2 ;;
  esac
done
max_ratio=0
for ratio in "${ratio_list[@]}"; do
  [[ "$ratio" =~ ^[0-9]+$ ]] && ((10#$ratio >= 1 && 10#$ratio <= 100)) || {
    echo "invalid ratio: $ratio" >&2; exit 2;
  }
  ratio_num=$((10#$ratio))
  if ((ratio_num > max_ratio)); then max_ratio=$ratio_num; fi
done
[[ "$(basename "$OUT")" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "batch directory name must contain only letters, digits, dots, dashes or underscores" >&2
  exit 2
}
[[ -f "$SITE" ]] || {
  if [[ "$DRY_RUN" = 1 ]]; then
    SITE="$ROOT/scripts/common/site.example.json"
  else
    echo "site config missing: $SITE (copy scripts/common/site.example.json to data/site.json)" >&2
    exit 2
  fi
}
[[ "$DRY_RUN" = 1 || ! -e "$OUT" ]] || {
  echo "refusing to overwrite batch: $OUT" >&2; exit 2;
}

if [[ "$DRY_RUN" != 1 ]]; then
  "$PLOT_PYTHON" -c 'import matplotlib, numpy' || {
    echo "install plotting dependencies before starting the batch" >&2; exit 2;
  }
  for app in "${app_list[@]}"; do
    for system in "${system_list[@]}"; do
      "$PYTHON_BIN" "$ROOT/scripts/common/run_case.py" \
        --app "$app" --system "$system" --ratio "$max_ratio" \
        --site "$SITE" --out "$OUT/runs/preflight-$app-$system" \
        --timeout "$TIMEOUT" --check-local
    done
  done
  mkdir -p "$OUT/runs"
fi

failures=0
for app in "${app_list[@]}"; do
  for system in "${system_list[@]}"; do
    for ratio in "${ratio_list[@]}"; do
      for ((rep=1; rep<=REPEATS; rep++)); do
        run_id="${app}-${system}-${ratio}-r${rep}"
        case_out="$OUT/runs/$run_id"
        args=(--app "$app" --system "$system" --ratio "$ratio"
              --site "$SITE" --out "$case_out" --timeout "$TIMEOUT")
        [[ "$DRY_RUN" = 1 ]] && args+=(--dry-run)
        if [[ "$DRY_RUN" = 1 ]]; then
          "$PYTHON_BIN" "$ROOT/scripts/common/run_case.py" "${args[@]}" || failures=$((failures + 1))
        elif ! "$PYTHON_BIN" "$ROOT/scripts/common/run_case.py" "${args[@]}" 2>&1 |
             tee -a "$OUT/batch.log"; then
          failures=$((failures + 1))
          echo "FAILED $run_id" >&2
        fi
      done
    done
  done
done

if [[ "$DRY_RUN" = 1 ]]; then
  echo "dry-run: no files written, no servers contacted"
  exit "$((failures > 0))"
fi
"$PYTHON_BIN" "$ROOT/scripts/figure9/collect.py" \
  --runs-dir "$OUT/runs" --output "$OUT/figure9.csv"
"$PLOT_PYTHON" "$ROOT/scripts/figure9/plot.py" \
  --input "$OUT/figure9.csv" --output-dir "$OUT/figure"
# A wholly failed batch must not replace a previously useful public input.
if (( $(wc -l < "$OUT/figure9.csv") > 1 )); then
  mkdir -p "$ROOT/data"
  # Publish a complete copy for reviewers who plot separately. The original
  # batch CSV remains alongside the run records for provenance.
  published_tmp=$(mktemp "$ROOT/data/.figure9.csv.XXXXXX")
  trap 'rm -f -- "$published_tmp"' EXIT
  cp -- "$OUT/figure9.csv" "$published_tmp"
  mv -f -- "$published_tmp" "$ROOT/data/figure9.csv"
  trap - EXIT
  echo "latest Figure 9 CSV: $ROOT/data/figure9.csv"
else
  echo "no successful measurements; data/figure9.csv left unchanged" >&2
fi
echo "figure9 batch: $OUT (failed_cases=$failures)"
exit "$((failures > 0))"
