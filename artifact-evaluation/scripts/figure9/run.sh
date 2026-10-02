#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Figure 9 chooses a matrix; common/run_case.py owns every two-server run.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON:-python3}"
PLOT_PYTHON="$PYTHON_BIN"
[[ -x "$ROOT/.venv-plot/bin/python" ]] && PLOT_PYTHON="$ROOT/.venv-plot/bin/python"
RUN_CASE_PY="$ROOT/scripts/common/run_case.py"
BATCH_STATUS_PY="$ROOT/scripts/common/batch_status.py"
COLLECT_PY="$ROOT/scripts/figure9/collect.py"
PLOT_PY="$ROOT/scripts/figure9/plot.py"

APPS=llama,bfs,mg,wordcount,kv-b,kv-a,kv-s,nq
SYSTEMS=nonft,nonft-backup-off,starfish,hydra,carbink
RATIOS=13,25,50,75,100
REPEATS=1
TIMEOUT=1800
SITE="$ROOT/data/site.json"
OUT="$ROOT/results/figure9/batch-$(date -u +%Y%m%dT%H%M%SZ)"
BUILD_ROOT=""
STOP_ON_ERROR=0
DRY_RUN=0
LIST_SUPPORTED=0

usage() {
  cat <<'EOF'
Usage: scripts/figure9/run.sh [options]
  --apps llama,bfs,mg,wordcount,kv-b,kv-a,kv-s,nq
                                    applications (default: full Figure 9 matrix)
  --systems nonft,nonft-backup-off,starfish,hydra,carbink
                                    systems to select (default: all, in this order)
  --ratios 13,25,50,75,100          local memory percentages
  --repeats N                       independent runs per condition (default: 1)
  --site FILE                       local, ignored server/input configuration
  --out DIR                         new batch directory under results/
  --build-root DIR                 optional isolated build root passed to run_case
  --stop-on-error                  stop after the first safe failed case
  --timeout SEC                     client timeout (default: 1800)
  --dry-run                         show plans; do not write or contact servers
  --list-supported                  list source/recipe/adapter support, not runtime validation
Only verified measurements enter the CSV; accepted teardown warnings remain
visible. DATA_COMPLETE additionally requires correctness, Work traffic, remote
CPU and Work memory. It does not imply a matching NonFT-backup-off normalization
baseline exists. Install plot dependencies before running an actual batch.
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
    --build-root) BUILD_ROOT=$2; shift 2 ;;
    --stop-on-error) STOP_ON_ERROR=1; shift ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list-supported) LIST_SUPPORTED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -n "$BUILD_ROOT" && "$BUILD_ROOT" != /* ]]; then
  BUILD_ROOT="$PWD/$BUILD_ROOT"
fi

if [[ "$LIST_SUPPORTED" = 1 ]]; then
  exec "$PYTHON_BIN" "$ROOT/scripts/common/figure9_support.py" \
    --list-supported --apps "$APPS" --systems "$SYSTEMS"
fi

# Validate names and duplicate selections without rejecting a whole matrix
# merely because one system recipe or runtime is not installed yet. Individual
# cases are classified after the runner returns, so later systems still run.
"$PYTHON_BIN" "$ROOT/scripts/common/figure9_support.py" \
  --list-supported --apps "$APPS" --systems "$SYSTEMS" >/dev/null
[[ "$REPEATS" =~ ^[1-9][0-9]*$ && "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "--repeats and --timeout must be positive integers" >&2
  exit 2
}
IFS=',' read -r -a app_list <<< "$APPS"
IFS=',' read -r -a system_list <<< "$SYSTEMS"
IFS=',' read -r -a ratio_list <<< "$RATIOS"
(( ${#app_list[@]} && ${#system_list[@]} && ${#ratio_list[@]} )) || {
  echo "applications, systems, and ratios must be nonempty" >&2
  exit 2
}
declare -A seen_ratios=()
for ratio in "${ratio_list[@]}"; do
  [[ "$ratio" =~ ^[0-9]+$ ]] &&
    ((10#${ratio} >= 1 && 10#${ratio} <= 100)) || {
      echo "invalid ratio: $ratio" >&2
      exit 2
    }
  ratio_num=$((10#${ratio}))
  [[ ! ${seen_ratios[$ratio_num]+present} ]] || {
    echo "duplicate ratio: $ratio" >&2
    exit 2
  }
  seen_ratios[$ratio_num]=1
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
  echo "refusing to overwrite batch: $OUT" >&2
  exit 2
}

runner_scope_args=()
[[ -n "$BUILD_ROOT" ]] && runner_scope_args+=(--build-root "$BUILD_ROOT")

system_label() {
  case "$1" in
    nonft) printf '%s' 'NonFT' ;;
    starfish) printf '%s' 'Starfish' ;;
    hydra) printf '%s' 'Hydra' ;;
    carbink) printf '%s' 'Carbink' ;;
    nonft-backup-off) printf '%s' 'NonFT-backup-off' ;;
    *) printf '%s' "$1" ;;
  esac
}

plot_workload() {
  case "$1" in
    llama) printf '%s' 'llama' ;;
    bfs) printf '%s' 'bfs' ;;
    mg) printf '%s' 'mg' ;;
    wordcount) printf '%s' 'wc' ;;
    kv-b) printf '%s' 'kv_b' ;;
    kv-a) printf '%s' 'kv_a' ;;
    kv-s) printf '%s' 'kv_s' ;;
    nq) printf '%s' 'nq' ;;
    *) printf '%s' "$1" ;;
  esac
}

plot_system() {
  case "$1" in
    nonft|starfish|hydra|carbink) printf '%s' "$1" ;;
    nonft-backup-off) printf '%s' 'nonft' ;;
    *) printf '%s' "$1" ;;
  esac
}
selected_systems=""
for system in "${system_list[@]}"; do
  if [[ -n "$selected_systems" ]]; then selected_systems+=", "; fi
  selected_systems+="$(system_label "$system")"
done
printf 'This script will run %s sequentially.\n' "$selected_systems"

if [[ "$DRY_RUN" = 1 ]]; then
  # Keep dry-run read-only: no output directories, logs, CSVs, or temporary
  # files are created. The concise plan line replaces one JSON document per
  # case while preserving exact system/app/ratio/repeat traversal.
  dry_run_errors=0
  for system in "${system_list[@]}"; do
    for app in "${app_list[@]}"; do
      printf '%s App %s START\n' "${system^^}" "${app^^}"
      app_errors=0
      for ratio in "${ratio_list[@]}"; do
        for ((rep=1; rep<=REPEATS; rep++)); do
          run_id="${app}-${system}-${ratio}-r${rep}"
          if [[ "$system" = "nonft-backup-off" ]]; then
            runner_system=nonft
            baseline_args=(--baseline-variant nonft-backup-off)
          else
            runner_system="$system"
            baseline_args=()
          fi
          set +e
          "$PYTHON_BIN" "$RUN_CASE_PY" \
            --app "$app" --system "$runner_system" ${baseline_args[@]} \
            --collect-remote-cpu --collect-remote-memory --ratio "$ratio" --repeat "$rep" \
            --site "$SITE" --out "$OUT/runs/$run_id" "${runner_scope_args[@]}" \
            --timeout "$TIMEOUT" --dry-run >/dev/null 2>&1
          plan_rc=$?
          set -e
          if ((plan_rc == 0)); then
            printf 'PLAN %s %s %s%% r%s OK\n' \
              "${system^^}" "${app^^}" "$ratio" "$rep"
          else
            dry_run_errors=$((dry_run_errors + 1))
            app_errors=$((app_errors + 1))
            printf 'PLAN %s %s %s%% r%s ERROR (runner exit=%d)\n' \
              "${system^^}" "${app^^}" "$ratio" "$rep" "$plan_rc"
          fi
        done
      done
      printf '%s App %s FINISH! (failures=%d)\n' \
        "${system^^}" "${app^^}" "$app_errors"
    done
  done
  printf 'dry-run: plans only; no files written, no hosts contacted, no run validated (plan_errors=%d)\n' \
    "$dry_run_errors"
  exit "$((dry_run_errors > 0))"
fi

"$PLOT_PYTHON" -c 'import matplotlib, numpy' >/dev/null 2>&1 || {
  echo "ERROR: install plotting dependencies before starting the batch" >&2
  exit 2
}
mkdir -p "$OUT/runs" "$OUT/logs"
: > "$OUT/batch.log"
printf 'Figure 9 batch started: %s\n' "$OUT" >> "$OUT/batch.log"

passed=0
timeouts=0
errors=0
usable_warnings=0
data_complete_cases=0
data_incomplete_cases=0
unsafe_cleanup=0
stop_batch=0
for system in "${system_list[@]}"; do
  printf '%s Start...\n' "$(system_label "$system")"
  system_failures=0
  for app in "${app_list[@]}"; do
    printf '%s App %s START\n' "${system^^}" "${app^^}"
    app_failures=0
    for ratio in "${ratio_list[@]}"; do
      for ((rep=1; rep<=REPEATS; rep++)); do
        run_id="${app}-${system}-${ratio}-r${rep}"
        if [[ "$system" = "nonft-backup-off" ]]; then
          runner_system=nonft
          baseline_args=(--baseline-variant nonft-backup-off)
        else
          runner_system="$system"
          baseline_args=()
        fi
        case_out="$OUT/runs/$run_id"
        runner_log="$OUT/logs/$run_id.runner.log"
        args=(--app "$app" --system "$runner_system" ${baseline_args[@]}
              --collect-remote-cpu --collect-remote-memory --ratio "$ratio" --repeat "$rep"
              --site "$SITE" --out "$case_out" "${runner_scope_args[@]}"
              --timeout "$TIMEOUT")
        {
          printf '\n=== RUN %s (%s/%s/%s%%/r%s) ===\n' \
            "$run_id" "$system" "$app" "$ratio" "$rep"
        } >> "$OUT/batch.log"
        set +e
        "$PYTHON_BIN" "$RUN_CASE_PY" "${args[@]}" \
          >"$runner_log" 2>&1
        runner_rc=$?
        set -e
        cat "$runner_log" >> "$OUT/batch.log"
        printf '=== END RUN %s (runner_exit=%d) ===\n' "$run_id" "$runner_rc" \
          >> "$OUT/batch.log"

        # The shell form is the control-plane API. Detailed JSON is retained
        # in batch.log for auditability and never dumped into the terminal.
        set +e
        status_line=$("$PYTHON_BIN" "$BATCH_STATUS_PY" \
          --run-dir "$case_out" --exit-status "$runner_rc" --format shell 2>>"$OUT/batch.log")
        status_rc=$?
        set -e
        status=error
        safe=0
        if ((status_rc == 0)); then
          read -r status safe <<< "$status_line" || {
            status=error
            safe=0
            status_rc=1
          }
        fi
        set +e
        details=$("$PYTHON_BIN" "$BATCH_STATUS_PY" \
          --run-dir "$case_out" --exit-status "$runner_rc" --format json 2>&1)
        details_rc=$?
        set -e
        printf 'BATCH_STATUS %s: %s\n' "$run_id" "$details" >> "$OUT/batch.log"
        reason=""
        data_complete=0
        data_reason="data completeness classifier failed"
        if ((details_rc == 0)); then
          reason=$(printf '%s\n' "$details" | "$PYTHON_BIN" -c \
            'import json,sys; print(json.load(sys.stdin).get("reason", ""))' 2>/dev/null) || reason=""
          data_complete=$(printf '%s\n' "$details" | "$PYTHON_BIN" -c \
            'import json,sys; print(int(json.load(sys.stdin).get("data_complete") is True))' 2>/dev/null) || data_complete=0
          data_reason=$(printf '%s\n' "$details" | "$PYTHON_BIN" -c \
            'import json,sys; print(json.load(sys.stdin).get("data_complete_reason", "missing completeness verdict"))' 2>/dev/null) || data_reason="data completeness classifier failed"
        fi
        if ((status_rc != 0 || details_rc != 0)); then
          status=error
          safe=0
          reason="batch status classifier failed"
        fi

        case "$status" in
          passed)
            passed=$((passed + 1))
            printf '%s %s %s%% finish\n' "${system^^}" "${app^^}" "$ratio"
            ;;
          timeout)
            timeouts=$((timeouts + 1))
            app_failures=$((app_failures + 1))
            reason=${reason:-client timeout}
            printf '%s %s %s%% WARNING: %s\n' \
              "${system^^}" "${app^^}" "$ratio" "$reason"
            ;;
          warning)
            usable_warnings=$((usable_warnings + 1))
            app_failures=$((app_failures + 1))
            printf '%s %s %s%% WARNING: %s\n' \
              "${system^^}" "${app^^}" "$ratio" "$reason"
            ;;
          *)
            errors=$((errors + 1))
            app_failures=$((app_failures + 1))
            reason=${reason:-runner exit $runner_rc without a verified result}
            printf '%s %s %s%% ERROR: %s\n' \
              "${system^^}" "${app^^}" "$ratio" "$reason"
            ;;
        esac
        if [[ "$data_complete" = 1 ]]; then
          data_complete_cases=$((data_complete_cases + 1))
          printf 'DATA_COMPLETE %s: %s\n' "$run_id" "$data_reason"
        else
          data_incomplete_cases=$((data_incomplete_cases + 1))
          [[ "$status" != "passed" ]] || app_failures=$((app_failures + 1))
          printf 'DATA_INCOMPLETE %s: %s (Figure 9 usability is reported separately)\n' \
            "$run_id" "$data_reason"
        fi
        if [[ "$safe" != 1 ]]; then
          unsafe_cleanup=$((unsafe_cleanup + 1))
          stop_batch=1
          printf 'ERROR: stopping Figure 9 batch after %s; cleanup is unsafe (%s)\n' \
            "$run_id" "${reason:-cleanup evidence unavailable}" >&2
          break
        fi
        if ((STOP_ON_ERROR)) && [[ "$status" != "passed" || "$data_complete" != 1 ]]; then
          stop_batch=1
          printf 'ERROR: stopping Figure 9 batch after %s (--stop-on-error)\n' \
            "$run_id" >&2
          break
        fi
      done
      ((stop_batch)) && break
    done
    system_failures=$((system_failures + app_failures))
    if ((stop_batch)); then
      printf '%s App %s ABORTED! (failures=%d)\n' \
        "${system^^}" "${app^^}" "$app_failures"
    else
      printf '%s App %s FINISH! (failures=%d)\n' \
        "${system^^}" "${app^^}" "$app_failures"
    fi
    ((stop_batch)) && break
  done
  if ((stop_batch)); then
    printf '%s ABORTED! (failures=%d)\n' "${system^^}" "$system_failures"
  else
    printf '%s FINISH! (failures=%d)\n' "${system^^}" "$system_failures"
  fi
  ((stop_batch)) && break
done

postprocess_failures=0
set +e
"$PYTHON_BIN" "$COLLECT_PY" --runs-dir "$OUT/runs" --output "$OUT/figure9.csv" \
  >"$OUT/logs/collect.log" 2>&1
collect_rc=$?
set -e
cat "$OUT/logs/collect.log" >> "$OUT/batch.log"
if ((collect_rc != 0)); then
  postprocess_failures=$((postprocess_failures + 1))
  echo "ERROR: Figure 9 collection failed (see $OUT/logs/collect.log)" >&2
else
  plot_args=(--input "$OUT/figure9.csv" --output-dir "$OUT/figure"
             --ratios "${ratio_list[@]}")
  plot_workloads=()
  for app in "${app_list[@]}"; do
    plot_workloads+=("$(plot_workload "$app")")
  done
  plot_systems=()
  declare -A seen_plot_systems=()
  for system in "${system_list[@]}"; do
    if plotted=$(plot_system "$system"); then
      if [[ ! ${seen_plot_systems[$plotted]+present} ]]; then
        plot_systems+=("$plotted")
        seen_plot_systems[$plotted]=1
      fi
    fi
  done
  plot_args+=(--workloads "${plot_workloads[@]}"
              --systems "${plot_systems[@]}")
  set +e
  "$PLOT_PYTHON" "$PLOT_PY" "${plot_args[@]}" \
    >"$OUT/logs/plot.log" 2>&1
  plot_rc=$?
  set -e
  cat "$OUT/logs/plot.log" >> "$OUT/batch.log"
  if ((plot_rc != 0)); then
    postprocess_failures=$((postprocess_failures + 1))
    echo "ERROR: Figure 9 plotting failed (see $OUT/logs/plot.log)" >&2
  fi
fi

# A batch publishes into this clone's own data/ directory. Different clones
# have different ROOT values; remote case paths are additionally hashed from
# the resolved local batch root by run_case.py.
if ((collect_rc == 0)) && [[ -f "$OUT/figure9.csv" ]] &&
   (( $(wc -l < "$OUT/figure9.csv") > 1 )); then
  mkdir -p "$ROOT/data"
  published_tmp=$(mktemp "$ROOT/data/.figure9.csv.XXXXXX")
  if cp -- "$OUT/figure9.csv" "$published_tmp" &&
       mv -f -- "$published_tmp" "$ROOT/data/figure9.csv"; then
    echo "latest Figure 9 CSV: $ROOT/data/figure9.csv"
  else
    rm -f -- "$published_tmp"
    postprocess_failures=$((postprocess_failures + 1))
    echo "ERROR: could not publish $ROOT/data/figure9.csv" >&2
  fi
else
  echo "no successful measurements; data/figure9.csv left unchanged" >&2
fi

printf 'Figure 9 summary: passed=%d timeouts=%d errors=%d usable_warnings=%d unsafe_cleanup=%d postprocess_failures=%d\n' \
  "$passed" "$timeouts" "$errors" "$usable_warnings" "$unsafe_cleanup" "$postprocess_failures"
echo "figure9 batch: $OUT (execution_failed_or_warning_cases=$((timeouts + errors + usable_warnings)))"
printf 'Raw data summary: complete=%d incomplete=%d; normalized Figure 11 additionally requires matching NonFT-backup-off baselines.\n' \
  "$data_complete_cases" "$data_incomplete_cases"
if ((timeouts + errors + usable_warnings + unsafe_cleanup + postprocess_failures + data_incomplete_cases > 0)); then
  exit 1
fi
exit 0
