#!/usr/bin/env bash
# Existing parallel Mark2/Evict4 with cursor-based, worker-owned batches.
set -euo pipefail
test "$(hostname)" = cpuserver58
run=$(realpath "$1")
case "$run" in /mnt/nfs/xiayanwen/research/carbink-compaction-20260928/runs/*) ;; *) exit 2;; esac
export FARLIB_OPT_READY_QUEUE=0
export FARLIB_OPT_SERIALIZE_MARK_EVICT=0
export FARLIB_OPT_FULL_PRIME_MARK=1
export FARLIB_OPT_LEGACY_EXCLUSIVE_PIPELINE=1
export FARLIB_EXCLUSIVE_OWNED_BATCH=1
export FARLIB_LEGACY_SCAN_CURSORS=1
# Carbink currently retains the inherited64-region batch and notifyOFF.
env | LC_ALL=C sort | grep -E '^(FARLIB_OPT_|FARLIB_EXCLUSIVE_OWNED_BATCH=|FARLIB_LEGACY_SCAN_CURSORS=)' >"$run/evacuator-policy.txt"
runner=$(dirname "$(realpath "$0")")/object-verify-ict58.sh
exec bash "$runner" "$@"
