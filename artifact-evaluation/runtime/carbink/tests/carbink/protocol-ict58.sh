#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = cpuserver58
run=$(realpath "$1")
case "$run" in /mnt/nfs/xiayanwen/research/carbink-compaction-20260928/runs/*) ;; *) exit 2 ;; esac
date -Is >"$run/client-start.txt"
set +e
(
set -x
timeout -k 10 120 env FARLIB_PIN_RDMA_THREAD=0 FARLIB_RDMA_READ_BATCH=1 \
    numactl --physcpubind=0 --membind=0 \
    "$run/bin/test_carbink_remote_compact" "$run/client.config" \
    >"$run/client.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$run/client-timeout.pid"
wait "$pid"
) 2>"$run/client-command.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$run/client.exit"
date -Is >"$run/client-end.txt"
tail -24 "$run/client.log"
test "$rc" -eq 0
