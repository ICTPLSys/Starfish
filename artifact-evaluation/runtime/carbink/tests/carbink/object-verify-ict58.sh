#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = cpuserver58
run=$(realpath "$1")
verify_mode=${2:-1}
case "$verify_mode" in 0|1) ;; *) exit 2 ;; esac
case "$run" in /mnt/nfs/xiayanwen/research/carbink-compaction-20260928/runs/*) ;; *) exit 2 ;; esac
date -Is >"$run/client-start.txt"
lscpu -e=CPU,NODE,SOCKET,CORE >"$run/topology-58.txt"
grep -E 'MemAvailable|HugePages|Hugepagesize' /proc/meminfo >"$run/memory-before-58.txt"
set +e
(
set -x
timeout -k 10 600 env \
    FibreCpuSet=0-23 FibreWorkerCount=24 \
    FARLIB_SEPARATE_BACKGROUND_CLUSTER=1 FARLIB_BACKGROUND_CPU_BASE=24 \
    FARLIB_PIN_RDMA_THREAD=0 FARLIB_RDMA_READ_BATCH=1 \
    FARLIB_OBJECT_SIZE_TOTAL_MEMORY_BYTES=17179869184 \
    FARLIB_OBJECT_SIZE_BENCHMARK_SECS=20 FARLIB_OBJECT_SIZE_PHASES=1 \
    FARLIB_OBJECT_SIZE_WARMUP_TIMEOUT_S=120 FARLIB_OBJECT_SIZE_VERIFY="$verify_mode" \
    numactl --physcpubind=0-34 --membind=0 \
    "$run/bin/carbink_object_size" "$run/client.config" "$run/latency-cycles.txt" 0 0 \
    >"$run/client.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$run/client-timeout.pid"
wait "$pid"
) 2>"$run/client-command.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$run/client.exit"
date -Is >"$run/client-end.txt"
tail -40 "$run/client.log"
test "$rc" -eq 0
