#!/usr/bin/env bash
# Run on ictcpu74 only. Own exactly six processes under the supplied run.
set -euo pipefail
test "$(hostname)" = cpuserver74
action=$1
run=$(realpath "$2")
binary=$(realpath "$run/bin/carbink_server")
case "$run" in /mnt/nfs/xiayanwen/research/carbink-compaction-20260928/runs/*) ;; *) exit 2 ;; esac
if test "$action" = start; then
    if ss -ltn | grep -Eq ':1940[0-5][[:space:]]'; then
        echo "Requested port is already occupied; not touching it." >&2
        exit 2
    fi
    for i in 0 1 2 3 4 5; do
        port=$((19400+i))
        nohup env FARLIB_RDMA_DEVICE=mlx5_1 SERVER_PORT="$port" \
            numactl --physcpubind=0-23 --interleave=0,1 \
            "$binary" "$run/server.config" \
            >"$run/server-$i.log" 2>&1 </dev/null &
        printf '%s\n' "$!" >"$run/server-$i.pid"
    done
    date -Is >"$run/server-start.txt"
elif test "$action" = stop; then
    for i in 0 1 2 3 4 5; do
        pid=$(<"$run/server-$i.pid")
        test "$pid" -gt 1
        if test -d "/proc/$pid"; then
            actual=$(readlink "/proc/$pid/exe") || continue
            test "$actual" = "$binary" || { echo "PID identity mismatch: $pid"; exit 2; }
            kill -TERM "$pid"
        fi
    done
    date -Is >"$run/server-stop-request.txt"
    # Legacy RQ_STOP ends the RDMA session but the outer server loop can
    # re-enter blocking TCP accept, where its SIGTERM summary flag is not
    # consumed. Bound cleanup; recheck exact executable identity before KILL.
    for attempt in $(seq 1 20); do
        remaining=0
        for i in 0 1 2 3 4 5; do
            pid=$(<"$run/server-$i.pid")
            if test "$(readlink "/proc/$pid/exe" 2>/dev/null || true)" = "$binary"; then
                remaining=$((remaining+1))
            fi
        done
        test "$remaining" -eq 0 && break
        sleep 0.1
    done
    for i in 0 1 2 3 4 5; do
        pid=$(<"$run/server-$i.pid")
        if test "$(readlink "/proc/$pid/exe" 2>/dev/null || true)" = "$binary"; then
            printf 'force-clean owned server after TERM grace: pid=%s\n' "$pid" >>"$run/cleanup-74.txt"
            kill -KILL "$pid"
        fi
    done
else
    exit 2
fi
