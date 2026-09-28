# Starfish recovery integration status

## Source and scope

The integration starts from stable commit
`388fed85e8f55a1cc5d2291c118d278ca56ac4b1`.
The accepted Starfish recovery source is v13 from
`cpuserver58:/home/xiayanwen/research/starfish-kv-recovery-20260930/`.

The v13-to-v6 patch alone is insufficient when starting from the stable
artifact: v6 already contains prerequisites for the recovery RMW fallback,
source-borrow gate and bounded replacement allocation. Integration reconciles
the complete v13 runtime source against the public common baseline
`907c2af7d400a27970c8fa637976fbdbf41c3862`, preserving the stable measurement
hooks and the Figure 12 metadata/EC observers. Historical experiment source,
logs and failed attempts remain unchanged.

The stable Starfish WordCount build definition is retained. Backup credits
remain explicit opt-in (`FARLIB_BACKUP_CREDITS` defaults to false), rather than
adopting the isolated v13 tree's different default. Consequently this merged
build is not identical to the frozen v13 benchmark binary; do not transfer
its throughput or recovery timing to the merged build without a new run.

## Runtime contract

- `ft_background_rebuild` defaults to false.
- `ft_rmw_read_failure_fallback` defaults to false.
- `ft_rebuild_bandwidth_mbps` defaults to 2500 decimal MB/s, independently
  limiting aggregate repair reads and writes.
- The background path requires exclusive EC mode, eager eviction, a standby
  endpoint and the separate background worker cluster.
- Ordinary QP timeout/retry defaults remain 8/7. The accepted historical run
  explicitly used 4/0; this is not silently promoted to the default.
- The implementation is one-shot, single-service-failure recovery per cache.
- Figure 12 metadata collection rejects background-recovery-enabled runs:
  its steady-state allocator accounting does not cover the active rebuild's
  worker-local recovery state. EC instrumentation alone is a separate switch.

This integration does not incorporate Hydra/Carbink recovery or change their
ongoing Figure 13 implementation work.

## Historical v13 evidence (not a run of this merged build)

Paths below are relative to the isolated source root on cpuserver58:

- `evidence/build-v13-manifest.json`
- `evidence/runtime-v13-vs-v6.patch`
- `evidence/ctest-v13.log`: 9/9 historical tests passed.
- `runs/background-fault-v13-fast-r1/recovery-result.json`
- `runs/background-fault-v13-fast-r1/background-summary.json`

The accepted trial completed 1 billion requests and 4096 value checks with
zero failures. It write-completed 11837 stripes, with zero final remaining,
in-flight or failed stripes, and recorded 2695073 foreground reads from
rebuilt data. Rebuild start to finish took 5.195696125 seconds, not less than
5 seconds.

Verification in that run means four survivor reads followed by successful
RDMA WRITE completion. It did not add parity re-encode/readback verification
(`verified_stripes=0`). The injected fault killed one owned service process,
not a physical machine. This is one trial, with no same-build healthy
overhead control.

Older isolated README sections and `evidence/validation-status.json` retain
v6-era descriptions. Use the v13 manifest and accepted run records above for
that version; do not delete the old evidence or reinterpret it as v13 results.
