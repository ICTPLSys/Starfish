#!/usr/bin/env bash
set -euo pipefail

# Save reproducibility facts for a build or run.  This intentionally records
# a small allow-list of variables and never dumps the complete environment,
# which could contain credentials or unrelated machine configuration.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SYSTEM="${STARFISH_SYSTEM:-unknown}"
RUNTIME_DIR=""
BUILD_DIR=""
OUTPUT=""

usage() {
  cat <<'EOF'
Usage: collect_environment.sh [options]

Options:
  --system NAME       runtime variant label
  --runtime-dir DIR   runtime source directory
  --build-dir DIR     build directory
  --output FILE       write the snapshot to FILE (default: stdout)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) SYSTEM=$2; shift 2 ;;
    --runtime-dir) RUNTIME_DIR=$2; shift 2 ;;
    --build-dir) BUILD_DIR=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUNTIME_DIR" && -f "$ROOT/runtime/$SYSTEM/CMakeLists.txt" ]]; then
  RUNTIME_DIR="$ROOT/runtime/$SYSTEM"
fi

write_snapshot() {
  export LC_ALL=C
  printf '%s\n' '--- artifact-evaluation environment ---'
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'system=%s\n' "$SYSTEM"
  printf 'runtime_dir=%s\n' "${RUNTIME_DIR:-unset}"
  printf 'build_dir=%s\n' "${BUILD_DIR:-unset}"
  printf '\n[os]\n'
  uname -a 2>/dev/null || true
  if [[ -r /etc/os-release ]]; then
    sed -n '1,8p' /etc/os-release
  fi
  printf '\n[hardware]\n'
  command -v lscpu >/dev/null 2>&1 && lscpu | grep -E '^(Architecture|CPU\(s\)|Model name|NUMA node|Thread)' || true
  command -v free >/dev/null 2>&1 && free -h || true
  command -v df >/dev/null 2>&1 && df -h "$ROOT" | tail -1 || true
  printf '\n[toolchain]\n'
  command -v cmake >/dev/null 2>&1 && cmake --version | head -1 || true
  command -v ninja >/dev/null 2>&1 && ninja --version || true
  if [[ -n "${CC:-}" ]] && command -v "$CC" >/dev/null 2>&1; then "$CC" --version | head -1; elif command -v cc >/dev/null 2>&1; then cc --version | head -1; fi
  if [[ -n "${CXX:-}" ]] && command -v "$CXX" >/dev/null 2>&1; then "$CXX" --version | head -1; elif command -v c++ >/dev/null 2>&1; then c++ --version | head -1; fi
  command -v python3 >/dev/null 2>&1 && python3 --version || true
  printf '\n[rdma]\n'
  command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo -l || true
  command -v ibdev2netdev >/dev/null 2>&1 && ibdev2netdev || true
  command -v ip >/dev/null 2>&1 && ip -br addr || true
  printf '\n[libfibre]\n'
  printf 'LIBFIBRE_DIR=%s\n' "${LIBFIBRE_DIR:-unset}"
  if [[ -n "${LIBFIBRE_DIR:-}" ]]; then
    git -C "$LIBFIBRE_DIR" rev-parse HEAD 2>/dev/null || true
    find "$LIBFIBRE_DIR" -maxdepth 3 -type f \( -name 'libfibre.so*' -o -name 'Fibre.h' \) -print 2>/dev/null | sort || true
  fi
  printf '\n[hdr_histogram]\n'
  printf 'HDR_HISTOGRAM_PREFIX=%s\n' "${HDR_HISTOGRAM_PREFIX:-unset}"
  if [[ -n "${HDR_HISTOGRAM_PREFIX:-}" ]]; then
    grep -h '^set(PACKAGE_VERSION ' \
      "$HDR_HISTOGRAM_PREFIX"/lib*/cmake/hdr_histogram/hdr_histogram-config-version.cmake \
      2>/dev/null || true
  fi
  printf '\n[ae_variables]\n'
  env | sort | grep -E '^(FARLIB_|Fibre|GAPBS_|STARFISH_(SYSTEM|BUILD_DIR|SERVER_|REMOTE_|MODEL|DATASET|IB_DEVICE|WORKERS|CPUSET|TIMEOUT|BFS_))=' || true
  printf '\n[source]\n'
  if command -v git >/dev/null 2>&1 &&
     git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null || true
    git -C "$ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$ROOT" status --short 2>/dev/null | head -40 || true
  fi
}

if [[ -n "$OUTPUT" ]]; then
  mkdir -p "$(dirname "$OUTPUT")"
  write_snapshot > "$OUTPUT"
  printf 'environment snapshot: %s\n' "$OUTPUT"
else
  write_snapshot
fi
