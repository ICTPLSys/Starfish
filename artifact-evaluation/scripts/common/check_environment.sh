#!/usr/bin/env bash
set -euo pipefail

# Check the host before a build or a reviewer run.  This script reports what
# is missing; it does not install packages or change kernel/RDMA settings.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SYSTEM="${STARFISH_SYSTEM:-nonft}"
MODE=build
REPORT=""
STRICT=0
REQUIRE_WORKLOADS=0
failures=0
warnings=0

usage() {
  cat <<'EOF'
Usage: check_environment.sh [options]

Options:
  --system NAME       runtime variant (default: nonft)
  --mode build|run|full  checks needed for the selected phase
  --require-workloads require STARFISH_MODEL and STARFISH_DATASET
  --report FILE       also save the non-secret environment snapshot
  --strict            treat warnings as failures
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) SYSTEM=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --require-workloads) REQUIRE_WORKLOADS=1; shift ;;
    --report) REPORT=$2; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SYSTEM" in nonft|starfish|carbink|hydra) ;; *) echo "unsupported system: $SYSTEM" >&2; exit 2 ;; esac
case "$MODE" in build|run|full) ;; *) echo "--mode must be build, run, or full" >&2; exit 2 ;; esac

pass() { printf '[PASS] %s\n' "$1"; }
warn() { warnings=$((warnings + 1)); printf '[WARN] %s\n' "$1"; }
fail() { failures=$((failures + 1)); printf '[FAIL] %s\n' "$1"; }

need_cmd() {
  local cmd=$1
  if command -v "$cmd" >/dev/null 2>&1; then pass "command: $cmd"; else fail "missing command: $cmd"; fi
}

optional_cmd() {
  local cmd=$1
  if command -v "$cmd" >/dev/null 2>&1; then pass "optional command: $cmd"; else warn "optional command not found: $cmd"; fi
}

need_file() {
  local label=$1
  shift
  local path
  for path in "$@"; do
    [[ -e "$path" ]] && { pass "$label: $path"; return; }
  done
  fail "$label not found (checked: $*)"
}

optional_file() {
  local label=$1
  shift
  local path
  for path in "$@"; do
    [[ -e "$path" ]] && { pass "$label: $path"; return; }
  done
  warn "$label not found (checked: $*)"
}

need_library() {
  local label=$1
  local name=$2
  if (command -v ldconfig >/dev/null 2>&1 &&
      ldconfig -p 2>/dev/null | grep "lib${name}\.so" >/dev/null) \
      || (cd / && find /usr/lib /usr/local/lib /lib /lib64 \
          -name "lib${name}.so*" -print -quit 2>/dev/null | grep . >/dev/null); then
    pass "library: lib${name}"
  else
    fail "$label library lib${name} not found"
  fi
}

version_ge() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]]
}

echo "Artifact Evaluation environment check"
echo "root=$ROOT"
echo "system=$SYSTEM mode=$MODE"

[[ "$(uname -s)" = Linux ]] && pass 'Linux host' || fail 'Linux host required (build/run on Windows is unsupported)'
need_cmd bash
need_cmd cmake
need_cmd python3
need_cmd c++
optional_cmd ninja
optional_cmd git

cmake_version=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}' || true)
if [[ -n "$cmake_version" ]] && version_ge "$cmake_version" '3.16.0'; then
  pass "CMake >= 3.16 ($cmake_version)"
else
  fail "CMake >= 3.16 required (found: ${cmake_version:-unknown})"
fi

runtime="$ROOT/runtime/$SYSTEM"
need_file "runtime source" "$runtime/CMakeLists.txt"
need_file "LLaMA application" "$ROOT/apps/llama/CMakeLists.txt" "$ROOT/apps/llama/run_chat_far.cpp"
need_file "BFS application" "$ROOT/apps/bfs/CMakeLists.txt" "$ROOT/apps/bfs/gapbs_bfs_chunked.cpp"

need_file 'RDMA verbs header' /usr/include/infiniband/verbs.h /usr/local/include/infiniband/verbs.h
need_library 'RDMA verbs' ibverbs
need_file 'Boost program_options header' /usr/include/boost/program_options.hpp /usr/local/include/boost/program_options.hpp
need_library 'Boost program_options' boost_program_options
need_file 'OpenSSL header' /usr/include/openssl/ssl.h /usr/local/include/openssl/ssl.h
need_library 'OpenSSL crypto' crypto
optional_file 'PAPI header (optional)' /usr/include/papi.h /usr/local/include/papi.h
optional_file 'ISA-L header (optional)' /usr/include/isa-l.h /usr/local/include/isa-l.h

if [[ -z "${HDR_HISTOGRAM_PREFIX:-}" && -d "$ROOT/deps/hdr-histogram/install" ]]; then
  export HDR_HISTOGRAM_PREFIX="$ROOT/deps/hdr-histogram/install"
fi
if [[ -n "${HDR_HISTOGRAM_PREFIX:-}" ]]; then
  need_file 'HdrHistogram CMake package' \
    "$HDR_HISTOGRAM_PREFIX/lib/cmake/hdr_histogram/hdr_histogram-config.cmake" \
    "$HDR_HISTOGRAM_PREFIX/lib64/cmake/hdr_histogram/hdr_histogram-config.cmake"
  need_file 'HdrHistogram header' "$HDR_HISTOGRAM_PREFIX/include/hdr/hdr_histogram.h"
else
  need_file 'HdrHistogram CMake package' \
    /usr/local/lib/cmake/hdr_histogram/hdr_histogram-config.cmake \
    /usr/lib/x86_64-linux-gnu/cmake/hdr_histogram/hdr_histogram-config.cmake
  need_file 'HdrHistogram header' \
    /usr/include/hdr/hdr_histogram.h /usr/include/hdr_histogram.h \
    /usr/local/include/hdr/hdr_histogram.h /usr/local/include/hdr_histogram.h
fi

if [[ -z "${LIBFIBRE_DIR:-}" && -f "$ROOT/deps/libfibre/src/libfibre.so" ]]; then
  export LIBFIBRE_DIR="$ROOT/deps/libfibre"
fi
if [[ -n "${LIBFIBRE_DIR:-}" ]]; then
  need_file 'libfibre shared library' "$LIBFIBRE_DIR/src/libfibre.so" "$LIBFIBRE_DIR/src/libfibre.so.0" "$LIBFIBRE_DIR/src/libfibre.a"
  need_file 'libfibre header' "$LIBFIBRE_DIR/src/libfibre/Fibre.h"
else
  fail 'libfibre missing: run scripts/common/setup_environment.sh or set LIBFIBRE_DIR'
fi

if [[ "$MODE" = run || "$MODE" = full ]]; then
  need_cmd ssh
  need_cmd scp
  need_cmd ibv_devinfo
  optional_cmd ibdev2netdev
  optional_cmd numactl
  optional_cmd taskset
  optional_cmd ip
  if [[ -z "${STARFISH_SERVER_HOST:-}" && "${STARFISH_NO_SERVER:-0}" != 1 ]]; then
    warn 'STARFISH_SERVER_HOST is unset; a run must provide a server or use STARFISH_NO_SERVER=1'
  fi
fi

if [[ "$REQUIRE_WORKLOADS" = 1 ]]; then
  need_file 'LLaMA model' "${STARFISH_MODEL:-/path/not/set}"
  need_file 'BFS dataset' "${STARFISH_DATASET:-/path/not/set}"
fi

if [[ -n "$REPORT" ]]; then
  "$ROOT/scripts/common/collect_environment.sh" --system "$SYSTEM" \
    --runtime-dir "$runtime" --output "$REPORT"
fi

if [[ "$STRICT" = 1 ]]; then
  failures=$((failures + warnings))
fi
printf 'summary: failures=%s warnings=%s\n' "$failures" "$warnings"
[[ "$failures" = 0 ]]
