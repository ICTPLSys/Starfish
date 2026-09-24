#!/usr/bin/env bash
set -euo pipefail

# Configure and build one AE runtime without touching the source tree.
# The build directory is deliberately separate for each system so that a
# reviewer cannot accidentally reuse a CMake cache from another variant.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SYSTEM="${STARFISH_SYSTEM:-nonft}"
RUNTIME_DIR=""
BUILD_DIR=""
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
DRY_RUN=0
ENABLE_PAPI=0
TARGETS=(server run_chat_far gapbs_bfs_chunked)

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --system NAME       runtime variant (default: nonft)
  --runtime-dir DIR   explicit runtime source directory
  --build-dir DIR     explicit CMake build directory
  --jobs N             parallel build jobs
  --targets LIST      comma-separated CMake target names
  --enable-papi       opt in to hardware event profiling (off by default)
  --dry-run            print commands without configuring or building

By default, runtime source is selected from artifact-evaluation/runtime/<name>.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) SYSTEM=$2; shift 2 ;;
    --runtime-dir) RUNTIME_DIR=$2; shift 2 ;;
    --build-dir) BUILD_DIR=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --targets) IFS=',' read -r -a TARGETS <<< "$2"; shift 2 ;;
    --enable-papi) ENABLE_PAPI=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SYSTEM" in
  nonft|starfish|carbink|hydra) ;;
  *) echo "unsupported system: $SYSTEM" >&2; exit 2 ;;
esac

if [[ -z "$RUNTIME_DIR" ]]; then
  RUNTIME_DIR="$ROOT/runtime/$SYSTEM"
fi
RUNTIME_DIR=$(cd "$RUNTIME_DIR" 2>/dev/null && pwd || true)
[[ -n "$RUNTIME_DIR" && -f "$RUNTIME_DIR/CMakeLists.txt" ]] || {
  echo "runtime source not found for system '$SYSTEM'" >&2
  echo "expected: $ROOT/runtime/$SYSTEM/CMakeLists.txt" >&2
  echo "use --runtime-dir to select an explicit checkout" >&2
  exit 2
}

if [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR="$ROOT/build/$SYSTEM"
fi
BUILD_DIR=$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd)

# setup_environment.sh installs these in ignored deps/. Explicit environment
# values take precedence over the bundled dependency paths.
if [[ -z "${LIBFIBRE_DIR:-}" && -f "$ROOT/deps/libfibre/src/libfibre.so" ]]; then
  export LIBFIBRE_DIR="$ROOT/deps/libfibre"
fi
if [[ -z "${HDR_HISTOGRAM_PREFIX:-}" && -d "$ROOT/deps/hdr-histogram/install" ]]; then
  export HDR_HISTOGRAM_PREFIX="$ROOT/deps/hdr-histogram/install"
fi
HDR_OPTIONS=()
if [[ -n "${HDR_HISTOGRAM_PREFIX:-}" ]]; then
  export CMAKE_PREFIX_PATH="$HDR_HISTOGRAM_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  for libdir in lib lib64; do
    hdr_dir="$HDR_HISTOGRAM_PREFIX/$libdir/cmake/hdr_histogram"
    if [[ -f "$hdr_dir/hdr_histogram-config.cmake" ]]; then
      HDR_OPTIONS=(-Dhdr_histogram_DIR:PATH="$hdr_dir")
      break
    fi
  done
  ((${#HDR_OPTIONS[@]})) || {
    echo "HdrHistogram CMake package not found under $HDR_HISTOGRAM_PREFIX" >&2
    exit 2
  }
fi

command -v cmake >/dev/null 2>&1 || { echo 'cmake is required' >&2; exit 2; }
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ -n "${CXX:-}" ]]; then
    command -v "$CXX" >/dev/null 2>&1 || { echo "CXX not found: $CXX" >&2; exit 2; }
  else
    command -v c++ >/dev/null 2>&1 || { echo 'a C++ compiler is required' >&2; exit 2; }
  fi
  [[ -n "${LIBFIBRE_DIR:-}" ]] || {
    echo 'libfibre missing: run scripts/common/setup_environment.sh or set LIBFIBRE_DIR' >&2
    exit 2
  }
fi

if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
else
  GENERATOR=()
fi

if [[ "$ENABLE_PAPI" = 1 ]]; then
  # Unset cached OFF values so CMake can discover PAPI afresh.
  PAPI_OPTIONS=(-UPAPI_INCLUDE_DIR -UPAPI_LIBRARY)
else
  PAPI_OPTIONS=(-DPAPI_INCLUDE_DIR:PATH=OFF -DPAPI_LIBRARY:FILEPATH=OFF)
fi
configure=(cmake -S "$RUNTIME_DIR" -B "$BUILD_DIR" "${GENERATOR[@]}"
  -DCMAKE_BUILD_TYPE=Release -DFARLIB_BUILD_TESTS=OFF
  "${PAPI_OPTIONS[@]}" "${HDR_OPTIONS[@]}")
build=(cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j "$JOBS")

printf 'system: %s\nruntime: %s\nbuild: %s\ntargets: %s\njobs: %s\n' \
  "$SYSTEM" "$RUNTIME_DIR" "$BUILD_DIR" "${TARGETS[*]}" "$JOBS"
printf 'configure:'; printf ' %q' "${configure[@]}"; printf '\n'
printf 'build:'; printf ' %q' "${build[@]}"; printf '\n'

if [[ "$DRY_RUN" = 1 ]]; then
  exit 0
fi

"${configure[@]}"
"${build[@]}"

SNAPSHOT="$BUILD_DIR/environment.txt"
if [[ -x "$ROOT/scripts/common/collect_environment.sh" ]]; then
  "$ROOT/scripts/common/collect_environment.sh" \
    --system "$SYSTEM" --runtime-dir "$RUNTIME_DIR" --build-dir "$BUILD_DIR" \
    --output "$SNAPSHOT" || echo "warning: could not write $SNAPSHOT" >&2
fi
printf 'build: PASS\n'
