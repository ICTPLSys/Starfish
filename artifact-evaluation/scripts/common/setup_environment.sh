#!/usr/bin/env bash
set -euo pipefail

# Prepare build dependencies without silently changing a shared server.
# Source checkouts and installations live under ignored artifact-evaluation/deps/.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEPS="$ROOT/deps"
LIBFIBRE_URL=https://github.com/Crazylqx/libfibre.git
LIBFIBRE_REV=885d74dfb6746966a911ff6c821cc4a88fda3b91
HDR_URL=https://github.com/HdrHistogram/HdrHistogram_c.git
HDR_REV=8dcce8f68512fca460b171bccc3a5afce0048779 # 0.11.8
HDR_VERSION=0.11.8
LIBFIBRE_DIR="$DEPS/libfibre"
HDR_SOURCE="$DEPS/hdr-histogram/src"
HDR_PREFIX="$DEPS/hdr-histogram/install"
EXTERNAL_LIBFIBRE=0
EXTERNAL_HDR=0
INSTALL_APT=0
WITH_PLOT=0
DRY_RUN=0
JOBS=4

APT_PACKAGES=(build-essential cmake ninja-build git pkg-config python3
  libibverbs-dev libboost-program-options-dev libssl-dev zlib1g-dev
  openssh-client ibverbs-utils numactl)

usage() {
  cat <<'EOF'
Usage: setup_environment.sh [options]

Prepare pinned libfibre and HdrHistogram_c under ignored deps/ (no sudo).
The script does not configure RDMA, HugePages, WireGuard, or workload data.

Options:
  --install-apt          explicitly install Ubuntu/Debian packages with apt
  --libfibre-dir DIR     reuse an existing checkout at the pinned revision
  --hdr-prefix DIR       reuse an existing HdrHistogram 0.11.8 installation
  --with-plot            also create .venv-plot and install plot requirements
  --jobs N               maximum build parallelism (default: 4)
  --dry-run              print plan; no downloads, builds, writes, or sudo
  -h, --help             show this help
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

hdr_config() {
  local prefix=$1
  for libdir in lib lib64; do
    local file="$prefix/$libdir/cmake/hdr_histogram/hdr_histogram-config-version.cmake"
    if [[ -f "$file" ]] && grep -Fq "set(PACKAGE_VERSION \"$HDR_VERSION\")" "$file"; then
      printf '%s\n' "$file"
      return 0
    fi
  done
  return 1
}

while (($#)); do
  case "$1" in
    --install-apt) INSTALL_APT=1; shift ;;
    --libfibre-dir) (($# >= 2)) || die 'missing --libfibre-dir value'
      LIBFIBRE_DIR=$2; EXTERNAL_LIBFIBRE=1; shift 2 ;;
    --hdr-prefix) (($# >= 2)) || die 'missing --hdr-prefix value'
      HDR_PREFIX=$2; EXTERNAL_HDR=1; shift 2 ;;
    --with-plot) WITH_PLOT=1; shift ;;
    --jobs) (($# >= 2)) || die 'missing --jobs value'
      JOBS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die '--jobs must be a positive integer'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
  die 'this AE setup requires Linux x86-64'

if ((DRY_RUN)); then
  printf 'apt: %s\n' "$([[ $INSTALL_APT == 1 ]] && printf 'explicit install requested' || printf 'unchanged')"
  printf 'libfibre: %s @ %s -> %s\n' "$LIBFIBRE_URL" "$LIBFIBRE_REV" "$LIBFIBRE_DIR"
  printf 'HdrHistogram_c: %s @ %s -> %s\n' "$HDR_URL" "$HDR_REV" "$HDR_PREFIX"
  printf 'reuse existing libfibre=%s, HDR=%s; build jobs=%s; plot=%s\n' \
    "$EXTERNAL_LIBFIBRE" "$EXTERNAL_HDR" "$JOBS" "$WITH_PLOT"
  printf 'dry-run: no files written, no packages installed, no network access\n'
  exit 0
fi

if ((INSTALL_APT)); then
  command -v apt-get >/dev/null || die '--install-apt requires apt-get'
  if ((EUID == 0)); then APT=(apt-get); else
    command -v sudo >/dev/null || die '--install-apt requires sudo or root'
    APT=(sudo apt-get)
  fi
  "${APT[@]}" update
  "${APT[@]}" install -y "${APT_PACKAGES[@]}"
  if ((WITH_PLOT)); then "${APT[@]}" install -y python3-venv; fi
fi

for cmd in git cmake make c++ cc python3; do
  command -v "$cmd" >/dev/null || die "missing $cmd; install prerequisites (or use --install-apt)"
done
[[ -f /usr/include/infiniband/verbs.h || -f /usr/local/include/infiniband/verbs.h ]] ||
  die 'missing RDMA verbs headers; install libibverbs-dev'
[[ -f /usr/include/boost/program_options.hpp || -f /usr/local/include/boost/program_options.hpp ]] ||
  die 'missing Boost program_options headers; install libboost-program-options-dev'
[[ -f /usr/include/openssl/ssl.h || -f /usr/local/include/openssl/ssl.h ]] ||
  die 'missing OpenSSL headers; install libssl-dev'
[[ -f /usr/include/zlib.h || -f /usr/local/include/zlib.h ]] ||
  die 'missing zlib headers; install zlib1g-dev'

if ((EXTERNAL_HDR)); then
  [[ -d "$HDR_PREFIX" ]] || die "existing HdrHistogram prefix not found: $HDR_PREFIX"
  HDR_PREFIX=$(cd "$HDR_PREFIX" && pwd)
  hdr_config "$HDR_PREFIX" >/dev/null ||
    die "expected HdrHistogram $HDR_VERSION CMake package under $HDR_PREFIX"
fi

if ((EXTERNAL_LIBFIBRE)); then
  [[ -d "$LIBFIBRE_DIR/.git" || -f "$LIBFIBRE_DIR/.git" ]] ||
    die "existing libfibre checkout not found: $LIBFIBRE_DIR"
  LIBFIBRE_DIR=$(cd "$LIBFIBRE_DIR" && pwd)
else
  mkdir -p "$DEPS"
  if [[ ! -d "$LIBFIBRE_DIR/.git" ]]; then
    [[ ! -e "$LIBFIBRE_DIR" ]] || die "refusing to overwrite: $LIBFIBRE_DIR"
    git clone --branch dev-hh --single-branch --no-checkout "$LIBFIBRE_URL" "$LIBFIBRE_DIR"
    git -C "$LIBFIBRE_DIR" checkout --detach "$LIBFIBRE_REV"
  fi
fi
[[ "$(git -C "$LIBFIBRE_DIR" rev-parse HEAD)" == "$LIBFIBRE_REV" ]] ||
  die "libfibre must be at $LIBFIBRE_REV; refusing to change an existing checkout"
if ((EXTERNAL_LIBFIBRE)); then
  expected_submodule=$(git -C "$LIBFIBRE_DIR" rev-parse HEAD:src/errnoname)
  actual_submodule=$(git -C "$LIBFIBRE_DIR/src/errnoname" rev-parse HEAD 2>/dev/null || true)
  [[ "$expected_submodule" == "$actual_submodule" ]] ||
    die 'existing libfibre needs its pinned src/errnoname submodule initialized'
else
  git -C "$LIBFIBRE_DIR" submodule update --init --recursive -- src/errnoname
fi
make -C "$LIBFIBRE_DIR/src" -j "$JOBS" all
[[ -f "$LIBFIBRE_DIR/src/libfibre.so" && -f "$LIBFIBRE_DIR/src/libfibre/Fibre.h" ]] ||
  die 'libfibre build did not produce the expected library and header'

if ((!EXTERNAL_HDR)); then
  if [[ ! -d "$HDR_SOURCE/.git" ]]; then
    [[ ! -e "$HDR_SOURCE" ]] || die "refusing to overwrite: $HDR_SOURCE"
    mkdir -p "$(dirname "$HDR_SOURCE")"
    git clone --branch "$HDR_VERSION" --single-branch --no-checkout "$HDR_URL" "$HDR_SOURCE"
    git -C "$HDR_SOURCE" checkout --detach "$HDR_REV"
  fi
  [[ "$(git -C "$HDR_SOURCE" rev-parse HEAD)" == "$HDR_REV" ]] ||
    die "HdrHistogram source must be at $HDR_REV; refusing to change an existing checkout"
  cmake -S "$HDR_SOURCE" -B "$HDR_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HDR_PREFIX" \
    -DHDR_HISTOGRAM_BUILD_PROGRAMS=OFF -DBUILD_TESTING=OFF
  cmake --build "$HDR_SOURCE/build" -j "$JOBS"
  cmake --install "$HDR_SOURCE/build"
  hdr_config "$HDR_PREFIX" >/dev/null || die 'HdrHistogram installation has no expected CMake package'
fi

if ((WITH_PLOT)); then
  python3 -m venv "$ROOT/.venv-plot" || die 'python3-venv is required for --with-plot'
  "$ROOT/.venv-plot/bin/python" -m pip install -r "$ROOT/scripts/common/requirements-plot.txt"
fi

LIBFIBRE_DIR="$LIBFIBRE_DIR" HDR_HISTOGRAM_PREFIX="$HDR_PREFIX" \
  "$ROOT/scripts/common/check_environment.sh" --system nonft --mode build
printf 'AE dependencies ready. Build with:\n'
printf '  LIBFIBRE_DIR=%q HDR_HISTOGRAM_PREFIX=%q bash scripts/common/build.sh --system nonft\n' \
  "$LIBFIBRE_DIR" "$HDR_PREFIX"
