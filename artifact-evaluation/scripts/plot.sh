#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:---help}" in
  figure9) shift; exec bash "$HERE/figure9/plot.sh" "$@" ;;
  figure10) shift; exec bash "$HERE/figure10/plot.sh" "$@" ;;
  figure11) shift; exec bash "$HERE/figure11/plot.sh" "$@" ;;
  figure12) shift; exec bash "$HERE/figure12/plot.sh" "$@" ;;
  figure13) shift; exec bash "$HERE/figure13/plot.sh" "$@" ;;
  appendix) shift; exec bash "$HERE/appendix/plot.sh" "$@" ;;
  -h|--help)
    echo 'Usage: scripts/plot.sh {figure9|figure10|figure11|figure12|figure13|appendix} [figure-specific CSV options]'
    echo 'Use scripts/plot.sh FIGURE --help for figure-specific options.' ;;
  *) echo "unknown figure: $1 (available: figure9, figure10, figure11, figure12, figure13, appendix)" >&2; exit 2 ;;
esac
