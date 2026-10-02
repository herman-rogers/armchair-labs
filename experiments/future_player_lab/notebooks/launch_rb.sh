#!/usr/bin/env bash
set -euo pipefail
RB_NOTEBOOK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MARIMO_NOTEBOOK="$RB_NOTEBOOK_DIR/rb_stats.py"
export MARIMO_PORT="${MARIMO_PORT:-2719}"
exec bash "$RB_NOTEBOOK_DIR/launch.sh" "$@"
