#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd -- "$NOTEBOOK_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$LAB_DIR/../.." && pwd)"
NOTEBOOK_ENV="$LAB_DIR/.notebook-venv"

if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
  echo "Create the repository environment first: uv sync" >&2
  exit 1
fi
if [[ ! -x "$NOTEBOOK_ENV/bin/python" ]]; then
  uv venv --python "$REPO_DIR/.venv/bin/python" "$NOTEBOOK_ENV"
fi
uv pip install --python "$NOTEBOOK_ENV/bin/python" -r "$NOTEBOOK_DIR/requirements.txt"
# Reuse the scientific environment without replacing application dependencies.
"$NOTEBOOK_ENV/bin/python" - "$REPO_DIR/.venv/bin/python" <<'PY'
import subprocess
import sys
import sysconfig
from pathlib import Path
base = subprocess.check_output(
    [sys.argv[1], '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
    text=True,
).strip()
(Path(sysconfig.get_path('purelib')) / 'engine_workspace.pth').write_text(base + '\n')
PY

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd -- "$REPO_DIR"
exec "$NOTEBOOK_ENV/bin/marimo" edit "${MARIMO_NOTEBOOK:-$NOTEBOOK_DIR}" \
  --host 127.0.0.1 --port "${MARIMO_PORT:-2718}" --no-token --no-sandbox \
  --skip-update-check "$@"
