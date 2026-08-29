# Patron Saints analytics engine.
# Every recipe is a plain command — run them directly if you'd rather not install just.

default:
    @just --list

# Install Python and Node dependencies.
setup:
    uv sync
    cd web && npm install

# Fast offline unit suite. Must pass without network access.
test:
    uv run pytest

# Everything, including the network-marked schema, parity, and end-to-end tests.
test-all:
    uv run pytest -m "network or not network"

# Only the tests that hit live nflverse.
test-network:
    uv run pytest -m network

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

typecheck:
    uv run mypy

# Sign in to ESPN. Opens Chrome; sign in there and it captures the session.
auth-login *ARGS:
    uv run patron auth login {{ARGS}}

# Check the stored ESPN session still works.
auth-status:
    uv run patron auth status

# Build the draft board into data/outputs/.
board *ARGS:
    uv run patron board {{ARGS}}

# Rebuild the rolling v2 metric backtest and frontend report.
metric-report *ARGS:
    uv run patron metric-report {{ARGS}}

# Rebuild ignoring every cache.
rebuild:
    uv run patron board --force

# Compare a fresh build against the published 2026 draft board.
validate:
    uv run patron validate

# Drop cached nflverse downloads and derived artifacts.
refresh:
    uv run patron refresh

# Serve the API at :8000. It owns ESPN syncing; the frontend reads from it.
api:
    uv run patron serve --reload

# React dev server at :5173, proxying /api to the backend.
web:
    cd web && npm run dev

# Backend and frontend together.
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    uv run patron serve --reload &
    (cd web && npm run dev) &
    wait

# Everything CI would run.
check: lint typecheck test
