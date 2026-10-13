# Armchair Labs analytics engine.
# Every recipe is a plain command — run them directly if you'd rather not install just.

default:
    @just --list

# Prepare dependencies, data, query tables, and ESPN without starting services.
setup:
    uv run engine setup

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
    uv run engine auth login {{ARGS}}

# Check the stored ESPN session still works.
auth-status:
    uv run engine auth status

# Pin and capture the longest public college player-stat history (no credentials).
college-capture version through_season:
    uv run python research/capture_college.py --version {{version}} --through-season {{through_season}}

# Preserve captured sources, enrich offline, and validate a create-only observation batch.
data-build *ARGS:
    uv run engine data build {{ARGS}}

# Verify a table batch and every raw/enriched dependency.
data-verify version:
    uv run engine data verify --version {{version}}

# Capture the latest completed week, build the observation batch, publish tables. Use --upload on the publisher.
data-refresh *ARGS:
    uv run --extra shared-data engine data refresh {{ARGS}}

# Drop cached nflverse downloads and derived artifacts.
clear-cache:
    uv run engine clear-cache

# Serve the API at :8000. It owns ESPN syncing; the frontend reads from it.
api:
    uv run engine serve --reload

# React dev server at :5173, proxying /api to the backend.
web:
    cd web && npm run dev

# Prepare the system and start backend and frontend together.
dev:
    uv run engine dev

# Alias for dev: prepare tables, re-score campaigns for the latest week, and start services.
start:
    uv run engine dev

# Stop supervised services and reclaim project-owned orphans from the old shell recipe.
stop:
    uv run engine dev-stop

# Safely replace any recorded or legacy API/Vite processes, then run in the foreground.
restart:
    uv run engine dev --restart

# Show the supervisor PID, ports, and live children.
dev-status:
    uv run engine dev-status

# Everything CI would run.
check: lint typecheck test

# Strip embedded Jupyter outputs; marimo source files already contain code only.
notebooks-clean:
    uv run python -m engine.data.notebook_clean

# Fail if versionable notebooks contain outputs or session caches.
notebooks-check:
    uv run python -m engine.data.notebook_clean --check

# Fetch a pinned shared release and restore its inputs to notebook-compatible paths.
source-fetch profile="notebooks" *ARGS:
    uv run --extra shared-data python -m engine.data.shared fetch --profile {{profile}} {{ARGS}}

# Verify the downloaded transport release, without contacting GCS.
source-verify profile="notebooks":
    uv run python -m engine.data.shared verify --offline --profile {{profile}}

# Freeze local inputs into a deduplicated snapshot (does not upload or train).
source-snapshot release:
    uv run python -m engine.data.shared snapshot --release {{release}} --output data/.shared/staging/{{release}}/manifest.json

# Upload immutable objects first, publish the completion manifest last.
source-upload release store:
    uv run --extra shared-data python -m engine.data.shared publish --manifest data/.shared/staging/{{release}}/manifest.json --store {{store}} --reference data/releases/{{release}}.json

# Named analytical tables: inspect, refresh, query, cache, publish, or fetch.
tables *ARGS:
    uv run --extra shared-data engine tables {{ARGS}}

# Local recipe-only refresh from captured inputs; source acquisition/upload use data-refresh.
tables-refresh:
    uv run engine tables refresh --due

# Fetch the published query catalog; no training/source archives required.
data-fetch *ARGS:
    uv run --extra shared-data engine tables fetch {{ARGS}}

# Inspect and verify ChairStats campaign bundles; every bundle is queryable through the API.
campaigns *ARGS:
    uv run --no-sync engine campaigns {{ARGS}}

# Train/evaluate a campaign in the prepared GPU research environment.
campaign-train *ARGS:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 uv run --no-sync python -m experiments.future_player_lab.campaigns.run {{ARGS}}

# Score saved weights and assemble a new immutable campaign API bundle.
campaign-score *ARGS:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 uv run --no-sync python -m experiments.future_player_lab.campaigns.forecast {{ARGS}}

# Add rolling-origin forecasts for the latest completed week to every campaign that can be scored.
campaign-rescore *ARGS:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 uv run --no-sync python -m experiments.future_player_lab.campaigns.rescore {{ARGS}}

# Reproduce predictions directly from a published campaign bundle, offline.
campaign-replay *ARGS:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 uv run --no-sync python -m experiments.future_player_lab.campaigns.replay {{ARGS}}

# Compare frozen campaigns with external forecasts; separate from fitting/selection.
campaign-benchmark *ARGS:
    uv run --no-sync engine campaigns benchmark {{ARGS}}

# Preserve pre-Opening-Drive QB/RB research in the shared campaign store; no fitting.
campaign-import-scout *ARGS:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 uv run --no-sync python -m experiments.future_player_lab.campaigns.scout_team {{ARGS}}
