# Armchair Labs analytics engine.
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
    uv run engine auth login {{ARGS}}

# Check the stored ESPN session still works.
auth-status:
    uv run engine auth status

# Build the draft board into data/outputs/.
board *ARGS:
    uv run engine board {{ARGS}}

# Rebuild the rolling v2 metric backtest and frontend report.
metric-report *ARGS:
    uv run engine metric-report {{ARGS}}

# Run report-only automated feature discovery on the retained historical folds.
feature-discovery *ARGS:
    uv run engine feature-discovery {{ARGS}}

# Refresh rookie production, historical analogs, and walk-forward checks.
rookie-watch history *ARGS:
    uv run python research/rookie_watch.py --history {{history}} {{ARGS}}

# Build a new immutable outlook version from repaired history and a captured NFL snapshot.
player-outlook history current_report version:
    uv run python research/player_outlook.py --history {{history}} --current-report {{current_report}} --version {{version}}

# Pin and capture the longest public college player-stat history (no credentials).
college-capture version through_season:
    uv run python research/capture_college.py --version {{version}} --through-season {{through_season}}

# Link college careers to audited NFL outcomes; publish research without changing boards.
college-translation source history version:
    uv run python research/college_translation.py --source {{source}} --history {{history}} --version {{version}}

# Build unified player histories from the accepted college and current outlook releases.
player-profiles version season="2026":
    uv run python research/player_profiles.py --version {{version}} --season {{season}}

# Preserve captured sources, replay enrichment, and validate a create-only table batch.
data-build *ARGS:
    uv run engine data build {{ARGS}}

# Build downstream products against one table batch, without publishing the catalog.
data-products version prefix:
    uv run engine data products --version {{version}} --prefix {{prefix}}

# Validate the complete data/product set and atomically make it current.
data-publish version prefix:
    uv run engine data publish --version {{version}} --prefix {{prefix}}

# Verify a table batch and every raw/enriched dependency.
data-verify version:
    uv run engine data verify --version {{version}}

# Full generation → validation → table build → optional GCS upload. Use --upload on the publisher.
data-refresh *ARGS:
    uv run --extra shared-data engine data refresh {{ARGS}}

# Compatibility alias for existing automation; same implementation as data-refresh.
nextgen-refresh *ARGS:
    @just data-refresh {{ARGS}}

# Fit cutoff-safe ranking candidates; retains every historical evaluation and decision.
nextgen-rankings version *ARGS:
    uv run python research/nextgen_rankings.py --version {{version}} {{ARGS}}

# Verify and publish a new delivery release, preserving the source research run.
nextgen-rankings-publish source version:
    uv run python research/publish_nextgen_rankings.py --source {{source}} --version {{version}} --publish

# Search QB rates/production and the cross-fitted fantasy bridge on all earlier history.
qb-variations version *ARGS:
    uv run python research/qb_variations.py --version {{version}} {{ARGS}}

# Publish only target/horizon policies that passed their frozen checks.
qb-variations-publish source version *ARGS:
    uv run python research/publish_qb_variations.py --source {{source}} --version {{version}} --publish {{ARGS}}

# Score the frozen deployed QB policies as later completed weeks become available.
qb-variations-score source version:
    uv run python research/score_qb_variations.py --source {{source}} --version {{version}}

# Grade the frozen 2026 prospective forecast against final outcomes (post-season only).
grade-prospective *ARGS:
    uv run engine grade-prospective {{ARGS}}

# Rebuild ignoring every cache.
rebuild:
    uv run engine board --force

# Compare a fresh build against the published 2026 draft board.
validate:
    uv run engine validate

# Drop cached nflverse downloads and derived artifacts.
clear-cache:
    uv run engine clear-cache

# Serve the API at :8000. It owns ESPN syncing; the frontend reads from it.
api:
    uv run engine serve --reload

# React dev server at :5173, proxying /api to the backend.
web:
    cd web && npm run dev

# Backend and frontend together.
dev:
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

# Precompute version-bound API profiles, historical summaries, rankings and comparisons.
data-serving *ARGS:
    uv run python research/build_read_models.py {{ARGS}}

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
