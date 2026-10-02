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

# Run report-only automated feature discovery on the retained historical folds.
feature-discovery *ARGS:
    uv run patron feature-discovery {{ARGS}}

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

# Preserve captured sources, replay enrichment, and validate a create-only gold release.
data-build *ARGS:
    uv run python research/data_pipeline.py build {{ARGS}}

# Build downstream products against one gold release, without publishing the catalog.
data-products version prefix:
    uv run python research/data_pipeline.py products --version {{version}} --prefix {{prefix}}

# Validate the complete data/product set and atomically make it current.
data-publish version prefix:
    uv run python research/data_pipeline.py publish --version {{version}} --prefix {{prefix}}

# Verify a gold release and every raw/enriched dependency.
data-verify version:
    uv run python research/data_pipeline.py verify --version {{version}}

# Refresh completed-week observations and all NextGen products; publish after validation.
nextgen-refresh *ARGS:
    uv run python research/refresh_nextgen.py {{ARGS}}

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
    uv run patron grade-prospective {{ARGS}}

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
    uv run patron dev

# Stop supervised services and reclaim project-owned orphans from the old shell recipe.
stop:
    uv run patron dev-stop

# Safely replace any recorded or legacy API/Vite processes, then run in the foreground.
restart:
    uv run patron dev --restart

# Show the supervisor PID, ports, and live children.
dev-status:
    uv run patron dev-status

# Everything CI would run.
check: lint typecheck test

# Precompute version-bound API profiles, historical summaries, rankings and comparisons.
data-serving *ARGS:
    uv run python research/build_read_models.py {{ARGS}}

# Strip embedded Jupyter outputs; marimo source files already contain code only.
notebooks-clean:
    uv run python -m patron.data.notebook_clean

# Fail if versionable notebooks contain outputs or session caches.
notebooks-check:
    uv run python -m patron.data.notebook_clean --check

# Fetch a pinned shared release and restore its inputs to notebook-compatible paths.
data-fetch profile="notebooks" *ARGS:
    uv run --extra shared-data python -m patron.data.shared fetch --profile {{profile}} {{ARGS}}

# Verify the downloaded transport release, without contacting GCS.
data-cache-verify profile="notebooks":
    uv run python -m patron.data.shared verify --offline --profile {{profile}}

# Freeze local inputs into a deduplicated snapshot (does not upload or train).
data-snapshot release:
    uv run python -m patron.data.shared snapshot --release {{release}} --output data/.shared/staging/{{release}}/manifest.json

# Upload immutable objects first, publish the completion manifest last.
data-upload release store:
    uv run --extra shared-data python -m patron.data.shared publish --manifest data/.shared/staging/{{release}}/manifest.json --store {{store}} --reference data/releases/{{release}}.json
