# Armchair Labs deployed image: API + built frontend in one process, and the same image runs the
# refresh/monitor jobs (squanchnet laptop/apps/armchair). Built on the desktop by squanchnet's
# `just app-deploy armchair`. State (data/, experiments/*/runs) is mounted, never baked in.
FROM node:24.22.1-alpine3.24 AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# Python patch version must match the sealed v4 runs (weekly_v4/artifacts.py environment()).
FROM python:3.12.14-slim-bookworm
# libgomp: OpenMP runtime for lightgbm/xgboost/catboost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
# Same path as the desktop checkout: manifests in data/ record absolute provenance paths, and
# REPO_ROOT (engine.config.settings) is derived from the package location.
WORKDIR /home/boomer/dev/armchair-labs
COPY pyproject.toml uv.lock README.md ./
# `research` extra: the Opening Drive campaigns score with xgboost/lightgbm/catboost (CPU).
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra research --no-install-project
COPY src ./src
COPY experiments ./experiments
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra research
COPY --from=web /web/dist ./web/dist
ENV ARMCHAIR_WEB_DIST=/home/boomer/dev/armchair-labs/web/dist HOME=/home/boomer
# DuckDB's icu extension (TimeZone setting) lives in ~/.duckdb; bake it in rather than
# downloading at run time.
RUN chown 1000:1000 /home/boomer
USER 1000:1000
RUN python -c "import duckdb; duckdb.sql('INSTALL icu; LOAD icu; SET TimeZone=\'America/New_York\'')"
EXPOSE 8000
# Serve only: no setup, table restore or rescoring at start (the jobs own those).
CMD ["engine", "serve", "--host", "0.0.0.0", "--port", "8000"]
