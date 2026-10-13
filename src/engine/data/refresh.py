"""Weekly observation refresh: detect a completed week, capture, enrich, build, publish.

Run daily: an unchanged week only checks table recipes and retries an unfinished
upload. Use --force to rebuild the same week after provider corrections. An
interrupted run resumes with the same --version. After the tables publish, the
campaign scorers are asked to add rolling-origin forecasts for the new week.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from engine.config.settings import REPO_ROOT, get_settings
from engine.data.capture import capture_raw, completed_week, default_season, latest_snapshot
from engine.data.enrich import enrich
from engine.data.pipeline import build_tables
from engine.data.releases import atomic_json, identifier, load_tables
from engine.tables.registry import read_registry
from engine.tables.transport import DEFAULT_STORE
from engine.tables.workflow import refresh_and_publish as refresh_named_tables

ROOT = REPO_ROOT
RESCORE = ["-m", "experiments.future_player_lab.campaigns.rescore", "--due"]


def refresh_local(*, rescore: bool = True) -> None:
    """Run the shared refresh outside API query scopes and model runtime state."""
    command = [sys.executable, "-m", "engine.data.pipeline", "refresh"]
    if not rescore:
        command.append("--no-rescore")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "POLARS_MAX_THREADS": "4",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = " | ".join((result.stderr or result.stdout).strip().splitlines()[-3:])
        raise RuntimeError(f"NFL data and forecast refresh failed: {detail}")


def needs_refresh(published: dict, season: int, cutoff: int, *, force: bool = False) -> bool:
    old = (published["season"], published["through_week"])
    new = (season, cutoff)
    if new[0] == old[0] and new[1] < old[1]:
        raise ValueError("Refusing to move the published observation cutoff backward")
    return force or new > old


def finish_tables(data: Path, args) -> dict:
    """Same build/upload path as engine tables refresh --upload, every refreshable recipe."""
    names = [name for name, spec in read_registry().items() if spec.refresh_hours is not None]
    return refresh_named_tables(
        data,
        names=names,
        upload=getattr(args, "upload", False),
        store=getattr(args, "store", DEFAULT_STORE),
    )


def rescore_campaigns(log_path: Path) -> dict:
    """Ask the campaign scorers for the new week; a failure never undoes the tables."""
    with log_path.open("a") as log:
        log.write("\nStage: rescore\n")
        log.flush()
        result = subprocess.run(
            [sys.executable, *RESCORE], cwd=ROOT, stdout=log, stderr=log, check=False
        )
    return {"returncode": result.returncode, "log": str(log_path)}


def finish_forecasts(state: dict, args, log_path: Path) -> None:
    if not getattr(args, "no_rescore", False):
        state["rescore"] = rescore_campaigns(log_path)
        if state["rescore"]["returncode"]:
            raise RuntimeError(
                f"Observation tables are current, but forecast generation failed. See {log_path}"
            )


def run(args):
    data = get_settings().data_dir.resolve()
    runtime = data / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    status_path = runtime / "observation_refresh.json"
    with (runtime / "observation_refresh.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An observation refresh is already running") from exc
        previous = latest_snapshot(data)
        source = load_tables(data)
        published = source.manifest["current_observations"]
        season = getattr(args, "season", None) or default_season()
        college = getattr(args, "college_source", None) or (previous or {}).get("college_source")
        if not college:
            raise ValueError("A college source capture is required for the first build")
        version = identifier(args.version or datetime.now(UTC).strftime("weekly_%Y%m%d_%H%M%S"))
        state = dict(version=version, status="checking", started_at=datetime.now(UTC).isoformat())
        atomic_json(status_path, state)
        log_path = runtime / f"{version}.log"
        state["log"] = str(log_path)
        try:
            batch = data / "tables/batches" / version / "manifest.json"
            if batch.exists():
                cutoff = load_tables(data, version).manifest["current_observations"]["through_week"]
            else:
                cutoff = completed_week(season)
                if not needs_refresh(published, season, cutoff, force=args.force):
                    state["tables"] = finish_tables(data, args)
                    state.update(status="current", season=season, through_week=cutoff)
                    finish_forecasts(state, args, log_path)
                    state.update(completed_at=datetime.now(UTC).isoformat())
                    atomic_json(status_path, state)
                    return state
                state.update(status="running", stage="capture", season=season, through_week=cutoff)
                atomic_json(status_path, state)
                if not (data / "raw/snapshots" / version / "manifest.json").exists():
                    capture_raw(data, version, season, college, previous=previous)
                state.update(stage="enrich")
                atomic_json(status_path, state)
                if not (data / "enriched/releases" / version / "manifest.json").exists():
                    enrich(data, version)
                state.update(stage="tables")
                atomic_json(status_path, state)
                build_tables(data, version)
            if load_tables(data).ref["version"] != version:
                raise ValueError("The new batch is not the latest accepted batch; aborting")
            state["tables"] = finish_tables(data, args)
            state.update(status="published", through_week=cutoff)
            atomic_json(status_path, state)
            finish_forecasts(state, args, log_path)
            state.update(completed_at=datetime.now(UTC).isoformat())
            atomic_json(status_path, state)
            return state
        except Exception as exc:
            state.update(
                status="failed", error=str(exc), completed_at=datetime.now(UTC).isoformat()
            )
            atomic_json(status_path, state)
            raise
