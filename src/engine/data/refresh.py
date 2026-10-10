"""Weekly observation refresh: capture the completed week, build a table batch, publish tables.

Run daily: an unchanged week is a cheap no-op that still checks table recipes and
retries an unfinished upload. Use --force for within-week source corrections.
An interrupted run resumes completed stages with the same --version.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from engine.config.settings import REPO_ROOT, get_settings
from engine.data.releases import atomic_json, digest, identifier, load_manifest, load_tables
from engine.data.weekly import capture
from engine.tables.registry import read_registry
from engine.tables.transport import DEFAULT_STORE
from engine.tables.workflow import refresh_and_publish as refresh_named_tables

ROOT = REPO_ROOT


def needs_refresh(published: dict, captured: dict, *, force: bool = False) -> bool:
    old = (published["season"], published["through_week"])
    new = (captured["season"], captured["through_week"])
    if new < old:
        raise ValueError("Refusing to move the published observation cutoff backward")
    if new[0] != old[0]:
        raise ValueError("A new season requires updated historical evidence first")
    return force or new > old


def build_command(version: str, history: str, college: str, report: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "engine.data.pipeline",
        "build",
        "--version",
        version,
        "--history",
        history,
        "--college-source",
        college,
        "--current-report",
        str(report),
    ]


def finish_tables(data: Path, args) -> dict:
    """Same build/upload path as engine tables refresh --upload, every refreshable recipe."""
    names = [name for name, spec in read_registry().items() if spec.refresh_hours is not None]
    return refresh_named_tables(
        data,
        names=names,
        upload=getattr(args, "upload", False),
        store=getattr(args, "store", DEFAULT_STORE),
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
        source = load_tables(data)
        published = source.manifest["current_observations"]
        # Avoid a warm provider cache hiding Monday's final game. Only this process
        # and its children change cache duration; preserved research inputs stay frozen.
        os.environ["NFLVERSE_CACHE_DURATION"] = "0"
        get_settings.cache_clear()
        version = identifier(args.version or datetime.now(UTC).strftime("weekly_%Y%m%d_%H%M%S"))
        plan_path = runtime / f"{version}.json"
        state = dict(version=version, status="checking", started_at=datetime.now(UTC).isoformat())
        atomic_json(status_path, state)
        try:
            if plan_path.exists():
                plan = json.loads(plan_path.read_text())
                report = Path(plan["report"])
                if digest(report) != plan["report_sha256"]:
                    raise ValueError("Captured input report changed during the run")
            else:
                report = args.current_report or capture(
                    published["season"], after_week=published["through_week"], force=args.force
                )
                if report is None or not needs_refresh(
                    published, json.loads(report.read_text()), force=args.force
                ):
                    state["tables"] = finish_tables(data, args)
                    state.update(status="current", through_week=published["through_week"])
                    atomic_json(status_path, state)
                    return state
                _, enriched = load_manifest(data, "enriched", source.manifest["input"])
                _, raw = load_manifest(data, "raw", enriched["input"])
                plan = dict(
                    report=str(report.resolve()),
                    report_sha256=digest(report),
                    history=source.manifest["history"]["version"],
                    college=raw["college_source"],
                    previous_batch=source.ref,
                )
                atomic_json(plan_path, plan)
            captured = json.loads(report.read_text())
            state.update(through_week=captured["through_week"], season=captured["season"])
            log_path = runtime / f"{version}.log"
            state["log"] = str(log_path)
            manifest = data / "tables/batches" / version / "manifest.json"
            if manifest.exists():
                staged = load_tables(data, version)
                if staged.manifest["current_observations"] != {
                    k: captured[k] for k in ("season", "through_week", "saved_at")
                }:
                    raise ValueError("Staged table batch has different captured observations")
            else:
                state.update(status="running", stage="tables")
                atomic_json(status_path, state)
                print(f"tables: {version}; log: {log_path}", flush=True)
                with log_path.open("a") as log:
                    log.write("\nStage: tables\n")
                    log.flush()
                    subprocess.run(
                        build_command(version, plan["history"], plan["college"], report),
                        cwd=ROOT,
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
            if load_tables(data).ref["version"] != version:
                raise ValueError("The new batch is not the latest accepted batch; aborting")
            state["tables"] = finish_tables(data, args)
            state.update(status="published", completed_at=datetime.now(UTC).isoformat())
            atomic_json(status_path, state)
            return state
        except Exception as exc:
            state.update(
                status="failed", error=str(exc), completed_at=datetime.now(UTC).isoformat()
            )
            atomic_json(status_path, state)
            raise
