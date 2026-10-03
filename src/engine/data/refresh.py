"""Refresh every NextGen product through the latest completed week, then publish atomically.

Run daily: unchanged weeks are cheap no-ops. Use --force for within-week revisions.
An interrupted run can resume completed stages with the same --version.
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
from engine.data.catalog import publish_catalog
from engine.data.nextgen import load_analysis
from engine.data.releases import (
    atomic_json,
    digest,
    identifier,
    load_manifest,
    load_tables,
    reference,
)
from engine.data.weekly import capture
from engine.tables.registry import read_registry
from engine.tables.transport import DEFAULT_STORE
from engine.tables.workflow import refresh_and_publish as refresh_named_tables

ROOT = REPO_ROOT
DATA = get_settings().data_dir.resolve()


def needs_refresh(published: dict, captured: dict, *, force: bool = False) -> bool:
    old = (published["season"], published["through_week"])
    new = (captured["season"], captured["through_week"])
    if new < old:
        raise ValueError("Refusing to move the published observation cutoff backward")
    if new[0] != old[0]:
        raise ValueError("A new season requires updated historical evidence first")
    return force or new > old


def stages(version: str, history: str, college: str, report: Path, evidence: str):
    def command(script, *args):
        if script == "data_pipeline.py":
            return [sys.executable, "-m", "engine.data.pipeline", *map(str, args)]
        return [sys.executable, str(ROOT / "research" / script), *map(str, args)]

    return [
        (
            "tables",
            command(
                "data_pipeline.py",
                "build",
                "--version",
                version,
                "--history",
                history,
                "--college-source",
                college,
                "--current-report",
                report,
            ),
            [DATA / "tables/batches" / version / "manifest.json"],
        ),
        (
            "products",
            command("data_pipeline.py", "products", "--version", version, "--prefix", version),
            [
                DATA / "research" / f"{version}_{suffix}" / "manifest.json"
                for suffix in ("college", "outlook", "profiles")
            ],
        ),
        (
            "analysis",
            command(
                "nextgen_system.py",
                "--version",
                version + "_base",
                "--gold",
                version,
                "--profiles",
                version + "_profiles",
                "--evidence",
                evidence,
            ),
            [DATA / "research" / (version + "_base") / "manifest.json"],
        ),
        (
            "rankings",
            command(
                "nextgen_rankings.py",
                "--version",
                version + "_rankings",
                "--analysis",
                version + "_base",
            ),
            [DATA / "research" / (version + "_rankings") / "manifest.json"],
        ),
        (
            "passing",
            command(
                "build_qb_passing.py",
                "--version",
                version + "_passing_research",
                "--analysis-version",
                version + "_rankings",
            ),
            [DATA / "research" / (version + "_passing_research") / "manifest.json"],
        ),
        (
            "passing_delivery",
            command(
                "publish_qb_passing.py",
                "--source",
                version + "_passing_research",
                "--version",
                version + "_passing",
                "--analysis-version",
                version + "_rankings",
            ),
            [DATA / "research" / (version + "_passing") / "manifest.json"],
        ),
        (
            "qb_variations",
            command(
                "qb_variations.py",
                "--version",
                version + "_qb_research",
                "--analysis-version",
                version + "_passing",
            ),
            [DATA / "research" / (version + "_qb_research") / "manifest.json"],
        ),
        (
            "delivery",
            command(
                "publish_qb_variations.py",
                "--source",
                version + "_qb_research",
                "--version",
                version + "_delivery",
                "--analysis-version",
                version + "_passing",
            ),
            [DATA / "research" / (version + "_delivery") / "manifest.json"],
        ),
        (
            "weekly_model",
            command(
                "build_weekly_points.py",
                "--version",
                version + "_weekly",
                "--analysis",
                version + "_delivery",
            ),
            [DATA / "research" / (version + "_weekly") / "manifest.json"],
        ),
    ]


def activate_weekly(version, plan):
    if "weekly_model" not in plan["completed"]:
        return
    for path, expected in plan["completed"]["weekly_model"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Completed weekly model changed")
    atomic_json(DATA / "weekly/current.json", reference(DATA / "research" / (version + "_weekly")))


def finish_tables(args):
    """Use the same build/upload path as engine tables refresh --upload.

    Include every refreshable recipe, not a page-specific hardcoded table list.
    Manual/frozen research imports are only rebuilt by explicit table commands.
    """
    names = [name for name, spec in read_registry().items() if spec.refresh_hours is not None]
    return refresh_named_tables(
        DATA,
        names=names,
        upload=getattr(args, "upload", False),
        store=getattr(args, "store", DEFAULT_STORE),
    )


def run(args):
    runtime = DATA / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    status_path = runtime / "nextgen_refresh.json"
    with (runtime / "nextgen_refresh.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A NextGen refresh is already running") from exc
        catalog_path = DATA / "current.json"
        original_catalog = digest(catalog_path)
        catalog = json.loads(catalog_path.read_text())
        source = load_tables(DATA)
        _, analysis = load_analysis(DATA)
        published = source.manifest["current_observations"]
        # Avoid a warm provider cache hiding Monday's final game. Only this process
        # and its children change cache duration; preserved research inputs stay frozen.
        os.environ["NFLVERSE_CACHE_DURATION"] = "0"
        from engine.config.settings import get_settings

        get_settings.cache_clear()
        version = identifier(args.version or datetime.now(UTC).strftime("weekly_%Y%m%d_%H%M%S"))
        plan_path = runtime / f"{version}.json"
        state = dict(version=version, status="checking", started_at=datetime.now(UTC).isoformat())
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        atomic_json(status_path, state)
        try:
            if (
                previous.get("status") == "season_complete"
                and previous.get("catalog_sha256") == original_catalog
                and not args.force
                and not args.current_report
            ):
                previous["tables"] = finish_tables(args)
                atomic_json(status_path, previous)
                return previous
            if plan_path.exists():
                plan = json.loads(plan_path.read_text())
                if plan["catalog_sha256"] != original_catalog:
                    if catalog["products"]["analysis"]["version"] == version + "_delivery":
                        activate_weekly(version, plan)
                        state["tables"] = finish_tables(args)
                        state.update(status="current", through_week=published["through_week"])
                        atomic_json(status_path, state)
                        return state
                    raise ValueError("Catalog changed since this run; start a new version")
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
                    state["tables"] = finish_tables(args)
                    state.update(status="current", through_week=published["through_week"])
                    atomic_json(status_path, state)
                    return state
                captured = json.loads(report.read_text())
                if captured["through_week"] == 18:
                    state["tables"] = finish_tables(args)
                    state.update(
                        status="season_complete",
                        season=captured["season"],
                        through_week=18,
                        capture=str(report.resolve()),
                        catalog_sha256=original_catalog,
                        reason="Final observations captured; no regular-season games remain.",
                    )
                    atomic_json(status_path, state)
                    return state
                _, enriched = load_manifest(DATA, "enriched", source.manifest["input"])
                _, raw = load_manifest(DATA, "raw", enriched["input"])
                plan = dict(
                    catalog_sha256=original_catalog,
                    report=str(report.resolve()),
                    report_sha256=digest(report),
                    history=source.manifest["history"]["version"],
                    college=raw["college_source"],
                    evidence=analysis["evidence"]["version"],
                    completed={},
                )
                atomic_json(plan_path, plan)
            captured = json.loads(report.read_text())
            state.update(through_week=captured["through_week"], season=captured["season"])
            log_path = runtime / f"{version}.log"
            state["log"] = str(log_path)
            for name, command, manifests in stages(
                version, plan["history"], plan["college"], report, plan["evidence"]
            ):
                if name in plan["completed"]:
                    if any(digest(Path(p)) != h for p, h in plan["completed"][name].items()):
                        raise ValueError(f"Completed stage changed: {name}")
                    continue
                state.update(status="running", stage=name)
                atomic_json(status_path, state)
                print(f"{name}: {version}; log: {log_path}", flush=True)
                # A complete externally staged table batch may be adopted after validation.
                if name == "tables" and manifests[0].exists():
                    staged = load_tables(DATA, version)
                    if staged.manifest["current_observations"] != {
                        k: captured[k] for k in ("season", "through_week", "saved_at")
                    }:
                        raise ValueError("Staged table batch has different captured observations")
                else:
                    with log_path.open("a") as log:
                        log.write(f"\nStage: {name}\n")
                        log.flush()
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True)
                plan["completed"][name] = {str(p): digest(p) for p in manifests}
                atomic_json(plan_path, plan)
            if digest(catalog_path) != plan["catalog_sha256"]:
                raise ValueError(
                    "Catalog changed during rebuild; staged results were not published"
                )
            products = {key: version + "_" + key for key in ("college", "profiles")}
            products.update(analysis=version + "_delivery")
            products[f"outlook_{captured['season']}"] = version + "_outlook"
            publish_catalog(DATA, version, products, expected_catalog_sha256=plan["catalog_sha256"])
            activate_weekly(version, plan)
            state["tables"] = finish_tables(args)
            state.update(status="published", completed_at=datetime.now(UTC).isoformat())
            atomic_json(status_path, state)
            return state
        except Exception as exc:
            state.update(
                status="failed", error=str(exc), completed_at=datetime.now(UTC).isoformat()
            )
            atomic_json(status_path, state)
            raise
