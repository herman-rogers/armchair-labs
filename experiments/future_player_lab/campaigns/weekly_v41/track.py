"""Track v4.1: early-week forecast per origin, optional pregame updates.

    python -m experiments.future_player_lab.campaigns.weekly_v41.track [--force] [--pregame]

Early week: if the latest accepted snapshot's origin is not yet published (or
``--force``), run ``live.forecast(..., "early_week")`` and publish.
Pregame: for games not yet kicked off whose both teams have admitted official
injury report captures (collected by the v3.2 monitor), recompute with the
``pregame_60m`` arm, replace those players' weekly rows and rebuild the whole
coherent projection from that players frame, then republish the same origin.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from pathlib import Path

import pandas as pd

from ..weekly_v3.data import ROOT, TEAM_ALIASES
from ..weekly_v31.artifacts import digest, now
from . import live
from .publisher import CAMPAIGN, OUTPUTS, _write, publish, save_outputs

CONFIG = Path(__file__).parent.parent / "configs/opening_drive_v41.json"
RESEARCH_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v41_20261027_001"
REGISTRY = ROOT / "data/campaigns"
SNAPSHOTS = ROOT / "data/raw/snapshots"
OFFICIAL_CAPTURES = ROOT / "data/research/opening_drive_v32_official_captures"


def latest_snapshot(root=SNAPSHOTS):
    """Accepted snapshot with the latest season/through_week, then latest saved_at."""
    best = None
    for manifest in Path(root).glob("*/manifest.json"):
        document = json.loads(manifest.read_text())
        current = document.get("current_observations")
        if document.get("status") != "accepted" or "through_week" not in (current or {}):
            continue
        key = (
            current["season"],
            current["through_week"],
            str(current.get("saved_at", "")),
            str(document.get("registered_at", "")),
        )
        if best is None or key > best[0]:
            best = (key, manifest)
    if best is None:
        raise FileNotFoundError("No accepted snapshot with current observations")
    return best[1]


def research_run():
    if CONFIG.exists():
        configured = json.loads(CONFIG.read_text()).get("research_run")
        if configured:
            return ROOT / configured
    return RESEARCH_RUN


def published(registry, season, origin):
    path = Path(registry) / CAMPAIGN / "manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    weekly = [m for m in manifest["models"] if m["horizon"] == "weekly"]
    return bool(weekly) and all(
        any((f["season"], f["origin_week"]) == (season, origin) for f in m["forecasts"])
        for m in weekly
    )


def _team(value):
    return TEAM_ALIASES.get(value, value)


def official_directories(root=OFFICIAL_CAPTURES):
    return sorted(p for p in Path(root).glob("*") if p.is_dir() and not p.name.startswith("."))


def merge_rows(base, update):
    """Replace whole rows for updated players; keep the base order and population."""
    update = update[update.player_id.isin(base.player_id)]
    merged = pd.concat([base[~base.player_id.isin(update.player_id)], update], ignore_index=True)
    return merged.set_index("player_id").loc[base.player_id].reset_index()


def pregame(snapshot, run, registry, outputs=OUTPUTS, captured_at=None, archive=None):
    document = json.loads(Path(snapshot).read_text())
    season, origin = (document["current_observations"][c] for c in ("season", "through_week"))
    week = origin + 1
    folder = Path(outputs) / "forecasts" / f"{season}_{week:02d}"
    early, late = folder / "early_week", folder / "pregame_60m"
    if not (early / "players.parquet").exists():
        return dict(status="no_early_forecast")
    captured_at = captured_at or now()
    from ..weekly_v32 import live as v32_live

    schedule = v32_live.schedule_times(v32_live.asset(document, "schedule"))
    games = schedule[
        schedule.season.eq(season)
        & schedule.week.eq(week)
        & schedule.game_type.eq("REG")
        & schedule.kickoff.gt(pd.Timestamp(captured_at))
    ]
    directories = official_directories()
    reports, audit = live.official_reports(directories, season, week, captured_at)
    covered = set()
    for item in audit:
        if item["admitted"]:
            summary = json.loads((Path(item["path"]) / "summary.json").read_text())
            covered.update(_team(t) for t in summary.get("teams", []))
    eligible = [
        g
        for g in games.to_dict("records")
        if {_team(g["home_team"]), _team(g["away_team"])} <= covered
    ]
    if not eligible:
        return dict(status="no_eligible_games", covered_teams=sorted(covered))
    teams = {_team(g[c]) for g in eligible for c in ("home_team", "away_team")}
    fingerprint = dict(
        snapshot_sha256=digest(snapshot),
        research_manifest_sha256=digest(Path(run) / "manifest.json"),
        early_sha256=digest(early / "players.parquet"),
        games=sorted(g["game_id"] for g in eligible),
        reports=sorted(i["manifest_sha256"] for i in audit if i["admitted"]),
    )
    state = late / "state.json"
    if state.exists() and json.loads(state.read_text()).get("fingerprint") == fingerprint:
        return dict(status="current", games=fingerprint["games"])
    base = pd.read_parquet(
        (late if (late / "players.parquet").exists() else early) / "players.parquet"
    )
    fresh, _, _ = live.forecast(snapshot, run, "pregame_60m", reports)
    update = fresh[fresh.team.map(_team).isin(teams) & fresh.is_bye.ne(1)]
    players = merge_rows(base, update)
    projection = live.projection(run, players, snapshot)
    save_outputs(players, projection, "pregame_60m", outputs)
    _write(state, dict(updated_at=captured_at.isoformat(), fingerprint=fingerprint))
    kwargs = {"archive": archive} if archive else {}
    publication = publish(registry, projection, players, run, snapshot, **kwargs)
    return dict(
        status="published",
        games=fingerprint["games"],
        updated_players=len(update),
        publication=publication,
    )


def track(snapshot=None, *, force=False, pregame_updates=False, registry=REGISTRY, run=None):
    if CONFIG.exists() and json.loads(CONFIG.read_text()).get("status") != "active":
        return dict(status="not_tracking")
    run = Path(run or research_run())
    if not (run / "manifest.json").exists():
        return dict(status="research_run_missing", research_run=str(run))
    snapshot = Path(snapshot or latest_snapshot())
    document = json.loads(snapshot.read_text())
    season, origin = (document["current_observations"][c] for c in ("season", "through_week"))
    if origin >= 18:
        return dict(status="regular_season_complete")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS / ".tracking.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="already_running")
        early = OUTPUTS / "forecasts" / f"{season}_{origin + 1:02d}" / "early_week"
        result = dict(snapshot=str(snapshot), season=season, origin_week=origin)
        if force or not published(registry, season, origin) or not early.exists():
            players, projection, receipt = live.forecast(snapshot, run, "early_week")
            save_outputs(players, projection, "early_week", OUTPUTS)
            # Pregame rows rebuild from the new early forecast.
            shutil.rmtree(early.parent / "pregame_60m", ignore_errors=True)
            result["early_week"] = publish(registry, projection, players, run, snapshot)
        else:
            result["early_week"] = dict(status="current")
        if pregame_updates:
            result["pregame"] = pregame(snapshot, run, registry)
        return result


def backfill(snapshot=None, *, registry=REGISTRY, run=None, limit=None):
    """Publish missing earlier origins of the current season (see ``campaigns.backfill``)."""
    from .. import backfill as shared

    if CONFIG.exists() and json.loads(CONFIG.read_text()).get("status") != "active":
        return dict(status="not_tracking")
    run = Path(run or research_run())
    if not (run / "manifest.json").exists():
        return dict(status="research_run_missing", research_run=str(run))

    def forecast(derived):
        players, projection, _ = live.forecast(derived, run, "early_week")
        save_outputs(players, projection, "early_week", OUTPUTS)
        return publish(registry, projection, players, run, derived, retrospective=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS / ".tracking.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="already_running")
        return shared.run(
            CAMPAIGN,
            forecast,
            snapshot=Path(snapshot or latest_snapshot()),
            registry=Path(registry),
            limit=limit,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Accepted snapshot manifest; default latest")
    parser.add_argument("--force", action="store_true", help="Recompute a published origin")
    parser.add_argument("--pregame", action="store_true", help="Apply pregame report updates")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--research-run", type=Path)
    args = parser.parse_args(argv)
    result = track(
        args.snapshot,
        force=args.force,
        pregame_updates=args.pregame,
        registry=args.registry,
        run=args.research_run,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
