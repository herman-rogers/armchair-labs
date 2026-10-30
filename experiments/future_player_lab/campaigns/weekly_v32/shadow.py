"""Capture immutable paired forecasts, update the preview and score completed weeks."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..weekly_v3.replay import rows_digest
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, timestamp, verify, write
from . import capture as provider
from . import data, live, official, uncertainty
from .integration import DEFAULT_INTEGRATION, check

DEFAULT_LOG = data.DATA / "outputs/opening_drive_v32_shadow"


def player_frame(rows):
    return pd.DataFrame(
        [
            dict(
                **r["distribution"],
                player_id=r["player_id"],
                player_name=r.get("player_name"),
                position=r["position"],
                team=r.get("team"),
                opponent=r.get("opponent"),
                season=r["season"],
                week=r["origin_week"] + 1,
                prediction=r["predictions"]["points"],
                is_bye=int(r["scheduled_games"] == 0),
            )
            for r in rows
        ]
    )


def game_window(schedule, captured_at):
    """First capture in the 90–60 minute window; never a post-cutoff reconstruction."""
    minutes = (schedule.kickoff - pd.Timestamp(captured_at)).dt.total_seconds() / 60
    return schedule[minutes.between(60, 90)]


def make_predictions(integration, panel, reports, captured_at, requested, covered_teams=()):
    out = panel.copy()
    reports = reports.copy()
    for column in ("player_id", "season", "week", "team"):
        reports[column] = reports[column].astype(out[column].dtype)
    out["injury_cutoff"] = (
        pd.to_datetime(out.feature_cutoff, utc=True)
        if requested == "early_week"
        else pd.Timestamp(captured_at)
    )
    out["injury_cutoff"] = pd.to_datetime(out.injury_cutoff, utc=True).astype("datetime64[ns, UTC]")
    reports["usable_at"] = pd.to_datetime(reports.usable_at, utc=True).astype("datetime64[ns, UTC]")
    out = data.teammate_features(data.attach_reports(out, reports, "injury_cutoff"))
    # The later policy requires a current official report table for BOTH sides of the game.
    supported = out.team.isin(covered_teams) & out.opponent.isin(covered_teams)
    late = requested == "pregame_60m" and bool(supported.all()) and len(out) > 0
    horizon = "pregame_60m" if late else "early_week"
    if requested == "pregame_60m" and not late:
        # A late fallback is the original early forecast, not an early model fed later reports.
        raise ValueError("Late fallback must reuse the locked early forecast")
    result, budgets = live.predict(integration, out, horizon)
    result["report_status_label"] = np.select(
        [
            result.report_out.eq(1),
            result.report_doubtful.eq(1),
            result.report_questionable.eq(1),
            result.report_known.eq(1),
        ],
        ["Out", "Doubtful", "Questionable", "Reported"],
        default="Unknown",
    )
    return result, budgets, out


def seal_capture(
    destination,
    integration,
    snapshot,
    panel,
    rows,
    reports,
    audit,
    players,
    inputs,
    budgets,
    requested,
    captured_at,
    deadline,
    game_id=None,
):
    if now() >= deadline:
        raise ValueError("Capture deadline passed during inference")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".v32-lock-") as tmp:
        staging = Path(tmp) / "capture"
        staging.mkdir()
        panel.to_parquet(staging / "baseline_panel.parquet", index=False)
        inputs.to_parquet(staging / "features.parquet", index=False)
        reports.to_parquet(staging / "reports.parquet", index=False)
        players.to_parquet(staging / "players.parquet", index=False)
        write(staging / "baseline_rows.json", rows)
        write(staging / "input_audit.json", audit)
        write(staging / "budgets.json", budgets)
        from .review import capture_lineups

        capture_lineups(staging, players, player_frame(rows), integration, captured_at)
        shutil.copy2(snapshot, staging / "snapshot.json")
        seal(
            staging,
            dict(
                status="prospective",
                generated_at=now().isoformat(),
                evidence_cutoff=captured_at.isoformat(),
                deadline=deadline.isoformat(),
                forecast_horizon=requested,
                game_id=game_id,
                season=int(players.season.iloc[0]),
                week=int(players.week.iloc[0]),
                integration=str(Path(integration).resolve()),
                integration_sha256=digest(Path(integration) / "manifest.json"),
                players=len(players),
                fallback=bool((players.forecast_horizon != requested).any()),
                runtime_sources={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            ),
        )
        if now() >= deadline:
            raise ValueError("Capture deadline passed before publication")
        publish_directory(staging, destination)


def replay(directory):
    directory = Path(directory)
    receipt = verify(directory)
    integration = Path(receipt["integration"])
    check(integration)
    if digest(integration / "manifest.json") != receipt["integration_sha256"]:
        raise ValueError("Capture refers to a different integration version")
    if not (
        timestamp(receipt["evidence_cutoff"])
        <= timestamp(receipt["generated_at"])
        < timestamp(receipt["deadline"])
    ):
        raise ValueError("Capture does not meet the prospective deadline")
    panel = pd.read_parquet(directory / "baseline_panel.parquet")
    expected = json.loads((directory / "baseline_rows.json").read_text())
    actual = live.replay_baseline(integration, panel, receipt["season"], receipt["week"] - 1)
    if rows_digest(sorted(actual, key=lambda r: r["player_id"])) != rows_digest(
        sorted(expected, key=lambda r: r["player_id"])
    ):
        raise ValueError("Frozen baseline replay differs")
    inputs = pd.read_parquet(directory / "features.parquet")
    saved = pd.read_parquet(directory / "players.parquet")
    scored = saved[saved.is_bye.eq(0)]
    horizon = scored.forecast_horizon.iloc[0]
    generated, budgets = live.predict(integration, inputs, horizon)
    columns = [
        "prediction",
        "std",
        "participation_probability",
        *uncertainty.QCOLS,
        *[f"{p}_{c}" for p in ("active", "expected") for c in data.OPPORTUNITIES],
    ]
    np.testing.assert_array_equal(generated[columns].to_numpy(), scored[columns].to_numpy())
    np.testing.assert_array_equal(np.stack(generated.samples), np.stack(scored.samples))
    if budgets != json.loads((directory / "budgets.json").read_text()):
        raise ValueError("Workload budget replay differs")
    return dict(status="passed", players=len(saved), baseline_players=len(expected))


def collect(season, week, identities, allow_network):
    errors, captures = [], []
    if allow_network:
        for name, operation in (
            ("provider", lambda: provider.capture(week)),
            ("official", lambda: official.capture(season, week, identities)),
        ):
            try:
                captures.append(dict(source=name, path=str(operation())))
            except (requests.RequestException, ValueError, KeyError) as error:
                errors.append(dict(source=name, error=str(error)))
    official_paths = sorted((data.DATA / "research/opening_drive_v32_official_captures").glob("*"))
    return [p for p in official_paths if p.is_dir() and not p.name.startswith(".")], dict(
        captures=captures, errors=errors, network=allow_network
    )


def _track(
    snapshot,
    registry=data.DATA / "campaigns",
    integration=DEFAULT_INTEGRATION,
    log=DEFAULT_LOG,
    allow_network=True,
):
    from .publisher import publish
    from .review import review, score

    snapshot, registry, integration, log = map(Path, (snapshot, registry, integration, log))
    contract = check(integration)
    document = json.loads(snapshot.read_text())
    current = document["current_observations"]
    season, week = current["season"], current["through_week"] + 1
    if season != contract["forecast_season"]:
        raise ValueError("A new season needs a new v3.2 integration")
    if week > 18:
        return dict(status="regular_season_complete", scorecard=score(log, snapshot, registry))
    schedule = live.schedule_times(live.asset(document, "schedule"))
    games = schedule[
        schedule.season.eq(season) & schedule.week.eq(week) & schedule.game_type.eq("REG")
    ]
    if games.empty or games.kickoff.isna().any():
        raise ValueError("Complete game times required")
    first = games.kickoff.min().to_pydatetime()
    early = log / "forecasts" / f"{season}_{week:02d}" / "early_week"
    captured_at = now()
    if not early.exists() and captured_at >= first:
        return dict(status="missed_early_deadline", scorecard=score(log, snapshot, registry))
    panel, rows, baseline_manifest = live.baseline(snapshot, registry, captured_at)
    frozen = json.loads((integration / "baseline/entries.json").read_text())
    for entry in baseline_manifest["models"]:
        if entry["horizon"] != "weekly":
            continue
        saved = next(r for r in frozen if r["position"] == entry["position"])
        for key in ("mean_model", "distribution_model"):
            if saved["artifacts"][key]["sha256"] != entry["artifacts"][key]["sha256"]:
                raise ValueError("Baseline models changed since v3.2 integration")
    paths, collection = collect(season, week, panel, allow_network)
    captured_at = now()
    reports, audit = live.usable_reports(paths, season, week, captured_at)
    covered_teams = set()
    for item in audit:
        if item["admitted"]:
            covered_teams.update(
                json.loads((Path(item["path"]) / "summary.json").read_text())["teams"]
            )
    added = []
    log.mkdir(parents=True, exist_ok=True)
    if not early.exists():
        with tempfile.TemporaryDirectory(dir=log, prefix=".v32-inputs-") as tmp:
            history = live.workload_history(snapshot, registry, Path(tmp), contract["research_run"])
            features = live.attach_workloads(panel, history)
        old = player_frame(rows)
        features["base_prediction"] = features.player_id.map(old.set_index("player_id").prediction)
        eligible = features[features.scheduled_game.eq(1)].copy()
        players, budgets, inputs = make_predictions(
            integration, eligible, reports, captured_at, "early_week"
        )
        bye = old[old.is_bye.eq(1)].copy()
        bye["forecast_horizon"], bye["chosen_method"], bye["report_status_label"] = (
            "early_week",
            "scheduled_bye",
            "Scheduled bye",
        )
        players = pd.concat([players, bye], ignore_index=True)
        seal_capture(
            early,
            integration,
            snapshot,
            panel,
            rows,
            reports,
            audit,
            players,
            inputs,
            budgets,
            "early_week",
            captured_at,
            first,
        )
        added.append(str(early))
    receipt = verify(early)
    if receipt["integration_sha256"] != digest(integration / "manifest.json"):
        raise ValueError("This week's first capture belongs to another integration")
    # Later captures always reuse the locked early feature population and baseline.
    original = pd.read_parquet(early / "features.parquet")
    original_panel = pd.read_parquet(early / "baseline_panel.parquet")
    baseline_rows = json.loads((early / "baseline_rows.json").read_text())
    early_players = pd.read_parquet(early / "players.parquet")
    for game in game_window(games, captured_at).to_dict("records"):
        destination = early.parent / f"pregame_{game['game_id']}"
        if destination.exists():
            verify(destination)
            continue
        subset = original[original.game_id.eq(game["game_id"])].copy()
        if subset.empty:
            continue
        supported = {
            data.TEAM_ALIASES.get(game[c], game[c]) for c in ("home_team", "away_team")
        } <= covered_teams
        if not supported:
            # Preserve explicit missing-evidence status; do not claim a later-model forecast.
            continue
        players, budgets, inputs = make_predictions(
            integration, subset, reports, captured_at, "pregame_60m", covered_teams
        )
        seal_capture(
            destination,
            integration,
            early / "snapshot.json",
            original_panel,
            baseline_rows,
            reports,
            audit,
            players,
            inputs,
            budgets,
            "pregame_60m",
            captured_at,
            game["kickoff"].to_pydatetime() - pd.Timedelta(hours=1),
            game["game_id"],
        )
        added.append(str(destination))
    forecasts = early_players.copy()
    captures = [early]
    for path in sorted(early.parent.glob("pregame_*")):
        verify(path)
        late = pd.read_parquet(path / "players.parquet")
        forecasts = merge_game_update(forecasts, late)
        captures.append(path)
    publication = publish(registry, integration, forecasts, captures, baseline_manifest)
    scored = score(log, snapshot, registry)
    result = dict(
        status="tracking",
        added=added,
        collection=collection,
        usable_report_rows=len(reports),
        covered_report_teams=sorted(covered_teams),
        publication=publication,
        scorecard=scored,
        early_fallback_games=[
            r.game_id
            for r in games.itertuples()
            if not (early.parent / f"pregame_{r.game_id}").exists()
        ],
    )
    write(log / "status.json", dict(updated_at=now().isoformat(), **result))
    review(log, integration)
    return result


def merge_game_update(early, late):
    """Replace complete rows; don't cast fine timestamps into an all-missing old dtype."""
    if late.player_id.duplicated().any() or not set(late.player_id) <= set(early.player_id):
        raise ValueError("Late update must use unique members of the locked population")
    result = pd.concat([early[~early.player_id.isin(late.player_id)], late], ignore_index=True)
    return result.set_index("player_id").loc[early.player_id].reset_index()


def track(
    snapshot,
    registry=data.DATA / "campaigns",
    integration=DEFAULT_INTEGRATION,
    log=DEFAULT_LOG,
    allow_network=True,
):
    config = Path(__file__).parent.parent / "configs/opening_drive_v32.json"
    if config.exists() and json.loads(config.read_text()).get("status") != "shadow_preview":
        return dict(status="not_tracking")
    log = Path(log)
    log.mkdir(parents=True, exist_ok=True)
    with (log / ".tracking.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="already_running")
        return _track(snapshot, registry, integration, log, allow_network)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("track", "replay", "score"), nargs="?", default="track")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--registry", type=Path, default=data.DATA / "campaigns")
    parser.add_argument("--integration", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "replay":
        if not args.capture_dir:
            parser.error("replay requires --capture-dir")
        result = replay(args.capture_dir)
    else:
        from ..rescore import latest_raw_snapshot

        snapshot = args.snapshot or latest_raw_snapshot()
        if args.action == "score":
            from .review import score

            result = score(args.log, snapshot, args.registry)
        else:
            result = track(snapshot, args.registry, args.integration, args.log, not args.no_capture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
