"""Lock paired forecasts before the weekly deadline; replay and score them later."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats import CampaignStore
from engine.metrics import weekly_distribution

from ..weekly_v3 import data, modeling
from ..weekly_v3 import lineups as v3_lineups
from ..weekly_v3 import rescore as v3_rescore
from ..weekly_v3.lineups import aggregate
from ..weekly_v3.replay import rows_digest
from ..weekly_v3.rescore import _rows, validate_snapshot
from . import calibration, league
from .artifacts import digest, now, publish_directory, seal, timestamp, verify, write
from .run import DEFAULT_LOG, DEFAULT_RUN, summarize


class DeadlinePassed(ValueError):
    """No pregame capture can be admitted for this origin."""


def deadline(schedule, season, week):
    games = schedule[
        schedule.season.eq(season) & schedule.week.eq(week) & schedule.game_type.eq("REG")
    ]
    if games.empty or games.gameday.isna().any():
        raise ValueError("Complete dated schedule required for prospective capture")
    if games.duplicated("game_id").any():
        raise ValueError("Duplicate scheduled games")
    # Deliberately earlier than kickoff; matches the existing league forecast log.
    return timestamp(f"{min(str(d)[:10] for d in games.gameday)}T00:00:00+00:00")


def check_timing(document, schedule, captured_at):
    current = document["current_observations"]
    cutoff = deadline(schedule, current["season"], current["through_week"] + 1)
    if captured_at >= cutoff:
        raise DeadlinePassed("Weekly pregame deadline has passed; no prospective backfill")
    for value in (current["saved_at"], document["registered_at"]):
        if timestamp(value) > captured_at or timestamp(value) >= cutoff:
            raise ValueError("Snapshot timestamp is after capture or deadline")
    games = schedule[
        schedule.season.eq(current["season"]) & schedule.week.eq(current["through_week"] + 1)
    ]
    for name in ("home_score", "away_score"):
        if name in games and games[name].notna().any():
            raise ValueError("Forecast-week game outcomes are already present")
    return cutoff


def baseline_rows(directory, panel, season, origin):
    """Replay locally generated, checksummed frozen v3 models on saved inputs."""
    directory = Path(directory)
    artifacts = json.loads((directory / "baseline_models.json").read_text())
    rows = []
    for position, refs in sorted(artifacts.items()):
        for name, ref in refs.items():
            if digest(directory / ref["path"]) != ref["sha256"]:
                raise ValueError(f"Frozen baseline changed: {name}")
        for name, module in (
            ("source_modeling", modeling),
            ("source_data", data),
            ("source_lineup_runtime", weekly_distribution),
            ("source_lineup_research", v3_lineups),
            ("source_rescore", v3_rescore),
        ):
            if (directory / refs[name]["path"]).read_bytes() != Path(module.__file__).read_bytes():
                raise ValueError("Baseline inference source differs from frozen v3")
        contract = json.loads((directory / refs["run_manifest"]["path"]).read_text())
        for name, version in contract["packages"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Baseline environment changed: {name}")
        mean = pickle.loads((directory / refs["mean_model"]["path"]).read_bytes())
        spread = pickle.loads((directory / refs["distribution_model"]["path"]).read_bytes())
        rows.extend(_rows(panel, {"position": position}, mean, spread, season, origin))
    return rows


def verify_challenger_sources(directory, contract):
    for path in Path(__file__).parent.glob("*.py"):
        if (Path(directory) / "source" / path.name).read_bytes() != path.read_bytes():
            raise ValueError(f"Challenger source changed: {path.name}; use a new run")
    for name, version in contract.get("packages", {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Challenger environment changed: {name}")


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
                target_points=np.nan,
            )
            for r in rows
        ]
    )


def artificial_lineups(frame, dependence):
    """Freeze the same two forecast-only draft designs used in the v3 experiment."""
    season, week = int(frame.season.iloc[0]), int(frame.week.iloc[0])
    estimates = {
        (r["relationship"], *sorted([r["position_a"], r["position_b"]])): r
        for r in dependence["correlations"]
    }
    if dependence["covariance_before_season"] > season:
        raise ValueError("Future lineup dependence calibration")
    output = []
    for design, rank_key in (("v3_ranked", "prediction"), ("v2_ranked", "v2_weekly_mean")):
        pools = {
            p: sorted(
                frame[frame.position.eq(p) & frame.is_bye.eq(0)].to_dict("records"),
                key=lambda r: (-r[rank_key], r["player_id"]),
            )
            for p in ("QB", "RB", "WR", "TE")
        }
        rng = np.random.default_rng(season * 100 + week)
        rosters = [[] for _ in range(10)]
        for position, count in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1)):
            for _ in range(count):
                for i in rng.permutation(10):
                    if pools[position]:
                        rosters[i].append(pools[position].pop(0))
        flex = sorted(
            pools["RB"] + pools["WR"] + pools["TE"], key=lambda r: (-r[rank_key], r["player_id"])
        )
        for i in rng.permutation(10):
            if flex:
                rosters[i].append(flex.pop(0))
        for i, roster in enumerate(rosters):
            if len(roster) != 8:
                continue
            # Marginals and dependence remain v3; lineup bounds are corrected directly.
            result = aggregate(roster, estimates, seed=season * 10000 + week * 10 + i)
            output.append(
                dict(
                    result,
                    season=season,
                    week=week,
                    position="LINEUP",
                    design=design,
                    team_id=i,
                    player_ids=",".join(r["player_id"] for r in roster),
                    target_points=np.nan,
                    has_qb_receiver_stack=any(
                        a["position"] == "QB"
                        and b["position"] in ("WR", "TE")
                        and a.get("team")
                        and a["team"] == b.get("team")
                        for a in roster
                        for b in roster
                    ),
                )
            )
    return pd.DataFrame(output)


def predictions(directory, panel, season, origin):
    rows = baseline_rows(directory, panel, season, origin)
    frame = player_frame(rows)
    lookup = panel.set_index("player_id").v2_weekly_mean
    frame["v2_weekly_mean"] = frame.player_id.map(lookup)
    calibrators = json.loads((Path(directory) / "calibrators.json").read_text())
    players = calibration.apply(frame, calibrators["players"])
    models = json.loads((Path(directory) / "baseline_models.json").read_text())
    dependence = json.loads(
        (Path(directory) / models["QB"]["lineup_calibration"]["path"]).read_text()
    )
    lineups = artificial_lineups(frame, dependence)
    if lineups.empty:
        raise ValueError("Insufficient population for the locked artificial lineup designs")
    return rows, players, calibration.apply(lineups, calibrators["lineups"])


def league_predictions(directory, players):
    models = json.loads((Path(directory) / "baseline_models.json").read_text())
    dependence = json.loads(
        (Path(directory) / models["QB"]["lineup_calibration"]["path"]).read_text()
    )
    calibrator = json.loads((Path(directory) / "calibrators.json").read_text())["lineups"]
    return league.predict(directory, players, dependence, calibrator)


def capture(run_dir, registry, snapshot, log_dir=DEFAULT_LOG):
    run_dir, registry, snapshot, log_dir = map(Path, (run_dir, registry, snapshot, log_dir))
    contract = verify(run_dir)
    verify_challenger_sources(run_dir, contract)
    document = json.loads(snapshot.read_text())
    current = document["current_observations"]
    season, origin = current["season"], current["through_week"]
    validate_snapshot(snapshot, season, origin)
    if season != contract["forecast_season"]:
        raise ValueError("Challenger calibration is for a different season")
    destination = log_dir / "forecasts" / f"{season}_{origin + 1:02d}"
    if destination.exists():
        receipt = verify(destination)
        if receipt["challenger_manifest_sha256"] != digest(run_dir / "manifest.json"):
            raise ValueError("An origin is already locked to another challenger")
        return dict(status="already_locked", path=str(destination))
    asset = document["assets"]["current/schedule.parquet"]
    schedule_path = data.ROOT / "data" / asset["object"]
    if digest(schedule_path) != asset["sha256"]:
        raise ValueError("Schedule checksum mismatch")
    schedule = pd.read_parquet(schedule_path)
    cutoff = check_timing(document, schedule, now())
    if timestamp(contract["generated_at"]) > now():
        raise ValueError("Challenger has a future timestamp")
    store = CampaignStore(registry)
    store.check("opening_drive_v3")
    baseline = store.campaign("opening_drive_v3")
    frozen = json.loads((run_dir / "baseline_models.json").read_text())
    payloads, matrix_paths, datasets = [], set(), []
    baseline_dir = registry / "opening_drive_v3"
    for entry in baseline["models"]:
        if entry["horizon"] != "weekly":
            continue
        for name in ("mean_model", "distribution_model", "lineup_calibration"):
            if entry["artifacts"][name]["sha256"] != frozen[entry["position"]][name]["sha256"]:
                raise ValueError("Published v3 no longer matches the frozen comparison baseline")
        forecast = next(
            (f for f in entry["forecasts"] if f["season"] == season and f["origin_week"] == origin),
            None,
        )
        if forecast is None:
            raise ValueError("Refresh all four v3 positions before prospective capture")
        payload = json.loads((baseline_dir / forecast["path"]).read_text())
        generated = timestamp(payload["generated_at"])
        if generated >= cutoff or generated > now():
            raise ValueError("Baseline forecast was not generated before the deadline")
        if payload.get("snapshot_sha256") != digest(snapshot):
            raise ValueError("Baseline forecast does not reference this accepted snapshot")
        key = forecast.get("feature_matrix_artifact") or payload.get("feature_matrix_artifact")
        if not key:
            raise ValueError("Prospective capture requires a dated weekly refresh matrix")
        matrix_paths.add(baseline_dir / entry["artifacts"][key]["path"])
        dataset = baseline_dir / entry["artifacts"][f"forecast_dataset_{season}_{origin}"]["path"]
        datasets.append(dataset)
        payloads.append(payload)
    if (
        len(payloads) != 4
        or len({digest(p) for p in matrix_paths}) != 1
        or len({digest(p) for p in datasets}) != 1
    ):
        raise ValueError("Complete paired forecasts must share one feature matrix and receipt")
    matrix_path = matrix_paths.pop()
    panel = pd.read_parquet(matrix_path)
    receipt = json.loads(datasets[0].read_text())
    if (
        receipt["panel_sha256"] != digest(matrix_path)
        or receipt["current_through_week"] != origin
        or receipt["current_season"] != season
    ):
        raise ValueError("Matrix receipt does not match the forecast cutoff")
    if (
        panel.target_points.notna().any()
        or panel.duplicated("player_id").any()
        or not (panel.season.eq(season) & panel.week.eq(origin + 1)).all()
    ):
        raise ValueError("Live matrix has outcomes, duplicate identities, or a wrong week")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capture-", dir=destination.parent) as tmp:
        staging = Path(tmp) / destination.name
        staging.mkdir()
        for name in ("baseline_models.json", "calibrators.json"):
            shutil.copy2(run_dir / name, staging / name)
        shutil.copytree(run_dir / "baseline", staging / "baseline")
        shutil.copytree(run_dir / "source", staging / "source")
        shutil.copy2(run_dir / "manifest.json", staging / "challenger_manifest.json")
        shutil.copy2(snapshot, staging / "snapshot.json")
        shutil.copy2(schedule_path, staging / "schedule.parquet")
        shutil.copy2(matrix_path, staging / "panel.parquet")
        shutil.copy2(datasets[0], staging / "dataset.json")
        write(staging / "baseline_payloads.json", payloads)
        rows, players, lineups = predictions(staging, panel, season, origin)
        league_status = league.freeze_inputs(staging, data.ROOT, season, origin + 1, now())
        league_rows, unavailable = league_predictions(staging, players)
        if not league_rows.empty:
            league_rows.to_parquet(staging / "league_lineups.parquet", index=False)
            league_rows.to_csv(staging / "league_lineups.csv", index=False)
        write(staging / "league_support.json", dict(status=league_status, unavailable=unavailable))
        expected = sorted([r for p in payloads for r in p["rows"]], key=lambda r: r["player_id"])
        if rows_digest(sorted(rows, key=lambda r: r["player_id"])) != rows_digest(expected):
            raise ValueError("Frozen v3 replay differs from the published forecast")
        players.to_parquet(staging / "players.parquet", index=False)
        players.drop(columns="samples").to_csv(staging / "players.csv", index=False)
        lineups.to_parquet(staging / "lineups.parquet", index=False)
        lineups.to_csv(staging / "lineups.csv", index=False)
        captured = now()
        check_timing(document, schedule, captured)  # Work may have crossed the deadline.
        seal(
            staging,
            dict(
                schema_version=1,
                status="prospective",
                season=season,
                forecast_week=origin + 1,
                origin_week=origin,
                generated_at=captured.isoformat(),
                deadline=cutoff.isoformat(),
                challenger_manifest_sha256=digest(run_dir / "manifest.json"),
                baseline_manifest_sha256=digest(baseline_dir / "manifest.json"),
                players=len(players),
                artificial_lineups=len(lineups),
                league_offense_lineups=len(league_rows),
                selection_policy="First complete paired pregame capture; no replacement",
                baseline_note="Original v3 payloads keep their retrospective label; this capture "
                "independently verifies current-week timing and unobserved outcomes",
                lineup_scope="Artificial and available actual offense rosters scored separately; "
                "synthetic-to-league interval transfer is provisional; K/DST excluded",
            ),
        )
        if now() >= cutoff:
            raise DeadlinePassed("Deadline crossed before publication")
        publish_directory(staging, destination)
    return dict(status="locked", path=str(destination), players=len(players), lineups=len(lineups))


def replay(directory):
    directory = Path(directory)
    manifest = verify(directory)
    verify_challenger_sources(
        directory, json.loads((directory / "challenger_manifest.json").read_text())
    )
    for name in ("calibration.py", "prospective.py", "league.py"):
        if (directory / "source" / name).read_bytes() != Path(__file__).with_name(
            name
        ).read_bytes():
            raise ValueError("Prospective inference source changed since capture")
    snapshot = json.loads((directory / "snapshot.json").read_text())
    check_timing(
        snapshot,
        pd.read_parquet(directory / "schedule.parquet"),
        timestamp(manifest["generated_at"]),
    )
    _, players, lineups = predictions(
        directory,
        pd.read_parquet(directory / "panel.parquet"),
        manifest["season"],
        manifest["origin_week"],
    )
    for name, expected in (("players", players), ("lineups", lineups)):
        saved = pd.read_parquet(directory / f"{name}.parquet")
        if "samples" in expected:
            np.testing.assert_array_equal(np.stack(expected.samples), np.stack(saved.samples))
            expected, saved = expected.drop(columns="samples"), saved.drop(columns="samples")
        pd.testing.assert_frame_equal(expected, saved, check_exact=True)
    actual_league, _ = league_predictions(directory, players)
    if not actual_league.empty:
        pd.testing.assert_frame_equal(
            actual_league, pd.read_parquet(directory / "league_lineups.parquet"), check_exact=True
        )
    return dict(status="passed", players=len(players), lineups=len(lineups))


def score_captures(log_dir, outcomes):
    """Evaluate only locked populations; missing player labels keep lineups pending."""
    keys = ["season", "week", "player_id"]
    if outcomes.duplicated(keys).any():
        raise ValueError("Duplicate outcome identities")
    labels = outcomes[[*keys, "target_points"]]
    player_frames, lineup_frames, league_frames, support = [], [], [], []
    for directory in sorted((Path(log_dir) / "forecasts").glob("*")):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        manifest = verify(directory)
        if manifest["status"] != "prospective":
            raise ValueError("Only verified prospective captures may enter this scorecard")
        check_timing(
            json.loads((directory / "snapshot.json").read_text()),
            pd.read_parquet(directory / "schedule.parquet"),
            timestamp(manifest["generated_at"]),
        )
        players = pd.read_parquet(directory / "players.parquet").drop(columns="target_points")
        players = players.merge(labels, on=keys, how="left", validate="one_to_one")
        players.loc[players.is_bye.eq(1), "target_points"] = 0.0
        actual = players.set_index("player_id").target_points
        # Both methods are evaluated on the same captured non-bye rows.
        players = players[players.is_bye.eq(0)].copy()
        lineups = pd.read_parquet(directory / "lineups.parquet")
        lineups["target_points"] = [
            actual.reindex(ids.split(",")).sum(min_count=len(ids.split(",")))
            for ids in lineups.player_ids
        ]
        real_lineups = pd.DataFrame()
        if (directory / "league_lineups.parquet").exists():
            real_lineups = pd.read_parquet(directory / "league_lineups.parquet")
            real_lineups["target_points"] = [
                actual.reindex(ids.split(",")).sum(min_count=len(ids.split(",")))
                for ids in real_lineups.player_ids
            ]
            league_frames.append(real_lineups)
        support.append(
            dict(
                season=manifest["season"],
                week=manifest["forecast_week"],
                captured_at=manifest["generated_at"],
                players=len(players),
                observed_players=int(players.target_points.notna().sum()),
                pending_players=int(players.target_points.isna().sum()),
                lineups=len(lineups),
                observed_lineups=int(lineups.target_points.notna().sum()),
                league_offense_lineups=len(real_lineups),
                observed_league_offense_lineups=(
                    int(real_lineups.target_points.notna().sum()) if len(real_lineups) else 0
                ),
            )
        )
        player_frames.append(players)
        lineup_frames.append(lineups)
    scores = []
    for frames, kind, levels in (
        (player_frames, "player", (80, 90)),
        (lineup_frames, "artificial_lineup", (80,)),
        (league_frames, "league_offense", (80,)),
    ):
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        for method, adjusted in (("v3", False), ("v3.1", True)):
            periods = [("all", combined)] + [
                (f"{season}-W{week:02d}", group)
                for (season, week), group in combined.groupby(["season", "week"])
            ]
            for period, group in periods:
                scored = summarize(group, method, adjusted=adjusted, levels=levels)
                scored["kind"], scored["period"] = kind, period
                scores.append(scored)
    return pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(), support


def update_scorecard(log_dir, snapshot, registry=data.ROOT / "data/campaigns"):
    """Rebuild labels from accepted completed games using the original scoring contract."""
    log_dir, snapshot, registry = Path(log_dir), Path(snapshot), Path(registry)
    document = json.loads(snapshot.read_text())
    current = document["current_observations"]
    if (
        document.get("status") != "accepted"
        or current["season"] != 2026
        or not 0 <= current["through_week"] <= 18
    ):
        raise ValueError("Scoring requires an accepted 2026 snapshot through a regular-season week")
    captures = [
        p for p in (log_dir / "forecasts").glob("*") if p.is_dir() and not p.name.startswith(".")
    ]
    for directory in captures:
        verify(directory)
        models = json.loads((directory / "baseline_models.json").read_text())
        if models["QB"]["source_data"]["sha256"] != digest(Path(data.__file__)):
            raise ValueError("Outcome scoring code differs from the captured contract")
    revisions = log_dir / "evaluations"
    revisions.mkdir(parents=True, exist_ok=True)
    forecast_hashes = {p.name: digest(p / "manifest.json") for p in captures}
    previous_evaluations = sorted(
        p for p in revisions.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if previous_evaluations:
        previous_receipt = verify(previous_evaluations[-1])
        if (
            previous_receipt["snapshot_sha256"] == digest(snapshot)
            and previous_receipt["forecasts"] == forecast_hashes
        ):
            return dict(status="current", evaluation=str(previous_evaluations[-1]))
    with tempfile.TemporaryDirectory(prefix=".score-", dir=revisions) as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        previous = data.SNAPSHOT, data.CAMPAIGN
        try:
            data.SNAPSHOT, data.CAMPAIGN = snapshot, registry / "opening_drive_v2"
            panel, _, receipt = data.build_panel(
                output_dir=staging, start_year=current["season"], end_year=current["season"]
            )
        finally:
            data.SNAPSHOT, data.CAMPAIGN = previous
        outcomes = panel[panel.week.le(current["through_week"])]
        scores, support = score_captures(log_dir, outcomes)
        scores.to_csv(staging / "scores.csv", index=False)
        write(staging / "support.json", support)
        shutil.copy2(snapshot, staging / "snapshot.json")
        table = (
            scores[scores.season.eq("all")].to_html(index=False, border=0, float_format="%.3f")
            if len(scores)
            else "<p>No captured forecasts yet.</p>"
        )
        support_table = pd.DataFrame(support).to_html(index=False, border=0)
        (staging / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>V3.1 prospective scorecard</title>"
            "<style>body{font:15px system-ui;margin:32px;color:#152738}"
            "table{border-collapse:collapse}"
            "td,th{padding:8px;border-bottom:1px solid #ddd}div{overflow:auto}</style>"
            "<h1>Opening Drive v3.1 prospective scorecard</h1>"
            "<p>First complete paired capture before 00:00 UTC on the first NFL game date. "
            "Both versions use identical point forecasts. Missing results remain pending. "
            "Artificial and captured actual offense rosters are scored separately. "
            "Later lineup changes do not replace the captured rosters. K/DST is excluded.</p>"
            "<h2>Captured weeks and outcome support</h2><div>" + support_table + "</div>"
            "<h2>Paired metrics</h2><p>Lower interval score is better. Coverage and width must "
            "be considered together. No challenger is automatically promoted.</p><div>"
            + table
            + "</div>"
        )
        stamp = now().strftime("%Y%m%dT%H%M%S%fZ")
        seal(
            staging,
            dict(
                generated_at=now().isoformat(),
                season=current["season"],
                through_week=current["through_week"],
                snapshot_sha256=digest(snapshot),
                panel_sha256=receipt["panel_sha256"],
                forecasts=forecast_hashes,
            ),
        )
        destination = revisions / stamp
        publish_directory(staging, destination)
    display = log_dir / "scorecard"
    display.mkdir(exist_ok=True)
    for name in ("index.html", "scores.csv", "support.json", "manifest.json"):
        shutil.copy2(destination / name, display / name)
    return dict(status="scored", evaluation=str(destination), weeks=len(support))


def track(
    snapshot, registry=data.ROOT / "data/campaigns", run_dir=DEFAULT_RUN, log_dir=DEFAULT_LOG
):
    if not Path(run_dir).exists():
        return dict(status="not_configured")
    current = json.loads(Path(snapshot).read_text())["current_observations"]
    if current["through_week"] == 18:
        result = dict(status="regular_season_complete")
    else:
        try:
            result = capture(run_dir, registry, snapshot, log_dir)
        except DeadlinePassed as error:
            result = dict(status="missed_deadline", reason=str(error))
    result["scorecard"] = update_scorecard(log_dir, snapshot, registry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "score", "track", "replay"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--registry", type=Path, default=data.ROOT / "data/campaigns")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args(argv)
    if args.action == "replay":
        if args.capture_dir is None:
            parser.error("replay requires --capture-dir")
        result = replay(args.capture_dir)
    else:
        from ..rescore import latest_raw_snapshot

        snapshot = args.snapshot or latest_raw_snapshot()
        if args.action == "capture":
            result = capture(args.run, args.registry, snapshot, args.log)
        elif args.action == "score":
            result = update_scorecard(args.log, snapshot, args.registry)
        else:
            result = track(snapshot, args.registry, args.run, args.log)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    main()
