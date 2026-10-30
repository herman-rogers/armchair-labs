"""Paired live scorecard and a review of the integration's historical evidence."""
# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v3 import data as v3_data
from ..weekly_v3.lineups import NONSTARTERS, aggregate
from ..weekly_v31 import league
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, timestamp, verify, write
from . import data, uncertainty


def capture_lineups(directory, players, baseline, integration, captured_at):
    """Freeze memberships before outcomes; never select rosters using realized points."""
    season, week = int(players.season.iloc[0]), int(players.week.iloc[0])
    members = []
    active = players[players.is_bye.eq(0)].copy()
    active["base_prediction"] = active.player_id.map(baseline.set_index("player_id").prediction)
    for design, column in (("v3_ranked", "base_prediction"), ("v32_ranked", "prediction")):
        members.extend(
            dict(design=design, team_id=i, ids=ids)
            for i, ids in enumerate(uncertainty.draft(active, column))
        )
    support = league.freeze_inputs(directory, data.ROOT, season, week, captured_at)
    inputs_path = Path(directory) / "league_inputs.json"
    if inputs_path.exists():
        inputs = json.loads(inputs_path.read_text())
        identity = {int(k): v for k, v in inputs["identities"].items()}
        seen = set()
        for game in inputs["snapshot"]["week_lineups"]:
            if game["week"] != week:
                continue
            for side in ("home", "away"):
                team = game[f"{side}_team_id"]
                if team in seen:
                    raise ValueError("Duplicate actual league team")
                seen.add(team)
                starters = [
                    p
                    for p in game[f"{side}_lineup"]
                    if p["slot"] not in NONSTARTERS and p["position"] in {"QB", "RB", "WR", "TE"}
                ]
                ids = [identity.get(p["espn_id"]) for p in starters]
                if len(ids) == 8 and len(set(ids)) == 8 and set(ids) <= set(players.player_id):
                    members.append(dict(design="actual_league_offense", team_id=team, ids=ids))
    contracts = json.loads((Path(integration) / "dependence.json").read_text())
    estimates = uncertainty.estimates(contracts["early_week"])
    # Use the sealed v3 dependence for its comparison lineups.
    base_contract = json.loads((Path(integration) / "baseline_lineup_calibration.json").read_text())
    base_estimates = uncertainty.estimates(base_contract)
    frames = {
        "v3": baseline.set_index("player_id").to_dict("index"),
        "v3.2": players.set_index("player_id").to_dict("index"),
    }
    results = []
    for member in members:
        for name, mapping in frames.items():
            roster = [dict(player_id=p, **mapping[p]) for p in member["ids"]]
            for r in roster:
                r["samples"] = list(r["samples"])
            result = aggregate(
                roster,
                base_estimates if name == "v3" else estimates,
                seed=season * 10000 + week * 100 + member["team_id"],
            )
            results.append(
                dict(
                    season=season,
                    week=week,
                    design=member["design"],
                    team_id=member["team_id"],
                    method=name,
                    player_ids=",".join(member["ids"]),
                    **result,
                )
            )
    pd.DataFrame(results).to_parquet(Path(directory) / "lineups.parquet", index=False)
    write(
        Path(directory) / "lineup_support.json",
        dict(status=support, memberships=len(members), note="Offense only; K/DST excluded"),
    )


def score(log, snapshot, registry):
    log, snapshot, registry = Path(log), Path(snapshot), Path(registry)
    document = json.loads(snapshot.read_text())
    current = document["current_observations"]
    if document["status"] != "accepted":
        raise ValueError("Scoring requires an accepted snapshot")
    captures = sorted((log / "forecasts").glob("*/*/manifest.json"))
    hashes = {str(p.parent): digest(p) for p in captures}
    previous = log / "scorecard" / "manifest.json"
    if previous.exists():
        receipt = json.loads(previous.read_text())
        if receipt["snapshot_sha256"] == digest(snapshot) and receipt["captures"] == hashes:
            return dict(status="current", captures=len(captures))
    outcomes = pd.DataFrame(columns=["player_id", "season", "week", "target_points"])
    known = [verify(p.parent) for p in captures]
    if any(
        r["season"] == current["season"] and r["week"] <= current["through_week"] for r in known
    ):
        with tempfile.TemporaryDirectory(prefix=".v32-score-") as tmp:
            old = v3_data.SNAPSHOT, v3_data.CAMPAIGN
            try:
                v3_data.SNAPSHOT, v3_data.CAMPAIGN = snapshot, registry / "opening_drive_v2"
                panel, _, _ = v3_data.build_panel(
                    tmp, start_year=current["season"], end_year=current["season"]
                )
            finally:
                v3_data.SNAPSHOT, v3_data.CAMPAIGN = old
            outcomes = panel[panel.week.le(current["through_week"])][list(outcomes.columns)]
    score_rows, support, lineup_scores = [], [], []
    from .shadow import player_frame

    for path, receipt in zip(captures, known, strict=True):
        if receipt["status"] != "prospective" or timestamp(receipt["generated_at"]) >= timestamp(
            receipt["deadline"]
        ):
            raise ValueError("Non-prospective forecast cannot enter the scorecard")
        directory = path.parent
        new = pd.read_parquet(directory / "players.parquet")
        old = player_frame(json.loads((directory / "baseline_rows.json").read_text()))
        old = old[old.player_id.isin(new.player_id)].copy()
        frames = {}
        for name, f in (("v3", old), ("v3.2", new)):
            f = f.drop(columns=["target_points"], errors="ignore").merge(
                outcomes, on=["player_id", "season", "week"], how="left", validate="one_to_one"
            )
            f.loc[f.is_bye.eq(1), "target_points"] = 0.0
            frames[name] = f
            observed = f[f.is_bye.eq(0) & f.target_points.notna()]
            for position, part in observed.groupby("position"):
                draws = (
                    uncertainty.baseline_samples(part) if name == "v3" else np.stack(part.samples)
                )
                score_rows.append(
                    dict(
                        capture=str(directory),
                        horizon=receipt["forecast_horizon"],
                        method=name,
                        position=position,
                        **uncertainty.metrics(part, draws),
                    )
                )
        observed = frames["v3.2"]
        support.append(
            dict(
                capture=str(directory),
                season=receipt["season"],
                week=receipt["week"],
                horizon=receipt["forecast_horizon"],
                players=int(observed.is_bye.eq(0).sum()),
                observed=int((observed.is_bye.eq(0) & observed.target_points.notna()).sum()),
                pending=int((observed.is_bye.eq(0) & observed.target_points.isna()).sum()),
            )
        )
        lineups = pd.read_parquet(directory / "lineups.parquet")
        if not lineups.empty:
            actual = observed.set_index("player_id").target_points
            lineups["target_points"] = [
                actual.reindex(ids.split(",")).sum(min_count=len(ids.split(",")))
                for ids in lineups.player_ids
            ]
            lineups["capture"] = str(directory)
            lineup_scores.append(lineups)
    destination = log / "evaluations" / now().strftime("%Y%m%dT%H%M%S%fZ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".score-") as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        pd.DataFrame(score_rows).to_csv(staging / "scores.csv", index=False)
        pd.DataFrame(support).to_csv(staging / "support.csv", index=False)
        (pd.concat(lineup_scores, ignore_index=True) if lineup_scores else pd.DataFrame()).to_csv(
            staging / "lineups.csv", index=False
        )
        write(staging / "support.json", support)
        seal(
            staging,
            dict(
                generated_at=now().isoformat(),
                snapshot_sha256=digest(snapshot),
                captures=hashes,
                observed_players=sum(r["observed"] for r in support),
            ),
        )
        publish_directory(staging, destination)
    display = log / "scorecard"
    display.mkdir(exist_ok=True)
    for path in destination.iterdir():
        shutil.copy2(path, display / path.name)
    return dict(
        status="scored",
        observed_players=sum(r["observed"] for r in support),
        pending_players=sum(r["pending"] for r in support),
        evaluation=str(destination),
    )


def table(frame):
    return (
        '<div class="scroll">'
        + frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")
        + "</div>"
    )


def review(log, integration):
    log, integration = Path(log), Path(integration)
    scores = pd.read_csv(integration / "uncertainty_scores.csv")
    lineups = pd.read_parquet(integration / "lineup_predictions.parquet")
    lineups["coverage80"] = lineups.target_points.between(lineups.q10, lineups.q90)
    lineups["width80"] = lineups.q90 - lineups.q10
    lineups["interval_score80"] = (
        lineups.width80
        + 10 * (lineups.q10 - lineups.target_points).clip(lower=0)
        + 10 * (lineups.target_points - lineups.q90).clip(lower=0)
    )
    lineups["squared_error"] = (lineups.prediction - lineups.target_points) ** 2
    line_summary = (
        lineups.groupby(["horizon", "method"])[
            ["coverage80", "width80", "interval_score80", "crps", "squared_error"]
        ]
        .mean()
        .reset_index()
    )
    line_summary["rmse"] = np.sqrt(line_summary.pop("squared_error"))
    cols = ["coverage80", "width80", "interval_score80", "crps"]
    summary = (
        scores[scores.cohort.eq("base_points_5_plus")]
        .groupby(["horizon", "method"])[cols]
        .mean()
        .reset_index()
    )
    support = json.loads((log / "scorecard/support.json").read_text())
    captures = sorted((log / "forecasts").glob("*/early_week"))
    changes = pd.DataFrame()
    if captures:
        latest = captures[-1]
        players = pd.read_parquet(latest / "players.parquet")
        old = {
            r["player_id"]: r["predictions"]["points"]
            for r in json.loads((latest / "baseline_rows.json").read_text())
        }
        changes = players[
            [
                "player_name",
                "position",
                "player_id",
                "prediction",
                "participation_probability",
                "forecast_horizon",
            ]
        ].copy()
        changes["v3"] = changes.player_id.map(old)
        changes["change"] = changes.prediction - changes.v3
        changes = (
            changes.assign(magnitude=changes.change.abs())
            .sort_values("magnitude", ascending=False)
            .head(25)
            .drop(columns=["player_id", "magnitude"])
        )
    status = json.loads((log / "status.json").read_text()) if (log / "status.json").exists() else {}
    (
        log / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opening Drive v3.2 integration review</title>
<style>body{{font:16px/1.5 system-ui;background:#f5f7fa;color:#193249;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}section{{background:white;padding:22px;border:1px solid #d5e0e8;border-radius:12px;margin:20px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d9e2ea;text-align:right;white-space:nowrap}}a{{color:#155f9b}}@media(max-width:600px){{main{{padding:10px}}section{{padding:12px}}}}</style>
<main><h1>V3.2 is running alongside v3</h1><p>V3.2 is the application default, selected by the user on 2026-10-13. V3 remains available for comparison. Immutable prospective tracking continues, and v3.1 remains paused.</p>
<section><h2>Weekly, remaining and projected season points</h2><p>Since 2026-10-14, current remaining points sum a future-week curve that includes the saved v3.2 weekly prediction. Projected full-season totals add earned points. Averages and rankings use those totals. The carry-forward policy is evaluated separately and historical results are mixed.</p><p><a href="../opening_drive_v32_horizons/index.html">Horizon evaluation, live comparisons and verification</a></p></section>
<section><h2>Live evidence and fallback</h2><p>Usable current official report rows: <b>{status.get("usable_report_rows", 0)}</b>. Provider injury tags are archived separately and never treated as official active/inactive evidence. Wrong-week NFL pages are rejected. Missing evidence retains the early forecast.</p><p>Early forecasts lock before the first game. A per-game update may lock 90–60 minutes before kickoff when both teams have fresh official weekly report tables. No post-cutoff backfill occurs. The local timer and normal refresh share the same capture path.</p></section>
<section><h2>Prospective results</h2><p>Pending games have no accuracy result yet. Historical improvements below are separate from this live scorecard.</p>{table(pd.DataFrame(support))}<p><a href="scorecard/scores.csv">Player scores</a> · <a href="scorecard/lineups.csv">Captured offense lineups and results</a> · <a href="status.json">Latest collection status</a></p></section>
<section><h2>Ranges around the new predictions</h2><p>2023–2025 chronological checks; calibration uses earlier years only. Equal-weight averages over position/year cells for players with original v3 means ≥5. Nominal coverage is 80%; width, interval score and CRPS should be read together. Lower scores are better.</p>{table(summary)}<p>These are mean-preserving empirical residual distributions, fitted separately by position, timing and fixed projection bins. Point means are unchanged by this integration. This does not resume v3.1.</p></section>
<section><h2>Lineup results are mixed</h2><p>Matched eight-player offense lineups under two forecast-only draft designs, with earlier-year dependence estimates. Pregame results improve; early-week lineup point error is slightly worse despite better player RMSE. Artificial rosters do not establish gains for actual league choices.</p>{table(line_summary)}<p>K/DST means retain their existing source. Their uncertainty is withheld for this preview. Mixed early/pregame lineup ranges are also withheld until that combination has been evaluated.</p></section>
<section><h2>Largest changes in the first current-week capture</h2><p>Forecast differences, not realized improvements. Participation is the probability of offensive involvement, not a confirmed active-roster designation.</p>{table(changes)}</section>
<section><h2>Ongoing evaluation</h2><p>Review the dated live captures after their games complete, verify continued injury/workload coverage, and compare player and actual offense-lineup errors. Default selection does not change the historical evidence or pending live results.</p><p><a href="../opening_drive_v32_2026/index.html">Original v3.2 point-model investigation</a></p></section></main></html>""")
    summary.to_csv(log / "historical_uncertainty_summary.csv", index=False)
    line_summary.to_csv(log / "historical_lineup_summary.csv", index=False)
