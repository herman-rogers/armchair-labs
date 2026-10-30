"""Evaluate fixed-origin season aggregation; select decay on strictly earlier seasons."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from engine.chairstats import CampaignStore

from ..weekly_v3.data import HISTORY, ROOT, TEAM_ALIASES
from ..weekly_v31.artifacts import digest, now, seal, verify, write
from ..weekly_v32.run import DEFAULT_RUN as WEEKLY_RUN
from .curve import HALF_LIVES, durable_signal, future_curve

DEFAULT_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v32_horizons_20261014_002"


def select(frame, year, position):
    prior = frame[frame.season.lt(year) & frame.position.eq(position)].copy()
    if prior.empty:
        raise ValueError("Decay selection requires strictly earlier evaluation seasons")
    prior["square_error"] = (prior.prediction - prior.actual) ** 2
    scores = prior.groupby(["method", "season"]).square_error.mean().groupby("method").mean()
    choices = scores[[f"v32_decay_{h}" for h in HALF_LIVES]]
    return int(choices.idxmin().removeprefix("v32_decay_"))


def run(output=DEFAULT_RUN, registry=ROOT / "data/campaigns"):
    output, registry = Path(output), Path(registry)
    if output.exists():
        raise FileExistsError(output)
    verify(WEEKLY_RUN)
    store = CampaignStore(registry)
    store.check("opening_drive_v2")
    manifest = store.campaign("opening_drive_v2")
    predictions = pd.read_parquet(WEEKLY_RUN / "predictions.parquet")
    early = predictions[
        predictions.horizon.eq("early_week")
        & predictions.method.eq("selected")
        & predictions.week.eq(3)
    ].copy()
    panel = pd.read_parquet(WEEKLY_RUN / "panel.parquet")
    extra = panel[panel.horizon.eq("early_week") & panel.week.eq(3)]
    extra = extra[["row_id", "room_questionable_usage", "room_dnp_usage"]]
    early = early.merge(extra, on="row_id", validate="one_to_one")
    schedule_path = HISTORY / "schedule.parquet"
    history_receipt = json.loads((HISTORY / "manifest.json").read_text())
    if digest(schedule_path) != history_receipt["files"]["schedule.parquet"]:
        raise ValueError("Historical schedule checksum changed")
    schedule = pd.read_parquet(schedule_path)
    schedule = schedule[schedule.game_type.eq("REG")]
    for c in ("home_team", "away_team"):
        schedule[c] = schedule[c].replace(TEAM_ALIASES)
    sources = {
        str(WEEKLY_RUN / "manifest.json"): digest(WEEKLY_RUN / "manifest.json"),
        str(schedule_path): digest(schedule_path),
    }
    records, coverage = [], []
    for model in manifest["models"]:
        if model["horizon"] != "remaining":
            continue
        path = registry / "opening_drive_v2" / model["artifacts"]["evaluation_predictions"]["path"]
        sources[str(path)] = digest(path)
        evaluation = pd.read_parquet(path)
        evaluation = evaluation[evaluation.target.eq("points") & evaluation.recipe.eq("ensemble")]
        for season, group in evaluation.groupby("year"):
            games = schedule[schedule.season.eq(season) & schedule.week.gt(2)]
            weekly = early[early.season.eq(season) & early.position.eq(model["position"])]
            lookup = weekly.set_index("player_id").to_dict("index")
            matched = 0
            for r in group.to_dict("records"):
                team = TEAM_ALIASES.get(r["team"], r["team"])
                weeks = games.loc[
                    games.home_team.eq(team) | games.away_team.eq(team), "week"
                ].tolist()
                # Preserve the entire v2 candidate cohort, including unmatched weekly players.
                w = lookup.get(r["player_id"])
                supported = w is not None and w["team"] == team and bool(weeks)
                matched += int(supported)
                values = {"v2": r["prediction"]}
                for half_life in HALF_LIVES:
                    for version in ("v3", "v32"):
                        total = r["prediction"]
                        if supported:
                            mean = w["prediction"] if version == "v32" else w["base_prediction"]
                            signal = (
                                durable_signal(w, r["prediction"] / len(weeks))
                                if version == "v32"
                                else mean
                            )
                            curve = future_curve(
                                weeks, 3, mean, r["prediction"] / len(weeks), signal, half_life
                            )
                            total = sum(g["points"] for g in curve)
                        values[f"{version}_decay_{half_life}"] = total
                for method, value in values.items():
                    records.append(
                        dict(
                            player_id=r["player_id"],
                            player_name=r["player_display_name"],
                            season=int(season),
                            origin_week=2,
                            position=model["position"],
                            actual=float(r["actual"]),
                            prediction=float(value),
                            method=method,
                            weekly_supported=supported,
                        )
                    )
            coverage.append(
                dict(position=model["position"], season=int(season), n=len(group), matched=matched)
            )
    frame = pd.DataFrame(records)
    selected, policies = [], {}
    for year in (2023, 2024, 2025, 2026):
        policies[str(year)] = {}
        for position in ("QB", "RB", "WR", "TE"):
            half_life = select(frame, year, position)
            policies[str(year)][position] = half_life
            for version in ("v3", "v32"):
                part = frame[
                    frame.season.eq(year)
                    & frame.position.eq(position)
                    & frame.method.eq(f"{version}_decay_{half_life}")
                ].copy()
                part["method"] = f"{version}_selected_decay"
                part["selected_half_life"] = half_life
                selected.append(part)
    frame = pd.concat([frame, *selected], ignore_index=True)
    scores = []
    for (position, method), group in frame[frame.season.ge(2023)].groupby(["position", "method"]):
        for year in [None, *sorted(group.season.unique())]:
            part = group if year is None else group[group.season.eq(year)]
            error = part.prediction - part.actual
            rank = []
            for _, annual in part.groupby("season"):
                k = 10 if position in ("QB", "TE") else 20
                rank.append(float(annual.nlargest(k, "prediction").actual.mean()))
            scores.append(
                dict(
                    position=position,
                    method=method,
                    season=None if year is None else int(year),
                    n=len(part),
                    rmse=float(np.sqrt((error**2).mean())),
                    mae=float(error.abs().mean()),
                    bias=float(error.mean()),
                    top_k_actual_mean=float(np.mean(rank)),
                )
            )
    output.mkdir(parents=True)
    frame.to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
    write(output / "coverage.json", coverage)
    write(
        output / "policy.json",
        dict(
            version=1,
            forecast_season=2026,
            training_through_season=2025,
            half_lives=policies["2026"],
            historical_selections=policies,
            selector="Minimum earlier-season equal-year MSE; 2021–22 seed 2023 evaluation",
            formula="Current week equals saved weekly mean; later weeks blend early signal "
            "toward v2 anchor with exponential decay",
            injury_rule="Weekly report/teammate injury effects are confined to this week; "
            "affected early signals use the long-horizon anchor for later weeks",
            bye_rule="No weekly signal on a bye; retain long anchor for future games",
            limitations=[
                "Reused historical development years; "
                "not a pristine holdout or prospective validation.",
                "Exact archived v2 ensemble comparison at origin Week 2 only; "
                "later live cutoffs are a contract extension.",
                "Full-season totals add known actuals to remaining forecasts; "
                "preseason model is preserved separately at origin zero.",
                "Future schedule assumes current team; no future transfers, injury durations "
                "or opponent-specific feature updates are predicted.",
                "Long-horizon uncertainty is withheld; "
                "weekly interval widths cannot be added as independent errors.",
            ],
        ),
    )
    write(output / "sources.json", sources)
    for path in Path(__file__).parent.glob("*.py"):
        target = output / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    comparison = pd.DataFrame(scores)
    comparison = comparison[
        comparison.season.isna()
        & comparison.method.isin(["v2", "v3_selected_decay", "v32_selected_decay"])
    ]
    (output / "index.html").write_text(
        '<!doctype html><html><meta charset="utf-8">'
        "<title>Opening Drive v3.2 horizon evaluation</title>"
        "<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px}"
        "td,th{padding:8px;text-align:left}table{border-collapse:collapse}"
        "tr{border-bottom:1px solid #ccc}</style>"
        "<h1>Opening Drive v3.2: weekly to season projections</h1>"
        "<p>Fixed-origin forecasts: no later weekly predictions or outcomes enter the curve. "
        "2021–22 seed decay selection; 2023–25 evaluation selects using earlier seasons only. "
        "The comparator is the exact archived v2 ensemble at Week 2, "
        "including unmatched candidates. "
        "V3 uses the same selected decay as v3.2 to isolate the weekly model difference.</p>"
        + comparison.drop(columns="season").to_html(index=False, float_format=lambda x: f"{x:.3f}")
        + "<h2>2026 decay half-life (calendar weeks)</h2><pre>"
        + json.dumps(policies["2026"], indent=2)
        + "</pre><p>0 means only this week changes; 10000 approximates no decay. "
        "Current-week injury effects do not propagate to later weeks. Live later cutoffs remain "
        "an unvalidated extension. These are historical development results, "
        "not proof of live improvement.</p></html>"
    )
    seal(output, dict(kind="opening_drive_horizon_policy", generated_at=now().isoformat()))
    print(comparison.to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(run(args.output))
