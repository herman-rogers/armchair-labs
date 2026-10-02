"""Publish exploratory rookie intelligence from audited history and current NFL stats.

Run: .venv/bin/python research/rookie_watch.py --history data/research/VERSION
No draft boards or frozen forecasts are rewritten. Season 2026 includes Jan 2027.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml
from artifact_inputs import require_audited_version
from data_integrity_audit import digest

from patron.config.league import get_league
from patron.config.settings import CONFIG_DIR, get_settings
from patron.data import nflverse
from patron.metrics.positions import canonical_positions
from patron.metrics.rookies import analog_forecast, backtest, completed_cutoff, rookie_rows
from patron.scoring.bonuses import extract_touchdown_bonuses
from patron.scoring.engine import score_components


def mapped_snaps(raw: pl.DataFrame, identities: pl.DataFrame) -> pl.DataFrame:
    mapping = identities.select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    if mapping["pfr_id"].is_duplicated().any():
        raise ValueError("Ambiguous PFR snap identity")
    return (
        raw.filter(pl.col("game_type") == "REG")
        .join(mapping, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1")
        .select("player_id", "season", "week", "offense_snaps", "offense_pct")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--season", type=int, default=get_league().draft_season)
    parser.add_argument("--through-week", type=int, default=None)
    args = parser.parse_args()
    provenance = require_audited_version(args.history)
    historical_scoring = args.history / "implementation/patron/config/scoring.yaml"
    if yaml.safe_load(historical_scoring.read_text()) != yaml.safe_load(
        (CONFIG_DIR / "scoring.yaml").read_text()
    ):
        raise ValueError("Historical and current league scoring differ; rebuild history first")
    print("Verified historical audit; loading current production and usage", flush=True)
    settings = get_settings()
    identities = nflverse.load_players()
    current = nflverse.load_player_weeks([args.season])
    schedule = nflverse.load_schedules(args.season)
    cutoff = min(completed_cutoff(schedule, current, args.season), 13)
    if args.through_week is not None:
        if not 1 <= args.through_week <= cutoff:
            raise ValueError(
                f"Requested week lacks complete data; latest complete cutoff is {cutoff}"
            )
        cutoff = args.through_week
    if cutoff < 1:
        raise ValueError("No fully completed week with NFL stat coverage is available")
    current = current.filter(pl.col("week") <= cutoff)
    plays = nflverse.load_touchdown_plays([args.season]).filter(pl.col("week") <= cutoff)
    bonuses, _ = extract_touchdown_bonuses(plays)
    current = (
        score_components(current)
        .join(bonuses, on=["player_id", "season", "week"], how="left", validate="1:1")
        .with_columns(
            (pl.col("component_pts") + pl.col("bonus_pts").fill_null(0)).alias("league_points")
        )
    )
    # Points/targets/carries are official box-score quantities; snaps are separately
    # observed and never described as charted routes.
    current_snaps = mapped_snaps(nflverse.load_snap_counts([args.season]), identities).filter(
        pl.col("week") <= cutoff
    )
    rookie_identity = (
        canonical_positions(identities)
        .filter(
            (pl.coalesce("rookie_season", "draft_year") == args.season)
            & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
            & pl.col("gsis_id").is_not_null()
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("player_display_name"),
            "position",
            pl.lit(args.season).alias("season"),
            "draft_pick",
            pl.col("espn_id").cast(pl.Int64, strict=False),
            pl.col("latest_team").alias("team"),
        )
    )
    history_path = args.history / "outputs/metric_backtest_predictions.parquet"
    historical_rookies = (
        pl.read_parquet(history_path)
        .filter(
            (pl.col("player_population") == "rookie")
            & pl.col("forecast_season").is_between(2013, args.season - 1)
            & pl.col("outcome_complete")
        )
        .select(
            "player_id",
            "player_display_name",
            "position",
            pl.col("forecast_season").alias("season"),
            pl.col("rookie_draft_pick").alias("draft_pick"),
        )
    )
    historical_points = pl.read_parquet(
        args.history / "outputs/research_audited_weekly_points.parquet"
    )
    snap_paths, identity_paths = [], []
    for path in (args.history / "cache/nflverse").glob("*.parquet"):
        columns = set(pl.scan_parquet(path).collect_schema())
        if {"pfr_player_id", "offense_snaps", "offense_pct"}.issubset(columns):
            snap_paths.append(path)
        if {"gsis_id", "pfr_id", "rookie_season"}.issubset(columns):
            identity_paths.append(path)
    frozen_identities = pl.concat([pl.read_parquet(p) for p in identity_paths])
    historical_snaps = mapped_snaps(
        pl.concat([pl.read_parquet(p) for p in snap_paths], how="diagonal_relaxed"),
        frozen_identities,
    ).join(historical_rookies.select("player_id", "season"), on=["player_id", "season"], how="semi")
    completed = set(historical_rookies["season"].to_list())
    history = rookie_rows(
        historical_rookies,
        historical_points,
        historical_snaps,
        cutoff,
        completed_seasons=completed,
    )
    live = rookie_rows(rookie_identity, current, current_snaps, cutoff, completed_seasons=set())
    print(
        f"Comparing {len(live)} rookies after Week {cutoff}; validating earlier seasons", flush=True
    )
    predictions = [{**r, **analog_forecast(r, history)} for r in live]
    predictions.sort(key=lambda r: (-(r["forecast_next4"] or 0), r["player_display_name"]))
    for position in ("QB", "RB", "WR", "TE"):
        ranked = [
            r for r in predictions if r["position"] == position and r["forecast_next4"] is not None
        ]
        for rank, row in enumerate(ranked, 1):
            row["position_rank"] = rank
    saved_at = datetime.now(UTC).isoformat()
    archive = (
        settings.outputs_dir
        / "rookie_snapshots"
        / f"{args.season}_week{cutoff}_{saved_at.replace(':', '-')}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    for name, frame in [
        ("players", rookie_identity),
        ("weeks", current),
        ("snaps", current_snaps),
        ("schedule", schedule),
        ("touchdown_plays", plays),
    ]:
        frame.write_parquet(archive / f"{name}.parquet")
    report = {
        "season": args.season,
        "through_week": cutoff,
        "horizon": [cutoff + 1, cutoff + 4],
        "saved_at": saved_at,
        "status": "experimental",
        "history": provenance,
        "history_seasons": sorted(completed),
        "current_source": "nflverse weekly stats, touchdown plays, schedules, and PFR snaps",
        "snapshot_path": str(archive.relative_to(settings.outputs_dir)),
        "source_sha256": {p.name: digest(p) for p in archive.glob("*.parquet")},
        "implementation_sha256": {
            "runner": digest(Path(__file__)),
            "model": digest(Path(__file__).parents[1] / "src/patron/metrics/rookies.py"),
        },
        "players": predictions,
        "backtest": backtest(history),
        "method": (
            "Mean next-four-calendar-week league points of the 25 nearest earlier rookies "
            "at the same position and completed-week cutoff (minimum 20 historical players). "
            "Uses scoring pace, targets/carries and latest-week usage, observed offensive snap "
            "share, and draft capital. Scales learned only from earlier seasons."
        ),
        "limitations": [
            "Small samples: early production is evidence, not a settled role.",
            "The 10th–90th percentile is the spread of historical analog outcomes, "
            "not a calibrated prediction interval.",
            "Four-week totals include historical absences and byes; "
            "no current injury, matchup, or future bye adjustment.",
            "Retrospectively revised historical feeds and player metadata "
            "are not original publication-time snapshots.",
            "Missing box-score weeks are not games played. "
            "No weekly production or offensive snap observation means no forecast.",
            "Walk-forward results test forecast error, "
            "not waiver profit or an advantage over market prices.",
            "Research only: draft boards, production forecasts, "
            "and frozen experiments are unchanged.",
        ],
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (archive / "report.json").write_text(encoded)
    destination = settings.outputs_dir / f"rookie_watch_{args.season}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(encoded)
    temporary.replace(destination)
    print(f"Published {destination}", flush=True)
    for row in predictions:
        if row["position"] == "WR" and row.get("position_rank", 999) <= 8:
            print(
                row["player_display_name"], round(row["forecast_next4"], 1), "next-four-week points"
            )
    print(json.dumps(report["backtest"]["positions"], indent=2))


if __name__ == "__main__":
    main()
