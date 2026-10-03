"""Corrected gold and full-career adapters for the individual-statistic audit."""

from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from engine.data.player_profiles import prepare_weeks
from engine.data.releases import digest, identifier, load_gold, write_json
from engine.data.target_quality import correct_tables, mask_columns
from engine.metrics.player_profile import KEYS, TOTALS, summarize
from engine.metrics.projection import _age_factor
from engine.tables.contracts import table_spec, validate_forecasts

ROOT = Path(__file__).resolve().parents[1]


def corrected_gold(data, version):
    parent = load_gold(data, "canonical_20260923_r5")
    if parent.ref["version"] != "canonical_20260923_r5":
        raise ValueError("Protocol requires published canonical_20260923_r5")
    root = data / "gold/releases" / identifier(version)
    root.mkdir(parents=True, exist_ok=False)
    (root / "tables").mkdir()
    tables = {name: parent.read(name) for name in parent.manifest["tables"]}
    fixed, correction = correct_tables(tables)
    validate_forecasts(fixed["preseason_features"], fixed["season_outcomes"])
    assert fixed["season_outcomes"].equals(tables["season_outcomes"])
    assert all(fixed[n].height == tables[n].height for n in tables)
    for name in ["nfl_player_weeks", "nfl_player_seasons", "preseason_features"]:
        assert not fixed[name].filter(pl.col("receptions") > pl.col("targets")).height
    specs = {}
    for name, frame in fixed.items():
        old = parent.manifest["tables"][name]
        path = root / "tables" / f"{name}.parquet"
        frame.sort(old["primary_key"]).write_parquet(path)
        specs[name] = {
            **table_spec(frame, old["primary_key"], old["role"], old["description"]),
            "path": str(path.relative_to(root)),
            "sha256": digest(path),
            "enriched_release": parent.manifest["input"]["version"],
        }
    write_json(root / "target_quality.json", correction)
    write_json(root / "parent_gold.json", parent.ref)
    write_json(
        root / "quality.json",
        {
            "quality_passed": True,
            "coverage_complete": False,
            "checks": {
                "labels_and_candidate_counts_unchanged": True,
                "known_receptions_do_not_exceed_known_targets": True,
                "preseason_contract": True,
                "transitive_target_mask": True,
            },
        },
    )
    shutil.copy2(Path(__file__), root / "individual_stat_data.py")
    manifest = {
        k: v
        for k, v in parent.manifest.items()
        if k not in ["files", "tables", "version", "generated_at"]
    }
    manifest.update(
        version=version,
        generated_at=datetime.now(UTC).isoformat(),
        tables=specs,
        parent_gold=parent.ref,
        research_only=True,
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", manifest)
    return load_gold(data, version)


def canonical_weeks(gold, candidates):
    candidate_ids = candidates["player_id"].unique()
    points = gold.read("nfl_player_weeks").filter(
        pl.col("player_id").is_in(candidate_ids.implode())
    )
    points = points.select(*KEYS, "team", "position", *TOTALS)
    ids = gold.read("players").select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    snaps = (
        gold.read("nfl_snap_counts")
        .filter(pl.col("game_type") == "REG")
        .join(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1")
        .filter(pl.col("player_id").is_in(candidate_ids.implode()))
        .select(*KEYS, "team", "position", "offense_snaps", "offense_pct")
    )
    return prepare_weeks(points, points, snaps)


def complete_summary(observations, prefix):
    """Do not turn a partial career total into a complete exposure estimate."""
    from profile_history import SUMMARY_FIELDS, VOLUMES

    s = summarize(observations)
    result = {f"{prefix}_{k}": s[k] for k in SUMMARY_FIELDS}
    for field in VOLUMES:
        complete = s[f"{field}_observations"] == s["observed_weeks"] and s["observed_weeks"] > 0
        result[f"{prefix}_{field}"] = s[field] if complete else None
        result[f"{prefix}_{field}_per_week"] = s[field] / s["observed_weeks"] if complete else None
    return result


def full_features(gold):
    """Rebuild the old profile design using only corrected gold source tables."""
    import profile_history as ph

    # Explicit corrected transform, local to this offline runner. Archived code/results stay intact.
    original_summary = ph.summary_features
    ph.summary_features = complete_summary
    original_college = ph.college_features

    def college(history, candidate, identity, methods, nfl):
        values = original_college(history, candidate, identity, methods, nfl)
        exposure = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}[
            candidate["position"]
        ]
        if nfl.get(f"career_{exposure}") is None and nfl["career_observed_weeks"] > 0:
            values["college_prior_weight"] = None
            for field in ("passing_yards", "rushing_yards", "receiving_yards"):
                values[f"college_weighted_{field}"] = None
        return values

    ph.college_features = college
    candidates = (
        gold.read("preseason_features")
        .join(
            gold.read("season_outcomes"),
            on=["player_id", "forecast_season", "forecast_cutoff_date"],
            validate="1:1",
        )
        .filter(pl.col("outcome_complete"))
    )
    candidates = candidates.with_columns(
        pl.col(
            "forecast_cutoff_date", "availability_latest_known_on", "cutoff_last_transaction_date"
        ).cast(pl.String)
    )
    players = gold.read("players").rename({"gsis_id": "player_id"})
    weeks = canonical_weeks(gold, candidates)
    injuries = (
        gold.read("nfl_injuries")
        .filter(pl.col("game_type") == "REG")
        .rename({"gsis_id": "player_id"})
    )
    # The manually collected news ledger is not part of gold. Keep its old formulas registered,
    # but do not smuggle a separate source into an experiment declared gold-only.
    try:
        rows = ph.build_features(
            candidates,
            weeks,
            players,
            gold.read("college_annual"),
            gold.read("college_identity_links"),
            injuries,
            [],
            progress=True,
        )
    finally:
        ph.summary_features = original_summary
        ph.college_features = original_college
    for r in rows:
        for key in ("news_starter", "news_competition", "news_backup"):
            r[key] = None
    frame = pl.DataFrame(rows, infer_schema_length=None)
    dispersion = dispersion_features(weeks.to_dicts(), rows)
    frame = frame.hstack(pl.DataFrame(dispersion, infer_schema_length=None))
    # Named legacy deterministic formulas whose inputs have gold contracts.
    formulas = {
        "age_factor": pl.struct("position", "age_at_season").map_elements(
            lambda r: (
                _age_factor(r["position"], r["age_at_season"])
                if r["age_at_season"] is not None
                else None
            ),
            return_dtype=pl.Float64,
        ),
        "rb_age_cliff": ((pl.col("position") == "RB") & (pl.col("age_at_season") >= 27.5)).cast(
            pl.Float64
        ),
        "early_career_score": ((4 - pl.col("player_experience")) / 3).clip(0, 1),
        "source_opportunities_pg_sq": pl.col("source_opportunities_pg").pow(2),
        "career_opportunities_log": pl.col("career_opportunities").log1p(),
        "career_pass_attempts_log": pl.col("career_pass_attempts").log1p(),
        "wtd_opp": pl.col("carries") + 2.2 * pl.col("targets"),
        "wopr": 1.5 * pl.col("target_share") + 0.7 * pl.col("air_yards_share"),
    }
    frame = frame.with_columns(expr.alias(k) for k, expr in formulas.items())
    frame = frame.with_columns(
        (pl.col("early_career_score") * pl.col("combine_speed_score")).alias("young_speed_score"),
        (pl.col("early_career_score") * pl.col("combine_burst_score")).alias("young_burst_score"),
    )
    frame = frame.with_columns(
        pl.when(pl.col("season_pts") > 0)
        .then(
            pl.col("entering_sophomore")
            * (
                (0.1 * pl.col("rushing_yards") + 6 * pl.col("rushing_tds")) / pl.col("season_pts")
            ).clip(0, 1)
        )
        .otherwise(0.0)
        .alias("sophomore_rush_share")
    )
    for numerator, denominator, name in [
        ("targets", "games", "targets_pg"),
        ("carries", "games", "carries_pg"),
        ("attempts", "games", "attempts_pg"),
        ("receptions", "games", "receptions_pg"),
    ]:
        frame = frame.with_columns(
            pl.when(pl.col(denominator) > 0)
            .then(pl.col(numerator) / pl.col(denominator))
            .alias(name)
        )
    return (
        frame,
        profile_specifications(rows),
        list(formulas)
        + [
            "young_speed_score",
            "young_burst_score",
            "sophomore_rush_share",
            "targets_pg",
            "carries_pg",
            "attempts_pg",
            "receptions_pg",
        ]
        + list(dispersion[0]),
    )


def profile_specifications(rows):
    from profile_history import specifications

    specs = specifications(rows)
    market = (
        "market_position_log_rank",
        "market_position_inverse_rank",
        "market_position_dispersion",
        "market_overall_log_rank",
        "market_overall_inverse_rank",
        "market_overall_dispersion",
    )
    specs["basic_market"] = (*specs["basic"], *market)
    specs["profile_market"] = (*specs["profile"], *market)
    return specs


def additional_legacy_features(frame):
    """Exact deterministic legacy interactions, with unknown dependencies preserved."""
    expressions = {
        "preseason_status_score": pl.col("cutoff_preseason_status_score"),
        "preseason_rostered": pl.col("cutoff_preseason_rostered"),
        "preseason_reserve": pl.col("cutoff_preseason_reserve"),
        "contract_depth_security": pl.col("contract_years_remaining") * pl.col("depth_role_factor"),
    }
    access = (
        pl.when(pl.col("depth_chart_rank") > 0)
        .then(1.0 / pl.col("depth_chart_rank"))
        .otherwise(None)
    )
    for kind in ["target", "carry"]:
        expression = pl.col(f"team_vacated_{kind}_share") * access
        expressions[f"vacated_{kind}_opportunity"] = expression
        expressions[f"vacated_{kind}_early_career"] = expression * pl.col("early_career_score")
    return frame.with_columns(v.alias(k) for k, v in expressions.items()), list(expressions)


def scoring_distribution(values):
    """Observed-week scores: retain zero/negative values and expose the sample size."""
    x = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    result = {
        k: None
        for k in (
            "mean",
            "median",
            "std",
            "variance",
            "cv",
            "top2_positive_share",
            "mean_without_top2",
            "q25",
        )
    }
    result["observations"] = len(x)
    if len(x):
        result.update(
            mean=float(x.mean()), median=float(np.median(x)), q25=float(np.quantile(x, 0.25))
        )
    if len(x) >= 2:
        result.update(
            std=float(x.std()),
            variance=float(x.var()),
            cv=float(x.std() / x.mean()) if x.mean() > 0 else None,
        )
        positive = np.maximum(x, 0)
        result["top2_positive_share"] = (
            float(np.sort(positive)[-2:].sum() / positive.sum()) if positive.sum() else None
        )
    if len(x) >= 3:
        result["mean_without_top2"] = float(np.sort(x)[:-2].mean())
    return result


def dispersion_features(weeks, candidates):
    history = defaultdict(list)
    for row in weeks:
        if row.get("stat_recorded") or (row.get("offense_snaps") or 0) > 0:
            history[row["player_id"]].append(row)
    results = []
    for row in candidates:
        year = row["forecast_season"]
        previous = [w for w in history[row["player_id"]] if w["season"] < year]
        values = {}
        for label, subset in (
            ("prior", [w for w in previous if w["season"] == year - 1]),
            ("recent3", [w for w in previous if w["season"] >= year - 3]),
            ("career", previous),
        ):
            values.update(
                {
                    f"distribution_{label}_{k}": v
                    for k, v in scoring_distribution(
                        [w.get("league_points") for w in subset]
                    ).items()
                }
            )
        yearly = defaultdict(list)
        for w in previous:
            if w.get("league_points") is not None:
                yearly[w["season"]].append(w["league_points"])
        means = [float(np.mean(v)) for v in yearly.values()]
        values["distribution_career_season_mean_std"] = (
            float(np.std(means)) if len(means) >= 2 else None
        )
        values["distribution_career_season_observations"] = len(means)
        results.append(values)
    return results


def legacy_projection_features(gold, frame):
    """Recompute the legacy projection formulas from gold, retaining unknown dependencies."""
    from engine.board.builder import build_board
    from engine.board.overrides import OverrideSet
    from engine.config.league import get_league
    from engine.metrics.projection import ProjectionAssumptions, build_projection_board

    seasons = gold.read("nfl_player_seasons")
    birthdays = gold.read("players").select(pl.col("gsis_id").alias("player_id"), "birth_date")
    frames = []
    protected = {
        "historical_ppg_prior",
        "individual_prior_ppg",
        "prior_branch_ppg",
        "age_factor",
        "depth_role_factor",
        "effective_games",
        "projection_confidence",
        "availability_confidence",
        "projected_availability",
        "expected_games",
        "historical_injury_report_weeks",
        "injury_missed_equivalents",
        "historical_vor",
        "historical_repl_ppg",
        "opportunity_multiplier",
    }
    for year in sorted(frame["forecast_season"].unique()):
        history = seasons.filter(pl.col("season").is_between(year - 3, year - 1))
        config = get_league().model_copy(
            update={
                "seasons": list(range(year - 3, year)),
                "board_season": year - 1,
                "draft_season": year,
            }
        )
        board = build_board(
            history, birthdays, config=config, override_set=OverrideSet(), strict_overrides=False
        )
        current = frame.filter(pl.col("forecast_season") == year).select(
            "player_id",
            pl.col("cutoff_preseason_team").alias("current_team"),
            "depth_chart_rank",
            "depth_chart_position",
            "depth_chart_position_group",
            "depth_chart_date",
        )
        result = build_projection_board(
            history, board, config, assumptions=ProjectionAssumptions(), current_players=current
        )
        new = [
            c
            for c, dtype in result.schema.items()
            if (dtype.is_numeric() or dtype == pl.Boolean)
            and c not in frame.columns
            and c not in ["rank", "override_delta"]
        ]
        # Formula implementation fills missing receiving values with zero. Such composites
        # cannot be admitted when their position-level prior or player history lacks targets.
        if history.filter(pl.col("targets").is_null()).height:
            result = mask_columns(result, pl.lit(True), [c for c in new if c not in protected])
        frames.append(
            result.select("player_id", *new).with_columns(
                pl.lit(year).cast(pl.Int32).alias("forecast_season")
            )
        )
    legacy = pl.concat(frames, how="diagonal_relaxed")
    names = [c for c in legacy.columns if c not in ["player_id", "forecast_season"]]
    return frame.join(
        legacy, on=["player_id", "forecast_season"], how="left", validate="1:1"
    ), names


def write_features(data, root, gold_version):
    """Create a new research matrix from a verified corrected gold release."""
    root.mkdir(parents=True, exist_ok=False)
    gold = load_gold(data, gold_version)
    if (
        not gold.manifest.get("research_only")
        or "target_quality.json" not in gold.manifest["files"]
    ):
        raise ValueError("Individual-stat study requires the corrected research gold release")
    frame, specs, formulas = full_features(gold)
    frame, legacy = legacy_projection_features(gold, frame)
    frame, extra = additional_legacy_features(frame)
    frame.write_parquet(root / "features.parquet")
    write_json(root / "specs.json", specs)
    write_json(root / "formulas.json", formulas + extra)
    write_json(root / "legacy_formulas.json", legacy)
    write_json(root / "gold.json", gold.ref)
    current = (
        gold.read("preseason_features")
        .filter(pl.col("forecast_season") == 2026)
        .sort("position", "player_id")
    )
    weeks = canonical_weeks(gold, current)
    distributions = pl.DataFrame(
        dispersion_features(weeks.to_dicts(), current.to_dicts()), infer_schema_length=None
    )
    current = current.select(
        "player_id",
        "player_display_name",
        "position",
        "forecast_season",
        "player_population",
        "market_ecr",
        "age_at_season",
        "ppg",
        "floor",
        "volatility",
        "games",
    ).hstack(distributions)
    current.write_parquet(root / "draft_consistency_2026.parquet")
    current.write_csv(root / "draft_consistency_2026.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--gold-version", default="canonical_targets_20260923_r3")
    args = parser.parse_args()
    write_features(args.data, args.root, args.gold_version)
