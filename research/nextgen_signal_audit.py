"""Full-history observation-family comparisons sourced only from verified enriched gold."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from patron.data.gold import FORECAST_KEY, unique
from patron.data.player_profiles import prepare_weeks
from patron.data.releases import digest, load_gold, write_json
from patron.metrics.player_profile import KEYS, TOTALS, career_features

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/nextgen_signal_protocol.md"
POSITIONS = ("QB", "RB", "WR", "TE")
ERAS = {"all": (2004, 2025), "early": (2007, 2012), "middle": (2013, 2018), "modern": (2019, 2025)}
POINTS = ["ppg", "games"]
USAGE = [
    *POINTS,
    "attempts_pg",
    "carries_pg",
    "targets_pg",
    "receptions_pg",
    "target_share",
    "air_yards_share",
]
MARKET = [
    "ecr_log",
    "ecr_inverse",
    "overall_ecr_log",
    "overall_ecr_inverse",
    "market_ecr_sd",
    "market_overall_ecr_sd",
]
DRAFT = ["rookie_draft_capital_score", "rookie_was_drafted"]
FAMILIES = {
    "history": [
        "recent3_points_per_observed_week",
        "recent3_observed_weeks",
        "career_points_per_observed_week",
        "completed_observed_weeks",
        "completed_seasons_observed",
        "career_targets",
        "career_carries",
        "career_attempts",
    ],
    "age": ["age_at_season", "player_experience", "entering_sophomore"],
    "snaps": [
        "source_offense_snap_pct",
        "late_offense_snap_pct",
        "source_offense_snaps_pg",
        "offense_snap_pct_trend",
        "high_points_per_observed_week",
        "high_observed_weeks",
        "rotation_points_per_observed_week",
        "rotation_observed_weeks",
        "limited_points_per_observed_week",
        "limited_observed_weeks",
    ],
    "weekly_shape": [
        f"rich_s0_{series}_{stat}"
        for series in ("points", "opportunities")
        for stat in ("std", "slope", "last4_delta", "late_delta", "zero_rate")
    ],
    "scoring_opportunity": [
        "red_zone_carries_pg",
        "goal_line_carries_pg",
        "red_zone_targets_pg",
        "end_zone_targets_pg",
        "designed_carries_pg",
        "scramble_carries_pg",
    ],
    "efficiency": [
        "passing_yards_per_attempt",
        "rushing_yards_per_carry",
        "receiving_yards_per_target",
        "passing_epa_per_attempt",
        "passing_cpoe",
        "passing_first_down_rate",
        "rushing_first_down_rate",
        "receiving_first_down_rate",
    ],
    "team_environment": [
        "neutral_pass_rate",
        "neutral_epa_per_play",
        "neutral_pass_oe",
        "neutral_plays_pg",
    ],
    "availability": [
        "injury_report_weeks",
        "out_report_weeks",
        "doubtful_report_weeks",
        "cutoff_state_observed",
        "cutoff_preseason_rostered",
        "cutoff_preseason_reserve",
        "cutoff_injured_reserve",
        "cutoff_pup_nfi",
        "cutoff_suspended",
        "cutoff_reserve_other",
        "known_absence_games",
        "known_available_games_cap",
        "team_changed",
    ],
    "routes": ["route_participation", "targets_per_route_opportunity"],
    "tracking": ["ngs_cpoe", "ngs_separation", "ngs_yac_oe", "ngs_ryoe_per_att"],
    "draft_combine": [
        "draft_capital_score",
        "was_drafted",
        "combine_weight",
        "combine_speed_score",
        "combine_burst_score",
        "combine_agility_score",
    ],
}
COLLEGE = [
    "college_latest_passing_yards",
    "college_latest_rushing_yards",
    "college_latest_receiving_yards",
    "college_latest_receptions",
    "college_latest_share_receiving_yards",
    "college_latest_share_carries",
    "college_best_share_receiving_yards",
    "college_best_share_carries",
    "college_complete_seasons",
]
ROOKIE_FAMILIES = {
    "age_combine": [
        "rookie_age",
        "combine_weight",
        "combine_speed_score",
        "combine_burst_score",
        "combine_agility_score",
    ],
    "college": COLLEGE,
}
SUPPORT = {
    **FAMILIES,
    **ROOKIE_FAMILIES,
    "history": ["recent3_points_per_observed_week"],
    "snaps": ["source_offense_snap_pct", "high_points_per_observed_week"],
    "availability": ["cutoff_preseason_rostered", "known_available_games_cap"],
    "college": ["college_latest_passing_yards"],
    "market": ["ecr_log", "overall_ecr_log"],
}


def nominal_games(year):
    return 17 if year >= 2021 else 16


def quarantine_targets(features, weeks):
    """An enriched-derived view: retain rows, mask demonstrably broken source seasons."""
    audit = (
        weeks.filter(pl.col("position").is_in(POSITIONS))
        .group_by("season")
        .agg(
            pl.len().alias("weeks"),
            pl.col("targets").sum().alias("targets"),
            pl.col("receptions").sum().alias("receptions"),
            (pl.col("receptions") > pl.col("targets")).sum().alias("impossible_pairs"),
        )
        .sort("season")
    )
    bad = audit.filter(
        (pl.col("targets") < pl.col("receptions")) & (pl.col("impossible_pairs") >= 10)
    )["season"].to_list()
    affected = features.filter(pl.col("source_season").is_in(bad))
    columns = ["targets", "target_share", "air_yards_share", "receiving_first_down_rate"]
    columns += [c for c in features.columns if c.startswith("rich_s0_opportunities_")]
    view = features.with_columns(
        pl.when(pl.col("source_season").is_in(bad)).then(None).otherwise(pl.col(c)).alias(c)
        for c in columns
    )
    observations = weeks.with_columns(
        pl.when(pl.col("season").is_in(bad))
        .then(None)
        .otherwise(pl.col("targets"))
        .alias("targets")
    )
    report = dict(
        source_seasons=bad,
        season_audit=audit.to_dicts(),
        affected_candidate_rows=affected.height,
        impossible_candidate_rows=affected.filter(pl.col("receptions") > pl.col("targets")).height,
        masked_forecast_fields=columns,
        masked_weekly_fields=["targets"],
        policy="Retain every candidate and every season; unknown target counts are not zero. "
        "Do not infer targets from receptions. No upstream release is rewritten.",
    )
    return view, observations, report


def finite_columns(frame, columns):
    return pl.any_horizontal(pl.col(columns).cast(pl.Float64).is_finite().fill_null(False))


def admitted(train, columns):
    observed = train.filter(finite_columns(train, columns))
    return observed.group_by("forecast_season").len().filter(pl.col("len") >= 10).height >= 3


def college_features(candidates, annual, links):
    """Only complete pre-entry seasons; ambiguous multi-link years stay unknown."""
    by_id, linked = defaultdict(list), defaultdict(list)
    for row in annual.to_dicts():
        by_id[row["college_id"]].append(row)
    for row in links.filter(pl.col("status") == "linked").to_dicts():
        linked[row["player_id"]].append(row["college_id"])
    result = []
    for row in candidates.filter(pl.col("player_population") == "rookie").to_dicts():
        prior = [
            r
            for cid in linked[row["player_id"]]
            for r in by_id[cid]
            if r["season"] < row["forecast_season"]
            and str(r["season_end_date"])[:10] <= str(row["forecast_cutoff_date"])
        ]
        unambiguous = len({r["season"] for r in prior}) == len(prior)
        good = (
            sorted(
                [r for r in prior if r["complete_team_season"] and r["team_count"] == 1],
                key=lambda r: r["season"],
            )
            if unambiguous
            else []
        )
        last = good[-1] if good else {}
        values = {
            "player_id": row["player_id"],
            "forecast_season": row["forecast_season"],
            "college_complete_seasons": len(good),
        }
        for name in COLLEGE:
            if name.startswith("college_latest_"):
                values[name] = last.get(name.removeprefix("college_latest_"))
            elif name.startswith("college_best_"):
                numbers = [r[name.removeprefix("college_best_")] for r in good]
                values[name] = max((v for v in numbers if v is not None), default=None)
        result.append(values)
    return pl.DataFrame(result, infer_schema_length=None)


def build_features(gold):
    features = gold.read("preseason_features")
    labels = gold.read("season_outcomes")
    unique(features, FORECAST_KEY, "features")
    frame = features.join(labels, on=FORECAST_KEY, validate="1:1").filter(
        pl.col("outcome_complete")
    )
    if frame["forecast_season"].max() > 2025:
        raise ValueError("Pending outcomes must not be graded")
    ids = frame["player_id"].unique()
    points = gold.read("nfl_player_weeks").filter(pl.col("player_id").is_in(ids.implode()))
    points = points.select(*KEYS, "team", "position", *TOTALS)
    frame, points, target_quality = quarantine_targets(frame, points)
    players = (
        gold.read("players").select(pl.col("gsis_id").alias("player_id"), "pfr_id").drop_nulls()
    )
    unique(players, ["pfr_id"], "snap crosswalk")
    snaps = (
        gold.read("nfl_snap_counts")
        .filter(pl.col("game_type") == "REG")
        .join(players, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1")
        .filter(pl.col("player_id").is_in(ids.implode()))
        .select(*KEYS, "team", "position", "offense_snaps", "offense_pct")
    )
    weeks = prepare_weeks(points, points, snaps)
    by_id = defaultdict(list)
    for row in weeks.to_dicts():
        by_id[row["player_id"]].append(row)
    profiles = []
    for (year,), group in frame.sort("forecast_season").group_by(
        "forecast_season", maintain_order=True
    ):
        print(f"Aggregating completed player histories for {year}", flush=True)
        for row in group.select("player_id", "forecast_season").to_dicts():
            history = by_id[row["player_id"]]
            values = career_features(history, year)
            if any(r.get("targets") is None for r in history if r["season"] < year):
                values["career_targets"] = None
            profiles.append({**row, **values})
    frame = frame.join(
        pl.DataFrame(profiles, infer_schema_length=None),
        on=["player_id", "forecast_season"],
        validate="1:1",
    )
    frame = frame.join(
        college_features(frame, gold.read("college_annual"), gold.read("college_identity_links")),
        on=["player_id", "forecast_season"],
        how="left",
        validate="1:1",
    )
    for field in (
        "attempts",
        "carries",
        "targets",
        "receptions",
        "red_zone_carries",
        "goal_line_carries",
        "red_zone_targets",
        "end_zone_targets",
        "designed_carries",
        "scramble_carries",
    ):
        frame = frame.with_columns(
            pl.when(pl.col("games") > 0).then(pl.col(field) / pl.col("games")).alias(field + "_pg")
        )
    for numerator, denominator, name in [
        ("passing_yards", "attempts", "passing_yards_per_attempt"),
        ("rushing_yards", "carries", "rushing_yards_per_carry"),
        ("receiving_yards", "targets", "receiving_yards_per_target"),
    ]:
        frame = frame.with_columns(
            pl.when(pl.col(denominator) > 0)
            .then(pl.col(numerator) / pl.col(denominator))
            .alias(name)
        )
    for field, prefix in [("market_ecr", "ecr"), ("market_overall_ecr", "overall_ecr")]:
        frame = frame.with_columns(
            pl.when(pl.col(field) > 0).then(pl.col(field).log()).alias(prefix + "_log"),
            pl.when(pl.col(field) > 0).then(1 / pl.col(field)).alias(prefix + "_inverse"),
        )
    frame = frame.with_columns(
        (
            pl.col("actual_season_points")
            / pl.when(pl.col("forecast_season") >= 2021).then(17).otherwise(16)
        ).alias("target")
    )
    return frame, target_quality


def coverage(frame, columns):
    result = []
    for (year, pos, pop), group in frame.group_by(
        "forecast_season", "position", "player_population"
    ):
        for column in columns:
            values = group[column]
            observed = values.sum() if values.dtype == pl.Boolean else values.is_finite().sum()
            result.append(
                dict(
                    season=year,
                    position=pos,
                    population=pop,
                    feature=column,
                    candidates=group.height,
                    observed=int(observed),
                    semantics="true" if values.dtype == pl.Boolean else "finite",
                )
            )
    return pl.DataFrame(result).sort("feature", "season", "position", "population")


def fit_predict(train, test, columns):
    if train["forecast_season"].max() >= test["forecast_season"].min():
        raise ValueError("Training overlaps held-out season")
    if train["position"].n_unique() != 1 or train["player_population"].n_unique() != 1:
        raise ValueError("Mixed training population")
    if set(train["position"]) != set(test["position"]) or set(train["player_population"]) != set(
        test["player_population"]
    ):
        raise ValueError("Training/test population mismatch")
    if columns:
        columns = list(dict.fromkeys(columns))
        x = train.select(pl.col(columns).cast(pl.Float64)).to_numpy()
        xt = test.select(pl.col(columns).cast(pl.Float64)).to_numpy()
        x, xt = np.where(np.isfinite(x), x, np.nan), np.where(np.isfinite(xt), xt, np.nan)
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=100),
        )
        with threadpool_limits(limits=1):
            model.fit(x, train["target"].to_numpy())
            values = model.predict(xt)
    else:
        values = np.full(test.height, train["target"].mean())
    values = np.maximum(values, 0) * nominal_games(test["forecast_season"][0])
    values[test["known_available_games_cap"].fill_null(-1).to_numpy() == 0] = 0
    return values


def predict(frame):
    predictions, folds = [], []
    for population, families, base in [
        ("returner", FAMILIES, USAGE),
        ("rookie", ROOKIE_FAMILIES, DRAFT),
    ]:
        for position in POSITIONS:
            pool = frame.filter(
                (pl.col("position") == position) & (pl.col("player_population") == population)
            )
            for year in sorted(pool["forecast_season"].unique()):
                train, test = (
                    pool.filter(pl.col("forecast_season") < year),
                    pool.filter(pl.col("forecast_season") == year),
                )
                if train.height < 30 or train["forecast_season"].n_unique() < 3:
                    continue
                values = {
                    "simple": fit_predict(train, test, POINTS if population == "returner" else []),
                    "base": fit_predict(train, test, base),
                }
                market_active = admitted(train, SUPPORT["market"])
                values["market"] = (
                    fit_predict(train, test, base + MARKET) if market_active else values["base"]
                )
                active = {name: admitted(train, SUPPORT[name]) for name in families}
                for name, columns in families.items():
                    values[name] = (
                        fit_predict(train, test, base + columns) if active[name] else values["base"]
                    )
                    values[name + "_market"] = (
                        fit_predict(train, test, base + MARKET + columns)
                        if active[name] and market_active
                        else values["market"]
                    )
                all_columns = [c for name, cols in families.items() if active[name] for c in cols]
                values["all_families"] = fit_predict(train, test, base + all_columns)
                values["all_families_market"] = (
                    fit_predict(train, test, base + MARKET + all_columns)
                    if market_active
                    else values["all_families"]
                )
                active.update(
                    all_families=bool(all_columns), simple=True, base=True, market=market_active
                )
                row = dict(
                    season=year,
                    position=position,
                    population=population,
                    train_rows=train.height,
                    test_rows=test.height,
                    first_train_year=train["forecast_season"].min(),
                    last_train_year=train["forecast_season"].max(),
                    active=active,
                )
                folds.append(row)
                metadata = test.select(
                    *FORECAST_KEY,
                    "player_display_name",
                    "position",
                    "player_population",
                    "actual_season_points",
                    "games",
                    "market_ecr",
                    "market_overall_ecr",
                )
                metadata = metadata.with_columns(
                    [pl.Series("pred_" + k, v) for k, v in values.items()]
                )
                predictions.append(
                    metadata.with_columns(
                        [pl.lit(v).alias("active_" + k) for k, v in active.items()]
                    )
                )
            print(f"Finished {population} {position}: all earlier seasons retained", flush=True)
    return pl.concat(predictions, how="diagonal_relaxed"), folds


def season_summary(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite season differences")
    interval = None
    leave_one_out = None
    if len(x) > 1:
        boot = np.random.default_rng(20260923).choice(x, (10000, len(x))).mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
        omitted = (x.sum() - x) / (len(x) - 1)
        leave_one_out = [float(omitted.min()), float(omitted.max())]
    return dict(
        mean=float(x.mean()),
        bootstrap_95=interval,
        seasons=len(x),
        positive_seasons=int((x > 1e-9).sum()),
        leave_one_year_out=leave_one_out,
    )


def comparisons(predictions):
    results = []
    for population, families in [("returner", FAMILIES), ("rookie", ROOKIE_FAMILIES)]:
        comparisons = [("base", "simple", "base", False), ("market", "base", "market", True)]
        comparisons += [(name, "base", name, False) for name in [*families, "all_families"]]
        if population == "returner":
            comparisons += [(name, "simple", name, False) for name in [*families, "all_families"]]
        comparisons += [
            (name + "_market", "market", name, True) for name in [*families, "all_families"]
        ]
        for position in POSITIONS:
            pool = predictions.filter(
                (pl.col("player_population") == population) & (pl.col("position") == position)
            )
            for challenger, baseline, active, market in comparisons:
                pair = pool.filter(pl.col("active_" + active))
                if market:
                    pair = pair.filter(
                        pl.col("active_market")
                        & finite_columns(pair, ["market_ecr", "market_overall_ecr"])
                    )
                for era, (start, end) in ERAS.items():
                    for cohort in ["all", "small_prior"] if population == "returner" else ["all"]:
                        sample = pair.filter(pl.col("forecast_season").is_between(start, end))
                        if cohort == "small_prior":
                            sample = sample.filter(pl.col("games") < 10)
                        if sample.is_empty():
                            continue
                        by_year = []
                        for (year,), group in sample.group_by("forecast_season"):
                            y = group["actual_season_points"].to_numpy()
                            b, c = (group["pred_" + k].to_numpy() for k in (baseline, challenger))
                            if not np.isfinite(np.concatenate([y, b, c])).all():
                                raise ValueError("Nonfinite paired prediction")
                            by_year.append(
                                dict(
                                    season=year,
                                    n=len(y),
                                    baseline_mae=float(np.abs(y - b).mean()),
                                    challenger_mae=float(np.abs(y - c).mean()),
                                    mae_gain=float((np.abs(y - b) - np.abs(y - c)).mean()),
                                    mse_gain=float(((y - b) ** 2 - (y - c) ** 2).mean()),
                                )
                            )
                        by_year.sort(key=lambda r: r["season"])
                        results.append(
                            dict(
                                population=population,
                                position=position,
                                era=era,
                                cohort=cohort,
                                challenger=challenger,
                                baseline=baseline,
                                rows=sample.height,
                                baseline_mae=float(np.mean([r["baseline_mae"] for r in by_year])),
                                challenger_mae=float(
                                    np.mean([r["challenger_mae"] for r in by_year])
                                ),
                                mae_gain=season_summary([r["mae_gain"] for r in by_year]),
                                mse_gain=season_summary([r["mse_gain"] for r in by_year]),
                                folds=by_year,
                            )
                        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    gold = load_gold(ROOT / "data", args.gold)
    if gold.manifest["input"]["version"] != "canonical_20260923_r4":
        raise ValueError("Protocol pins enriched canonical_20260923_r4")
    from patron.data.releases import identifier

    output = ROOT / "data/research" / identifier(args.version)
    output.mkdir(parents=True, exist_ok=False)
    protected_paths = list((ROOT / "data/outputs").glob("board*.json"))
    protected_paths += [ROOT / "src/patron/config/experimental_freeze_2026.yaml"]
    protected_paths += [ROOT / "data/static/experimental_2026_predictions.csv"]
    protected = {str(p.relative_to(ROOT)): digest(p) for p in protected_paths if p.exists()}
    implementation = output / "implementation"
    implementation.mkdir()
    sources = [
        Path(__file__),
        PROTOCOL,
        ROOT / "research/test_nextgen_signal_audit.py",
        ROOT / "src/patron/metrics/player_profile.py",
        ROOT / "src/patron/data/player_profiles.py",
        ROOT / "src/patron/data/releases.py",
        ROOT / "src/patron/data/gold.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        shutil.copy2(source, implementation / source.name)
    frame, target_quality = build_features(gold)
    write_json(output / "target_quality.json", target_quality)
    feature_columns = sorted(
        set(
            USAGE
            + MARKET
            + DRAFT
            + COLLEGE
            + [c for cols in [*FAMILIES.values(), *ROOKIE_FAMILIES.values()] for c in cols]
        )
    )
    extra_coverage = [
        "contract_apy_cap_pct",
        "team_vacated_target_share",
        "team_vacated_carry_share",
        "depth_chart_rank",
    ]
    coverage(frame, feature_columns + extra_coverage).write_parquet(output / "coverage.parquet")
    keep = list(
        dict.fromkeys(
            [
                *FORECAST_KEY,
                "player_display_name",
                "position",
                "player_population",
                "actual_season_points",
                "target",
                "known_available_games_cap",
                *feature_columns,
            ]
        )
    )
    frame.select(keep).write_parquet(output / "features.parquet")
    predictions, folds = predict(frame)
    predictions.sort("player_population", "position", "forecast_season", "player_id").write_parquet(
        output / "predictions.parquet"
    )
    report = dict(
        generated_at=datetime.now(UTC).isoformat(),
        research_only=True,
        gold=gold.ref,
        enriched=gold.manifest["input"],
        protocol_sha256=digest(PROTOCOL),
        candidates=frame.height,
        predicted_candidates=predictions.height,
        quarantined_candidates=gold.manifest["tables"]["forecast_quarantine"]["rows"],
        populations=frame.group_by("player_population").len().to_dicts(),
        families=FAMILIES,
        rookie_families=ROOKIE_FAMILIES,
        support_columns=SUPPORT,
        folds=folds,
        comparisons=comparisons(predictions),
        limitations=[
            "Retrospective exploratory comparisons; intervals not adjusted for many tests.",
            "Fixed ridge family tests do not isolate individual feature value or causal impact.",
            "Gold acceptance verifies reconstruction, not original publication vintages.",
            "Market lane uses learned ECR covariates, not raw ECR ranking or executable ADP.",
            "Market-only candidates inventoried but not modeled; no 2026 forecast is fitted.",
            "Previously reviewed seasons are not an independent confirmation set.",
        ],
    )
    write_json(output / "report.json", report)
    if any(digest(ROOT / p) != h for p, h in protected.items()):
        raise ValueError("Protected artifacts changed during audit")
    if any(digest(ROOT / p) != h for p, h in code_hashes.items()):
        raise ValueError("Implementation changed during audit")
    # Re-verify the pinned dependency closure before sealing results.
    if load_gold(ROOT / "data", args.gold).ref != gold.ref:
        raise ValueError("Gold changed during audit")
    write_json(
        output / "manifest.json",
        dict(
            status="complete",
            gold=gold.ref,
            enriched=gold.manifest["input"],
            implementation_sha256=code_hashes,
            protected_sha256=protected,
            protected_unchanged=True,
            packages={p: version(p) for p in ("numpy", "polars", "scikit-learn")},
            output_sha256={
                str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
            },
        ),
    )
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
