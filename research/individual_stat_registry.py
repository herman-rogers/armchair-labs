"""Exhaustive inventory: absence from a fit is never evidence that a stat is harmful."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from patron.data.gold import RICH_SERIES

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {
    "td_over_exp",
    "BUY",
    "td_luck",
    "td_regression_adjustment",
    "projected_bonus_pg",
    "bonus_regression_adjustment",
    "team_context_factor",
    "qb_context",
    "wopr",
    "wtd_opp",
    "v2_score",
    "floor_vor",
    "ceiling_vor",
    "availability_adjusted_vor",
    "expected_season_points",
    "rosterable_pool_prior",
    "projected_ceiling",
    "teammate_competition_target_availability",
}
METADATA = {
    "forecast_season",
    "source_season",
    "season",
    "prior_season",
    "eligible",
    "role_games",
    "profile_identity_known",
    "profile_weeks_known",
    "rank",
    "outcome_complete",
    "review_cohort",
}


def is_market(name):
    return name.startswith("market_") or name in {"log_ecr", "inverse_ecr"}


def units_and_dependencies(name, legacy):
    unit = "source-defined; see formula and pinned schema"
    if name.startswith("distribution_"):
        unit = (
            "observed weeks/seasons"
            if name.endswith("observations")
            else "fraction"
            if name.endswith(("_cv", "_share"))
            else "points squared"
            if name.endswith("_variance")
            else "league points per observed week"
        )
        dependency = (
            "gold.nfl_player_weeks.league_points; gold.nfl_snap_counts; earlier seasons only"
        )
    elif name.startswith("rich_"):
        unit = (
            "fraction"
            if name.endswith(
                ("_coverage", "_zero_rate", "_lag1", "_entropy", "_peak_week", "_end_streak")
            )
            else "named weekly-series units; see formula"
        )
        dependency = "gold.nfl_weekly_usage named series; preceding source seasons s0/s1/s2"
    elif name in legacy:
        dependency = (
            "gold.nfl_player_seasons previous three seasons; birth dates; cutoff depth; "
            "pinned league/projection config"
        )
    elif name.startswith("college_"):
        dependency = "gold.college_annual; college_identity_links; NFL entry year; cutoff"
    elif name.startswith("injury_"):
        dependency = "gold.nfl_injuries before forecast season; source feed availability"
    elif name.startswith(("career_", "basic_prior_")):
        dependency = "gold.nfl_player_weeks; nfl_snap_counts; player identity; forecast cutoff"
    else:
        dependency = "gold.preseason_features or stated deterministic transform; see definition_ref"
    if name in ["ppg", "floor", "volatility", "historical_ppg_prior", "proj_ppg"]:
        unit = "league points per game (source-specific denominator)"
    elif name in ["age_at_season", "basic_age", "player_experience", "basic_experience"]:
        unit = "years"
    elif name.endswith("_factor") or name == "rb_age_cliff":
        unit = "dimensionless multiplier/indicator"
    return unit, dependency


def family(name, catalog):
    if is_market(name):
        return "market"
    if name.startswith("rich_"):
        for s in sorted(RICH_SERIES, key=len, reverse=True):
            if name.startswith(f"rich_{s}_") or any(
                name.startswith(f"rich_s{i}_{s}_") for i in range(3)
            ):
                return "weekly_" + s
    for prefix in [
        "distribution_prior",
        "distribution_recent3",
        "distribution_career",
        "basic_prior",
        "basic",
        "career_recent3",
        "career_high",
        "career_rotation",
        "career_limited",
        "career_unknown",
        "career",
        "college",
        "injury",
        "context",
        "market",
        "news",
    ]:
        if name.startswith(prefix + "_"):
            return prefix
    if name in catalog:
        return catalog[name]["group"]
    if any(s in name for s in ["snap", "route"]):
        return "participation"
    if any(s in name for s in ["age", "draft", "rookie", "combine", "experience", "sophomore"]):
        return "age_capital_athleticism"
    if any(
        s in name for s in ["cutoff", "absence", "suspension", "available", "contract", "vacated"]
    ):
        return "preseason_context"
    return "production_usage"


def definition(name, catalog):
    if name in catalog:
        return catalog[name]["description"], "src/patron/config/metric_report.yaml"
    if name.startswith("distribution_"):
        suffix = name.removeprefix("distribution_").split("_", 1)[1]
        meanings = dict(
            mean="Arithmetic mean of observed weekly league points",
            median="Median observed weekly league points",
            std="Population standard deviation of observed weekly league points; >=2 observations",
            variance="Population variance of observed weekly league points; >=2 observations",
            cv="Population SD / arithmetic mean, only when mean >0 and >=2 observations",
            top2_positive_share=(
                "Sum of largest two max(points,0) / sum max(points,0); "
                ">=2 observations and positive total"
            ),
            mean_without_top2=(
                "Mean weekly points after removing the highest two observations; >=3 observations"
            ),
            q25="25th percentile of observed weekly points (linear interpolation)",
            observations=(
                "Count of finite scoring observations; includes observed zero and negative scores"
            ),
            season_mean_std="Population SD across prior season means; >=2 seasons",
            season_observations="Count of earlier seasons with finite scoring observations",
        )
        return (
            meanings[suffix] + "; prior=previous season, recent3=previous three, "
            "career=all available prior seasons.",
            "research/individual_stat_data.py:scoring_distribution/dispersion_features",
        )
    if name.startswith("rich_"):
        signal = next(
            s
            for s in sorted(RICH_SERIES, key=len, reverse=True)
            if name.startswith(f"rich_{s}_")
            or any(name.startswith(f"rich_s{i}_{s}_") for i in range(3))
        )
        suffix = name.split(signal + "_", 1)[1]
        formulas = {
            "coverage": "count(finite x)/18",
            "mean": "mean(finite x)",
            "std": "population SD(finite x)",
            "slope": "OLS slope of finite x against calendar week; 0 with one observation",
            "last4": "mean(finite x in calendar weeks 15-18)",
            "last4_delta": "mean(weeks 15-18) - mean(weeks 11-14)",
            "late_delta": "mean(weeks 10-18) - mean(weeks 1-9)",
            "max": "max(finite x)",
            "q75": "75th percentile(finite x)",
            "zero_rate": "count(x==0)/count(finite x)",
            "lag1": "adjacent-week centered cross-product / product of centered norms",
            "entropy": "-sum(p*ln(p))/ln(18), p=abs(x)/sum(abs(x)); 0 if all zero",
            "peak_week": "first calendar week of maximum finite x / 18",
            "max_jump4": "maximum adjacent change in rolling four-calendar-week means",
            "min_jump4": "minimum adjacent change in rolling four-calendar-week means",
            "end_streak": "consecutive positive observations ending at week 18 / 18",
            "mean_yoy": "s0 mean - s1 mean",
            "last4_yoy": "s0 last4 - s1 last4",
            "three_year_trend": "OLS slope of finite season means in chronological order s2,s1,s0",
            "year_stability": "population SD of finite means from s2,s1,s0",
        }
        return (
            f"x={signal}; {formulas[suffix]}. "
            "s0/s1/s2 are preceding 1/2/3 seasons; cross-year terms use these three. "
            "Calendar-week coverage and zero-fill follow the pinned rich_weekly implementation, "
            "not observed-game denominators.",
            "src/patron/metrics/rich_weekly.py",
        )
    summary_prefixes = [
        "basic_prior",
        "career_recent3",
        "career_high",
        "career_rotation",
        "career_limited",
        "career_unknown",
        "career",
    ]
    for prefix in summary_prefixes:
        if not name.startswith(prefix + "_"):
            continue
        suffix = name[len(prefix) + 1 :]
        period = {
            "basic_prior": "previous season",
            "career_recent3": "previous three seasons",
            "career_high": "all prior weeks with offensive snap share >=70%",
            "career_rotation": "all prior weeks with offensive snap share >=35% and <70%",
            "career_limited": "all prior weeks with offensive snap share <35%",
            "career_unknown": "all prior weeks without snap-share evidence",
            "career": "all available prior seasons",
        }[prefix]
        formulas = {
            "observed_weeks": "N=count(scoring record or positive offensive snaps)",
            "snap_observations": "count(observed weeks with known snap share)",
            "snap_share": "mean of known offensive snap shares across observed weeks",
            "points_per_observed_week": "sum(league points)/N, complete points required",
            "targets_per_observed_week": "sum(targets)/N, complete targets required",
            "carries_per_observed_week": "sum(carries)/N, complete carries required",
            "passing_yards_per_attempt": "sum(passing yards)/sum(attempts) on paired known weeks",
            "rushing_yards_per_carry": "sum(rushing yards)/sum(carries) on paired known weeks",
            "receiving_yards_per_target": "sum(receiving yards)/sum(targets) on paired known weeks",
        }
        for quantity in ["attempts", "carries", "targets", "receptions", "league_points"]:
            formulas[quantity] = f"sum({quantity}), only with complete observed-week coverage"
            formulas[quantity + "_per_week"] = formulas[quantity] + " / N observed weeks"
        if suffix in formulas:
            return (
                formulas[suffix] + "; period=" + period + ".",
                "research/individual_stat_data.py:complete_summary + "
                "research/profile_history.py:nfl_features",
            )
        break
    if name.startswith("context_"):
        return "Alias of cutoff-safe gold field " + name.removeprefix(
            "context_"
        ) + ".", "research/profile_history.py:build_features"
    if name.startswith(("college_career_", "college_latest_", "college_best_")):
        period, field = name.removeprefix("college_").split("_", 1)
        operation = {"career": "sum", "latest": "latest accepted season value", "best": "maximum"}[
            period
        ]
        return (
            f"{operation} of {field} over linked college seasons before NFL entry and "
            "forecast cutoff; require complete single-team season and no same-year ambiguity."
        ), "research/profile_history.py:college_features"
    explicit = {
        "career_recent3_weighted_ppg": (
            "sum(weekly points * .55**season_lag) / sum(.55**season_lag), "
            "previous three seasons, observed weeks"
        ),
        "age_factor": (
            "clip(.78,1.03): RB 1.02 if age<24 else 1-.035*max(age-26,0); "
            "WR 1.01 if age<24 else 1-.02*max(age-29,0); "
            "TE 1.01 if age<25 else 1-.018*max(age-30,0); "
            "QB 1-.015*max(age-35,0). Unknown age remains unknown."
        ),
        "wtd_opp": "carries + 2.2*targets; retired composite, diagnostic research only",
        "wopr": (
            "1.5*target_share + .7*air_yards_share; retired composite, diagnostic research only"
        ),
        "rb_age_cliff": "1 if RB and age_at_season >=27.5, else 0; unknown age preserved",
    }
    if name in explicit:
        return explicit[name], "research/individual_stat_data.py + research/profile_history.py"
    if name.startswith(
        (
            "basic_",
            "career_",
            "college_",
            "injury_",
            "context_",
            "news_",
            "market_position_",
            "market_overall_",
        )
    ):
        return (
            "Cutoff-safe profile transform "
            + name
            + "; full career unless prior/recent3/role-band is named. "
            "Complete totals require complete observations; rates use their declared denominator.",
            "research/profile_history.py + "
            "research/individual_stat_data.py:complete_summary/full_features",
        )
    return (
        "Named historical gold observation or deterministic formula: "
        + name
        + ". See pinned implementation and source table schema; missing remains unknown.",
        "src/patron/data/gold.py + src/patron/metrics/backtest.py + "
        "src/patron/metrics/projection.py",
    )


def build_registry(frame, root):
    catalog = {
        r["key"]: r
        for r in yaml.safe_load((ROOT / "src/patron/config/metric_report.yaml").read_text())[
            "metrics"
        ]
    }
    enriched_root = ROOT / "data/enriched/releases/canonical_20260923_r4"
    manifest = json.loads((enriched_root / "manifest.json").read_text())
    old_schema = pl.read_parquet_schema(enriched_root / manifest["tables"]["historical_inputs"])
    old_numeric = {c for c, t in old_schema.items() if t.is_numeric() or t == pl.Boolean}
    gold_ref = json.loads((root / "gold.json").read_text())
    gold_schema = pl.read_parquet_schema(
        ROOT / "data/gold/releases" / gold_ref["version"] / "tables/preseason_features.parquet"
    )
    profile = set(json.loads((root / "specs.json").read_text())["profile_market"])
    formulas = set(json.loads((root / "formulas.json").read_text()))
    legacy = set(json.loads((root / "legacy_formulas.json").read_text()))
    numeric = {
        c for c, t in frame.schema.items() if t.is_numeric() or t == pl.Boolean or t == pl.Null
    }
    keys = sorted(numeric | old_numeric | set(catalog) | RETIRED)
    registry, fingerprints = [], {}
    for key in keys:
        sources = [
            label
            for label, names in [
                ("gold", gold_schema),
                ("legacy_enriched", old_numeric),
                ("metric_catalog", catalog),
                ("profile", profile),
                ("recomputed_formula", formulas | legacy),
                ("retired", RETIRED),
            ]
            if key in names
        ]
        status, reason = (
            "admitted_research_only",
            "Cutoff-safe gold observation or deterministic recomputation; no serving approval.",
        )
        if key.startswith("actual_") or key == "outcome_complete":
            status, reason = "outcome_only", "Held-out outcome; never a predictor."
        elif key in METADATA or key.startswith("evidence_"):
            status, reason = (
                "metadata",
                "Bookkeeping; use explicitly defined baseline era/population variables.",
            )
        elif key.startswith(("fitted_", "adaptive_", "candidate_", "week1_proxy_")):
            status, reason = (
                "unsafe_or_fitted_output",
                "Saved fitted output or future proxy requires a separate leakage-safe rebuild.",
            )
        elif key not in numeric:
            status, reason = (
                "unreconstructed",
                "Historical formula/output lacks an admitted gold reconstruction; "
                "archived, not declared ineffective.",
            )
        finite = (
            frame[key].cast(pl.Float64).is_finite().fill_null(False) if key in numeric else None
        )
        count = int(finite.sum()) if finite is not None else 0
        years = sorted(frame.filter(finite)["forecast_season"].unique().to_list()) if count else []
        alias = None
        if status == "admitted_research_only":
            if not count:
                status, reason = (
                    "unavailable_in_gold",
                    "No finite value after validity masks; missing is not zero.",
                )
            elif frame.filter(finite)[key].n_unique() <= 1 and count == frame.height:
                status, reason = (
                    "constant",
                    "No observed variation in this completed historical universe.",
                )
            else:
                raw = frame[key].cast(pl.Float64).to_numpy().copy()
                raw[~np.isfinite(raw)] = np.nan
                fingerprint = raw.tobytes()
                alias = fingerprints.get(fingerprint)
                fingerprints.setdefault(fingerprint, key)
        meaning, ref = definition(key, catalog)
        unit, dependencies = units_and_dependencies(key, legacy)
        registry.append(
            dict(
                stat=key,
                family=family(key, catalog),
                status=status,
                reason=reason,
                origins=";".join(sources),
                definition=meaning,
                definition_ref=ref,
                unit=unit,
                dependencies=dependencies,
                finite_rows=count,
                total_rows=frame.height,
                first_year=min(years) if years else None,
                last_year=max(years) if years else None,
                finite_years=len(years),
                exact_alias_of=alias,
                market_input=is_market(key),
                retired=key in RETIRED,
                data_version=gold_ref["version"],
                definition_version=root.name,
                target="next-season league points",
                serving="research_only",
                original_profile=key in profile,
            )
        )
    return registry
