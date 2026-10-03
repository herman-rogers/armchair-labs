"""Run the preregistered individual-stat and combination audit on corrected gold."""

# ruff: noqa: B023
# The fold-local helpers below are called synchronously inside the same iteration.

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from individual_stat_engine import (
    Context,
    bh_adjust,
    deduplicate,
    point_prediction,
    prepare,
    uncertainty,
)
from individual_stat_registry import build_registry, is_market
from profile_history import fit_predict
from threadpoolctl import threadpool_limits

from engine.data.releases import write_json

POSITIONS = ["QB", "RB", "WR", "TE"]


def recipe_names(specs, position):
    recipes = {k: list(v) for k, v in specs.items()}
    for lane in ["", "_market"]:
        profile = list(specs["profile" + lane])
        recipes["profile_no_cross_position_passing" + lane] = [
            c for c in profile if position == "QB" or "attempts" not in c
        ]
        recipes["profile_no_college_weight" + lane] = [
            c
            for c in profile
            if not c.startswith("college_weighted_") and c != "college_prior_weight"
        ]
        recipes["profile_no_career_totals" + lane] = [
            c
            for c in profile
            if not (
                c.startswith("career_")
                and any(
                    c.endswith("_" + n)
                    for n in ["attempts", "carries", "targets", "receptions", "league_points"]
                )
            )
        ]
    return recipes


def losses(y, before, after, masks):
    ea, eb = np.abs(y - before), np.abs(y - after)
    sa, sb = (y - before) ** 2, (y - after) ** 2
    return [
        dict(
            population=pop,
            n=int(mask.sum()),
            before_mae=float(ea[mask].mean()),
            after_mae=float(eb[mask].mean()),
            mae_gain=float((ea - eb)[mask].mean()),
            mse_gain=float((sa - sb)[mask].mean()),
        )
        for pop, mask in masks.items()
        if mask.any()
    ]


def run(root, *, benchmarks=True):
    frame = pl.read_parquet(root / "features.parquet").sort(
        "forecast_season", "position", "player_id"
    )
    specs = json.loads((root / "specs.json").read_text())
    registry = build_registry(frame, root)
    pl.DataFrame(registry).write_parquet(root / "registry.parquet")
    pl.DataFrame(registry).write_csv(root / "registry.csv")
    write_json(root / "registry.json", registry)
    admitted = [r for r in registry if r["status"] == "admitted_research_only"]
    names = sorted(set(r["stat"] for r in admitted) | set(specs["profile_market"]))
    names = [n for n in names if n in frame.columns]
    indices = {n: i for i, n in enumerate(names)}
    families = defaultdict(list)
    for r in admitted:
        families[r["family"]].append(r["stat"])
    parts = root / "fold_parts"
    parts.mkdir(exist_ok=True)
    print(
        f"Inventory {len(registry)} entries; {len(admitted)} admitted; {len(families)} families",
        flush=True,
    )
    for year in sorted(frame["forecast_season"].unique()):
        for pos in POSITIONS:
            dest = parts / f"{year}_{pos}.parquet"
            if dest.exists():
                continue
            train = frame.filter((pl.col("forecast_season") < year) & (pl.col("position") == pos))
            test = frame.filter((pl.col("forecast_season") == year) & (pl.col("position") == pos))
            if train.height < 30 or not test.height:
                continue
            assert train["forecast_season"].max() < test["forecast_season"].min()
            print(
                f"Individual screen {year} {pos}: training {train.height}, testing {test.height}",
                flush=True,
            )
            raw = train.select(names).cast(pl.Float64).to_numpy()
            raw_test = test.select(names).cast(pl.Float64).to_numpy()
            x, xt, blocks = prepare(raw, raw_test, names)
            years, test_years = (
                train["forecast_season"].to_numpy(),
                test["forecast_season"].to_numpy(),
            )
            observed = np.isfinite(raw)
            eligible_years = np.sum(
                [observed[years == y].sum(axis=0) >= 10 for y in np.unique(years)], axis=0
            )
            y = train["actual_points_per_scheduled_game"].to_numpy()
            actual = test["actual_season_points"].to_numpy()
            caps = test["known_available_games_cap"].cast(pl.Float64).to_numpy()
            populations = test["player_population"].to_numpy()
            all_masks = {
                "all": np.ones(test.height, dtype=bool),
                **{p: populations == p for p in ["rookie", "returner", "market_only"]},
            }
            market_fields = [
                indices[c] for c in ["market_position_log_rank", "market_overall_log_rank"]
            ]
            market_train = np.isfinite(raw[:, market_fields]).any(axis=1)
            market_test = np.isfinite(raw_test[:, market_fields]).any(axis=1)
            market_ready = (
                sum(market_train[years == yr].sum() >= 10 for yr in np.unique(years)) >= 3
            )
            market_masks = {k: v & market_test for k, v in all_masks.items()}
            cache, records = {}, []

            def context(columns):
                key = tuple(dict.fromkeys(columns))
                if key not in cache:
                    cache[key] = Context(x, xt, y, blocks, key)
                return cache[key]

            def score(r, lane, action, context_name, before, after):
                j = indices[r["stat"]]
                rows = losses(
                    actual,
                    point_prediction(before, test_years, caps),
                    point_prediction(after, test_years, caps),
                    all_masks if lane == "own" else market_masks,
                )
                for values in rows:
                    records.append(
                        dict(
                            stat=r["stat"],
                            family=r["family"],
                            position=pos,
                            season=year,
                            lane=lane,
                            action=action,
                            context=context_name,
                            training_n=train.height,
                            training_first_year=int(years.min()),
                            covered_training_years=int(eligible_years[j]),
                            finite_train=int(observed[:, j].sum()),
                            finite_test=int(np.isfinite(raw_test[:, j]).sum()),
                            **values,
                        )
                    )

            for lane, base_name, profile_name in [
                ("own", "basic", "profile"),
                ("market", "basic_market", "profile_market"),
            ]:
                if lane == "market" and (not market_ready or not market_test.any()):
                    continue
                base_names = list(specs[base_name])
                base = context(base_names)
                base.prepare_additions(x, xt)
                full = context(specs[profile_name])
                for r in admitted:
                    name = r["stat"]
                    if lane == "own" and r["market_input"]:
                        continue
                    if eligible_years[indices[name]] < 3 or not blocks[name]:
                        continue
                    if name in base_names:
                        score(
                            r,
                            lane,
                            "remove_baseline",
                            base_name,
                            base.prediction,
                            base.remove(name),
                        )
                    else:
                        score(r, lane, "add", base_name, base.prediction, base.add(blocks[name]))
                        members = [
                            n for n in families[r["family"]] if lane == "market" or not is_market(n)
                        ]
                        combo = context(base_names + members)
                        score(
                            r,
                            lane,
                            "remove_family",
                            r["family"],
                            combo.prediction,
                            combo.remove(name),
                        )
                    if name in specs[profile_name]:
                        score(
                            r,
                            lane,
                            "remove_profile",
                            profile_name,
                            full.prediction,
                            full.remove(name),
                        )
            pl.DataFrame(records).write_parquet(dest)
            if benchmarks:
                print(f"Combination refits {year} {pos}", flush=True)
                recipes = recipe_names(specs, pos)
                for lane in ["", "_market"]:
                    recipes["profile_deduplicated" + lane] = deduplicate(
                        x, list(specs["profile" + lane]), blocks
                    )
                train_rows, test_rows = train.to_dicts(), test.to_dicts()
                prediction = test.select(
                    "player_id",
                    "forecast_season",
                    "position",
                    "player_population",
                    "actual_season_points",
                    "market_ecr",
                    "market_overall_ecr",
                    "basic_prior_observed_weeks",
                    "career_observed_weeks",
                )
                for architecture in ["ridge", "boost"]:
                    for label, columns in recipes.items():
                        values = (
                            point_prediction(context(columns).prediction, test_years, caps)
                            if architecture == "ridge"
                            else fit_predict(train_rows, test_rows, columns, architecture)
                        )
                        prediction = prediction.with_columns(
                            pl.Series(architecture + "__" + label, values)
                        )
                prediction.write_parquet(parts / f"predictions_{year}_{pos}.parquet")
    folds = pl.concat(
        [pl.read_parquet(p) for p in sorted(parts.glob("[0-9]*.parquet"))], how="diagonal_relaxed"
    )
    folds.write_parquet(root / "individual_folds.parquet")
    predictions = pl.concat(
        [pl.read_parquet(p) for p in sorted(parts.glob("predictions_*.parquet"))],
        how="diagonal_relaxed",
    )
    predictions.write_parquet(root / "predictions.parquet")
    print("Summarizing uncertainty and multiple comparisons", flush=True)
    summarize(root, folds, registry)
    summarize_combinations(root, predictions)


def reuse_benchmarks(root, previous):
    """Reuse only predictions whose complete feature matrices and labels are identical."""
    specs = json.loads((root / "specs.json").read_text())
    assert specs == json.loads((previous / "specs.json").read_text())
    columns = sorted(
        set(sum(specs.values(), []))
        | {
            "player_id",
            "forecast_season",
            "position",
            "player_population",
            "actual_points_per_scheduled_game",
            "actual_season_points",
            "known_available_games_cap",
            "market_ecr",
            "market_overall_ecr",
        }
    )
    a = (
        pl.read_parquet(root / "features.parquet")
        .select(columns)
        .sort("forecast_season", "player_id")
    )
    b = (
        pl.read_parquet(previous / "features.parquet")
        .select(columns)
        .sort("forecast_season", "player_id")
    )
    assert a.equals(b)
    paths = sorted((previous / "fold_parts").glob("predictions_*.parquet"))
    assert len(paths) == 84
    (root / "fold_parts").mkdir(exist_ok=True)
    for path in paths:
        shutil.copy2(path, root / "fold_parts" / path.name)


def summarize(root, folds, registry):
    keys = ["stat", "family", "position", "lane", "action", "context"]
    results = []
    details = {r["stat"]: r for r in registry}
    specs = json.loads((root / "specs.json").read_text())

    def identity(name):
        item = details.get(name, {})
        return item.get("exact_alias_of") or name

    def aliases(row):
        if row["action"] == "remove_profile":
            columns = specs[row["context"]]
        elif row["action"] == "remove_family":
            columns = list(specs["basic_market" if row["lane"] == "market" else "basic"])
            columns += [r["stat"] for r in registry if r["family"] == row["family"]]
        else:
            columns = specs[row["context"]]
        return ";".join(
            c for c in columns if c != row["stat"] and identity(c) == identity(row["stat"])
        )

    for _, group in folds.filter(pl.col("population") == "all").group_by(keys, maintain_order=True):
        row = group.select(keys).row(0, named=True)
        gains = group.sort("season")["mae_gain"].to_numpy()
        values = uncertainty(gains)
        item = details[row["stat"]]
        results.append(
            dict(
                **row,
                seasons=len(gains),
                first_season=int(group["season"].min()),
                last_season=int(group["season"].max()),
                player_seasons=int(group["n"].sum()),
                mean_mae_gain=float(gains.mean()),
                mean_mse_gain=float(group["mse_gain"].mean()),
                before_mae=float(group["before_mae"].mean()),
                after_mae=float(group["after_mae"].mean()),
                positive_seasons=int((gains > 0).sum()),
                alias_of=item["exact_alias_of"],
                exact_alias_in_context=aliases(row),
                retired=item["retired"],
                **values,
            )
        )
    qvalues = bh_adjust([r["p_value"] for r in results])
    for r, q in zip(results, qvalues, strict=True):
        r["q_value"] = float(q)
        r["passes_screen"] = bool(
            r["seasons"] >= 5
            and r["mean_mae_gain"] >= 1
            and r["ci_low"] > 0
            and q <= 0.05
            and r["mean_mse_gain"] >= 0
            and r["loo_min"] > 0
        )
        r["evidence"] = (
            "research_candidate" if r["passes_screen"] else "does_not_pass_preregistered_screen"
        )
        if r["exact_alias_in_context"]:
            r["interpretation"] = "Alias/regularization sensitivity; not independent information"
        elif r["retired"]:
            r["interpretation"] = "Retired formula diagnostic; no automatic reinstatement"
        else:
            r["interpretation"] = (
                "Conditional model-specific contribution; correlated substitutes remain"
            )
    result = pl.DataFrame(results).sort(
        "lane", "action", "position", "mean_mae_gain", descending=[False, False, False, True]
    )
    result.write_parquet(root / "individual_results.parquet")
    result.write_csv(root / "individual_results.csv")
    strata = []
    for era, lo, hi in [
        ("all", 2005, 2025),
        ("2005-2012", 2005, 2012),
        ("2013-2018", 2013, 2018),
        ("2019-2025", 2019, 2025),
    ]:
        subset = folds.filter(pl.col("season").is_between(lo, hi))
        strata.append(
            subset.group_by(*keys, "population")
            .agg(
                pl.len().alias("seasons"),
                pl.col("n").sum().alias("player_seasons"),
                pl.col("mae_gain").mean().alias("mean_mae_gain"),
                pl.col("mse_gain").mean().alias("mean_mse_gain"),
            )
            .with_columns(pl.lit(era).alias("era"))
        )
    pl.concat(strata).write_parquet(root / "individual_subgroups.parquet")


def summarize_combinations(root, predictions):
    records = []
    for position in POSITIONS:
        frame = predictions.filter(pl.col("position") == position)
        for architecture in ["ridge", "boost"]:
            for col in frame.columns:
                if not col.startswith(architecture + "__"):
                    continue
                recipe = col.split("__", 1)[1]
                market = recipe.endswith("_market")
                baseline = architecture + "__" + ("basic_market" if market else "basic")
                if col == baseline:
                    continue
                reference = (
                    architecture + "__" + ("profile_market" if market else "profile")
                    if recipe.startswith("profile_") and recipe not in ["profile_market"]
                    else baseline
                )
                subset = frame
                if market:
                    # Same market-era gate as the individual screen: >=3 earlier covered years.
                    subset = subset.filter(
                        (
                            pl.col("market_ecr").is_not_null()
                            | pl.col("market_overall_ecr").is_not_null()
                        )
                        & (pl.col("forecast_season") >= 2014)
                    )
                for pop in ["all", "returner", "rookie", "market_only"]:
                    group = (
                        subset
                        if pop == "all"
                        else subset.filter(pl.col("player_population") == pop)
                    )
                    if not group.height:
                        continue
                    for era, lo, hi in [
                        ("all", 2005, 2025),
                        ("2005-2012", 2005, 2012),
                        ("2013-2018", 2013, 2018),
                        ("2019-2025", 2019, 2025),
                    ]:
                        eligible = group.filter(pl.col("forecast_season").is_between(lo, hi))
                        if not eligible.height:
                            continue
                        g = (
                            eligible.group_by("forecast_season")
                            .agg(
                                (pl.col("actual_season_points") - pl.col(reference))
                                .abs()
                                .mean()
                                .alias("before"),
                                (pl.col("actual_season_points") - pl.col(col))
                                .abs()
                                .mean()
                                .alias("after"),
                                (
                                    (pl.col("actual_season_points") - pl.col(reference)).pow(2)
                                    - (pl.col("actual_season_points") - pl.col(col)).pow(2)
                                )
                                .mean()
                                .alias("mse_gain"),
                            )
                            .sort("forecast_season")
                        )
                        gains = (g["before"] - g["after"]).to_numpy()
                        records.append(
                            dict(
                                position=position,
                                architecture=architecture,
                                recipe=recipe,
                                baseline=reference,
                                population=pop,
                                era=era,
                                seasons=len(gains),
                                player_seasons=eligible.height,
                                mean_mae_gain=float(gains.mean()),
                                before_mae=float(g["before"].mean()),
                                after_mae=float(g["after"].mean()),
                                mean_mse_gain=float(g["mse_gain"].mean()),
                                **uncertainty(gains),
                            )
                        )
    pl.DataFrame(records).write_parquet(root / "combination_results.parquet")
    pl.DataFrame(records).write_csv(root / "combination_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--reuse-benchmarks", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.reuse_benchmarks:
            reuse_benchmarks(args.root, args.reuse_benchmarks)
        run(args.root, benchmarks=args.reuse_benchmarks is None)
