"""Predeclared, bounded QB efficiency/opportunity comparison for Opening Drive v2."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from engine.chairstats import CampaignStore

from ..gpu_lab.run import digest, json_write, packages
from .data import ROOT, matrix
from .qb_features import augment_qb
from .qb_policy import PredictionPolicy
from .run import RECIPES, clamp, columns, estimator, metrics, predict, select_members

TARGETS = ("attempts", "passing_yards")
YEARS = tuple(range(2015, 2026))
BASE = ROOT / "data/campaigns/opening_drive_v2"
OLD_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_v2_001"
PROTOCOL = {
    "campaign_id": "opening_drive_v2",
    "scope": "Retrain attempts and passing_yards only; preserve other seven target weights.",
    "features": "Add explicit past-only QB efficiency and opportunity to frozen base columns.",
    "candidates": {
        "attempts": ["control", "features"],
        "passing_yards": ["control", "features", "component_control", "component_features"],
    },
    "tree_selection": "Existing four recipes, equal-weight top two previous-three-year MSE.",
    "policy_selection": (
        "Only the previous three years: qualify an alternative if equal-year mean MAE and "
        "MSE each improve at least 1% versus control, and prior-season-300-attempt cohort "
        "equal-year MSE does not worsen. Require all three cohort years; choose lowest "
        "MSE then MAE then name. Otherwise control. Never use current/future outcomes."
    ),
    "warmup": "Fit recipe folds 2015–2025; construct policies 2018–2025; evaluate 2021–2025.",
    "rate": (
        "Ridge alpha1000 on explicit QB features + age/draft/availability; impute training "
        "means with indicators and standardize. Train future yards/attempts where attempts>0; "
        "attempt sample weights balanced by year. Clip rate0–20; 50% Ridge +50% "
        "past career YPA with two-year half-life and 400 league-prior attempts."
    ),
    "normalization": "Preserve calendar-week exposure for all tree candidates and controls.",
    "missing_observations": (
        "Missing player records or null attempts are unknown, not observed zero. Explicit "
        "observation-known and scheduled-window coverage features; incomplete current windows "
        "have null opportunity/log-volume fields; last/last3 require their own observed slots. "
        "Same historical/current rule. Shrunk efficiency retains league-prior fallback. "
        "Team totals describe captured observations, not a claim of provider completeness."
    ),
    "development_caveat": "Reused historical development seasons; no untouched test claim.",
}


def select_policy(annual, target, forecast_year):
    years = set(range(forecast_year - 3, forecast_year))
    scores = {}
    for candidate in PROTOCOL["candidates"][target]:
        rows = [
            r
            for r in annual
            if r["target"] == target and r["recipe"] == candidate and r["year"] in years
        ]
        if len(rows) != 3 or {r["year"] for r in rows} != years:
            raise ValueError("Incomplete strictly-prior policy selection history")
        if any(not np.isfinite(r[k]) for r in rows for k in ("mae", "mse")):
            raise ValueError("Nonfinite policy selection loss")
        scores[candidate] = {k: float(np.mean([r[k] for r in rows])) for k in ("mae", "mse")}
        scores[candidate]["starter_mse"] = (
            float(np.mean([r["starter_mse"] for r in rows]))
            if all(
                r["starter_n"] > 0
                and r["starter_mse"] is not None
                and np.isfinite(r["starter_mse"])
                for r in rows
            )
            else None
        )
    base = scores["control"]
    qualified = [
        name
        for name, value in scores.items()
        if name != "control"
        and value["mae"] <= 0.99 * base["mae"]
        and value["mse"] <= 0.99 * base["mse"]
        and value["starter_mse"] is not None
        and base["starter_mse"] is not None
        and value["starter_mse"] <= base["starter_mse"]
    ]
    winner = (
        min(qualified, key=lambda n: (scores[n]["mse"], scores[n]["mae"], n))
        if qualified
        else "control"
    )
    return winner, {
        "validation_years": sorted(years),
        "scores": scores,
        "qualified": qualified,
        "winner": winner,
    }


def fit_rate(x, panel, train, indices):
    attempts = panel["y_attempts"].to_numpy().astype(float)
    yards = panel["y_passing_yards"].to_numpy().astype(float)
    mask = train & (attempts > 0) & np.isfinite(attempts) & np.isfinite(yards)
    years = panel["year"].to_numpy()[mask]
    weights = attempts[mask].copy()
    for year in set(years):
        year_mask = years == year
        weights[year_mask] /= weights[year_mask].sum()
    weights *= len(weights) / weights.sum()
    model = make_pipeline(
        SimpleImputer(add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=1000),
    )
    model.fit(x[mask][:, indices], yards[mask] / attempts[mask], ridge__sample_weight=weights)
    return model


def candidate_summary(target, recipe, year, actual, values, starter):
    subgroup = metrics(actual[starter], values[starter])
    return {
        "target": target,
        "recipe": recipe,
        "year": year,
        **metrics(actual, values),
        "starter_n": subgroup["n"],
        "starter_mse": subgroup["mse"],
    }


def fit_horizon(horizon, output, cache, device="cuda"):
    from . import revision_data

    started = time.monotonic()
    model_id = "qb_" + horizon
    directory = Path(output) / model_id
    directory.mkdir(parents=True, exist_ok=False)
    old = BASE / model_id / "artifacts"
    old_manifest = json.loads((old / "model_manifest.json").read_text())
    old_names = json.loads((old / "feature_contract.json").read_text())
    original_weights = joblib.load(old / "weights.joblib")
    panel, names = revision_data.load_dataset("QB", horizon, cache)
    if names != old_names:
        raise ValueError("Corrected control feature order differs from frozen v2")
    # Prove the original forecast checkpoints use this exact historical matrix/order.
    parent = pl.read_parquet(OLD_RUN / "inputs" / model_id / "panel.parquet")
    if not parent.equals(panel):
        raise ValueError("Historical candidate matrix changed; borrowed controls are invalid")
    panel = augment_qb(panel, revision_data.history_tables())
    additions = [c for c in panel.columns if c.startswith("qb__")]
    names = old_names + additions
    panel.write_parquet(directory / "training_features.parquet")
    current, current_names, provenance = revision_data.current_panel("QB", horizon)
    if current_names != old_names:
        raise ValueError("Current base feature order differs")
    current = augment_qb(current, revision_data.current_tables())
    current.write_parquet(directory / "current_features.parquet")
    json_write(directory / "current_provenance.json", provenance)
    json_write(directory / "feature_names.json", names)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("qb_features.py"),
        Path(__file__).with_name("qb_policy.py"),
        Path(revision_data.__file__),
        Path(__file__).with_name("run.py"),
        Path(__file__).with_name("data.py"),
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    fit_contract = {
        "protocol": PROTOCOL,
        "sources": sources,
        "original_model_manifest_sha256": digest(old / "model_manifest.json"),
        "original_weights_sha256": digest(old / "weights.joblib"),
        "original_feature_contract_sha256": digest(old / "feature_contract.json"),
        "dataset_sha256": digest(directory / "training_features.parquet"),
        "current_features_sha256": digest(directory / "current_features.parquet"),
        "historical_control_matrix_exactly_equal": True,
    }
    json_write(directory / "fit_contract.json", fit_contract)
    x = matrix(panel, names)
    years = panel["year"].to_numpy()
    exposure = panel["weeks"].to_numpy()
    starters = panel["qb__prior_attempts"].to_numpy() >= 300
    rate_names = additions + [
        c for c in old_names if c == "age" or c.startswith(("draft_event__", "availability__"))
    ]
    rate_indices = [names.index(c) for c in rate_names]
    prior_index = names.index("qb__career_ypa_h2_p400")
    annual, candidate_annual, oof, choices, borrowed = [], [], [], [], []
    final_choices = {}
    for year in YEARS:
        test, past = years == year, years < year
        predictions = {}
        for target in TARGETS:
            y = panel[f"y_{target}"].to_numpy().astype(float)
            train = past & np.isfinite(y)
            for view in ("control", "features"):
                recipe_predictions = {}
                for recipe in RECIPES:
                    available = old_names if view == "control" else names
                    indices = columns(available, recipe)
                    prior = OLD_RUN / model_id / "checkpoints" / f"{target}_{year}_{recipe}.npy"
                    if view == "control" and prior.exists():
                        receipt = json.loads(prior.with_suffix(".json").read_text())
                        if digest(prior) != receipt["sha256"]:
                            raise ValueError("Original checkpoint checksum mismatch")
                        values = np.load(prior, allow_pickle=False)
                        if values.shape != (int(test.sum()),):
                            raise ValueError("Original checkpoint row count differs")
                        borrowed.append(
                            {"path": str(prior.relative_to(ROOT)), "sha256": digest(prior)}
                        )
                    else:
                        model = estimator(recipe, device)
                        model.fit(x[train][:, indices], y[train] / exposure[train])
                        values = clamp(model.predict(x[test][:, indices]) * exposure[test], target)
                    recipe_predictions[recipe] = values
                    annual.append(
                        {
                            "view": view,
                            "target": target,
                            "recipe": recipe,
                            "year": year,
                            **metrics(y[test], values),
                        }
                    )
                    checkpoint = directory / "checkpoints" / f"{view}_{target}_{year}_{recipe}.npy"
                    checkpoint.parent.mkdir(exist_ok=True)
                    np.save(checkpoint, values, allow_pickle=False)
                if year >= 2018:
                    selected = select_members(
                        [r for r in annual if r["view"] == view], target, year
                    )
                    predictions[target, view] = np.mean(
                        [recipe_predictions[r] for r in selected], axis=0
                    )
        if year >= 2018:
            rate = fit_rate(x, panel, past, rate_indices)
            rates = (
                0.5 * np.clip(rate.predict(x[test][:, rate_indices]), 0, 20)
                + 0.5 * x[test, prior_index]
            )
            for view in ("control", "features"):
                predictions["passing_yards", "component_" + view] = (
                    predictions["attempts", view] * rates
                )
            for target in TARGETS:
                actual = panel[f"y_{target}"].to_numpy()[test].astype(float)
                for recipe in PROTOCOL["candidates"][target]:
                    values = predictions[target, recipe]
                    candidate_annual.append(
                        candidate_summary(target, recipe, year, actual, values, starters[test])
                    )
                    oof.append(
                        panel.filter(pl.Series(test))
                        .select(
                            "player_id",
                            "player_display_name",
                            "year",
                            "origin",
                            "team",
                            "history_games",
                            "history_weeks",
                        )
                        .with_columns(
                            pl.lit(target).alias("target"),
                            pl.lit(recipe).alias("recipe"),
                            pl.Series("actual", actual),
                            pl.Series("prediction", values),
                            pl.Series("high_prior_usage", starters[test]),
                        )
                    )
                if year >= 2021:
                    winner, choice = select_policy(candidate_annual, target, year)
                    choices.append({"target": target, "year": year, **choice})
                    selected_rows = oof[-len(PROTOCOL["candidates"][target]) :]
                    chosen = next(r for r in selected_rows if r["recipe"][0] == winner)
                    oof.append(chosen.with_columns(pl.lit("ensemble").alias("recipe")))
        print(f"QB challenge {horizon}/{year} complete", flush=True)
    final = copy.deepcopy(original_weights)
    final["feature_names"] = names
    rate = fit_rate(x, panel, years <= 2025, rate_indices)
    for target in TARGETS:
        winner, choice = select_policy(candidate_annual, target, 2026)
        final_choices[target] = choice
        view = "features" if "features" in winner else "control"
        upstream_target = "attempts" if winner.startswith("component_") else target
        chosen = select_members([r for r in annual if r["view"] == view], upstream_target, 2026)
        members = []
        for recipe in chosen:
            indices = columns(old_names if view == "control" else names, recipe)
            if view == "control":
                existing = next(
                    m for m in original_weights["models"][upstream_target] if m["recipe"] == recipe
                )
                members.append(existing)
            else:
                y = panel[f"y_{upstream_target}"].to_numpy().astype(float)
                train = np.isfinite(y)
                model = estimator(recipe, device)
                model.fit(x[train][:, indices], y[train] / exposure[train])
                members.append({"recipe": recipe, "indices": indices, "model": model})
        wrapper = PredictionPolicy(
            members,
            count=upstream_target == "attempts",
            rate=rate if winner.startswith("component_") else None,
            rate_indices=rate_indices,
            prior_index=prior_index,
        )
        final["models"][target] = [
            {"recipe": winner, "indices": list(range(len(names))), "model": wrapper}
        ]
    final["qb_revision"] = {"protocol": PROTOCOL, "final_choices": final_choices}
    joblib.dump(final, directory / "models.joblib", compress=3)
    predictions = predict(final, current)
    replayed = predict(joblib.load(directory / "models.joblib"), current)
    for target in final["targets"]:
        np.testing.assert_array_equal(predictions[target], replayed[target])
    # Preserve the seven untouched targets' original evidence explicitly.
    old_oof = pl.read_parquet(old / "evaluation_predictions.parquet").filter(
        ~pl.col("target").is_in(TARGETS)
    )
    oof = pl.concat(oof, how="diagonal_relaxed")
    oof.write_parquet(directory / "challenge_predictions.parquet")
    combined = pl.concat([old_oof, oof.filter(pl.col("year") >= 2021)], how="diagonal_relaxed")
    combined.write_parquet(directory / "evaluation_predictions.parquet")
    old_report = json.loads((BASE / model_id / "metrics.json").read_text())
    rows = [r for r in old_report["rows"] if r["target"] not in TARGETS]
    for (target, recipe), block in (
        oof.filter(pl.col("year") >= 2021).partition_by(["target", "recipe"], as_dict=True).items()
    ):
        for cohort, group in [
            ("all_candidates", block),
            ("high_prior_usage", block.filter(pl.col("high_prior_usage"))),
            (
                "established_appearances",
                block.filter(pl.col("history_games") / pl.col("history_weeks") >= 0.5),
            ),
        ]:
            for year, subset in [
                (None, group),
                *[(k[0], v) for k, v in group.partition_by("year", as_dict=True).items()],
            ]:
                rows.append(
                    {
                        "target": target,
                        "recipe": recipe,
                        "cohort": cohort,
                        "year": year,
                        **metrics(subset["actual"].to_numpy(), subset["prediction"].to_numpy()),
                    }
                )
    report = {
        **old_report,
        "rows": rows,
        "qb_revision": {
            "protocol": PROTOCOL,
            "choices": choices,
            "final_choices": final_choices,
            "recipe_annual": annual,
            "candidate_annual": candidate_annual,
            "scope": "Two targets reevaluated; seven targets retain original v2 evidence/weights.",
        },
    }
    json_write(directory / "metrics.json", report)
    json_write(directory / "borrowed_control_receipts.json", borrowed)
    if any(digest(ROOT / p) != h for p, h in sources.items()):
        raise ValueError("Training source changed during fit")
    manifest = {
        **old_manifest,
        "feature_count": len(names),
        "source_files": sources,
        "packages": packages(),
        "runtime_seconds": time.monotonic() - started,
        "training_rows": panel.height,
        "dataset_manifest_sha256": digest(directory / "fit_contract.json"),
        "final_members": {
            t: [m["recipe"] for m in members] for t, members in final["models"].items()
        },
        "qb_revision": final["qb_revision"],
        "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    }
    json_write(directory / "manifest.json", manifest)
    print(f"Sealed QB challenge {horizon}: {final_choices}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    CampaignStore(BASE.parent).check("opening_drive_v2")
    args.output.mkdir(parents=True, exist_ok=False)
    json_write(args.output / "protocol.json", PROTOCOL)
    with threadpool_limits(limits=4):
        for horizon in ("season", "remaining"):
            fit_horizon(horizon, args.output, args.cache, args.device)


if __name__ == "__main__":
    main()
