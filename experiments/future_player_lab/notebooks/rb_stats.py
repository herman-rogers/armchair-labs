"""Running back stat prediction: data, editable models, and chronological diagnostics."""
# ruff: noqa: E501, B018

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl
    from experiments.future_player_lab.notebooks import rb_availability as availability
    from experiments.future_player_lab.notebooks import rb_probability as probability

    # Reuse verified data loading and outcome definitions, not the old experiments.
    from experiments.future_player_lab.notebooks import workbench as data
    from threadpoolctl import threadpool_limits

    px.defaults.template = "plotly_white"
    return (
        availability,
        data,
        json,
        mo,
        np,
        pl,
        probability,
        px,
        threadpool_limits,
    )


@app.cell
def _(mo):
    mo.md("""
    # Predict running back stats

    Choose a model, tune its settings, and press **Train RB models**.
    Predict carries, rushing yards/TDs, targets, receptions, and receiving yards/TDs.
    Nothing trains on opening this notebook. No saved experiments run automatically.

    We use **all 2,929 prepared inputs plus dated availability evidence**, including player history, weekly usage,
    college data, age/draft/availability, team context, and dated market inputs.
    The historical snapshot is supplemented by the versioned injury/news archive.
    Inputs respect the forecast cutoff; incomplete target observations are excluded.

    For test season Y: **train through Y−2 → validate on Y−1 → test on Y**.
    Validation selects tree count; test outcomes never select settings.
    A final refit through Y−1 produces the player forecasts. These are historical
    future-period tests, not live 2026 projections.

    **Availability:** only confirmed absences covering the entire forecast period
    force a zero. The same rule applies to the baseline. Partial absences become
    inputs; totals are not discounted a second time. Unknown reports do not mean
    healthy, and actual outcomes are never rewritten. Original and adjusted
    forecasts remain available for comparison. News uses its reviewed knowledge
    date; corroborated reports use the later publication/update timestamp.
    In-season injury evidence is cut off at the end of the day before the first
    predicted NFL week starts; preseason uses the saved forecast cutoff date.
    """)
    return


@app.cell(hide_code=True)
def _(availability, mo):
    _summary, _coverage = availability.archive_status()
    mo.accordion(
        {
            "Injury archive: coverage and data rules": mo.vstack(
                [
                    mo.md(
                        f"**{_summary['original_reports']:,} original reports**, "
                        f"**{_summary['early_official_rows']:,} additional early records**, and "
                        f"**{_summary['news_candidates']:,} captured RB news candidates**. "
                        f"Dated articles corroborate **{_summary['corroborated_original_records']:,} "
                        f"previously undated records** ({_summary['corroborated_rb_records']} RB records). "
                        "This is an incomplete archive. Unreviewed news, undated early records, "
                        "and unresolved identities cannot force a zero. Retrospective source captures "
                        "are not original publication snapshots. Reviewed historical announcements "
                        "retain their reconstruction basis; automatically matched reports use the later "
                        "publication/update timestamp. Partial and uncertain absences become inputs."
                    ),
                    mo.ui.table(_coverage, selection=None, label="Injury coverage by season"),
                    mo.download(
                        _coverage.write_csv(),
                        filename="injury_coverage_by_season.csv",
                        label="Download injury coverage",
                    ),
                ]
            )
        }
    )
    return


@app.cell
def _(mo):
    model_choice = mo.ui.dropdown(
        ["Histogram boosting", "LightGBM", "XGBoost", "CatBoost", "Random forest", "Extra trees"],
        value="Histogram boosting",
        label="Model",
    )
    model_choice
    return (model_choice,)


@app.cell
def _(data, mo, model_choice):
    _forest = model_choice.value in {"Random forest", "Extra trees"}
    _cat = model_choice.value == "CatBoost"
    _xgb = model_choice.value == "XGBoost"
    _controls = {
        "targets": mo.ui.multiselect(
            data.POSITION_TARGETS["RB"], value=data.POSITION_TARGETS["RB"], label="Stats to predict"
        ),
        "horizon": mo.ui.dropdown(
            {
                "Rest of season (after Week 2)": "remaining",
                "Next four weeks (after Week 2)": "next_four",
                "Next week (after Week 2)": "next_week",
                "Full season (preseason)": "season",
            },
            value="Rest of season (after Week 2)",
            label="Prediction period",
        ),
        "year": mo.ui.dropdown(list(range(2019, 2026)), value=2025, label="Test season"),
        "rounds": mo.ui.number(10, 4096, value=120, step=1, label="Maximum trees / rounds"),
        "depth": mo.ui.number(1, 12, value=6, step=1, label="Maximum depth"),
        "features": mo.ui.number(0.1, 1.0, value=1.0, step=0.1, label="Feature fraction"),
        "objective": mo.ui.dropdown(
            ["rmse", "mae", "crps"], value="rmse", label="Validation metric"
        ),
        "calibration_seasons": mo.ui.number(
            1, 5, value=3, step=1, label="Earlier calibration seasons (extra fits per stat)"
        ),
        "distribution_neighbors": mo.ui.number(
            10, 1000, value=100, step=10, label="Nearest calibration predictions per distribution"
        ),
        "seed": mo.ui.number(0, 2147483647, value=20260925, step=1, label="Random seed"),
        "learning_curve": mo.ui.checkbox(
            value=True, label="Also compute training-history learning curve (extra fits)"
        ),
    }
    if not _forest:
        _controls["rate"] = mo.ui.number(0.001, 1.0, value=0.05, step=0.001, label="Learning rate")
        _controls["l2"] = mo.ui.number(0, 1000, value=10.0, step=1, label="L2 regularization")
    if not _cat:
        _controls["leaf"] = mo.ui.number(
            1,
            200,
            value=30,
            step=1,
            label="Minimum child weight" if _xgb else "Minimum samples per leaf",
        )
    if model_choice.value in {"Histogram boosting", "LightGBM"}:
        _controls["leaves"] = mo.ui.number(2, 127, value=15, step=1, label="Maximum leaves")
    _names = [n for n in _controls if n not in {"targets", "learning_curve"}]
    _layout = "{targets}\n\n| Settings | Settings |\n|---|---|\n"
    for _i in range(0, len(_names), 2):
        _left = "{" + _names[_i] + "}"
        _right = "{" + _names[_i + 1] + "}" if _i + 1 < len(_names) else ""
        _layout += f"| {_left} | {_right} |\n"
    _layout += "\n{learning_curve}"
    settings = mo.md(_layout).batch(**_controls).form(submit_button_label="Train RB models")
    mo.vstack(
        [
            mo.md(
                "## 1 · Settings\nChanging the model resets this form. Trees fit squared error; the validation metric selects their count. Training all seven stats takes longer than training one."
            ),
            settings,
        ]
    )
    return (settings,)


@app.cell
def _():
    # Edit these constructors directly to change the model implementation.
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer

    def make_model(c, rounds):
        common = dict(random_state=int(c["seed"]), max_depth=int(c["depth"]))
        engine = c["model"]
        if engine == "Histogram boosting":
            return HistGradientBoostingRegressor(
                max_iter=rounds,
                learning_rate=float(c["rate"]),
                max_leaf_nodes=int(c["leaves"]),
                min_samples_leaf=int(c["leaf"]),
                l2_regularization=float(c["l2"]),
                max_features=float(c["features"]),
                early_stopping=False,
                **common,
            )
        if engine == "LightGBM":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=rounds,
                learning_rate=float(c["rate"]),
                num_leaves=int(c["leaves"]),
                min_child_samples=int(c["leaf"]),
                reg_lambda=float(c["l2"]),
                colsample_bytree=float(c["features"]),
                verbosity=-1,
                n_jobs=1,
                **common,
            )
        if engine == "XGBoost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=rounds,
                learning_rate=float(c["rate"]),
                min_child_weight=float(c["leaf"]),
                reg_lambda=float(c["l2"]),
                colsample_bytree=float(c["features"]),
                tree_method="hist",
                n_jobs=1,
                **common,
            )
        if engine == "CatBoost":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=rounds,
                learning_rate=float(c["rate"]),
                depth=int(c["depth"]),
                l2_leaf_reg=float(c["l2"]),
                rsm=float(c["features"]),
                random_seed=int(c["seed"]),
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
        cls = RandomForestRegressor if engine == "Random forest" else ExtraTreesRegressor
        return cls(
            n_estimators=rounds,
            min_samples_leaf=int(c["leaf"]),
            max_features=float(c["features"]),
            n_jobs=1,
            **common,
        )

    return SimpleImputer, make_model


@app.cell
def _(SimpleImputer, make_model, np):
    def fit_model(x, y, c, rounds):
        # All transformations learn from this fit's training rows only.
        active = np.flatnonzero(np.any(np.isfinite(x), axis=0))
        if not len(active):
            raise ValueError("No observed training features")
        train = x[:, active]
        imputer = None
        if c["model"] in {"Random forest", "Extra trees"}:
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            train = imputer.fit_transform(train)
        model = make_model(c, rounds)
        model.fit(train, y)  # The actual training call; y is the stat per calendar week.
        return model, active, imputer

    def predict_prefixes(fitted, x, c, counts, target):
        model, active, imputer = fitted
        values = x[:, active]
        if imputer is not None:
            values = imputer.transform(values)
        engine = c["model"]
        if engine == "Histogram boosting":
            preds = {i: p for i, p in enumerate(model.staged_predict(values), 1) if i in counts}
        elif engine in {"Random forest", "Extra trees"}:
            preds, total = {}, np.zeros(len(values))
            for i, tree in enumerate(model.estimators_, 1):
                total += tree.predict(values)
                if i in counts:
                    preds[i] = total.copy() / i
        else:
            preds = {
                n: model.predict(
                    values,
                    **(
                        {"num_iteration": n}
                        if engine == "LightGBM"
                        else {"iteration_range": (0, n)}
                        if engine == "XGBoost"
                        else {"ntree_end": n}
                    ),
                )
                for n in counts
            }
        # Count forecasts are nonnegative expected values; yardage may be negative.
        return {n: p if "yards" in target else np.maximum(0, p) for n, p in preds.items()}

    def error_metrics(actual, predicted):
        valid = np.isfinite(actual) & np.isfinite(predicted)
        error = predicted[valid] - actual[valid]
        return dict(
            rows=int(valid.sum()),
            rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
            mae=float(np.mean(abs(error))) if len(error) else None,
            bias=float(np.mean(error)) if len(error) else None,
        )

    return error_metrics, fit_model, predict_prefixes


@app.cell
def _(
    availability,
    data,
    error_metrics,
    fit_model,
    np,
    pl,
    predict_prefixes,
    probability,
):
    def train_stat(panel, x, names, weeks, target, c):
        labeled = data.stat_panel(panel, weeks, target)
        known = np.isfinite(labeled["actual"].to_numpy())
        excluded = labeled.filter(pl.Series(~known)).group_by("year").len().sort("year")
        labeled, x = labeled.filter(pl.Series(known)), x[known]
        years = labeled["year"].to_numpy()
        actual, exposure = labeled["actual"].to_numpy(), labeled["exposure"].to_numpy()
        force_zero = labeled["availability_force_zero"].to_numpy()
        rate = actual / exposure
        masks = {
            "Train": years < c["year"] - 1,
            "Validation": years == c["year"] - 1,
            "Test": years == c["year"],
        }
        if any(not m.any() for m in masks.values()):
            raise ValueError(f"{target}: an evaluation split has no known outcomes")
        maximum = int(c["rounds"])
        counts = sorted(
            {min(maximum, n) for n in [10, 20, 40, 80, 120, 240, 480, 960, 1920, 4096]} | {maximum}
        )
        train = masks["Train"]
        # Chronological out-of-sample residuals, strictly before validation.
        # Each calibration year's model sees only earlier seasons. Never use
        # in-sample training residuals or test outcomes to fit uncertainty.
        calibration_years = sorted(set(years[train]))[1:][-int(c.get("calibration_seasons", 3)) :]
        calibration_predictions = {n: [] for n in counts}
        calibration_actual, calibration_baseline, calibration_rows = [], [], []
        baseline_rate = labeled["persistence_prediction"].to_numpy() / exposure
        neighbors = int(c.get("distribution_neighbors", 100))
        for calibration_year in calibration_years:
            past = years < calibration_year
            holdout = (years == calibration_year) & np.isfinite(baseline_rate)
            if not holdout.any():
                continue
            historical_fit = fit_model(x[past], rate[past], c, maximum)
            historical_predictions = predict_prefixes(historical_fit, x[holdout], c, counts, target)
            for n in counts:
                calibration_predictions[n].append(historical_predictions[n])
            calibration_actual.append(rate[holdout])
            calibration_baseline.append(baseline_rate[holdout])
            calibration_rows.append(
                labeled.filter(pl.Series(holdout))
                .select(*data.KEYS, "name", "actual", "exposure", "persistence_prediction")
                .with_columns(pl.lit(int(years[past].max())).alias("fit_last_year"))
            )
        if not calibration_actual:
            raise ValueError("Need an earlier held-out season to calibrate distributions")
        calibration_actual = np.concatenate(calibration_actual)
        calibration_baseline = np.concatenate(calibration_baseline)
        calibration_predictions = {n: np.concatenate(p) for n, p in calibration_predictions.items()}
        fitted = fit_model(x[train], rate[train], c, maximum)
        curves, predictions = [], {}
        for split, mask in masks.items():
            prefixes = predict_prefixes(fitted, x[mask], c, counts, target)
            predictions[split] = prefixes
            for n, pred in prefixes.items():
                raw = pred * exposure[mask]
                draws = probability.residual_distribution(
                    pred,
                    exposure[mask],
                    calibration_predictions[n],
                    calibration_actual,
                    neighbors,
                    "yards" not in target,
                )
                adjusted_draws = probability.constrain(draws, force_zero[mask])
                curves.append(
                    dict(
                        split=split,
                        trees=n,
                        **error_metrics(actual[mask], np.where(force_zero[mask], 0.0, raw)),
                        raw_rmse=error_metrics(actual[mask], raw)["rmse"],
                        crps=float(probability.empirical_crps(actual[mask], adjusted_draws).mean()),
                        raw_crps=float(probability.empirical_crps(actual[mask], draws).mean()),
                    )
                )
        selected = min(
            (r for r in curves if r["split"] == "Validation"),
            key=lambda r: (r[c["objective"]], r["trees"]),
        )["trees"]
        details = []
        for split, mask in masks.items():
            details.append(
                labeled.filter(pl.Series(mask))
                .select(*data.KEYS, "name", "population", "actual")
                .with_columns(
                    pl.Series("raw_prediction", predictions[split][selected] * exposure[mask]),
                    pl.Series(
                        "prediction",
                        np.where(
                            force_zero[mask], 0.0, predictions[split][selected] * exposure[mask]
                        ),
                    ),
                    pl.lit(split).alias("split"),
                )
            )
        learning = []
        if c["learning_curve"]:
            for start in sorted(
                {
                    int(years[train].min()),
                    *[max(int(years[train].min()), c["year"] - 1 - n) for n in [3, 5, 10]],
                }
            ):
                take = train & (years >= start)
                small = fit_model(x[take], rate[take], c, selected)
                for split, mask in {**masks, "Train": take}.items():
                    pred = (
                        predict_prefixes(small, x[mask], c, [selected], target)[selected]
                        * exposure[mask]
                    )
                    learning.append(
                        dict(
                            first_year=start,
                            training_rows=int(take.sum()),
                            split=split,
                            **error_metrics(actual[mask], np.where(force_zero[mask], 0.0, pred)),
                        )
                    )
        # Final refit includes validation; the test season remains excluded.
        refit = train | masks["Validation"]
        final_model = fit_model(x[refit], rate[refit], c, selected)
        test = masks["Test"]
        pred = (
            predict_prefixes(final_model, x[test], c, [selected], target)[selected] * exposure[test]
        )
        forecast = (
            labeled.filter(pl.Series(test))
            .select(*data.KEYS, "name", "population", "actual", "persistence_prediction")
            .with_columns(pl.Series("prediction", pred))
        )
        forecast = availability.adjust_forecasts(
            forecast,
            labeled.filter(pl.Series(test)).select(pl.col("^availability_.*$")),
        )
        scores = [
            dict(
                model="Final refit · adjusted",
                **error_metrics(actual[test], forecast["prediction"].to_numpy()),
            ),
            dict(
                model="Prior production rate · adjusted",
                **error_metrics(actual[test], forecast["persistence_prediction"].to_numpy()),
            ),
            dict(model="Final refit · raw", **error_metrics(actual[test], pred)),
            dict(
                model="Prior production rate · raw",
                **error_metrics(actual[test], forecast["raw_persistence_prediction"].to_numpy()),
            ),
        ]
        probabilistic = probability.evaluate(
            actual[test],
            pred / exposure[test],
            baseline_rate[test],
            exposure[test],
            calibration_predictions[selected],
            calibration_baseline,
            calibration_actual,
            force_zero[test],
            target,
            neighbors,
        )
        probabilistic["calibration"] = pl.concat(calibration_rows).with_columns(
            pl.Series("model_prediction_rate", calibration_predictions[selected]),
            pl.Series("actual_rate", calibration_actual),
            pl.Series("baseline_prediction_rate", calibration_baseline),
        )
        return dict(
            capacity=pl.DataFrame(curves),
            learning=pl.DataFrame(learning),
            splits=pl.concat(details),
            forecasts=forecast,
            scores=pl.DataFrame(scores).join(
                probabilistic["scores"].select("model", "crps"),
                on="model",
                how="left",
                maintain_order="left",
            ),
            probability=probabilistic,
            selected=selected,
            excluded=excluded,
            fitted=final_model,
            active_features=[names[j] for j in final_model[1]],
            train_years=[int(years[train].min()), int(years[train].max())],
        )

    return (train_stat,)


@app.cell
def _(
    availability,
    data,
    mo,
    model_choice,
    settings,
    threadpool_limits,
    train_stat,
):
    mo.stop(
        settings.value is None,
        mo.md(
            "**Ready.** Submit the settings above to load the data and train. The model construction and `.fit()` calls are editable in the preceding code cells."
        ),
    )
    config = {**settings.value, "model": model_choice.value}
    mo.stop(not config["targets"], mo.callout("Choose at least one RB stat.", kind="warn"))
    # No fitting, data loading, or cached forecast display occurs before submission.
    rb_panel, rb_x, rb_features, _original = data.sandbox_data("RB", config["horizon"], "market")
    _annotations = availability.annotate(rb_panel, *availability.load_inputs())
    rb_x, rb_features = availability.append_features(rb_x, rb_features, _annotations)
    rb_panel = rb_panel.hstack(_annotations)
    config["availability_archive"] = "injury_archive_20260925_r1"
    config["availability_manifest_sha256"] = availability.ARCHIVE_MANIFEST_SHA256
    config["availability_rule"] = "confirmed_full_period_zero_v1"
    config["distribution_method"] = "nearest_historical_residuals_v1"
    config["distribution_score"] = "empirical_crps"
    weekly_outcomes = data.target_weeks()
    results = {}
    with mo.status.spinner(title="Training selected RB stats…"), threadpool_limits(limits=1):
        for _target in config["targets"]:
            results[_target] = train_stat(
                rb_panel, rb_x, rb_features, weekly_outcomes, _target, config
            )
    return config, rb_features, rb_panel, results


@app.cell(hide_code=True)
def _(config, data, json, mo, pl, rb_features, rb_panel, results):
    _rows, _forecasts = [], []
    for _target, _r in results.items():
        for _s in _r["scores"].to_dicts():
            _rows.append(
                {
                    "stat": data.target_label(_target),
                    "units": data.target_unit(_target),
                    "trees": _r["selected"],
                    **_s,
                }
            )
        _forecasts.append(
            _r["forecasts"]
            .select(*data.KEYS, "name", "prediction")
            .with_columns(pl.lit(_target).alias("stat"))
        )
    predicted_stats = (
        pl.concat(_forecasts)
        .pivot(on="stat", index=data.KEYS + ["name"], values="prediction")
        .sort("name")
    )
    inspect_stat = mo.ui.dropdown(
        {data.target_label(t): t for t in results},
        value=data.target_label(next(iter(results))),
        label="Stat to inspect",
    )
    mo.vstack(
        [
            mo.md(
                f"## 2 · Predictions\n**RB · {config['model']} · {config['year']} · {config['horizon'].replace('_', ' ')}**. "
                f"Loaded **{rb_panel.height:,} player-season examples** and **{len(rb_features):,} inputs**. "
                "Counts are fractional expected totals. Separate models do not enforce joint constraints such as receptions ≤ targets."
            ),
            mo.ui.table(
                predicted_stats.select("name", *results)
                .with_columns(pl.col(t).round(2) for t in results)
                .rename({"name": "Player", **{t: data.target_label(t) for t in results}}),
                selection=None,
            ),
            mo.ui.table(
                pl.DataFrame(_rows).with_columns(
                    pl.col(c).round(2) for c in ["rmse", "mae", "bias", "crps"] if c in _rows[0]
                ),
                selection=None,
                label="Final test accuracy, by stat and baseline (units differ)",
            ),
            mo.hstack(
                [
                    mo.download(
                        predicted_stats.write_csv(),
                        filename="rb_predicted_stats.csv",
                        label="Download predictions",
                    ),
                    mo.download(
                        json.dumps(config, indent=2),
                        filename="rb_model_settings.json",
                        label="Download settings",
                    ),
                ]
            ),
            inspect_stat,
        ]
    )
    return (inspect_stat,)


@app.cell(hide_code=True)
def _(data, mo, pl, results):
    _audit = []
    for _stat, _result in results.items():
        _f = _result["forecasts"]
        if "raw_prediction" in _f.columns:
            _audit.append(_f.with_columns(pl.lit(data.target_label(_stat)).alias("stat")))
    if _audit:
        _rows = pl.concat(_audit)
        _display = _rows.select(
            "name",
            "year",
            "stat",
            "actual",
            "raw_prediction",
            "prediction",
            "raw_persistence_prediction",
            "persistence_prediction",
            "availability_cutoff",
            "availability_force_zero",
            pl.col("availability_evidence_ids").list.join("; "),
            pl.col("availability_conflicts").list.join("; "),
        )
        mo.output.append(
            mo.vstack(
                [
                    mo.md(
                        f"### Availability audit\n{_rows.filter(pl.col('availability_force_zero')).height} player/stat forecasts have a confirmed full-period absence. Raw and adjusted scores above use identical actual outcomes. Curves show adjusted forecasts; raw RMSE is retained in the capacity table. A zero override is a dated eligibility rule, not proof the model learned the absence."
                    ),
                    mo.ui.table(
                        _display,
                        selection=None,
                        label="Raw forecasts, adjusted forecasts, and evidence",
                    ),
                    mo.download(
                        _display.write_csv(),
                        filename="rb_availability_audit.csv",
                        label="Download availability audit",
                    ),
                ]
            )
        )
    else:
        mo.output.append(mo.md("Availability rules take effect on your next training run."))
    return


@app.cell(hide_code=True)
def _(data, inspect_stat, mo):
    from experiments.future_player_lab.notebooks.rb_accuracy import accuracy_context

    accuracy_cohort = mo.ui.dropdown(
        ["All evaluated RBs", "RBs with positive actual production"],
        value="All evaluated RBs",
        label="Accuracy population",
    )
    _stat = inspect_stat.value
    _default = (
        100 if "yards" in _stat else 1 if "tds" in _stat else 25 if _stat == "carries" else 10
    )
    accuracy_tolerance = mo.ui.number(
        0,
        1000,
        value=_default,
        step=1,
        label=f"Acceptable absolute miss ({data.target_unit(_stat)})",
    )
    mo.vstack(
        [
            mo.md(
                "### How good are these predictions?\n"
                "Compare errors with actual production and with the prior-production baseline. "
                "These controls analyze existing predictions; they do not retrain. "
                "The positive-production filter is a retrospective diagnostic, not a rule for choosing players in advance."
            ),
            mo.hstack([accuracy_cohort, accuracy_tolerance]),
        ]
    )
    return accuracy_cohort, accuracy_context, accuracy_tolerance


@app.cell(hide_code=True)
def _(
    accuracy_cohort,
    accuracy_context,
    accuracy_tolerance,
    data,
    inspect_stat,
    mo,
    pl,
    px,
    results,
):
    _positive = accuracy_cohort.value == "RBs with positive actual production"
    _rows = []
    for _t, _result in results.items():
        _f = _result["forecasts"]
        _s, _ = accuracy_context(
            _f["actual"], _f["prediction"], _f["persistence_prediction"], positive_only=_positive
        )
        _rows.append(
            {
                "Stat": data.target_label(_t),
                "Players": _s["players"],
                "Actual mean": _s["actual_mean"],
                "Actual median": _s["actual_median"],
                "MAE": _s["mae"],
                "Error / production %": _s["wape_pct"],
                "MAE improvement vs baseline %": _s["mae_improvement_pct"],
                "R²": _s["r2"],
            }
        )
    _target = inspect_stat.value
    _unit = data.target_unit(_target)
    _f = results[_target]["forecasts"]
    _s, _mask = accuracy_context(
        _f["actual"],
        _f["prediction"],
        _f["persistence_prediction"],
        tolerance=accuracy_tolerance.value,
        positive_only=_positive,
    )
    _table = pl.DataFrame(_rows, infer_schema_length=None)
    _table = _table.with_columns(
        pl.col(c).round(2) for c, dtype in _table.schema.items() if dtype == pl.Float64
    )
    mo.stop(
        _s["players"] == 0,
        mo.md("No matched players have production in this selection. Choose all evaluated RBs."),
    )

    def _fmt(value, suffix=""):
        return "undefined (zero denominator)" if value is None else f"{value:,.2f}{suffix}"

    _detail = (
        _f.filter(pl.Series(_mask))
        .with_columns(
            (pl.col("prediction") - pl.col("actual")).alias("Error"),
            (pl.col("prediction") - pl.col("actual")).abs().alias("Absolute error"),
            pl.when(pl.col("actual") != 0)
            .then(100 * (pl.col("prediction") - pl.col("actual")).abs() / pl.col("actual").abs())
            .otherwise(None)
            .alias("Absolute error %"),
        )
        .with_columns(
            (pl.col("Absolute error") <= accuracy_tolerance.value).alias("Within tolerance")
        )
    )
    _scatter = px.scatter(
        _detail,
        x="actual",
        y="prediction",
        color="population",
        hover_data=["name", "Absolute error", "Absolute error %"],
        title=f"Final test: predicted versus actual {data.target_label(_target).lower()}",
    )
    _lo = min(_detail["actual"].min(), _detail["prediction"].min())
    _hi = max(_detail["actual"].max(), _detail["prediction"].max())
    _scatter.add_shape(type="line", x0=_lo, y0=_lo, x1=_hi, y1=_hi, line_dash="dash")
    mo.vstack(
        [
            mo.ui.table(
                _table, selection=None, label="Accuracy in context: final test predictions"
            ),
            mo.md(
                f"**{data.target_label(_target)} — {_s['players']} RBs:** actual production averaged "
                f"**{_s['actual_mean']:,.1f} {_unit}** (median **{_s['actual_median']:,.1f}**, "
                f"90th percentile **{_s['actual_p90']:,.1f}**; **{_s['zero_actual_pct']:.1f}%** had zero). "
                f"The average absolute miss was **{_s['mae']:,.1f} {_unit}**. "
                f"Summed absolute errors were **{_fmt(_s['wape_pct'], '%')}** of summed absolute actual production.\n\n"
                f"MAE improvement over the simple baseline: **{_fmt(_s['mae_improvement_pct'], '%')}**; "
                f"RMSE improvement: **{_fmt(_s['rmse_improvement_pct'], '%')}**. Positive means better; negative means worse.\n\n"
                f"**{_s['within_tolerance_pct']:.1f}%** of predictions missed by at most **{accuracy_tolerance.value:g} {_unit}**, "
                f"versus **{_s['baseline_within_tolerance_pct']:.1f}%** for the baseline. "
                f"**R² = {_fmt(_s['r2'])}**."
            ),
            mo.accordion(
                {
                    "How to judge these numbers": mo.md(
                        "**MAE** is the average absolute miss; RMSE penalizes large misses more. "
                        "**Error / production (WAPE)** = sum of absolute errors / sum of absolute actuals. "
                        "Lower is better; it is neither the percentage of correct predictions nor the average player's percentage error. "
                        "It can exceed 100%. Zeros still contribute errors; the ratio is undefined when total production is zero.\n\n"
                        "**R²** compares squared error with predicting this evaluation group's actual mean for everybody: "
                        "1 is perfect, 0 matches that constant, negative is worse. This hindsight mean is a descriptive reference, "
                        "not a deployable forecast; the prior-production baseline is the practical comparison. "
                        "R² is undefined when actual production has no variation.\n\n"
                        "Use baseline improvement, a tolerance useful for your decision, and consistency across seasons together. "
                        "There is no universal 'good' error threshold. Many zero-production players can make aggregate errors look small; "
                        "inspect the positive-production group as well. This is one historical test season, not evidence of future reliability."
                    )
                }
            ),
            mo.ui.plotly(_scatter),
            mo.ui.table(
                _detail.select(
                    "name",
                    "actual",
                    "prediction",
                    "persistence_prediction",
                    "Error",
                    "Absolute error",
                    "Absolute error %",
                    "Within tolerance",
                )
                .sort("actual", descending=True)
                .with_columns(
                    pl.col(c).round(2)
                    for c in [
                        "actual",
                        "prediction",
                        "persistence_prediction",
                        "Error",
                        "Absolute error",
                        "Absolute error %",
                    ]
                ),
                selection=None,
                label="What happened versus what we predicted (percentage error blank when actual is zero)",
            ),
            mo.download(
                # CSV needs scalar fields; keep injury lists intact in the forecasts.
                _detail.with_columns(
                    pl.col(_column).list.join("; ")
                    for _column in ["availability_evidence_ids", "availability_conflicts"]
                    if _column in _detail.columns
                ).write_csv(),
                filename=f"rb_{_target}_prediction_vs_actual.csv",
                label="Download prediction versus actual",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(inspect_stat, mo, results):
    _result = results[inspect_stat.value]
    mo.stop(
        "probability" not in _result,
        mo.md(
            "**Distribution evaluation:** submit training again to build chronological calibration forecasts. Existing point predictions cannot establish a predictive distribution by themselves."
        ),
    )
    _players = {
        f"{r['name']} ({r['player_id']})": i for i, r in enumerate(_result["forecasts"].to_dicts())
    }
    distribution_player = mo.ui.dropdown(
        _players, value=next(iter(_players)), label="Player distribution to inspect"
    )
    distribution_variant = mo.ui.dropdown(
        list(_result["probability"]["samples"]),
        value="Final refit · adjusted",
        label="Distribution",
    )
    mo.hstack([distribution_player, distribution_variant])
    return distribution_player, distribution_variant


@app.cell(hide_code=True)
def _(
    config,
    data,
    distribution_player,
    distribution_variant,
    inspect_stat,
    mo,
    np,
    pl,
    probability,
    px,
    results,
):
    _target = inspect_stat.value
    _r = results[_target]
    _p = _r["probability"]
    _f = _r["forecasts"]
    _y = _f["actual"].to_numpy()
    _scores = _p["scores"]
    _cal = _p["calibration"]
    _model_crps = _scores.filter(pl.col("model") == "Final refit · adjusted")["crps"][0]
    _base_crps = _scores.filter(pl.col("model") == "Prior production rate · adjusted")["crps"][0]
    _skill = (
        f"{100 * (1 - _model_crps / _base_crps):.1f}%"
        if _base_crps > 0
        else "undefined (baseline CRPS is zero)"
    )
    _coverage = []
    for _name, _draws in _p["samples"].items():
        for _level in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            _lo, _hi = probability.interval(_draws, _level)
            _coverage.append(
                dict(
                    model=_name,
                    nominal=100 * _level,
                    observed=100 * float(np.mean((_y >= _lo) & (_y <= _hi))),
                )
            )
    _reliability = px.line(
        pl.DataFrame(_coverage),
        x="nominal",
        y="observed",
        color="model",
        markers=True,
        title="Final test: interval coverage (percent)",
        labels={"nominal": "Requested coverage %", "observed": "Observed coverage %"},
    )
    _reliability.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line_dash="dash")
    _draws = _p["samples"][distribution_variant.value]
    _point_col = {
        "Final refit · raw": "raw_prediction",
        "Final refit · adjusted": "prediction",
        "Prior production rate · raw": "raw_persistence_prediction",
        "Prior production rate · adjusted": "persistence_prediction",
    }[distribution_variant.value]
    _lo, _hi = probability.interval(_draws, 0.8)
    _details = _f.select(
        *data.KEYS, "name", "actual", pl.col(_point_col).alias("point_prediction")
    ).with_columns(
        pl.Series("distribution_mean", _draws.mean(axis=1)),
        pl.Series("p10", _lo),
        pl.Series("p50", np.quantile(_draws, 0.5, axis=1, method="inverted_cdf")),
        pl.Series("p90", _hi),
        pl.Series("crps", probability.empirical_crps(_y, _draws)),
        pl.lit(distribution_variant.value).alias("distribution"),
    )
    _i = distribution_player.value
    _ecdf = px.ecdf(
        x=_draws[_i],
        title=f"{_f['name'][_i]}: predictive cumulative distribution",
        labels={"x": data.target_unit(_target)},
    )
    _ecdf.add_vline(x=float(_y[_i]), line_dash="dash", annotation_text="Actual")
    _draw_export = _f.select(*data.KEYS, "name").hstack(
        pl.DataFrame(_draws, schema=[f"draw_{i + 1}" for i in range(_draws.shape[1])])
    )
    mo.vstack(
        [
            mo.md(
                f"## Distribution quality: {data.target_label(_target)}\n"
                f"**Adjusted CRPS improvement vs baseline: {_skill}** (positive is better). "
                "CRPS scores the entire distribution and is in the stat's units; lower is better. "
                "It rewards distributions that are both concentrated and well placed, rather than simply wide. "
                f"Calibration uses **{_cal.height} earlier out-of-season predictions from "
                f"{_cal['year'].min()}–{_cal['year'].max()}**, each fitted only on prior seasons. "
                f"Validation is {config['year'] - 1}; final test is {config['year']}. "
                f"Each forecast uses {_draws.shape[1]} nearest calibration predictions by production rate. "
                "The final refit reuses this earlier residual calibration, so coverage must be checked rather than assumed."
            ),
            mo.ui.table(
                _scores.with_columns(
                    pl.col(c).round(3) for c, t in _scores.schema.items() if t == pl.Float64
                ),
                selection=None,
                label="CRPS, coverage (0–1), and average interval width — raw and adjusted",
            ),
            mo.ui.plotly(_reliability),
            mo.ui.plotly(_ecdf),
            mo.ui.table(_details, selection=None, label="Per-player distribution and test outcome"),
            mo.md(
                "50%, 80%, and 95% intervals should be checked for coverage **and** width. "
                "Atoms at zero and discrete outcomes can make coverage exceed the nominal rate. "
                "The distribution mean can differ from the original point prediction because residual "
                "calibration shifts it and count forecasts are censored at zero. A confirmed full-period "
                "absence becomes a point mass at zero for both model and baseline. "
                "No log-likelihood is shown: this empirical forecast has no assumed continuous density. "
                "These are marginal distributions for separate stats, not a joint simulation of a player's season."
            ),
            mo.hstack(
                [
                    mo.download(
                        _details.write_csv(),
                        filename=f"rb_{_target}_distribution_scores.csv",
                        label="Download distribution scores",
                    ),
                    mo.download(
                        _draw_export.write_csv(),
                        filename=f"rb_{_target}_predictive_samples.csv",
                        label="Download predictive samples",
                    ),
                    mo.download(
                        _cal.write_csv(),
                        filename=f"rb_{_target}_calibration.csv",
                        label="Download calibration forecasts",
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(config, data, inspect_stat, mo, pl, px, results):
    _target = inspect_stat.value
    _r = results[_target]
    _unit = data.target_unit(_target)
    _label = data.target_label(_target)
    _errors = _r["splits"].with_columns((pl.col("prediction") - pl.col("actual")).alias("error"))
    _capacity = px.line(
        _r["capacity"],
        x="trees",
        y="rmse",
        color="split",
        markers=True,
        title=f"{_label}: train / validation / test error versus tree count",
        labels={"rmse": f"RMSE ({_unit})"},
    ).add_vline(x=_r["selected"], line_dash="dash")
    _scatter = px.scatter(
        _errors,
        x="actual",
        y="prediction",
        facet_col="split",
        color="split",
        hover_data=["name", "year"],
        title=f"Predicted versus actual ({_unit})",
    )
    _low = min(_errors["actual"].min(), _errors["prediction"].min())
    _high = max(_errors["actual"].max(), _errors["prediction"].max())
    _scatter.add_shape(
        type="line", x0=_low, y0=_low, x1=_high, y1=_high, line_dash="dash", row="all", col="all"
    )
    _content = [
        mo.md(
            f"## 3 · Diagnostics: {_label}\nTrain: **{_r['train_years'][0]}–{_r['train_years'][1]}**; "
            f"validation: **{config['year'] - 1}**; test: **{config['year']}**. Validation selected **{_r['selected']} trees**. "
            "The three-split plots use the model fitted on training years only. The predictions above use the final refit including validation. "
            "All errors below are totals in this stat's units. Repeated tuning against the test season turns it into development evidence."
        ),
        mo.ui.plotly(_capacity),
        mo.ui.table(
            _r["capacity"].filter(pl.col("trees") == _r["selected"]),
            selection=None,
            label="Selected tree count: split metrics",
        ),
        mo.ui.plotly(_scatter),
        mo.ui.plotly(
            px.histogram(
                _errors,
                x="error",
                color="split",
                facet_col="split",
                nbins=40,
                title=f"Residual distributions: prediction − actual ({_unit})",
            )
        ),
        mo.ui.plotly(
            px.scatter(
                _errors,
                x="prediction",
                y="error",
                color="split",
                facet_col="split",
                hover_data=["name", "year"],
                title="Residuals versus prediction",
            ).add_hline(y=0, line_dash="dash")
        ),
    ]
    if "crps" in _r["capacity"].columns:
        _content.insert(
            2,
            mo.ui.plotly(
                px.line(
                    _r["capacity"],
                    x="trees",
                    y="crps",
                    color="split",
                    markers=True,
                    title="Distribution CRPS versus tree count (training is in-sample)",
                ).add_vline(x=_r["selected"], line_dash="dash")
            ),
        )
    if not _r["learning"].is_empty():
        _content.append(
            mo.ui.plotly(
                px.line(
                    _r["learning"].sort("training_rows"),
                    x="training_rows",
                    y="rmse",
                    color="split",
                    markers=True,
                    hover_data=["first_year"],
                    labels={"rmse": f"RMSE ({_unit})"},
                    title="Learning curve: add older training seasons; validation/test stay fixed",
                )
            )
        )
    _misses = (
        _r["forecasts"]
        .with_columns((pl.col("prediction") - pl.col("actual")).alias("error"))
        .with_columns(pl.col("error").abs().alias("absolute_error"))
        .sort("absolute_error", descending=True)
    )
    _content.extend(
        [
            mo.ui.table(_misses.head(25), selection=None, label="Largest final-test errors"),
            mo.accordion(
                {
                    "Data details": mo.vstack(
                        [
                            mo.md(
                                "Missing target totals were excluded, not filled with zero. All-missing input columns are dropped using training data only. Forest imputation is also fitted only on training rows."
                            ),
                            mo.ui.table(
                                _r["excluded"],
                                selection=None,
                                label="Unknown target examples excluded by season",
                            ),
                            mo.ui.table(
                                pl.DataFrame({"active_input": _r["active_features"]}),
                                selection=None,
                                label="Inputs available at final refit",
                            ),
                        ]
                    )
                }
            ),
        ]
    )
    mo.vstack(_content)
    return


if __name__ == "__main__":
    app.run()
