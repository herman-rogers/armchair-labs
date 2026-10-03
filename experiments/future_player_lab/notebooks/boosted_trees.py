"""Original boosters, new tree libraries, ensembles, and editable diagnostics."""
# ruff: noqa: E501, B018

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl
    from experiments.future_player_lab.notebooks import workbench as wb

    px.defaults.template = "plotly_white"
    return mo, np, pl, px, root, wb


@app.cell
def _(mo):
    mo.md("""
    # Boosted trees: from the original model to ensembles

    Start with **saved historical forecasts**, inspect where they succeed and fail,
    then run your own chronological experiment below. The defaults show running backs,
    remaining-season points, and the complete 2019–2025 comparison.

    **Original** means the pre-existing `profile_boost` and `enriched_boost` forecasts.
    **Published** is the policy actually served—it is not always a boosted model.
    **New** uses the expanded library; **combined** allows saved existing models too.
    All comparisons use the same players and years. Models train on earlier seasons.

    Browse freely: changing comparison filters does not retrain anything. The two
    experiment forms run only when submitted. Their results are research outputs.
    """)
    return


@app.cell
def _(mo):
    position = mo.ui.dropdown(["QB", "RB", "WR", "TE"], value="RB", label="Position")
    horizon = mo.ui.dropdown(
        ["season", "next_week", "next_four", "remaining"],
        value="remaining",
        label="Forecast horizon",
    )
    years = mo.ui.range_slider(
        2019, 2025, value=[2019, 2025], step=1, show_value=True, label="Evaluation seasons"
    )
    cohort = mo.ui.dropdown(
        ["all", "rookie", "returner", "market_only"], value="all", label="Player cohort"
    )
    mo.hstack([position, horizon, years, cohort], justify="start")
    return cohort, horizon, position, years


@app.cell
def _(horizon, mo, wb):
    _defaults = ["persistence", "lgb_04_t120", "new:top_3"]
    if horizon.value in {"next_four", "remaining"}:
        _defaults = [
            "existing:profile_boost",
            "existing:enriched_boost",
            "published",
            "lgb_04_t120",
            "new:top_3",
            "combined:top_3",
            "combined:policy_mae",
        ]
    models = mo.ui.multiselect(
        wb.model_menu(horizon.value), value=_defaults, label="Models to compare", full_width=True
    )
    mo.vstack(
        [
            models,
            mo.md(
                "Original production boosters have matching forecasts only for `next_four` and `remaining`. No week-8 or different-horizon results are substituted."
            ),
        ]
    )
    return (models,)


@app.cell
def _(mo, models):
    mo.stop(not models.value, mo.md("Select at least one model above."))
    reference = mo.ui.dropdown(
        models.value,
        value="published" if "published" in models.value else models.value[0],
        label="Comparison baseline",
    )
    plot_metric = mo.ui.dropdown(
        ["mae", "rmse", "ndcg24", "capture"], value="mae", label="Chart metric"
    )
    mo.hstack([reference, plot_metric], justify="start")
    return plot_metric, reference


@app.cell
def _(cohort, horizon, mo, models, position, reference, wb, years):
    historical_error = None
    try:
        historical = wb.comparison_frame(
            position.value,
            horizon.value,
            tuple(models.value),
            int(years.value[0]),
            int(years.value[1]),
            cohort.value,
        )
        annual = wb.annual_scores(historical)
        scores = wb.summary_scores(annual, reference.value)
    except ValueError as _error:
        historical_error = str(_error)
    mo.stop(historical_error is not None, mo.callout(historical_error or "", kind="warn"))
    return annual, historical, scores


@app.cell
def _(annual, mo, plot_metric, pl, px, scores, wb):
    _metric = plot_metric.value
    _annual = annual.with_columns(
        pl.col("model").map_elements(wb.label, return_dtype=pl.String).alias("label")
    )
    mo.vstack(
        [
            mo.md(
                "## 1 · Historical accuracy and ranking\nMAE/RMSE are total fantasy points (lower is better). NDCG and top-K capture are ranking scores (higher is better). Seasons receive equal weight; forecast rows are not independent samples."
            ),
            mo.ui.plotly(
                px.bar(
                    scores,
                    x="label",
                    y=_metric,
                    color="label",
                    title=f"Equal-season {_metric}",
                    labels={"label": "Model"},
                ).update_layout(showlegend=False)
            ),
            mo.ui.plotly(
                px.line(
                    _annual.sort("year"),
                    x="year",
                    y=_metric,
                    color="label",
                    markers=True,
                    title="Does the gain persist across seasons?",
                )
            ),
            mo.ui.table(
                scores,
                selection=None,
                label="Matched scores and MAE-gain intervals",
                show_data_types=False,
            ),
            mo.md(
                "Positive `mae_gain` and `mse_reduction_pct` favor the candidate. MAE intervals are descriptive 95% season bootstraps, unadjusted for multiple research comparisons. Cohort filters recompute ranking within that cohort. Canonical-label corrections are applied to the unchanged original forecasts."
            ),
        ]
    )
    return


@app.cell
def _(historical, mo, models):
    residual_model = mo.ui.dropdown(models.value, value=models.value[0], label="Inspect errors for")
    residual_model
    return (residual_model,)


@app.cell
def _(historical, mo, pl, px, residual_model, wb):
    _f = historical.filter(pl.col("model") == residual_model.value).with_columns(
        (pl.col("prediction") - pl.col("actual")).alias("error")
    )
    _scatter = px.scatter(
        _f,
        x="actual",
        y="prediction",
        color="population",
        hover_data=["name", "year"],
        title="Predicted versus actual points",
    )
    _limits = [
        min(_f["actual"].min(), _f["prediction"].min()),
        max(_f["actual"].max(), _f["prediction"].max()),
    ]
    _scatter.add_shape(
        type="line",
        x0=_limits[0],
        y0=_limits[0],
        x1=_limits[1],
        y1=_limits[1],
        line={"dash": "dash", "color": "gray"},
    )
    _cohorts = []
    for (_population,), _part in _f.partition_by("population", as_dict=True).items():
        _a = wb.annual_scores(_part)
        _cohorts.append(
            {
                "population": _population,
                "rows": _part.height,
                "MAE": _a["mae"].mean(),
                "mean_bias": _part["error"].mean(),
            }
        )
    mo.vstack(
        [
            mo.md(
                "## 2 · Calibration, cohorts, and failures\nPositive residuals mean overprediction. Large missed seasons and many small errors can favor different models."
            ),
            mo.ui.plotly(_scatter),
            mo.ui.plotly(
                px.scatter(
                    _f,
                    x="prediction",
                    y="error",
                    color="population",
                    hover_data=["name", "year"],
                    title="Residuals versus prediction",
                ).add_hline(y=0, line_dash="dash")
            ),
            mo.ui.table(
                _cohorts,
                selection=None,
                label="Cohort errors: MAE equally weights available years; bias pools rows",
            ),
            mo.ui.table(
                _f.sort(pl.col("error").abs(), descending=True).head(25),
                selection=None,
                label="Largest errors (raw, including negative forecasts)",
            ),
        ]
    )
    return


@app.cell
def _(horizon, mo, position, wb):
    panel, feature_coverage = wb.data_profile(position.value, horizon.value)
    feature_metadata = wb.read_json(wb.checked(wb.DATA, "features.json"))
    feature_name = mo.ui.dropdown(
        feature_metadata["names"],
        value="summary:16:mean:usage_points",
        searchable=True,
        label="Explore an input",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "## 3 · Data coverage and distributions\nThis section describes the full prepared task, 2004–2025, independently of the comparison-year and cohort filters above. Missing cells remain unknown. Derived features and aliases do not represent independent observations."
            ),
            feature_name,
        ]
    )
    return feature_coverage, feature_metadata, feature_name, panel


@app.cell
def _(
    feature_coverage, feature_metadata, feature_name, horizon, mo, np, panel, pl, position, px, wb
):
    _counts = panel.group_by("year", "population").len().sort("year")
    _groups = (
        feature_coverage.group_by("group")
        .agg(pl.len().alias("columns"), pl.col("missing_fraction").mean())
        .sort("missing_fraction", descending=True)
    )
    _j = feature_metadata["names"].index(feature_name.value)
    _, _x = wb.task_data(position.value, horizon.value)
    _feature = (
        panel.select("year", "population", "actual")
        .with_columns(pl.Series("input", _x[:, _j]))
        .filter(pl.col("input").is_finite())
    )
    mo.vstack(
        [
            mo.md(
                f"**{panel.height:,} candidate-season rows · {panel['player_id'].n_unique():,} players · {len(feature_metadata['names']):,} inputs.** Input observed fraction: {np.isfinite(_x[:, _j]).mean():.1%}."
            ),
            mo.ui.plotly(
                px.bar(
                    _counts,
                    x="year",
                    y="len",
                    color="population",
                    title="Candidate coverage by season",
                )
            ),
            mo.ui.plotly(
                px.histogram(
                    panel,
                    x="actual",
                    color="population",
                    nbins=60,
                    title="Future-point distribution, including zero outcomes",
                )
            ),
            mo.ui.plotly(
                px.bar(_groups, x="group", y="missing_fraction", title="Missingness by input group")
            ),
            mo.ui.plotly(
                px.histogram(
                    _feature, x="input", color="population", nbins=60, title=feature_name.value
                )
            ),
            mo.ui.table(
                feature_coverage.sort("missing_fraction", descending=True),
                selection=None,
                page_size=10,
                label="Searchable feature inventory",
            ),
        ]
    )
    return


@app.cell
def _(mo, wb):
    tree_paths = [
        p
        for p in wb.read_json(wb.FITS / "protocol.json")["paths"]
        if p["family"] in {"hist", "lgb", "xgb", "cat", "extra", "forest"}
    ]
    capacity_path = mo.ui.dropdown(
        [p["name"] for p in tree_paths], value="lgb_04", label="Saved tree recipe"
    )
    capacity_year = mo.ui.dropdown(
        list(range(2019, 2026)), value=2025, label="Saved evaluation year"
    )
    mo.vstack(
        [
            mo.md(
                "## 4 · Saved capacity and ensemble-size curves\nTraining error and later-season error below are both **points per exposure**. The training target was normalized this way; total-point test loss must not be overlaid on that training loss. A falling training curve with rising later-season error indicates overfitting."
            ),
            mo.hstack([capacity_path, capacity_year], justify="start"),
        ]
    )
    return capacity_path, capacity_year, tree_paths


@app.cell
def _(capacity_path, capacity_year, horizon, mo, pl, position, px, tree_paths, wb):
    saved_curves = wb.saved_capacity(position.value, horizon.value, capacity_path.value)
    _path = next(p for p in tree_paths if p["name"] == capacity_path.value)
    _budget = pl.DataFrame(wb.read_json(wb.checked(wb.DIAGNOSTICS, "budget_curves.json"))).filter(
        (pl.col("position") == position.value) & (pl.col("horizon") == horizon.value)
    )
    _budget = (
        _budget.group_by("candidates")
        .agg(pl.col("mse").mean().sqrt().alias("RMSE"))
        .sort("candidates")
    )
    mo.vstack(
        [
            mo.ui.plotly(
                px.line(
                    saved_curves.filter(pl.col("year") == capacity_year.value),
                    x="trees",
                    y="rmse_rate",
                    color="split",
                    markers=True,
                    title="Saved training versus later-season capacity curve",
                )
            ),
            mo.ui.plotly(
                px.line(
                    _budget,
                    x="candidates",
                    y="RMSE",
                    markers=True,
                    title="Fixed convex-blend library budgets: 2019–2025",
                )
            ),
            mo.accordion({"Exact saved recipe": mo.json(_path)}),
            mo.md(
                "The three saved tree checkpoints are 40/120/360. Budget subsets have a fixed ordering; this is not an exhaustive subset search. These charts diagnose completed experiments, not untouched validation. Use the sandbox below for denser curves and training-history sizes."
            ),
        ]
    )
    return (saved_curves,)


@app.cell
def _(horizon, mo, position, wb):
    _prefix = "published_library/" if horizon.value in {"next_four", "remaining"} else ""
    _choices = wb.read_json(
        wb.checked(wb.REPORT, f"{_prefix}{position.value}_{horizon.value}/choices.json")
    )
    _decisions = [
        {
            "year": r["year"],
            "policy": p,
            "method": d["chosen"],
            "earlier_validation_years": str(d["validation_years"]),
        }
        for r in _choices
        for p, d in r["selected"].items()
    ]
    _last = next(r for r in _choices if r["year"] == 2025)
    mo.vstack(
        [
            mo.md(
                "## 5 · What the ensembles actually selected\nA top-three blend chooses its three members using earlier errors. A policy selector chooses a combination method using earlier evaluations. Neither is the hindsight-best model for the year shown."
            ),
            mo.ui.table(_decisions, selection=None, label="Year-by-year ensemble decisions"),
            mo.accordion(
                {"2025 compositions, weights and stacking parameters": mo.json(_last["selected"])}
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6 · Model guide and tuning controls

    | Model | What it does | Most useful knobs |
    |---|---|---|
    | Original profile/enriched booster | sklearn histogram gradient boosting; enriched adds tracking/college context | 120 rounds, learning rate .05, 15 leaves, minimum 30 samples, L2 10, 63 bins; no early stopping; original seed 20260923 |
    | LightGBM | Leaf-wise boosted trees with native missing-value handling | Leaves/depth, leaf support, learning rate, rounds, L2, feature fraction |
    | XGBoost | Regularized histogram boosted trees | Depth, minimum child weight, learning rate, rounds, L2, feature fraction; this implementation subsamples rows at .8 |
    | CatBoost | Symmetric boosted trees, here using numeric inputs | Depth, learning rate, rounds, L2, feature fraction; 63 borders |
    | Random forest / extra trees | Averages independently randomized trees | Tree count, depth, leaf support, feature sampling; visible in saved comparisons |
    | Ensembles | Combine earlier-trained forecasts | Membership, equal/learned weights, regularization and objective |

    **Interpret knobs by engine:** leaf support is sample count for sklearn/LightGBM and
    minimum child weight for XGBoost; it is not passed to CatBoost. Leaves are a control
    for sklearn/LightGBM, not XGBoost/CatBoost. Depth 0 means unlimited for sklearn/LightGBM.
    L2 scales differ by engine. Fewer leaves, larger leaf support, or stronger L2 usually
    reduce complexity. Smaller learning rates often need more rounds. More rounds can overfit.

    **Feature discovery:** `selected` keeps train-selected inputs; `interactions` adds
    learned products and stabilized ratios; `leaf` adds tree-derived response coordinates.
    These transformations fit only on training rows. They are not causal explanations.
    Own-data `inventory` and `all` are aliases here. `market` adds dated market inputs.

    **Reading a learning curve:** training size below grows by adding older complete seasons,
    while validation/test seasons stay fixed. It measures the value of more historical data,
    including any regime changes—not an IID random-sample learning curve.
    """)
    return


@app.cell
def _(horizon, mo):
    _presets = ["New histogram booster", "LightGBM", "XGBoost", "CatBoost"]
    if horizon.value in {"next_four", "remaining"}:
        _presets += ["Original profile booster", "Original enriched booster"]
    preset = mo.ui.dropdown(_presets, value="New histogram booster", label="Load a recipe preset")
    preset
    return (preset,)


@app.cell
def _(horizon, mo, position, preset, wb):
    _views = ["basic", "summary", "raw", "all", "market"]
    if horizon.value in {"next_four", "remaining"}:
        _views = ["original_profile", "original_enriched", *_views]
    preset_config = {
        **wb.DEFAULT_TRIAL,
        "position": position.value,
        "horizon": horizon.value,
        "view": "all",
        "feature_scope": "position",
        "target": "all_stats",
    }
    if preset.value.startswith("Original"):
        preset_config.update(
            view="original_profile" if "profile" in preset.value else "original_enriched",
            depth=0,
            feature_scope="all",
            target="points",
        )
    elif preset.value in {"LightGBM", "XGBoost", "CatBoost"}:
        preset_config["engine"] = {"LightGBM": "lgb", "XGBoost": "xgb", "CatBoost": "cat"}[
            preset.value
        ]
    _controls = {
        "position": mo.ui.dropdown(
            [position.value], value=position.value, label="Trial position (set above)"
        ),
        "target": mo.ui.dropdown(
            {
                "All relevant stats (separate models)": "all_stats",
                "Fantasy points": "points",
                **{wb.target_label(t): t for t in wb.POSITION_TARGETS[position.value]},
            },
            value="All relevant stats (separate models)"
            if preset_config["target"] == "all_stats"
            else wb.target_label(preset_config["target"]),
            label="What to predict",
        ),
        "horizon": mo.ui.dropdown(
            [horizon.value], value=horizon.value, label="Trial horizon (set above)"
        ),
        "view": mo.ui.dropdown(_views, value=preset_config["view"], label="Input set"),
        "feature_scope": mo.ui.dropdown(
            {"Position-specific inputs": "position", "All input categories (comparison)": "all"},
            value="Position-specific inputs"
            if preset_config["feature_scope"] == "position"
            else "All input categories (comparison)",
            label="Player stat categories",
        ),
        "engine": mo.ui.dropdown(
            ["hist", "lgb", "xgb", "cat"], value=preset_config["engine"], label="Engine"
        ),
        "representation": mo.ui.dropdown(
            ["identity", "selected", "interactions", "leaf"],
            value="identity",
            label="Feature discovery",
        ),
        "year": mo.ui.dropdown(list(range(2019, 2026)), value=2025, label="Held-out test season"),
        "objective": mo.ui.dropdown(["mse", "mae"], value="mse", label="Validation objective"),
        "max_trees": mo.ui.number(2, 4096, value=120, step=1, label="Maximum rounds"),
        "learning_rate": mo.ui.number(0.001, 1, value=0.05, step=0.001, label="Learning rate"),
        "leaves": mo.ui.number(2, 127, value=15, step=1, label="Maximum leaves (hist/lgb)"),
        "depth": mo.ui.number(0, 12, value=preset_config["depth"], step=1, label="Maximum depth"),
        "leaf_samples": mo.ui.number(1, 200, value=30, step=1, label="Leaf support (not CatBoost)"),
        "l2": mo.ui.number(0, 1000, value=10, step=1, label="L2 regularization"),
        "feature_fraction": mo.ui.number(0.1, 1, value=1, step=0.1, label="Feature fraction"),
        "history_years": mo.ui.number(
            0, 20, value=0, step=1, label="Earlier training seasons (0 = all)"
        ),
        "half_life": mo.ui.number(
            0, 20, value=0, step=1, label="Recency half-life in years (0 = equal)"
        ),
        "seed": mo.ui.number(0, 2147483647, value=20260923, step=1, label="Random seed"),
    }
    trial_form = (
        mo.md("""
    | Dataset and evaluation | Tree settings |
    |---|---|
    | {position} | {engine} |
    | {horizon} | {max_trees} |
    | {view} | {learning_rate} |
    | {representation} | {leaves} |
    | {year} | {depth} |
    | {objective} | {leaf_samples} |
    | {history_years} | {l2} |
    | {half_life} | {feature_fraction} |
    | {seed} | |
    | {feature_scope} | |
    | {target} | |
    """)
        .batch(**_controls)
        .form(submit_button_label="Train tree experiment")
    )
    mo.vstack(
        [
            mo.md(
                "### Run a tree experiment\nChoose **What to predict**: an individual stat, all relevant stats, or fantasy points. "
                "Each stat gets its own model and validation-selected tree count. Errors are in that stat's units. "
                "Trees fit squared error; the validation objective chooses the number of rounds. "
                "The immediately earlier season selects tree count; the final refit includes it and is evaluated on the chosen test year. "
                "All-stats training runs several experiments and can take minutes. Identical trials reuse verified saved results."
            ),
            trial_form,
            mo.md(
                "**Position-specific inputs:** QB keeps passing and rushing; RB keeps rushing and receiving; "
                "WR/TE keep receiving and rushing (including hybrid roles). All retain age, draft, availability, "
                "workload, past fantasy production and team context. RB/WR/TE exclude player passing stats, "
                "including college/history derivatives; team passing volume remains available. "
                "These are candidate feature sets, not proven optimal sets. Input categories and prediction targets are separate choices. "
                "Choose `selected` feature discovery to learn a smaller set using training data only. "
                "Choose all categories for a controlled comparison. Original presets retain all inputs by default."
            ),
            mo.md(
                "To study the original recipe, choose `original_profile` or `original_enriched`, `hist`, `identity`, 120 rounds, .05 learning rate, 15 leaves, depth 0, leaf support 30, L2 10, feature fraction 1, all history and no recency weighting. Original inputs retain original training labels and nonnegative predictions; evaluation uses reconciled canonical outcomes. This sandbox adds a validation split, so it is a new experiment rather than the archived forecast above."
            ),
        ]
    )
    return preset_config, trial_form


@app.cell
def _(mo, preset_config, trial_form, wb):
    trial_error = None
    try:
        if trial_form.value is None:
            target_trials = wb.run_target_trials(preset_config, cached=True)
        else:
            target_trials = wb.run_target_trials(trial_form.value)
    except ValueError as _error:
        trial_error = str(_error)
    mo.stop(trial_error is not None, mo.callout(trial_error or "", kind="warn"))
    mo.stop(
        not target_trials,
        mo.md(
            "Submit the form to generate training-size and overfitting curves. No models train merely by opening this notebook. Previously saved default curves appear automatically when available."
        ),
    )
    return (target_trials,)


@app.cell
def _(mo, pl, target_trials, wb):
    curve_target = mo.ui.dropdown(
        {wb.target_label(t): t for t in target_trials},
        value=wb.target_label(next(iter(target_trials))),
        label="Stat to inspect in curves below",
    )
    _rows = []
    for _target, _result in target_trials.items():
        _error = _result["forecasts"]["prediction"] - _result["forecasts"]["actual"]
        _rows.append(
            {
                "target": wb.target_label(_target),
                "units": wb.target_unit(_target),
                "players": len(_error),
                "rmse": float((_error**2).mean() ** 0.5),
                "mae": float(_error.abs().mean()),
                "rounds": _result["metadata"]["selected_trees"],
                "unknown_examples_excluded": sum(
                    _result["metadata"].get("unknown_target_rows_excluded_by_year", {}).values()
                ),
            }
        )
    _lines = wb.stat_line(target_trials)
    _display_lines = (
        _lines.select("name", *target_trials)
        .with_columns(pl.col(t).round(2) for t in target_trials)
        .rename({"name": "Player", **{t: wb.target_label(t) for t in target_trials}})
    )
    _settings = next(iter(target_trials.values()))["metadata"]["provenance"]["config"]
    mo.vstack(
        [
            mo.md(
                "### Predicted player stat lines\nThese are held-out historical predictions for the selected season and horizon. "
                "Each column is a separately trained forecast. Fractional counts are expected values. "
                "Models do not yet enforce joint constraints such as receptions ≤ targets. "
                "The saved ranking comparisons above remain fantasy-point forecasts."
                f" Showing **{len(target_trials)} completed stat model(s)**; submit the form to compute any uncached selections."
                f" **{_settings['position']} · {_settings['year']} · {_settings['horizon'].replace('_', ' ')}**."
            ),
            mo.ui.table(
                pl.DataFrame(_rows).with_columns(pl.col("rmse", "mae").round(2)),
                selection=None,
                label="Accuracy by stat (units differ; do not average errors across stats)",
            ),
            mo.ui.table(_display_lines, selection=None, label="Predicted totals for each player"),
            mo.download(
                data=_lines.write_csv(),
                filename="predicted_player_stats.csv",
                label="Download player stat lines",
            ),
            curve_target,
        ]
    )
    return (curve_target,)


@app.cell
def _(curve_target, target_trials):
    trial = target_trials[curve_target.value]
    return (trial,)


@app.cell
def _(mo, pl, px, trial, wb):
    _meta = trial["metadata"]
    _pred = trial["forecasts"].with_columns(pl.lit("sandbox_refit").alias("model"))
    _metrics = wb.annual_scores(_pred)
    _matched_scores = wb.compare_trial(trial)
    mo.vstack(
        [
            mo.md(
                f"### Completed tree experiment: {wb.target_label(_meta['target'])}\n"
                f"Errors are in **{_meta['units']}** over the forecast period. "
                f"Training: **{_meta['train_first_year']}–{_meta['train_last_year']}** → validation: **{_meta['validation_year']}** → test: **{_meta['test_year']}**. Validation selected **{_meta['selected_trees']} rounds**. Final refit uses all permitted history through {_meta['refit_last_year']}."
            ),
            mo.ui.plotly(
                px.line(
                    trial["capacity"],
                    x="trees",
                    y="rmse",
                    color="split",
                    markers=True,
                    title="Capacity: training / earlier validation / later test",
                    labels={"rmse": f"RMSE ({_meta['units']})"},
                ).add_vline(x=_meta["selected_trees"], line_dash="dash")
            ),
            mo.ui.plotly(
                px.line(
                    trial["learning"].sort("train_rows"),
                    x="train_rows",
                    y="rmse",
                    color="split",
                    markers=True,
                    hover_data=["first_year", "last_train_year"],
                    title="Learning curve: add older seasons, keep validation and test fixed",
                    labels={"rmse": f"RMSE ({_meta['units']})"},
                )
            ),
            mo.ui.table(_metrics, selection=None, label="Final refit: test-season scores"),
            mo.ui.table(
                _matched_scores,
                selection=None,
                label="Final refit versus baseline: identical test players",
            ),
            mo.ui.table(trial["forecasts"], selection=None, label="Saved test forecasts"),
            mo.accordion(
                {
                    "Exact included and excluded input features": mo.ui.table(
                        trial["features"], selection=None, label="Position feature audit"
                    )
                }
            ),
            mo.accordion({"Submitted settings, selected features and provenance": mo.json(_meta)}),
            mo.md(
                f"Saved to `{trial['path'].relative_to(wb.ROOT)}`. The curves use the inner training model; the table uses the final refit. Repeatedly tuning after looking at this test season makes it development evidence. Freeze a recipe before evaluating genuinely later outcomes."
            ),
            mo.download(
                data=trial["forecasts"].write_csv(),
                filename="tree_trial_forecasts.csv",
                label="Download trial forecasts",
            ),
        ]
    )
    return


@app.cell
def _(horizon, mo, position, wb):
    _choices = wb.model_menu(horizon.value)
    _initial = ["lgb_04_t120", "auto_xgb_market", "persistence"]
    if "existing:profile_boost" in _choices:
        _initial = ["existing:profile_boost", "lgb_04_t120", "auto_xgb_market"]
    blend_form = mo.ui.dictionary(
        {
            "models": mo.ui.multiselect(
                _choices, value=_initial, label="Blend members", full_width=True
            ),
            "year": mo.ui.dropdown([2022, 2023, 2024, 2025], value=2025, label="Blend test year"),
            "method": mo.ui.dropdown(
                ["equal", "convex", "greedy"], value="convex", label="Weighting method"
            ),
            "shrink": mo.ui.number(
                0, 10, value=0.1, step=0.1, label="Convex shrinkage toward equal weights"
            ),
            "steps": mo.ui.number(1, 200, value=20, step=1, label="Greedy MSE selection steps"),
        }
    ).form(submit_button_label="Evaluate custom ensemble")
    mo.vstack(
        [
            mo.md(
                f"## 7 · Build your own ensemble\nThis form uses **{position.value} / {horizon.value}**, all available cohorts, and up to five earlier out-of-year seasons. The comparison filters do not silently shorten its training history. Weights use earlier MSE; the chosen test year's outcomes never fit them. Try adding the original booster or removing redundant models."
            ),
            blend_form,
        ]
    )
    return (blend_form,)


@app.cell
def _(blend_form, horizon, mo, pl, position, px, wb):
    mo.stop(
        blend_form.value is None,
        mo.md(
            "Select members and submit to compare a custom blend with its members on the same test players."
        ),
    )
    _c = blend_form.value
    _error_message = None
    try:
        _library = wb.comparison_frame(position.value, horizon.value, tuple(_c["models"]))
        _forecast, _weights, _training_years = wb.blend_trial(
            _library, _c["year"], _c["method"], _c["shrink"], _c["steps"]
        )
        _members = _library.filter(pl.col("year") == _c["year"]).select(*_forecast.columns, "model")
        _result = pl.concat(
            [_members, _forecast.with_columns(pl.lit("custom_blend").alias("model"))]
        )
        _scores = wb.annual_scores(_result)
    except ValueError as _error:
        _error_message = str(_error)
    mo.stop(_error_message is not None, mo.callout(_error_message or "", kind="warn"))
    mo.vstack(
        [
            mo.md(
                f"Weights learned on **{_training_years}**; test **{_c['year']}**. A blend with one member is that member."
            ),
            mo.ui.plotly(px.bar(_weights, x="model", y="weight", title="Custom ensemble weights")),
            mo.ui.table(_scores, selection=None, label="Members versus custom ensemble"),
            mo.download(
                data=_forecast.write_csv(),
                filename="custom_ensemble_forecasts.csv",
                label="Download custom forecasts",
            ),
            mo.accordion(
                {
                    "Reproduce this blend": mo.json(
                        {
                            "position": position.value,
                            "horizon": horizon.value,
                            **_c,
                            "training_years": _training_years,
                            "weights": _weights.to_dicts(),
                        }
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, wb):
    mo.md(f"""
    ## 8 · Reproduce and extend

    - Edit a notebook cell to change the analysis; marimo tracks dependent cells.
    - Change a submitted form to create a new tree trial. Each trial stores config,
      library versions, source/data hashes, curves and predictions under `runs/notebook_tree_*`.
    - For repeatable multi-year experiments, call `workbench.run_trial({{...}})` for each
      outer year. Define the candidate settings before reviewing those years' outcomes.
    - To extend the campaign itself, edit a copy of `deep/config.json` and the recipe
      generator in `deep/models.py`, then use **new** run IDs. Preserve completed runs.
    - To add predictors, update `deep/data.py` with a cutoff-safe definition and rebuild
      into a new prepared-data run. Do not insert current news into historical rows.

    Canonical publication still requires the documented older-history, uncertainty,
    cohort, eligibility and ranking checks. This notebook does not publish predictions.

    **Local sources:** `deep_readout.md`, `deep/README.md`,
    `src/engine/metrics/current_rankings.py`, and the manifests for
    `{wb.DATA.name}`, `{wb.FITS.name}`, `{wb.CAPACITY.name}`, `{wb.REPORT.name}`.

    **Model documentation:** [sklearn histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html),
    [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
    [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
    [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/),
    [marimo forms](https://docs.marimo.io/api/inputs/form/) and
    [expensive notebooks](https://docs.marimo.io/guides/expensive_notebooks/).
    """)
    return


if __name__ == "__main__":
    app.run()
