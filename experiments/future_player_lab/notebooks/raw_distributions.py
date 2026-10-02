"""Raw NFL data: opportunity, efficiency, boosted trees and probabilistic forecasts."""
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
    import plotly.graph_objects as go
    import polars as pl
    from experiments.future_player_lab.notebooks import raw_distribution_data as raw
    from experiments.future_player_lab.notebooks import raw_distribution_models as models
    from experiments.future_player_lab.notebooks import run_raw_distributions as runner

    px.defaults.template = "plotly_white"
    return go, json, mo, models, np, pl, px, raw, runner


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From raw football data to forecast distributions

    **An exploratory experiment, starting from the raw source snapshot.** No prepared
    feature matrix, previous predictions, or assumed winning feature set. Inventory every
    source; derive numeric historical summaries with explicit cutoffs; test the ideas.
    No Bayesian hierarchical models are used or needed by the boosted trees.

    Questions: Does opportunity persist more than efficiency? Does decomposing team volume,
    player share and per-touch production help? How much shrinkage helps? Does a learned age
    curve improve forecasts? Are Poisson or negative-binomial TD distributions useful?

    **Primary evaluation: CRPS for distributions, and negative log probability for TD counts.**
    Smaller is better. RMSE/MAE, coverage and width are supporting diagnostics. A narrow range
    is useful only if it also scores well. Never compare CRPS across different target units
    or horizons as if they were interchangeable.

    The completed local experiment appears below when available. Opening this notebook loads
    its tables but does **not** fit models. Submit the experiment form to run a new session.
    RB is the initial experiment; WR and TE can be explored with the same rush/receive structure.
    """)
    return


@app.cell
def _(raw):
    raw_catalog = raw.catalog()
    return (raw_catalog,)


@app.cell(hide_code=True)
def _(mo, pl, raw, raw_catalog):
    _summary = (
        raw_catalog.group_by("family", "policy")
        .agg(
            pl.len().alias("assets"),
            pl.col("rows").sum().alias("raw_rows"),
            (pl.col("bytes").sum() / 1024**2).alias("MiB"),
        )
        .sort("family")
    )
    mo.vstack(
        [
            mo.md(
                f"## 1. What do we actually have?\n\n**{raw_catalog.height:,} captured assets** in `{raw.SNAPSHOT.name}` / `{raw.SNAPSHOT.parent.name}`. Counts below describe source records, including overlapping captures; they are not independent training examples. Every raw column remains accessible in the browser."
            ),
            mo.ui.table(
                _summary,
                selection=None,
                label="Raw inventory and each source's time/identity policy",
            ),
            mo.download(
                raw_catalog.drop("path").write_csv(),
                filename="raw_source_inventory.csv",
                label="Download source inventory",
            ),
            mo.md("""
        The model starts with **all numeric historical fields** from compatible NFL sources,
        plus exact-ID college statistics and pre-draft measurements. It learns which columns
        are nonconstant using training data only. Numeric IDs and calendar identifiers are
        excluded; text and nested fields remain in the raw browser. Prior-season, current-to-origin
        and last-four-calendar-week means are explicit, editable representations—not proven features.
        Historical EPA/WP/NGS values are provider-derived summaries; the play-detail ablation removes
        the PBP and NGS families. There is no frame-level tracking tensor here.

        **Coverage is part of the experiment.** Source families without a usable temporal/identity
        contract are listed with their exclusion reason. College joins use unique exact ESPN IDs;
        ambiguous IDs and opaque `stat_N` columns are withheld. Provider publication vintages are
        incomplete, so these are reconstructed historical tests, not a claim to exact live replay.
        """),
        ]
    )
    return


@app.cell
def _(mo, pl, raw_catalog):
    asset_choice = mo.ui.dropdown(
        raw_catalog.filter(pl.col("asset").str.ends_with(".parquet"))["asset"].to_list(),
        value=raw_catalog.filter(pl.col("family") == "box")["asset"][0],
        label="Inspect an unmodified raw table",
    )
    asset_choice
    return (asset_choice,)


@app.cell
def _(asset_choice, mo, pl, raw):
    _preview = raw.raw_preview(asset_choice.value)
    mo.accordion(
        {
            "Raw rows (first 100; all columns)": mo.ui.table(_preview, selection=None),
            "Raw field names and types": mo.ui.table(
                pl.DataFrame(
                    {
                        "field": list(_preview.schema),
                        "dtype": [str(t) for t in _preview.schema.values()],
                    }
                ),
                selection=None,
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Freeze the forecast question and chronology

    At origin W, predictors use only weeks ≤ W and earlier seasons. Next-week targets cover W+1;
    remaining-season targets cover W+1 through the regular-season end. Preseason uses W=0.
    Candidates are prior-year participants, observed current-year players and exactly linked
    drafted combine entrants. Undrafted/unlinked preseason newcomers are missing from this cohort;
    there is no future-participation filter. Absent production is zero only in completed periods;
    recorded unknown target fields are excluded and counted. Team identity is the last observed
    team at the origin, or the draft team for a new entrant. Trades and departures remain difficult.

    For test year **Y**: **fit ≤Y−3 → select on Y−2 → refit ≤Y−2 → calibrate on Y−1 → test on Y**.
    Tree counts and efficiency shrinkage use validation scores. Residual donors and NB dispersion
    use only the separate calibration year. Validation selection uses fitting-period residuals;
    those can be optimistic, but neither calibration nor test is used to select the recipe.
    Repeatedly inspecting these test years makes them development evidence.

    The points target is explicitly **rushing/receiving PPR**:
    `0.1 × (rushing yards + receiving yards) + receptions + 6 × (rushing TD + receiving TD)`.
    It excludes passing, fumbles, returns, two-point conversions and league bonuses.
    Individual stat and TD distributions are also scored in their own units.

    Historical closing lines have no original publication timestamps. The optional **closing-line
    sensitivity** gives the next-week direct model those eventual lines and must be read as a
    timing-optimistic experiment. It is excluded from the main operational comparison. Season-long
    future lines are never offered as a preseason feature.
    """)
    return


@app.cell
def _(mo):
    experiment = (
        mo.md("""
    {position} {horizon} {engine}

    {years} {origins}

    {rounds} {draws} {ablations} {market}
    """)
        .batch(
            **{
                "position": mo.ui.dropdown(["RB", "WR", "TE"], value="RB", label="Position"),
                "horizon": mo.ui.dropdown(
                    ["week", "remaining", "season"], value="week", label="Horizon"
                ),
                "engine": mo.ui.dropdown(
                    ["lightgbm", "xgboost", "hist"], value="lightgbm", label="Tree engine"
                ),
                "years": mo.ui.multiselect(
                    list(range(2019, 2026)), value=[2025], label="Test seasons"
                ),
                "origins": mo.ui.multiselect(
                    list(range(1, 18)),
                    value=[2],
                    label="Completed weeks at origin (ignored preseason)",
                ),
                "rounds": mo.ui.multiselect(
                    [20, 40, 100, 200], value=[40, 100], label="Validation tree-count candidates"
                ),
                "draws": mo.ui.number(100, 5000, value=1000, step=100, label="Monte Carlo draws"),
                "ablations": mo.ui.checkbox(
                    value=True, label="Test opportunity-only and no-play-detail inputs"
                ),
                "market": mo.ui.checkbox(
                    value=False, label="Add closing-line sensitivity (next-week only)"
                ),
            }
        )
        .form(submit_button_label="Run raw-data distribution experiment", bordered=True)
    )
    experiment
    return (experiment,)


@app.cell
def _(experiment, mo, models, pl, raw):
    session = None
    if experiment.value is not None:
        _c = experiment.value
        mo.stop(
            not _c["years"]
            or not _c["rounds"]
            or (_c["horizon"] != "season" and not _c["origins"]),
            mo.md("Select a test year, tree count, and forecast origin."),
        )
        mo.stop(
            _c["market"] and _c["horizon"] != "week",
            mo.md("Closing-line sensitivity is limited to the next-week horizon."),
        )
        with mo.status.spinner(title="Building raw inputs and fitting chronological experiments"):
            _panel, _features, _audit = raw.make_panel(
                _c["position"], _c["horizon"], tuple(_c["origins"])
            )
            _folds = [
                models.run_fold(
                    _panel,
                    _features,
                    year=_y,
                    engine=_c["engine"],
                    rounds=tuple(_c["rounds"]),
                    draws=int(_c["draws"]),
                    ablations=_c["ablations"],
                    market=_c["market"],
                )
                for _y in sorted(_c["years"])
            ]
            _rows = pl.concat([_f["rows"] for _f in _folds])
            session = {
                "rows": _rows,
                "summary": models.summarize(_rows),
                "intervals": models.paired_crps_intervals(_rows),
                "folds": _folds,
                "panel": _panel,
                "features": _features,
                "audit": _audit,
                "config": _c,
            }
    return (session,)


@app.cell
def _(runner):
    saved = runner.load_saved() if (runner.DEFAULT_RUN / "manifest.json").exists() else None
    return (saved,)


@app.cell(hide_code=True)
def _(mo, raw, saved, session):
    _readout = raw.ROOT / "experiments/future_player_lab/notebooks/raw_distribution_readout.md"
    if saved is not None and session is None and _readout.exists():
        mo.output.replace(
            mo.accordion(
                {"Completed RB experiment: findings and limitations": mo.md(_readout.read_text())}
            )
        )
    return


@app.cell
def _(mo, saved, session):
    result = session if session is not None else saved
    mo.stop(
        result is None, mo.md("No completed local run yet. Submit the form to start an experiment.")
    )
    mo.md(
        "**Showing submitted session results.**"
        if session is not None
        else "**Showing the verified completed local experiment.** Its settings and chronology are recorded below; the unsubmitted form does not change these results."
    )
    return (result,)


@app.cell
def _(mo, result):
    _rows = result["rows"]
    result_filters = mo.ui.dictionary(
        {
            "horizon": mo.ui.dropdown(
                sorted(_rows["horizon"].unique().to_list()),
                value="week" if "week" in _rows["horizon"].to_list() else _rows["horizon"][0],
                label="Result horizon",
            ),
            "target": mo.ui.dropdown(
                sorted(_rows["target"].unique().to_list()), value="points", label="Scored target"
            ),
            "year": mo.ui.dropdown(
                ["All", *[str(x) for x in sorted(_rows["year"].unique().to_list())]],
                value="All",
                label="Test year",
            ),
        }
    )
    result_filters
    return (result_filters,)


@app.cell
def _(pl, result, result_filters):
    filtered = result["rows"].filter(
        (pl.col("horizon") == result_filters.value["horizon"])
        & (pl.col("target") == result_filters.value["target"])
    )
    if result_filters.value["year"] != "All":
        filtered = filtered.filter(pl.col("year") == int(result_filters.value["year"]))
    filtered = filtered.with_columns(pl.col("negative_log_score").fill_nan(None))
    scoreboard = (
        filtered.group_by("model")
        .agg(
            pl.len().alias("n"),
            pl.col("crps", "empirical_crps", "negative_log_score").mean(),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("covered50", "covered80", "covered90", "width80").mean(),
        )
        .sort("crps")
    )
    return filtered, scoreboard


@app.cell(hide_code=True)
def _(filtered, mo, pl, px, result, result_filters, scoreboard):
    _annual = filtered.group_by("year", "model").agg(pl.col("crps").mean())
    mo.vstack(
        [
            mo.md(
                "## 3. Score the distributions\n\nCRPS is the primary ranking here. Negative log score uses an analytic count PMF, so it is populated only for the explicit Poisson/NB TD models; we never take the log of Monte Carlo histogram frequencies. Blank log scores do not mean zero error."
            ),
            mo.ui.table(
                scoreboard,
                selection=None,
                label="Same candidates, target and test periods for every model",
            ),
            mo.ui.plotly(
                px.bar(
                    scoreboard, x="model", y="crps", title="Distribution error (lower is better)"
                )
            ),
            mo.ui.plotly(
                px.line(
                    _annual.sort("year"),
                    x="year",
                    y="crps",
                    color="model",
                    markers=True,
                    title="Does the result repeat across seasons?",
                )
            ),
            mo.accordion(
                {
                    "Points CRPS differences, pooled across all test years": mo.vstack(
                        [
                            mo.md(
                                "This table always uses points and all saved test years for the selected horizon. Negative differences favor the model. These are descriptive 95% player-cluster bootstrap intervals: repeated origins and years for a player stay together. Shared-team dependence and recipe search are not corrected; the annual plot matters."
                            ),
                            mo.ui.table(
                                result["intervals"].filter(
                                    pl.col("horizon") == result_filters.value["horizon"]
                                ),
                                selection=None,
                            ),
                        ]
                    )
                }
            ),
            mo.download(
                scoreboard.write_csv(),
                filename="distribution_scores.csv",
                label="Download these scores",
            ),
            mo.download(
                filtered.write_csv(),
                filename="individual_distribution_scores.csv",
                label="Download individual forecasts and scores",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(filtered, go, mo, pl, px):
    _coverage = (
        filtered.group_by("model")
        .agg(pl.col("covered50", "covered80", "covered90").mean())
        .unpivot(index="model", variable_name="interval", value_name="coverage")
    )
    _coverage = _coverage.with_columns(
        pl.col("interval")
        .replace_strict({"covered50": 0.5, "covered80": 0.8, "covered90": 0.9})
        .alias("nominal")
    )
    _fig = px.line(
        _coverage.sort("nominal"),
        x="nominal",
        y="coverage",
        color="model",
        markers=True,
        range_y=[0, 1],
        title="Nominal versus observed central-interval coverage",
    )
    _fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Reference",
            line={"dash": "dash", "color": "gray"},
        )
    )
    mo.vstack(
        [
            mo.md(
                "## 4. Calibration and sharpness\n\nCoverage alone is not a proper score: very wide forecasts can cover everything. Read coverage together with CRPS and interval width. Discrete TD intervals and zero-heavy points can over-cover nominal levels; the randomized rank histogram handles ties."
            ),
            mo.ui.plotly(_fig),
            mo.ui.plotly(
                px.histogram(
                    filtered,
                    x="pit",
                    facet_col="model",
                    facet_col_wrap=3,
                    nbins=10,
                    title="Randomized ensemble ranks (approximately uniform when calibrated)",
                )
            ),
            mo.ui.plotly(
                px.scatter(
                    filtered.group_by("model").agg(pl.col("width80", "crps").mean()),
                    x="width80",
                    y="crps",
                    text="model",
                    title="Width versus proper-score error",
                )
            ),
        ]
    )
    return


@app.cell
def _(filtered, mo):
    _choices = {
        f"{r['player_display_name']} · {r['year']} · after W{r['origin']}": f"{r['player_id']}|{r['year']}|{r['origin']}"
        for r in filtered.select("player_id", "player_display_name", "year", "origin")
        .unique()
        .sort("year", "player_display_name")
        .iter_rows(named=True)
    }
    player_choice = mo.ui.dropdown(
        _choices,
        value=next(iter(_choices)),
        label="Player / forecast origin",
        allow_select_none=False,
    )
    player_choice
    return (player_choice,)


@app.cell
def _(filtered, mo, np, pl, player_choice, px, result, result_filters, session):
    mo.stop(
        player_choice.value is None, mo.md("Choose a player to inspect the simulated distribution.")
    )
    _id, _year, _origin = player_choice.value.split("|")
    player_rows = filtered.filter(
        (pl.col("player_id") == _id)
        & (pl.col("year") == int(_year))
        & (pl.col("origin") == int(_origin))
    )
    _h = result_filters.value["horizon"]
    _target = result_filters.value["target"]
    if session is not None:
        _samples = next(
            f["draws"] for f in session["folds"] if f["chronology"]["test"] == int(_year)
        )
    else:
        _samples = np.load(result["path"] / f"{_h}_{_year}" / "draws.npz")
    _dist = pl.concat(
        [
            pl.DataFrame(
                {
                    "draw": _samples[f"{r['model']}/{_target}"][r["sample_index"]],
                    "model": r["model"],
                }
            )
            for r in player_rows.iter_rows(named=True)
        ]
    )
    if session is None:
        _samples.close()
    _chart = px.ecdf(
        _dist, x="draw", color="model", title="Forecast CDFs; vertical line is the observed outcome"
    )
    _chart.add_vline(x=player_rows["actual"][0], line_dash="dash")
    mo.vstack(
        [
            mo.md(
                "## 5. A player's range of outcomes\n\nP10/P50/P90 are forecast quantiles, not guaranteed floor/ceiling values. The mean need not equal the median."
            ),
            mo.ui.table(
                player_rows.select(
                    "player_display_name",
                    "year",
                    "origin",
                    "model",
                    "target",
                    "actual",
                    "mean",
                    "p10",
                    "median",
                    "p90",
                    "crps",
                    "negative_log_score",
                ),
                selection=None,
            ),
            mo.ui.plotly(_chart),
            mo.download(
                player_rows.write_csv(),
                filename="player_distributions.csv",
                label="Download player forecast summary",
            ),
            mo.download(
                _dist.write_csv(),
                filename="player_predictive_draws.csv",
                label="Download this player's Monte Carlo draws",
            ),
        ]
    )
    return (player_rows,)


@app.cell
def _(pl, raw, result, result_filters, session):
    _h = result_filters.value["horizon"]
    if session is not None:
        analysis_panel, candidate_features, exclusions = (
            session["panel"],
            session["features"],
            session["audit"],
        )
    else:
        analysis_panel = pl.read_parquet(result["path"] / f"panel_{_h}.parquet")
        candidate_features = [
            c
            for c in raw.numeric(analysis_panel)
            if not c.startswith("y_")
            and c not in {"target_records", "year", "market_implied", "market_spread"}
        ]
        exclusions = pl.read_parquet(result["path"] / f"unknown_targets_{_h}.parquet")
    return analysis_panel, candidate_features, exclusions


@app.cell(hide_code=True)
def _(analysis_panel, filtered, mo, pl):
    _history = analysis_panel.select(
        "player_id",
        "year",
        "origin",
        ((pl.col("history_carries") + pl.col("history_targets")) / pl.col("history_weeks")).alias(
            "past_opportunities_per_week"
        ),
    ).with_columns(
        pl.when(pl.col("past_opportunities_per_week") == 0)
        .then(pl.lit("0 observed"))
        .when(pl.col("past_opportunities_per_week") < 5)
        .then(pl.lit("0–5 per week"))
        .when(pl.col("past_opportunities_per_week") < 15)
        .then(pl.lit("5–15 per week"))
        .otherwise(pl.lit("15+ per week"))
        .alias("past_workload"),
    )
    _cohorts = filtered.join(_history, on=["player_id", "year", "origin"], validate="m:1")
    _cohorts = (
        _cohorts.group_by("past_workload", "model")
        .agg(
            pl.len().alias("n"),
            pl.col("crps").mean(),
            pl.col("covered80").mean(),
            pl.col("actual").mean().alias("actual_mean"),
        )
        .sort("past_workload", "crps")
    )
    mo.accordion(
        {
            "Distribution quality by workload known at the cutoff": mo.vstack(
                [
                    mo.md(
                        "These groups use past carries plus targets per calendar week, never future production. They help reveal whether scores are dominated by players with little observed workload."
                    ),
                    mo.ui.table(_cohorts, selection=None),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(analysis_panel, exclusions, mo, pl, px, raw):
    _stability = raw.stability(analysis_panel)
    _age = analysis_panel.filter(pl.col("history_carries") > 0).with_columns(
        pl.col("age").floor().alias("age_bin")
    )
    _age = _age.group_by("age_bin").agg(
        pl.len().alias("n"),
        pl.col("y_carries", "y_rushing_yards").sum(),
        (pl.col("y_points").sum() / pl.col("weeks").sum()).alias("points_per_calendar_week"),
    )
    _age = _age.with_columns(
        (pl.col("y_rushing_yards") / pl.col("y_carries")).alias("yards_per_carry")
    )
    mo.vstack(
        [
            mo.md(
                "## 6. Explore opportunity, efficiency, age and missingness\n\nThese are descriptive full-history views, not feature-selection evidence. Correlation does not prove stability for every player. Efficiency correlations require positive denominators in both periods; that conditions on survival/usage. The age chart also contains survivor and cohort effects. The model estimates its quadratic age curve from fitting years only and compares it with no age adjustment; no peak age is imposed."
            ),
            mo.ui.table(_stability, selection=None),
            mo.ui.plotly(
                px.scatter(
                    _age,
                    x="age_bin",
                    y="yards_per_carry",
                    size="n",
                    hover_data=["n"],
                    title="Observed exposure-weighted rushing efficiency by age",
                )
            ),
            mo.ui.table(
                analysis_panel.group_by("year")
                .agg(
                    pl.len().alias("candidate_origins"),
                    (pl.col("y_points") == 0).mean().alias("zero_production_fraction"),
                    pl.col("age").is_null().mean().alias("missing_age_fraction"),
                )
                .sort("year"),
                selection=None,
            ),
            mo.accordion(
                {"Unknown target rows excluded, by origin": mo.ui.table(exclusions, selection=None)}
            ),
        ]
    )
    return


@app.cell
def _(candidate_features, mo):
    feature_choice = mo.ui.dropdown(
        candidate_features, value="history_carries", label="Inspect a candidate predictor"
    )
    feature_choice
    return (feature_choice,)


@app.cell
def _(analysis_panel, candidate_features, feature_choice, mo, pl, px):
    _coverage = pl.DataFrame(
        {
            "feature": candidate_features,
            "missing_fraction": [
                analysis_panel[c].null_count() / analysis_panel.height for c in candidate_features
            ],
            "observed_unique_values": [
                analysis_panel[c].drop_nulls().n_unique() for c in candidate_features
            ],
        }
    )
    _f = feature_choice.value
    mo.vstack(
        [
            mo.md(
                f"**{len(candidate_features):,} candidate numeric inputs.** All remain inspectable; each fit drops only columns that are all missing or constant in its own fitting years. History observations are active/source-record means; the explicit `history_* / history_weeks` baseline includes calendar-week zeros. Injury/source absence stays unknown."
            ),
            mo.ui.plotly(
                px.scatter(
                    analysis_panel,
                    x=_f,
                    y="y_points",
                    color="year",
                    hover_name="player_display_name",
                    opacity=0.45,
                    title=f"Descriptive association: {_f} versus future points",
                )
            ),
            mo.accordion(
                {
                    "Full input coverage audit": mo.ui.table(
                        _coverage.sort("missing_fraction", descending=True), selection=None
                    )
                }
            ),
            mo.download(
                _coverage.write_csv(),
                filename="raw_candidate_feature_audit.csv",
                label="Download feature coverage",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(json, mo, result, session):
    _config = session["config"] if session is not None else result["manifest"]["config"]
    mo.vstack(
        [
            mo.md(r"""
        ## 7. What each model is testing

        - **Prior rate:** past points per calendar week, with conditional calibration-year residual draws.
        - **Raw all:** boosted trees over all eligible raw-history numeric summaries. Calibration residuals are sampled within predicted-mean bins; these are empirical predictive distributions, not a tree confidence interval.
        - **Without play detail / opportunity only:** explicit feature ablations against the full raw-input model.
        - **Decomposed:** separate trees for team plays, run fraction, targeted-pass fraction and player shares. Teams have shared volume draws; carries/targets are multinomial allocations with an unmodeled-player remainder. Receptions cannot exceed targets and TDs cannot exceed carries plus receptions.
        - **Unshrunk / shrunk / age:** raw per-touch rates versus validation-selected pseudo-exposure shrinkage toward fitting-data league means, optionally a learned quadratic age curve. This is frequentist regularization, with no prior/posterior inference. Team and player component residuals supply variation; one horizon-level efficiency draw retains persistent risk across the period. This remains a simplified injury/trade/role-change process, not an explicit injury model.
        - **TD squared error / Poisson:** compare tree losses with identical candidate inputs. Each predicted mean is scored under a Poisson PMF and an NB2 PMF (`variance = mean + alpha × mean²`). Alpha is maximum-likelihood calibrated on Y−1; it can reduce to Poisson. Prior-rate TD distributions are included too.

        Decomposed CRPS scores the actual simulator. Its unknown mixture density is never fabricated from a finite histogram. For IID Monte Carlo samples, the primary score is **fair CRPS**:

        \[
        \widehat{CRPS}=\frac{1}{M}\sum_i |x_i-y|-\frac{1}{2M(M-1)}\sum_{i\ne j}|x_i-x_j|.
        \]

        Empirical-CDF CRPS is also recorded; its second denominator is `2M²`. Fair CRPS can be slightly negative from finite simulation noise. Count negative log score is `−log p(y)` in natural-log units. Finite-sample simulation error, source vintage gaps and cohort limitations remain visible; these results do not establish a universal tree-versus-neural-network winner. No neural-network benchmark or causal aging claim is made.

        Computation is editable in `raw_distribution_data.py` and `raw_distribution_models.py` beside this notebook. The offline runner saves raw provenance hashes, configuration, software versions, individual forecasts, simulation draws and source/target audits. A new run never overwrites an existing run directory.

        References: [Grinsztajn et al., NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html),
        [Gneiting & Raftery, proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf),
        [fair ensemble CRPS](https://scoringrules.readthedocs.io/en/latest/crps_estimators.html),
        [LightGBM objectives](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
        """),
            mo.accordion({"Exact settings for these results": mo.json(_config)}),
            mo.download(
                json.dumps(_config, indent=2),
                filename="distribution_experiment_config.json",
                label="Download configuration",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
