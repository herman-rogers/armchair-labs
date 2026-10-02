"""Descriptive evaluation slices; never a model selection or serving decision."""

import polars as pl


def passing_workload_diagnostics(predictions: pl.DataFrame, features: pl.DataFrame) -> dict:
    """Partition matched forecasts using only attempts known at the preseason cutoff.

    Actual zero-yard seasons are outcome context, never used to select workload
    groups. Missing prior attempts remain distinct from observed zero.
    """
    matched = (
        predictions.filter(
            pl.col("complete")
            & pl.col("actual").is_not_null()
            & pl.col("prediction").is_not_null()
            & pl.col("baseline").is_not_null()
        )
        .join(
            features.select(
                "player_id", pl.col("forecast_season").alias("season"), "x_prior_attempts"
            ),
            on=["player_id", "season"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("prediction") - pl.col("actual")).abs().alias("absolute_error"),
            (pl.col("baseline") - pl.col("actual")).abs().alias("baseline_absolute_error"),
        )
    )
    groups = []
    for label, condition in (
        ("300+ prior-season pass attempts", pl.col("x_prior_attempts") >= 300),
        ("1–299 prior-season pass attempts", pl.col("x_prior_attempts").is_between(1, 299)),
        ("0 prior-season pass attempts", pl.col("x_prior_attempts") == 0),
        ("Prior attempts unknown", pl.col("x_prior_attempts").is_null()),
    ):
        subset = matched.filter(condition)
        if subset.is_empty():
            continue
        annual = subset.group_by("season").agg(
            pl.col("absolute_error").mean(), pl.col("baseline_absolute_error").mean()
        )
        error = annual["absolute_error"].mean()
        baseline_error = annual["baseline_absolute_error"].mean()
        groups.append(
            dict(
                label=label,
                n=subset.height,
                years=annual.height,
                error=error,
                baseline_error=baseline_error,
                improvement=baseline_error - error,
            )
        )
    return dict(
        n=matched.height,
        zero_outcomes=matched.filter(pl.col("actual") == 0).height,
        workload_groups=groups,
        note=(
            "Groups use prior-season attempts known at the forecast cutoff, not future "
            "starts, injuries or production. They describe prior workload, not confirmed "
            "starting jobs. Errors average seasons equally within each group; group "
            "averages need not recombine to the overall error."
        ),
    )
