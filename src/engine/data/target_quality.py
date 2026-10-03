"""Canonical target-count quarantine; preserve observations and candidate years."""

import polars as pl

from engine.metrics.player_profile import KEYS

TARGET_FIELDS = {
    "targets",
    "target_share",
    "air_yards_share",
    "wopr",
    "receiving_first_down_rate",
    "targets_per_route_opportunity",
}
TARGET_SERIES = {
    "opportunities",
    "target_share",
    "air_yards_share",
    "targets_per_route",
    "receiving_epa_rate",
}


def mask_columns(frame, condition, names):
    return frame.with_columns(
        pl.when(condition).then(None).otherwise(pl.col(c)).alias(c)
        for c in names
        if c in frame.columns
    )


def correct_tables(tables):
    """Propagate a source-year incident without deleting candidates or their outcomes."""
    weekly = tables["nfl_player_weeks"]
    audit = (
        weekly.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .group_by("season")
        .agg(
            pl.col("targets").sum().alias("targets"),
            pl.col("receptions").sum().alias("receptions"),
            (pl.col("receptions") > pl.col("targets")).sum().alias("violations"),
        )
        .sort("season")
    )
    bad = audit.filter((pl.col("targets") < pl.col("receptions")) & (pl.col("violations") >= 10))[
        "season"
    ].to_list()
    corrected = dict(tables)
    changes = []

    def save(name, frame):
        old = tables[name]
        affected = {
            c: frame[c].null_count() - old[c].null_count()
            for c in old.columns
            if frame[c].null_count() > old[c].null_count()
        }
        changes.append(dict(table=name, newly_unknown=affected))
        corrected[name] = frame

    for name in ["nfl_player_weeks", "nfl_player_weeks_quarantine", "nfl_player_seasons"]:
        if name in tables:
            save(
                name,
                mask_columns(
                    tables[name],
                    pl.col("season").is_in(bad) | (pl.col("receptions") > pl.col("targets")),
                    TARGET_FIELDS,
                ),
            )
    usage = tables["nfl_weekly_usage"]
    save("nfl_weekly_usage", mask_columns(usage, pl.col("season").is_in(bad), TARGET_SERIES))
    features = tables["preseason_features"]
    features = mask_columns(
        features, pl.col("source_season").is_in(bad), TARGET_FIELDS | {"source_opportunities_pg"}
    )
    # Team vacancy uses the previous TEAM season, including for rookies with no source season.
    features = mask_columns(
        features,
        (pl.col("forecast_season") - 1).is_in(bad),
        ["team_vacated_target_share", "known_vacated_target_share"],
    )
    for lag in range(3):
        names = [
            c
            for c in features.columns
            if any(c.startswith(f"rich_s{lag}_{signal}_") for signal in TARGET_SERIES)
        ]
        features = mask_columns(features, (pl.col("forecast_season") - 1 - lag).is_in(bad), names)
    names = [
        c
        for c in features.columns
        if any(c.startswith(f"rich_{signal}_") for signal in TARGET_SERIES)
    ]
    features = mask_columns(
        features,
        pl.any_horizontal((pl.col("forecast_season") - 1 - lag).is_in(bad) for lag in range(3)),
        names,
    )
    first_bad = (
        weekly.filter(pl.col("season").is_in(bad))
        .group_by("player_id")
        .agg(pl.col("season").min().alias("_first_bad_target_year"))
    )
    features = features.join(first_bad, on="player_id", how="left", validate="m:1")
    features = mask_columns(
        features,
        pl.col("forecast_season") > pl.col("_first_bad_target_year"),
        ["career_opportunities"],
    ).drop("_first_bad_target_year")
    save("preseason_features", features)
    return corrected, dict(
        source_seasons=bad,
        isolated_weekly_violations=weekly.filter(
            ~pl.col("season").is_in(bad) & (pl.col("receptions") > pl.col("targets"))
        )
        .select(*KEYS, "position", "receptions", "targets")
        .to_dicts(),
        source_audit=audit.to_dicts(),
        changes=changes,
        rule="Aggregate targets < receptions and >=10 weekly violations; "
        "unknown entire source-year target feed and transitive derivatives.",
    )
