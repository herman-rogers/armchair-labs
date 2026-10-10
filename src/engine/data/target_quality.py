"""Canonical target-count quarantine; preserve observations and candidate years."""

import polars as pl

KEYS = ["player_id", "season", "week"]

TARGET_FIELDS = {"targets", "target_share", "air_yards_share", "wopr"}


def mask_columns(frame, condition, names):
    return frame.with_columns(
        pl.when(condition).then(None).otherwise(pl.col(c)).alias(c)
        for c in names
        if c in frame.columns
    )


def correct_tables(tables):
    """Unknown an entire source-year target feed when it contradicts receptions."""
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
    for name in ["nfl_player_weeks", "nfl_player_weeks_quarantine", "nfl_player_seasons"]:
        if name not in tables:
            continue
        old = tables[name]
        frame = mask_columns(
            old,
            pl.col("season").is_in(bad) | (pl.col("receptions") > pl.col("targets")),
            TARGET_FIELDS,
        )
        affected = {
            c: frame[c].null_count() - old[c].null_count()
            for c in old.columns
            if frame[c].null_count() > old[c].null_count()
        }
        changes.append(dict(table=name, newly_unknown=affected))
        corrected[name] = frame
    return corrected, dict(
        source_seasons=bad,
        isolated_weekly_violations=weekly.filter(
            ~pl.col("season").is_in(bad) & (pl.col("receptions") > pl.col("targets"))
        )
        .select(*KEYS, "position", "receptions", "targets")
        .to_dicts(),
        source_audit=audit.to_dicts(),
        changes=changes,
        rule="Aggregate targets < receptions and >=10 weekly violations: "
        "unknown the entire source-year target feed.",
    )
