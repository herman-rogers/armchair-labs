"""Season-level NFL Next Gen Stats for descriptive player profiles."""

import polars as pl

from patron.metrics.player_profile import unique

TRACKING_METRICS = {
    "ngs_cpoe": {
        "label": "Completion percentage over expected",
        "unit": "percentage points",
        "positions": ["QB"],
        "definition": "NFL Next Gen Stats season completion percentage minus expected completion "
        "percentage. Positive values mean more completions than expected.",
    },
    "ngs_separation": {
        "label": "Average separation",
        "unit": "yards",
        "positions": ["WR", "TE"],
        "definition": "NFL Next Gen Stats season average distance from the nearest defender "
        "at the catch or incompletion. This describes receiving context, not isolated talent.",
    },
    "ngs_yac_oe": {
        "label": "Yards after catch over expected",
        "unit": "yards per reception",
        "positions": ["WR", "TE"],
        "definition": "NFL Next Gen Stats season average yards after catch minus expected "
        "yards after catch per reception.",
    },
    "ngs_ryoe_per_att": {
        "label": "Rushing yards over expected",
        "unit": "yards per carry",
        "positions": ["RB"],
        "definition": "NFL Next Gen Stats season rushing yards over expected per attempt. "
        "Positive values mean more rushing yards than expected.",
    },
}
TRACKING_SCHEMA: dict[str, pl.DataType] = {
    "player_id": pl.String(),
    "season": pl.Int32(),
    "position": pl.String(),
    **dict.fromkeys(TRACKING_METRICS, pl.Float64()),
}
TRACKING_NOTE = (
    "NFL Next Gen Stats via nflverse, using the provider's regular-season summaries. "
    "Captured coverage spans 2016–2025 and varies by player and measure. "
    "Tracking sample sizes are not retained in these gold summaries. Missing is unknown. "
    "Only completed seasons at the selected cutoff are shown; current-season tracking "
    "is not captured. These are descriptive observations, not a new forecast."
)


def tracking_seasons(seasons: pl.DataFrame) -> pl.DataFrame:
    """Copy the provider's season aggregates without averaging ratios or imputing gaps."""
    frame = (
        seasons.select(*TRACKING_SCHEMA)
        .cast(pl.Schema(TRACKING_SCHEMA))
        .sort("player_id", "season")
    )
    unique(frame, ["player_id", "season"])
    if frame.select(pl.any_horizontal(pl.col("player_id", "season").is_null()).any()).item():
        raise ValueError("Tracking summaries have missing player-season keys")
    for metric in TRACKING_METRICS:
        if frame.filter(pl.col(metric).is_not_null() & ~pl.col(metric).is_finite()).height:
            raise ValueError(f"Nonfinite tracking metric: {metric}")
    return frame
