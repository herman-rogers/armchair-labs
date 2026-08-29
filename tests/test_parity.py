"""Parity between the component scorer and nflverse's precomputed PPR total.

The pipeline takes base points from `fantasy_points_ppr` because the draft-prep work
observed that this league's base scoring matches nflverse's PPR exactly. That
observation is load-bearing — it is the difference between the board being right and
the board being plausibly wrong — and until now it was never tested.

This asserts it on real data. The day the two paths diverge is the day a league
setting drifted or nflverse changed a definition, and either way the right outcome is
a failing test rather than a quietly wrong board.

Network-marked; run with `just test-all`.
"""

from __future__ import annotations

import polars as pl
import pytest

from patron.data import nflverse
from patron.scoring.engine import (
    BASE_POINTS,
    COMPONENT_POINTS,
    base_points_from_nflverse,
    score_components,
)

pytestmark = pytest.mark.network

SEASON = 2025
TOLERANCE = 0.02


@pytest.fixture(scope="module")
def scored() -> pl.DataFrame:
    weeks = nflverse.load_player_weeks([SEASON]).filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    )
    return (
        weeks.pipe(score_components)
        .pipe(base_points_from_nflverse)
        .with_columns((pl.col(COMPONENT_POINTS) - pl.col(BASE_POINTS)).alias("delta"))
    )


class TestParity:
    def test_the_two_scoring_paths_agree(self, scored: pl.DataFrame) -> None:
        disagreements = scored.filter(pl.col("delta").abs() > TOLERANCE)

        if disagreements.height:
            worst = (
                disagreements.with_columns(pl.col("delta").abs().alias("abs_delta"))
                .sort("abs_delta", descending=True)
                .head(10)
                .select(
                    "player_display_name",
                    "position",
                    "week",
                    COMPONENT_POINTS,
                    BASE_POINTS,
                    "delta",
                )
            )
            pytest.fail(
                f"{disagreements.height} of {scored.height} player-weeks disagree by "
                f"more than {TOLERANCE}. Either a league setting drifted or nflverse "
                f"changed a definition.\n{worst}"
            )

    def test_the_sample_is_actually_large(self, scored: pl.DataFrame) -> None:
        """Guard against the test passing because it scored nothing."""
        assert scored.height > 5_000

    def test_season_totals_agree(self, scored: pl.DataFrame) -> None:
        totals = scored.select(
            pl.col(COMPONENT_POINTS).sum().alias("components"),
            pl.col(BASE_POINTS).sum().alias("nflverse"),
        ).to_dicts()[0]

        assert totals["components"] == pytest.approx(totals["nflverse"], rel=1e-4)
