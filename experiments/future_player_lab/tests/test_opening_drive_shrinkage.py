"""Post-hoc diagnostic weights still obey their declared chronological rule."""

import polars as pl
import pytest
from experiments.future_player_lab.campaigns.shrinkage import select_alpha


def history():
    return pl.DataFrame(
        [
            {
                "year": year,
                "high_prior_usage": high,
                "actual": 5.0 if high else 10.0,
                "raw": 5.0,
                "affine": 10.0,
            }
            for year in (2018, 2019, 2020)
            for high in (False, True)
        ]
    )


@pytest.mark.parametrize("cohort,expected", [(None, 0.5), (False, 0.0), (True, 1.0)])
def test_alpha_selection_cannot_use_current_or_future_outcomes(cohort, expected):
    past = history()
    future = pl.DataFrame(
        [
            {
                "year": year,
                "high_prior_usage": high,
                "actual": -999999.0,
                "raw": 999999.0,
                "affine": -999999.0,
            }
            for year in (2021, 2022)
            for high in (False, True)
        ]
    )
    assert select_alpha(past, 2021, cohort)[0] == expected
    assert select_alpha(pl.concat([past, future]), 2021, cohort) == select_alpha(past, 2021, cohort)


def test_empty_historical_cohort_cannot_borrow_current_rows():
    frame = history().filter(~((pl.col("year") == 2020) & pl.col("high_prior_usage")))
    with pytest.raises(ValueError, match="Incomplete"):
        select_alpha(frame, 2021, True)


def test_equal_validation_losses_prefer_affine_weight():
    frame = history().with_columns(pl.lit(10.0).alias("raw"))
    assert select_alpha(frame, 2021)[0] == 0.0
