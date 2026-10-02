"""Tests for temporal safety, unknown values, and exact conditional ridge estimates."""

import numpy as np
import polars as pl
import pytest
from individual_stat_data import (
    complete_summary,
    correct_tables,
    dispersion_features,
    scoring_distribution,
)
from individual_stat_engine import (
    Context,
    bh_adjust,
    deduplicate,
    point_prediction,
    prepare,
    supported_years,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize("seed", [12, 83, 209])
def test_exact_block_additions_and_removals_match_explicit_refits(seed):
    rng = np.random.default_rng(seed)
    train, test = rng.normal(size=(80, 6)), rng.normal(size=(15, 6))
    train[:, 1] = train[:, 0] + 0.0001 * train[:, 1]
    train[:20, 2] = np.nan
    train[:, 4] = np.nan
    train[:, 5] = 1
    test[:5, 2] = np.nan
    target = rng.normal(size=80) + np.nan_to_num(train[:, 2])
    names = list("abcdef")
    x, xt, blocks = prepare(train, test, names)

    def explicit(keys):
        indices = [names.index(k) for k in keys]
        return (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(),
                Ridge(alpha=100),
            )
            .fit(train[:, indices], target)
            .predict(test[:, indices])
        )

    base = Context(x, xt, target, blocks, ["a", "b", "f"])
    base.prepare_additions(x, xt)
    for name in ["c", "d", "e"]:
        np.testing.assert_allclose(
            base.add(blocks[name]), explicit(["a", "b", "f", name]), atol=1e-10
        )
    full = Context(x, xt, target, blocks, names)
    for name in names:
        np.testing.assert_allclose(
            full.remove(name), explicit([n for n in names if n != name]), atol=1e-10
        )


def test_distribution_distinguishes_two_big_weeks_and_respects_unknowns():
    a = scoring_distribution([40, 40] + [4] * 8)
    assert a["mean"] == 11.2
    assert a["median"] == a["mean_without_top2"] == 4
    assert a["top2_positive_share"] == pytest.approx(80 / 112)
    assert scoring_distribution([None])["mean"] is None
    assert scoring_distribution([0, 0])["mean"] == 0
    assert scoring_distribution([1, 2])["mean_without_top2"] is None
    assert scoring_distribution([-4, 0, 2])["top2_positive_share"] == 1
    assert scoring_distribution([-4, 0, 2])["cv"] is None


def test_distribution_keeps_full_career_but_excludes_forecast_year():
    weeks = [
        dict(player_id="a", season=y, stat_recorded=True, league_points=p)
        for y, p in [(2001, 1), (2018, 4), (2019, 1000)]
    ]
    candidate = dict(player_id="a", forecast_season=2019)
    result = dispersion_features(weeks, [candidate])[0]
    assert result["distribution_career_observations"] == 2
    assert result["distribution_career_mean"] == 2.5
    assert result["distribution_prior_mean"] == 4
    weeks[-1]["league_points"] = -99999
    assert dispersion_features(weeks, [candidate])[0] == result


def test_partial_career_totals_are_not_complete_exposure():
    rows = [
        dict(stat_recorded=True, targets=10, league_points=4),
        dict(stat_recorded=True, targets=None, league_points=3),
    ]
    result = complete_summary(rows, "career")
    assert result["career_targets"] is None
    assert result["career_targets_per_week"] is None
    assert result["career_league_points"] == 7


def test_bad_source_year_propagates_to_lags_cross_year_and_career():
    weeks = pl.DataFrame(
        [
            dict(player_id=str(i), season=2003, week=1, position="WR", receptions=2, targets=0)
            for i in range(10)
        ]
    )
    features = pl.DataFrame(
        [
            dict(
                player_id="0",
                forecast_season=y,
                source_season=y - 1,
                targets=0,
                receptions=2,
                source_opportunities_pg=2,
                career_opportunities=10,
                rich_s0_targets_per_route_mean=0.5,
                rich_s1_targets_per_route_mean=0.5,
                rich_s2_targets_per_route_mean=0.5,
                rich_targets_per_route_three_year_trend=0.1,
            )
            for y in [2004, 2005, 2006, 2007]
        ]
    )
    tables = dict(
        nfl_player_weeks=weeks,
        nfl_player_seasons=weeks,
        preseason_features=features,
        nfl_weekly_usage=weeks.with_columns(pl.lit(1.0).alias("opportunities")),
    )
    fixed, report = correct_tables(tables)
    assert report["source_seasons"] == [2003]
    f = fixed["preseason_features"]
    for i in range(3):
        assert f[f"rich_s{i}_targets_per_route_mean"][i] is None
        assert f["rich_targets_per_route_three_year_trend"][i] is None
    assert f["rich_targets_per_route_three_year_trend"][3] == 0.1
    assert f["career_opportunities"].null_count() == 4
    assert fixed["nfl_weekly_usage"]["opportunities"].null_count() == 10


def test_coverage_is_measured_on_training_years_not_row_count():
    years = np.repeat([2004, 2005, 2006], 11)
    x = np.ones(33)
    x[:2] = np.nan
    assert supported_years(x, years) == 2


def test_dedup_and_scaling_never_use_test_values():
    train = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    test = np.array([[200, 999]], dtype=float)
    x, xt, blocks = prepare(train, test, ["a", "b"])
    assert deduplicate(x, ["a", "b"], blocks) == ["a"]
    assert abs(x.mean()) < 1e-10
    assert xt[0, 0] > 100


def test_point_conversion_and_multiple_testing():
    np.testing.assert_equal(
        point_prediction(
            np.array([-1.0, 10, 10]), np.array([2019, 2021, 2021]), np.array([16, 17, 0])
        ),
        [0, 170, 0],
    )
    np.testing.assert_allclose(bh_adjust([0.01, 0.04, 0.2]), [0.03, 0.06, 0.2])


def test_constant_decimal_column_is_not_a_spurious_predictor():
    # Floating summation can give std(0.1,0.1,...) > 0 at machine precision.
    raw = np.full((99, 1), 0.1)
    x, xt, blocks = prepare(raw, np.array([[400.0]]), ["constant"])
    assert blocks["constant"] == []
    assert x.shape == (99, 0)
    model = Context(x, xt, np.arange(99.0), blocks, ["constant"])
    assert model.prediction[0] == 49


def test_target_incident_masks_team_vacancy_even_for_rookies():
    weeks = pl.DataFrame(
        [
            dict(player_id=str(i), season=2003, week=1, position="WR", receptions=1, targets=0)
            for i in range(10)
        ]
    )
    features = pl.DataFrame(
        [
            dict(
                player_id="rookie",
                forecast_season=2004,
                source_season=None,
                known_vacated_target_share=0.0,
                team_vacated_target_share=0.0,
            )
        ],
        schema_overrides={"source_season": pl.Int32},
    )
    fixed, _ = correct_tables(
        dict(
            nfl_player_weeks=weeks,
            nfl_player_seasons=weeks,
            nfl_weekly_usage=weeks,
            preseason_features=features,
        )
    )
    assert fixed["preseason_features"]["known_vacated_target_share"][0] is None
    assert fixed["preseason_features"]["team_vacated_target_share"][0] is None


def test_market_aliases_and_review_cohort_are_not_own_data():
    from individual_stat_registry import METADATA, family, is_market

    assert "review_cohort" in METADATA
    for name in ["log_ecr", "inverse_ecr", "market_ecr", "market_position_log_rank"]:
        assert is_market(name)
        assert family(name, {}) == "market"
    assert not is_market("career_recent3_weighted_ppg")
