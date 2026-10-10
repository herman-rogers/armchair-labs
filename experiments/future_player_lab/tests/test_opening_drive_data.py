"""Independent cutoff and scoring checks for the multi-position campaign."""

from datetime import date

import numpy as np
import polars as pl
import pytest
from experiments.future_player_lab.campaigns import data
from experiments.future_player_lab.notebooks import raw_distribution_data as raw


@pytest.fixture
def campaign_tables(monkeypatch):
    def build(position):
        stats = set(raw.STATS) | {
            "attempts",
            "completions",
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
        }
        box = pl.DataFrame(
            {
                "player_id": ["returner", "returner", "returner", "future_only"],
                "season": [2024, 2025, 2025, 2025],
                "week": [1, 1, 3, 3],
                "team": ["ATL"] * 4,
                "position": [position] * 4,
                "player_display_name": ["Returner"] * 3 + ["Future player"],
                **{s: [2.0, 3.0, 50.0, 1000.0] for s in stats},
            }
        )
        player = raw.prefixed(box, raw.KEY, "box").with_columns(
            pl.lit(9.0).alias("pbp_pass__participation_pressure")
        )
        return {
            "box": box,
            "player": player,
            "team": pl.DataFrame(
                {
                    "team": ["ATL"] * 3,
                    "season": [2024, 2025, 2025],
                    "week": [1, 1, 3],
                    "attempts": [20.0, 30.0, 1000.0],
                }
            ),
            "schedule": pl.DataFrame(
                {
                    "season": [2024, 2025, 2025, 2025],
                    "week": [18, 1, 3, 18],
                    "home_team": ["ATL"] * 4,
                    "away_team": ["CAR"] * 4,
                    "home_score": [0.0] * 4,
                    "away_score": [0.0] * 4,
                }
            ),
            "combine": pl.DataFrame(
                {
                    "pfr_id": ["rookie"],
                    "draft_year": [2025],
                    "pos": [position],
                    "draft_team": ["ATL"],
                    "player_name": ["Rookie"],
                    "season": [2025],
                    "forty": [4.5],
                }
            ),
            "mapping": pl.DataFrame({"pfr_id": ["rookie"], "player_id": ["rookie"]}),
            "identity": pl.DataFrame(
                {
                    "gsis_id": ["returner", "rookie"],
                    "birth_date": [date(2000, 1, 1), date(2003, 1, 1)],
                }
            ),
            "college": pl.DataFrame(
                {
                    "player_id": ["rookie", "rookie"],
                    "season": [2024, 2025],
                    "yards": [100.0, 99999.0],
                }
            ),
        }

    monkeypatch.setattr(data, "draft_sources", lambda: (None, None, None))
    monkeypatch.setattr(
        data,
        "draft_features",
        lambda panel, *_: panel.with_columns(
            [pl.lit(None, pl.Float64).alias(c) for c in data.ADDED]
        ),
    )
    return build


@pytest.mark.parametrize("position", ["RB", "WR", "QB", "TE"])
@pytest.mark.parametrize("origin", [0, 2])
def test_future_stats_players_and_scores_do_not_change_features(campaign_tables, position, origin):
    tables = campaign_tables(position)
    expected = data.feature_rows(tables, position, 2025, origin)
    changed = dict(tables)
    for name in ("box", "player", "team"):
        frame = tables[name]
        changed[name] = frame.with_columns(
            [
                pl.when((pl.col("season") == 2025) & (pl.col("week") > origin))
                .then(999999.0)
                .otherwise(pl.col(c))
                .alias(c)
                for c in raw.numeric(frame, ["player_id"])
            ]
        )
    changed["schedule"] = tables["schedule"].with_columns(
        pl.lit(None, pl.Float64).alias("home_score"),
        pl.lit(None, pl.Float64).alias("away_score"),
    )
    actual = data.feature_rows(changed, position, 2025, origin)
    assert actual.equals(expected)
    assert set(actual["player_id"]) == {"returner", "rookie"}
    assert not any(c.startswith("y_") for c in actual.columns)
    assert actual.filter(pl.col("player_id") == "rookie")["college__yards"][0] == 100.0
    row = actual.filter(pl.col("player_id") == "returner").row(0, named=True)
    assert row["history_weeks"] == (18 if origin == 0 else 2)
    assert row["history_carries"] == (2 if origin == 0 else 3)
    assert row["prior__pbp_pass__participation_pressure"] == 9
    assert row["current__pbp_pass__participation_pressure"] is None
    if position == "QB":
        assert row["history_passing_yards"] == (2 if origin == 0 else 3)


@pytest.mark.parametrize("position", ["RB", "WR", "QB", "TE"])
def test_recorded_unknown_is_not_zero_but_absence_is_zero(campaign_tables, position):
    tables = campaign_tables(position)
    stat = "passing_yards" if position == "QB" else "receiving_yards"
    tables["box"] = tables["box"].with_columns(
        pl.when((pl.col("player_id") == "returner") & (pl.col("week") == 3))
        .then(None)
        .otherwise(pl.col(stat))
        .alias(stat)
    )
    features = data.feature_rows(tables, position, 2025, 2)
    labeled = data.attach_targets(features, tables, position, 2025, 2)
    returner = labeled.filter(pl.col("player_id") == "returner").row(0, named=True)
    rookie = labeled.filter(pl.col("player_id") == "rookie").row(0, named=True)
    assert returner[f"y_{stat}"] is None
    assert returner["y_points"] is None
    assert returner["y_carries"] == 50
    assert rookie[f"y_{stat}"] == rookie["y_points"] == 0
    assert set(labeled["player_id"]) == {"returner", "rookie"}
    assert not any(c.startswith("y_") for c in data.feature_names(labeled))


def test_incomplete_schedule_cannot_be_treated_as_zero_labels(campaign_tables):
    tables = campaign_tables("WR")
    panel = data.feature_rows(tables, "WR", 2025, 2)
    tables["schedule"] = tables["schedule"].with_columns(
        pl.when(pl.col("week") == 18).then(None).otherwise(pl.col("home_score")).alias("home_score")
    )
    with pytest.raises(ValueError, match="incomplete"):
        data.attach_targets(panel, tables, "WR", 2025, 2)


def test_position_scoring_is_explicit_and_unknowns_propagate():
    qb = dict(
        passing_yards=250.0,
        passing_tds=2.0,
        passing_interceptions=1.0,
        rushing_yards=30.0,
        rushing_tds=1.0,
    )
    assert data.points("QB", qb) == 25.0
    qb["passing_yards"] = np.nan
    assert np.isnan(data.points("QB", qb))
    receiver = dict(
        rushing_yards=10.0,
        receiving_yards=100.0,
        receptions=5.0,
        rushing_tds=0.0,
        receiving_tds=1.0,
    )
    assert (
        data.points("WR", receiver)
        == data.points("RB", receiver)
        == data.points("TE", receiver)
        == 22.0
    )


def test_matrix_rejects_outcomes_identifiers_and_missing_definitions():
    frame = pl.DataFrame(
        {"age": [22.0, None], "y_points": [100.0, 200.0], "market_implied": [30.0, 40.0]}
    )
    assert np.isnan(data.matrix(frame, ["age"])[1, 0])
    for forbidden in ("y_points", "market_implied"):
        with pytest.raises(ValueError, match="Outcomes|future"):
            data.matrix(frame, [forbidden])
    with pytest.raises(ValueError, match="Missing feature"):
        data.matrix(frame, ["history_attempts"])
