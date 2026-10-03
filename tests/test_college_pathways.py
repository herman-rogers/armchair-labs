from copy import deepcopy

import polars as pl
import pytest

from engine.data.college import normalize_season
from engine.data.college_identity import build_links
from engine.metrics.college_translation import (
    COLLEGE_FEATURES,
    NFL_TARGETS,
    features,
    nfl_backtest,
    nfl_cohort,
)


def college_sources():
    raw = pl.DataFrame(
        [
            {
                "season": 2024,
                "game_id": 1,
                "team_id": 10,
                "athlete_id": 100,
                "athlete_name": "Passer",
                "category": "passing",
                "passingYards": "100",
                "passingTouchdowns": "1",
                "interceptions": "0",
            },
            {
                "season": 2024,
                "game_id": 1,
                "team_id": 10,
                "athlete_id": 101,
                "athlete_name": "Receiver",
                "category": "receiving",
                "receptions": "5",
                "receivingYards": "100",
                "receivingTouchdowns": "1",
            },
        ],
        infer_schema_length=None,
    )
    roster = pl.DataFrame(
        [
            {
                "season": 2024,
                "team_id": 10,
                "athlete_id": pid,
                "athlete_display_name": name,
                "position_abbreviation": "WR",
                "division": "fbs",
                "team_location": "Washington",
                "date_of_birth": "2002-01-01",
            }
            for pid, name in [(101, "Receiver"), (102, "Bench Player")]
        ]
    )
    schedule = pl.DataFrame(
        {
            "season": [2024],
            "game_id": [1],
            "home_id": [10],
            "away_id": [20],
            "completed": [True],
            "start_date": ["2024-10-01"],
        }
    )
    return raw, roster, schedule


def test_college_normalization_keeps_non_pros_and_zero_production_roster_players():
    raw, roster, schedule = college_sources()
    seasons, games, audit = normalize_season(raw, roster, schedule)
    receiver = seasons.filter(pl.col("college_id") == "101").row(0, named=True)
    assert receiver["receiving_yards"] == 100
    assert receiver["share_receiving_yards"] == 1
    assert receiver["complete_team_season"]
    bench = seasons.filter(pl.col("college_id") == "102").row(0, named=True)
    assert bench["receiving_yards"] == 0
    assert bench["observed_stat_games"] == 0  # NOT games played
    assert games.height == 2 and audit["unrostered_stat_stints"] == 1


def test_missing_scheduled_games_are_not_mislabeled_complete():
    raw, roster, schedule = college_sources()
    schedule = pl.concat(
        [schedule, schedule.with_columns(pl.lit(2, dtype=pl.Int64).alias("game_id"))]
    )
    seasons, _, _ = normalize_season(raw, roster, schedule)
    assert seasons["coverage"].to_list() == [0.5] * seasons.height
    assert not seasons["complete_team_season"].any()


def test_team_totals_must_reconcile_even_with_all_games_captured():
    raw, roster, schedule = college_sources()
    raw = raw.with_columns(
        pl.when(pl.col("category") == "receiving")
        .then(pl.lit("80"))
        .otherwise(pl.col("receivingYards"))
        .alias("receivingYards")
    )
    seasons, _, audit = normalize_season(raw, roster, schedule)
    assert audit["unreconciled_team_games"] == 1
    assert seasons["coverage"].min() == 1
    assert not seasons["complete_team_season"].any()


def test_unnamed_stats_are_quarantined_not_imputed_zero():
    raw, roster, schedule = college_sources()
    raw = raw.with_columns(pl.lit(None, dtype=pl.String).alias("passingYards"))
    seasons, _, audit = normalize_season(raw, roster, schedule)
    assert audit["quarantined_team_games"] == 1
    assert not seasons["complete_team_season"].any()
    assert seasons["receiving_yards"].null_count() == seasons.height


def test_conflicting_stats_fail_but_identical_duplicate_records_are_audited():
    raw, roster, schedule = college_sources()
    _, _, audit = normalize_season(pl.concat([raw, raw]), roster, schedule)
    assert audit["identical_rows_removed"] == raw.height
    bad = pl.concat([raw, raw.head(1).with_columns(pl.lit("999").alias("passingYards"))])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_season(bad, roster, schedule)


def test_nonfinite_statistics_and_placeholder_ids_never_become_valid_features():
    raw, roster, schedule = college_sources()
    bad = raw.with_columns(pl.lit("NaN").alias("passingYards"))
    seasons, _, audit = normalize_season(bad, roster, schedule)
    assert audit["quarantined_team_games"] == 1
    assert not seasons["complete_team_season"].any()
    with pytest.raises(ValueError, match="Placeholder"):
        normalize_season(raw.with_columns(pl.lit(0).alias("athlete_id")), roster, schedule)


def identity_inputs():
    college = pl.DataFrame(
        {
            "college_id": ["101", "101", "102"],
            "season": [2023, 2024, 2024],
            "player_name": ["Alpha Player"] * 2 + ["Unknown Player"],
            "college_team": ["Oregon", "Washington", "Washington"],
            "birth_date": ["2002-01-01"] * 3,
        }
    )
    nfl = pl.DataFrame(
        {
            "gsis_id": ["n1"],
            "espn_id": [101],
            "display_name": ["Alpha Player"],
            "rookie_season": [2025],
            "draft_year": [2025],
            "college_name": ["Washington"],
            "birth_date": ["2002-01-01"],
        }
    )
    crosswalk = pl.DataFrame(
        schema={
            "gsis_id": pl.String,
            "espn_id": pl.String,
            "cfbref_id": pl.String,
            "name": pl.String,
            "college": pl.String,
        }
    )
    return college, nfl, crosswalk


def test_stable_provider_id_links_transfers_without_name_only_matching():
    college, nfl, crosswalk = identity_inputs()
    links, registry, audit = build_links(college, nfl, crosswalk)
    linked = links.filter(pl.col("status") == "linked")
    assert linked["college_id"].to_list() == ["101"]
    assert linked["method"][0] == "provider_id_name_chronology"
    assert audit["college_players"] == 2
    assert registry.filter(pl.col("namespace") == "espn_college").height == 2
    assert links.filter(pl.col("college_id") == "102")["player_id"][0] is None


def test_exact_id_dob_or_name_conflict_requires_review():
    college, nfl, crosswalk = identity_inputs()
    for column, value in [("birth_date", "1990-01-01"), ("display_name", "Different Person")]:
        links, _, _ = build_links(college, nfl.with_columns(pl.lit(value).alias(column)), crosswalk)
        assert links.filter(pl.col("college_id") == "101")["status"][0] == "review"


def test_no_silent_many_to_one_identity_merge():
    college, nfl, crosswalk = identity_inputs()
    duplicate = college.filter(pl.col("college_id") == "101").with_columns(
        pl.lit("999").alias("college_id")
    )
    links, _, _ = build_links(pl.concat([college, duplicate]), nfl, crosswalk)
    assert links.filter(pl.col("status") == "linked").height == 0
    assert links.filter(pl.col("status") == "review").height == 2


def test_name_only_is_not_sufficient_and_overrides_require_evidence():
    college, nfl, crosswalk = identity_inputs()
    nfl = nfl.with_columns(
        pl.lit(999).alias("espn_id"),
        pl.lit(None).alias("birth_date"),
        pl.lit("Unrelated").alias("college_name"),
    )
    links, _, _ = build_links(college, nfl, crosswalk)
    assert links.filter(pl.col("status") == "linked").height == 0
    with pytest.raises(ValueError, match="reason"):
        build_links(college, nfl, crosswalk, [{"college_id": "101", "player_id": "n1"}])


def season_row(year=2024):
    return {
        "college_id": "101",
        "season": year,
        "player_name": "Alpha Player",
        "complete_team_season": True,
        "team_count": 1,
        "teams": ["Washington"],
        "birth_date": "2002-01-01",
        "scheduled_games": 12,
        "passing_yards": 0,
        "rushing_yards": 20,
        "receiving_yards": 500,
        "share_receiving_yards": 0.2,
        "share_rushing_yards": 0.01,
    }


def test_future_college_data_does_not_change_features():
    history = [season_row(2023), season_row(2024)]
    before = features(history, 2025)
    future = {**season_row(2025), "receiving_yards": 1000000}
    assert before == features([*history, future], 2025)
    assert before["latest_receiving_yards"] == 500
    assert not set(COLLEGE_FEATURES) & {"draft_capital", "was_drafted", "draft_pick"}


def test_incomplete_college_history_withholds_forecasts():
    history = [{**season_row(), "complete_team_season": False}]
    assert not features(history, 2025)["eligible"]
    assert not features([season_row(2023)], 2025)["eligible"]
    assert not features([], 2025)["eligible"]


def test_nfl_unlinked_is_unscored_but_verified_nfl_no_production_can_be_zero():
    candidates = pl.DataFrame(
        [
            {
                "player_id": pid,
                "player_display_name": "Player",
                "position": "WR",
                "forecast_season": 2025,
                "rookie_draft_pick": None,
                "outcome_complete": True,
                "actual_season_points": None,
                "actual_games": 0,
            }
            for pid in ["n1", "n2"]
        ]
    )
    links = pl.DataFrame(
        {
            "college_id": ["101"],
            "player_id": ["n1"],
            "status": ["linked"],
            "method": ["provider_id_name_chronology"],
        }
    )
    outcomes = pl.DataFrame(
        schema={"player_id": pl.String, "season": pl.Int64, "league_points": pl.Float64}
    )
    rows = nfl_cohort([season_row()], links, candidates, outcomes, 2025)
    assert rows[0]["eligible"] and rows[0]["nfl_year1_points"] == 0
    assert rows[0]["nfl_first3_points"] is None
    assert not rows[1]["eligible"] and rows[1]["college_ids"] == []


def test_walk_forward_maturity_and_own_outcome_cannot_change_prediction():
    rows = []
    for year in range(2005, 2013):
        for i in range(12):
            rows.append(
                {
                    "player_id": f"{year}-{i}",
                    "player_display_name": "Player",
                    "college_ids": [str(i)],
                    "position": "WR",
                    "eligible": True,
                    "forecast_year": year,
                    "draft_capital": -float(i),
                    "was_drafted": 1,
                    **{c: float(i + 1) for c in COLLEGE_FEATURES},
                    **{t: float(i * 2 + 1) for t in NFL_TARGETS},
                }
            )
    predictions, _ = nfl_backtest(rows)
    assert predictions
    assert all(r["train_latest_outcome_year"] < r["forecast_year"] for r in predictions)
    triples = [r for r in predictions if r["target"] == "nfl_first3_points"]
    assert triples and all(r["train_latest_entry"] <= r["forecast_year"] - 3 for r in triples)
    changed = deepcopy(rows)
    for row in changed:
        if row["forecast_year"] == 2012:
            row["nfl_year1_points"] = 1e9
    after, _ = nfl_backtest(changed)
    before_values = [r["college_plus_draft"] for r in predictions if r["forecast_year"] == 2012]
    after_values = [r["college_plus_draft"] for r in after if r["forecast_year"] == 2012]
    assert before_values == after_values
