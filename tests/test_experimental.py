"""Pure tests for report-only additive feature stacks."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from patron.metrics.experimental import (
    build_combine_features,
    build_contract_features,
    build_forecast_features,
    build_player_background,
    build_rookie_features,
    build_snap_features,
    build_transaction_features,
)


def test_combine_features_build_weight_adjusted_speed_and_burst() -> None:
    combine = pl.DataFrame(
        {
            "pfr_id": ["P"],
            "pos": ["TE"],
            "wt": [250.0],
            "forty": [4.6],
            "vertical": [36.0],
            "broad_jump": [120.0],
            "cone": [7.0],
            "shuttle": [4.2],
        }
    )
    players = pl.DataFrame({"gsis_id": ["p"], "pfr_id": ["P"]})

    row = build_combine_features(combine, players).to_dicts()[0]

    assert row["player_id"] == "p"
    assert row["combine_speed_score"] == pytest.approx(250 * 200 / 4.6**4)
    assert row["combine_burst_score"] == pytest.approx(48.0)
    assert row["combine_agility_score"] == pytest.approx(-11.2)


def test_rookie_features_create_a_distinct_preseason_population() -> None:
    players = pl.DataFrame(
        {
            "gsis_id": ["rookie", "veteran"],
            "display_name": ["Rookie Runner", "Veteran Runner"],
            "position": ["HB", "RB"],
            "birth_date": ["2003-05-01", "1995-05-01"],
            "college_conference": ["SEC", "SEC"],
            "rookie_season": [2026, 2017],
            "draft_year": [2026, 2017],
            "draft_round": [2, 1],
            "draft_pick": [40, None],
        }
    )
    combine = pl.DataFrame(
        {
            "player_id": ["rookie"],
            "combine_weight": [215.0],
            "combine_speed_score": [108.0],
            "combine_burst_score": [47.0],
            "combine_agility_score": [-11.0],
        }
    )

    row = build_rookie_features(players, combine, [2026]).to_dicts()[0]

    assert row["player_id"] == "rookie"
    assert row["position"] == "RB"
    assert row["player_population"] == "rookie"
    assert row["rookie_indicator"] == 1.0
    assert row["rookie_day_two"] == 1.0
    assert row["college_major_conference"] == 1.0
    assert row["rookie_age"] == pytest.approx(23.34, abs=0.02)

    rows = build_rookie_features(players, combine, [2017, 2026])
    undrafted = rows.filter(pl.col("player_id") == "veteran").to_dicts()[0]
    assert undrafted["rookie_was_drafted"] == 0.0
    assert undrafted["rookie_draft_capital_score"] == pytest.approx(-math.log(300.0))


def test_snap_features_join_ids_and_measure_late_role() -> None:
    snaps = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2025] * 4,
            "game_type": ["REG"] * 4,
            "week": [1, 15, 16, 17],
            "pfr_player_id": ["P"] * 4,
            "position": ["WR"] * 4,
            "offense_snaps": [20.0, 40.0, 50.0, 60.0],
            "offense_pct": [0.25, 0.50, 0.75, 1.00],
        }
    )
    players = pl.DataFrame({"gsis_id": ["p"], "pfr_id": ["P"]})

    row = build_snap_features(snaps, players).to_dicts()[0]

    assert row["player_id"] == "p"
    assert row["source_offense_snaps_pg"] == pytest.approx(42.5)
    assert row["source_offense_snap_pct"] == pytest.approx(0.625)
    assert row["late_offense_snap_pct"] == pytest.approx(0.75)
    assert row["offense_snap_pct_trend"] == pytest.approx(0.125)


def test_contract_features_reconstruct_historical_active_term_without_current_status() -> None:
    contracts = pl.DataFrame(
        {
            "gsis_id": ["p", "p"],
            "year_signed": [2020, 2024],
            "years": [4, 3],
            "apy_cap_pct": [0.02, 0.05],
            "guaranteed": [4.0, 20.0],
            "is_active": [False, True],
        }
    )

    rows = build_contract_features(contracts, [2022, 2025]).sort("forecast_season")

    assert rows["contract_apy_cap_pct_proxy"].to_list() == [0.02, 0.05]
    assert rows["contract_years_remaining_proxy"].to_list() == [2.0, 2.0]
    assert rows["contract_guaranteed_log_proxy"][0] == pytest.approx(math.log1p(4.0))
    assert rows["contract_years_remaining"].null_count() == 2
    assert rows["contract_apy_cap_pct"].null_count() == 2


def test_forecast_features_capture_roster_move_and_vacated_role() -> None:
    seasons = pl.DataFrame(
        {
            "player_id": ["mover", "stayer", "other"],
            "season": [2024, 2024, 2024],
            "position": ["WR", "WR", "WR"],
            "team": ["A", "A", "B"],
            "targets": [100.0, 50.0, 20.0],
            "carries": [0.0, 0.0, 0.0],
        }
    )
    rosters = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "game_type": ["REG", "REG", "REG"],
            "team": ["B", "A", "B"],
            "position": ["WR", "WR", "WR"],
            "gsis_id": ["mover", "stayer", "other"],
            "status": ["ACT", "ACT", "ACT"],
            "years_exp": [3, 2, 4],
        }
    )
    depth = pl.DataFrame(
        {
            "dt": ["2025-08-31T00:00:00Z"] * 3,
            "team": ["B", "A", "B"],
            "gsis_id": ["mover", "stayer", "other"],
            "pos_abb": ["WR", "WR", "WR"],
            "pos_name": ["Wide Receiver"] * 3,
            "pos_rank": [2, 1, 1],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["mover", "stayer", "other"],
            "draft_year": [2022, 2023, 2021],
            "draft_round": [1, 2, None],
            "draft_pick": [10, 40, None],
        }
    )

    frame = build_forecast_features(
        rosters,
        seasons,
        [2025],
        "08-31",
        depth_charts=depth,
        player_background=build_player_background(players),
    )
    mover = frame.filter(pl.col("player_id") == "mover").to_dicts()[0]
    stayer = frame.filter(pl.col("player_id") == "stayer").to_dicts()[0]

    assert mover["team_changed"] == 1.0
    assert mover["team_vacated_target_share"] == 0.0
    assert stayer["team_changed"] == 0.0
    assert stayer["team_vacated_target_share"] == pytest.approx(2 / 3)
    assert stayer["vacated_target_opportunity"] == pytest.approx(2 / 3)
    assert stayer["player_experience"] == 2.0
    assert stayer["draft_capital_score"] == pytest.approx(-math.log(40))


def test_forecast_features_prefer_cutoff_transactions_over_week_one_proxy() -> None:
    seasons = pl.DataFrame(
        {
            "player_id": ["released"],
            "season": [2024],
            "position": ["RB"],
            "team": ["BUF"],
            "targets": [10.0],
            "carries": [50.0],
        }
    )
    rosters = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "team": ["MIA"],
            "position": ["RB"],
            "gsis_id": ["released"],
            "status": ["ACT"],
            "years_exp": [3],
        }
    )
    cutoff = pl.DataFrame(
        {
            "forecast_season": [2025],
            "player_id": ["released"],
            "cutoff_preseason_team": ["BUF"],
            "cutoff_preseason_status": ["off"],
            "cutoff_preseason_status_score": [0.0],
            "cutoff_preseason_rostered": [0.0],
            "cutoff_preseason_reserve": [0.0],
        }
    )
    background = build_player_background(
        pl.DataFrame(
            {
                "gsis_id": ["released"],
                "draft_year": [2022],
                "draft_round": [3],
                "draft_pick": [90],
            }
        )
    )

    row = build_forecast_features(
        rosters,
        seasons,
        [2025],
        "08-31",
        player_background=background,
        transaction_features=cutoff,
    ).to_dicts()[0]

    assert row["cutoff_preseason_status"] == "off"
    assert row["preseason_status_score"] == 0.0
    assert row["preseason_rostered"] == 0.0
    assert row["week1_proxy_status"] == "ACT"
    assert row["week1_proxy_rostered"] == 1.0
    assert row["team_changed"] == 0.0


def test_dated_transactions_reconstruct_cutoff_roster_state() -> None:
    player_seasons = pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "player_id": ["cut", "move", "pup"],
            "player_display_name": ["Cut Player", "Move Player", "Pup Player"],
            "position": ["RB", "WR", "TE"],
            "team": ["BUF", "TEN", "MIA"],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["cut", "move", "pup"],
            "display_name": ["Cut Player", "Move Player", "Pup Player"],
        }
    )
    transactions = pl.DataFrame(
        {
            "transaction_date": [date(2024, 8, 27), date(2024, 8, 29), date(2024, 8, 28)],
            "transaction_year": [2024, 2024, 2024],
            "category": ["waivers", "trades", "reserve-list"],
            "from_team": ["BUF", "TEN", "MIA"],
            "to_team": [None, "LAC", "MIA"],
            "player_name": ["Cut Player", "Move Player", "Pup Player"],
            "description": [
                "Terminated Via Waivers, all contracts",
                "Traded",
                "Reserve/Physically Unable to Perform",
            ],
        }
    )

    cutoff = build_transaction_features(transactions, player_seasons, players, [2024], "08-31")
    by_id = {row["player_id"]: row for row in cutoff.to_dicts()}

    assert by_id["cut"]["cutoff_preseason_rostered"] == 0.0
    assert by_id["cut"]["cutoff_preseason_team"] is None
    assert by_id["move"]["cutoff_preseason_team"] == "LAC"
    assert by_id["move"]["cutoff_preseason_status"] == "active"
    assert by_id["pup"]["cutoff_preseason_reserve"] == 1.0
    assert all(row["cutoff_transaction_matched"] == 1.0 for row in by_id.values())


def test_market_snapshot_date_controls_transactions_and_preserves_reserve_kind() -> None:
    player_seasons = pl.DataFrame(
        {
            "season": [2020, 2020],
            "player_id": ["known", "too-late"],
            "player_display_name": ["Known Pup", "Late Ir"],
            "position": ["RB", "WR"],
            "team": ["BUF", "BAL"],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["known", "too-late"],
            "display_name": ["Known Pup", "Late Ir"],
        }
    )
    transactions = pl.DataFrame(
        {
            "transaction_date": [date(2021, 8, 27), date(2021, 8, 28)],
            "transaction_year": [2021, 2021],
            "category": ["reserve-list", "reserve-list"],
            "from_team": ["BUF", "BAL"],
            "to_team": ["BUF", "BAL"],
            "player_name": ["Known Pup", "Late Ir"],
            "description": [
                "Reserve/Physically Unable to Perform",
                "Placed on Reserve/Injured",
            ],
        }
    )

    rows = build_transaction_features(
        transactions,
        player_seasons,
        players,
        [2021],
        "08-31",
        cutoff_by_season={2021: date(2021, 8, 27)},
    )
    by_id = {row["player_id"]: row for row in rows.to_dicts()}

    assert by_id["known"]["forecast_cutoff_date"] == date(2021, 8, 27)
    assert by_id["known"]["cutoff_availability_class"] == "pup_nfi"
    assert by_id["known"]["cutoff_pup_nfi"] == 1.0
    assert by_id["too-late"]["cutoff_availability_class"] == "unknown"
    assert by_id["too-late"]["cutoff_transaction_matched"] == 0.0


def test_team_level_transaction_prose_matches_each_player_clause() -> None:
    player_seasons = pl.DataFrame(
        {
            "season": [2023, 2023],
            "player_id": ["signed", "released"],
            "player_display_name": ["Adoree Jackson", "Jakob Johnson"],
            "position": ["WR", "RB"],
            "team": ["TEN", "LV"],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": ["signed", "released"],
            "display_name": ["Adoree' Jackson", "Jakob Johnson"],
        }
    )
    transactions = pl.DataFrame(
        {
            "transaction_date": [date(2024, 8, 31)],
            "transaction_year": [2024],
            "category": ["espn"],
            "from_team": [None],
            "to_team": ["NYG"],
            "player_name": [None],
            "description": [
                "Signed DB Adoree' Jackson to the active roster. Released FB Jakob Johnson."
            ],
        }
    )

    cutoff = build_transaction_features(transactions, player_seasons, players, [2024], "08-31")
    by_id = {row["player_id"]: row for row in cutoff.to_dicts()}

    assert by_id["signed"]["cutoff_preseason_team"] == "NYG"
    assert by_id["signed"]["cutoff_preseason_rostered"] == 1.0
    assert by_id["released"]["cutoff_preseason_team"] is None
    assert by_id["released"]["cutoff_preseason_rostered"] == 0.0


@pytest.mark.parametrize(
    "description",
    [
        "Signed CB Michael Jackson.",
        "Signed OL Jackson Dennis.",
        "Released QB Lamar Jacksonson.",
        "Signed QB L. Jackson.",
    ],
)
def test_transaction_prose_does_not_match_partial_names(description: str) -> None:
    players = pl.DataFrame(
        {
            "gsis_id": ["lamar"],
            "display_name": ["Lamar Jackson"],
            "short_name": ["L.Jackson"],
        }
    )
    seasons = pl.DataFrame(
        {
            "season": [2019],
            "player_id": ["lamar"],
            "player_display_name": ["Lamar Jackson"],
            "position": ["QB"],
            "team": ["BAL"],
        }
    )
    events = pl.DataFrame(
        {
            "transaction_date": [date(2020, 8, 1)],
            "transaction_year": [2020],
            "category": ["espn"],
            "from_team": [None],
            "to_team": ["NE"],
            "player_name": [None],
            "description": [description],
        }
    )
    row = build_transaction_features(events, seasons, players, [2020], "08-31").row(0, named=True)
    assert row["cutoff_preseason_team"] == "BAL"
    assert row["cutoff_transaction_matched"] == 0.0


@pytest.mark.parametrize("named", [False, True])
def test_transaction_namesake_outside_fold_still_requires_team_match(named: bool) -> None:
    players = pl.DataFrame(
        {
            "gsis_id": ["qb", "cb"],
            "display_name": ["Lamar Jackson", "Lamar Jackson"],
        }
    )
    seasons = pl.DataFrame(
        {
            "season": [2019],
            "player_id": ["qb"],
            "player_display_name": ["Lamar Jackson"],
            "position": ["QB"],
            "team": ["BAL"],
        }
    )
    events = pl.DataFrame(
        {
            "transaction_date": [date(2020, 8, 1)],
            "transaction_year": [2020],
            "category": ["espn"],
            "from_team": [None],
            "to_team": ["NYJ"],
            "player_name": ["Lamar Jackson" if named else None],
            "description": ["Signed CB Lamar Jackson."],
        }
    )
    row = build_transaction_features(events, seasons, players, [2020], "08-31").row(0, named=True)
    assert row["cutoff_preseason_team"] == "BAL"
    assert row["cutoff_transaction_matched"] == 0.0
