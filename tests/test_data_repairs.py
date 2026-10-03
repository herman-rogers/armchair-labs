"""Regression cases from the September 2026 raw-data integrity audit."""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from engine.metrics.backtest import candidate_outcomes
from engine.metrics.experimental import build_contract_features, build_transaction_features
from engine.metrics.positions import (
    canonical_positions,
    historical_positions,
    repair_player_week_positions,
)
from engine.metrics.transaction_events import normalize_transaction_sources


def replay(name, prior_team, descriptions):
    players = pl.DataFrame({"gsis_id": ["p"], "display_name": [name]})
    seasons = pl.DataFrame(
        {
            "player_id": ["p"],
            "player_display_name": [name],
            "season": [2024],
            "team": [prior_team],
            "position": ["RB"],
        }
    )
    events = pl.DataFrame(
        [
            {
                "transaction_date": date(2025, 8, 11),
                "transaction_year": 2025,
                "category": "official",
                "from_team": None,
                "to_team": team,
                "player_name": None,
                "description": text,
                "source_url": "fixture",
            }
            for team, text in descriptions
        ]
    )
    result = build_transaction_features(events, seasons, players, [2025], "08-31")
    reverse = build_transaction_features(events.reverse(), seasons, players, [2025], "08-31")
    assert_frame_equal(result, reverse)
    assert_frame_equal(
        normalize_transaction_sources(events),
        normalize_transaction_sources(normalize_transaction_sources(events)),
    )
    return result.row(0, named=True)


@pytest.mark.parametrize(
    "name,team,text,expected_team,expected_status,expected_class",
    [
        (
            "Devin Funchess",
            "GB",
            "WR Devin Funchess placed on reserve/opt-out",
            "GB",
            "reserve",
            "opt_out",
        ),
        (
            "Jason Vander Laan",
            "NO",
            "Placed TEs Cole Wick and Jason Vander Laan on Reserve/Did Not Report/Covid Opt-Out.",
            "NO",
            "reserve",
            "opt_out",
        ),
        (
            "Julio Jones",
            "TEN",
            "Traded a second-round pick to the Atlanta Falcons for WR Julio Jones.",
            "TEN",
            "active",
            "active",
        ),
        (
            "Nick Chubb",
            "CLE",
            "Placed RB Nick Chubb on reserve/PUP, RB Nyheim Hines on reserve/NFI, "
            "and DT Michael Hall Jr. on the commissioner's exempt list.",
            "CLE",
            "reserve",
            "pup_nfi",
        ),
        (
            "Robert Tonyan",
            "MIN",
            "Placed CB Mekhi Blackmon on reserve/injured and WR Malik Knowles "
            "on exempt/commissioner permission.Signed TE Robert Tonyan.",
            "MIN",
            "active",
            "active",
        ),
        (
            "Josh Oliver",
            "BAL",
            "Placed TE Josh Oliver on exempt/COVID-19.",
            "BAL",
            "reserve",
            "covid",
        ),
        (
            "George Kittle",
            "SF",
            "Agreed to terms with TE George Kittle on a contract extension.",
            "SF",
            "unknown",
            "unknown",
        ),
    ],
)
def test_backfill_player_local_interpretation(
    name, team, text, expected_team, expected_status, expected_class
):
    row = replay(name, None, [(team, text)])
    assert row["cutoff_preseason_team"] == expected_team
    assert row["cutoff_preseason_status"] == expected_status
    assert row["cutoff_availability_class"] == expected_class


@pytest.mark.parametrize(
    "name,prior,events,team,status",
    [
        (
            "Deebo Samuel",
            "SF",
            [
                (
                    "SF",
                    "The 49ers have traded WR Deebo Samuel Sr. to the Washington Commanders "
                    "in exchange for a 2025 fifth-round draft choice. In addition, the team "
                    "has released DL Maliek Collins and DL Javon Hargrave.",
                )
            ],
            "WAS",
            "active",
        ),
        (
            "Ameer Abdullah",
            "LV",
            [
                (
                    "SF",
                    "The San Francisco 49ers have signed DL Trevis Gipson and RB Jeff Wilson "
                    "Jr. to one-year deals. The team placed RB Ameer Abdullah on the Injured "
                    "Reserve List, waived QB Tanner Mordecai, and activated OL Andre Dillard "
                    "off the Active/Physically Unable to Perform List.",
                )
            ],
            "SF",
            "reserve",
        ),
        (
            "Brian Robinson",
            "WAS",
            [
                (
                    "SF",
                    "The 49ers have acquired RB Brian Robinson Jr. from the Washington "
                    "Commanders in exchange for a sixth-round pick in 2026. In order to make "
                    "room on the roster, the team has waived WR Malik Knowles.",
                ),
                ("WAS", "Traded RB Brian Robinson Jr."),
            ],
            "SF",
            "active",
        ),
        (
            "Malik Willis",
            "TEN",
            [
                (
                    "GB",
                    "QB Malik Willis acquired in trade with Tennessee for 2025 seventh-round "
                    "draft pick; TE Tyler Davis and RB AJ Dillon placed on injured reserve; "
                    "DL Jonathan Ford on injured reserve/designated for return; CB LJ Davis "
                    "and LB Ralen Goforth waived/injured",
                ),
                ("TEN", "Traded QB Malik Willis to the Green Bay Packers."),
            ],
            "GB",
            "active",
        ),
        (
            "D'Andre Swift",
            "DET",
            [("DET", "The Detroit Lions trade RB D'Andre Swift to the Philadelphia Eagles.")],
            "PHI",
            "active",
        ),
        (
            "Joe Mixon",
            "CIN",
            [
                (
                    "CIN",
                    "Re-signed LB Akeem Davis-Gaither to a contract extension; Acquired the "
                    "No. 224 selection in Round 7 of the 2024 NFL Draft in a trade with "
                    "Houston for HB Joe Mixon; Terminated the contract of S Nick Scott",
                )
            ],
            "HOU",
            "active",
        ),
        (
            "Skyy Moore",
            "KC",
            [
                ("KC", "Traded WR Skyy Moore."),
                ("SF", "Acquired WR Skyy Moore from the Kansas City Chiefs."),
            ],
            "SF",
            "active",
        ),
        (
            "Aaron Rodgers",
            "GB",
            [
                ("GB", "Traded QB Aaron Rodgers to the New York Jets."),
                ("NYJ", "Acquired QB Aaron Rodgers in a trade with the Green Bay Packers."),
            ],
            "NYJ",
            "active",
        ),
        (
            "Test Player",
            "SF",
            [("SF", "Activated RB Test Player from the practice squad.")],
            "SF",
            "active",
        ),
    ],
)
def test_real_transaction_clauses(name, prior, events, team, status):
    row = replay(name, prior, events)
    assert row["cutoff_preseason_team"] == team
    assert row["cutoff_preseason_status"] == status
    assert row["cutoff_state_observed"]


def test_missing_trade_destination_and_same_day_conflict_are_unknown():
    missing = replay("Test Player", "SF", [("SF", "Traded RB Test Player.")])
    assert missing["cutoff_preseason_team"] is None
    assert missing["cutoff_preseason_rostered"] is None
    conflict = replay(
        "Test Player", "SF", [("SF", "Released RB Test Player."), ("SF", "Signed RB Test Player.")]
    )
    assert conflict["cutoff_state_resolution"] == "conflicting_same_day"
    assert conflict["cutoff_preseason_status_score"] is None
    reserve_conflict = replay(
        "Test Player",
        "SF",
        [
            ("SF", "Placed RB Test Player on injured reserve."),
            ("SF", "Placed RB Test Player on the PUP list."),
        ],
    )
    assert reserve_conflict["cutoff_state_resolution"] == "conflicting_same_day"
    assert reserve_conflict["cutoff_injured_reserve"] is None
    assert len(reserve_conflict["cutoff_evidence"]) == 2


@pytest.mark.parametrize("action", ["Promoted", "Elevated", "Signed"])
def test_practice_squad_exit_is_not_practice_status(action):
    row = replay(
        "Test Player",
        "SF",
        [
            ("SF", f"{action} RB Test Player from the practice squad."),
        ],
    )
    assert row["cutoff_preseason_status"] == "active"


def test_contract_cutoff_and_effective_start_not_signing_year():
    contracts = pl.DataFrame(
        {
            "gsis_id": ["before", "after", "unpublished", "undated", "old"],
            "year_signed": [2024, 2024, 2024, 2024, 2023],
            "years": [4] * 5,
            "apy_cap_pct": [0.1] * 5,
            "guaranteed": [50.0] * 5,
            "signing_date": [date(2024, 8, 1), date(2024, 9, 8), date(2024, 8, 1), None, None],
            "available_date": [date(2024, 8, 2), date(2024, 9, 8), date(2024, 9, 1), None, None],
            "effective_start_year": [2025, 2025, 2025, None, None],
        }
    )
    rows = build_contract_features(contracts, [2024], cutoff_by_season={2024: date(2024, 8, 30)})
    assert set(rows["player_id"]) == {"before", "old"}
    before = rows.filter(pl.col("player_id") == "before").row(0, named=True)
    assert before["contract_years_remaining"] == 4  # extension starts NEXT year
    assert before["contract_point_in_time_verified"]
    assert rows.filter(pl.col("player_id") == "old")["contract_apy_cap_pct"].item() is None


def test_same_year_undated_zeke_and_dak_contracts_excluded():
    rows = build_contract_features(
        pl.DataFrame(
            {
                "gsis_id": ["zeke", "dak"],
                "year_signed": [2019, 2024],
                "years": [6, 4],
                "apy_cap_pct": [0.08, 0.2],
                "guaranteed": [50.0, 100.0],
            }
        ),
        [2019, 2024],
        cutoff_by_season={2019: date(2019, 7, 17), 2024: date(2024, 8, 30)},
    )
    assert rows.filter(
        (pl.col("player_id") == "zeke") & (pl.col("forecast_season") == 2019)
    ).is_empty()
    assert rows.filter(
        (pl.col("player_id") == "dak") & (pl.col("forecast_season") == 2024)
    ).is_empty()


def test_research_position_policy_is_explicit_and_does_not_mutate_v1_input():
    original = pl.DataFrame({"position": ["FB", "HB", "RB", "TE"]})
    assert canonical_positions(original)["position"].to_list() == ["RB", "RB", "RB", "TE"]
    assert original["position"].to_list() == ["FB", "HB", "RB", "TE"]


def test_historical_roster_position_overrides_retrospective_stat_label():
    weeks = pl.DataFrame(
        {
            "player_id": ["thomas", "bradford"],
            "season": [2019, 2011],
            "week": [1, 1],
            "position": ["LB", "LB"],
        }
    )
    rosters = pl.DataFrame(
        {
            "gsis_id": ["thomas", "bradford"],
            "season": [2019, 2011],
            "week": [1, 1],
            "game_type": ["REG"] * 2,
            "position": ["TE", "RB"],
        }
    )
    result = repair_player_week_positions(weeks, historical_positions(rosters)).sort("player_id")
    assert result["position"].to_list() == ["RB", "TE"]
    assert result["raw_position"].to_list() == ["LB", "LB"]
    assert result["position_source"].to_list() == ["weekly_roster", "weekly_roster"]
    overridden = repair_player_week_positions(
        weeks,
        historical_positions(rosters),
        overrides={"thomas": "WR"},
    )
    assert overridden.filter(pl.col("player_id") == "thomas")["position"].item() == "WR"
    assert (
        overridden.filter(pl.col("player_id") == "thomas")["position_source"].item()
        == "league_override"
    )


def test_candidate_outcomes_do_not_erase_points_after_a_position_change():
    actual = pl.DataFrame(
        {
            "player_id": ["p"],
            "position": ["LB"],
            "ppg": [15.0],
            "ppg_denominator_games": [2.0],
            "season_pts": [30.0],
        }
    )
    candidates = pl.DataFrame({"player_id": ["p"], "position": ["TE"]})
    result = candidate_outcomes(actual, candidates, {"TE": 10.0}, 17).row(0, named=True)
    assert result["actual_season_points"] == 30.0
    assert result["actual_ppg"] == 15.0
    assert result["actual_vor"] == 5.0
