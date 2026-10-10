"""Causal clock, identity, and retirement regressions for the v2 population."""

from datetime import UTC, datetime

import polars as pl
import pytest
from experiments.future_player_lab.campaigns.availability import espn, layer


def fixtures(origin=0):
    panel = pl.DataFrame(
        {
            "year": [2021] * 4,
            "origin": [origin] * 4,
            "player_id": ["retired", "injured", "unknown", "returning"],
            "team": ["NO"] * 4,
            "base_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2020, 2021, 2021, 2021],
            "week": [17, 1, 2, 3],
            "gameday": ["2021-01-03", "2021-09-09", "2021-09-20", "2021-09-27"],
        }
    )
    rows = []
    for player, status, day in [
        ("retired", "RET", "2021-06-11"),
        ("injured", "RES", "2021-08-30"),
        ("returning", "RET", "2021-03-01"),
        ("returning", "ACT", "2021-09-01"),
    ]:
        rows.append(
            dict(
                player_id=player,
                status=status,
                available_at=layer.following_day(day),
                team="NO",
                source="transaction",
                source_url="https://example.com/archive",
                description=status,
            )
        )
    return panel, pl.DataFrame(rows, schema=layer.EVENT_SCHEMA), pl.DataFrame(), schedule


def test_only_dated_retirement_excludes_and_reinstatement_restores():
    panel, events, injuries, schedule = fixtures()
    result = layer.attach_availability(panel, events, injuries, schedule)
    assert set(result.filter(pl.col("availability_eligible"))["player_id"]) == {
        "injured",
        "unknown",
        "returning",
    }
    assert result.select(panel.columns).equals(panel.sort("year", "player_id"))
    assert result.filter(pl.col("player_id") == "unknown")["availability__known"][0] == 0


@pytest.mark.parametrize("origin", [0, 2])
def test_future_transactions_and_weekly_rosters_cannot_change_past(origin):
    panel, events, injuries, schedule = fixtures(origin)
    expected = layer.attach_availability(panel, events, injuries, schedule)
    future = events.with_columns(
        pl.lit(datetime(2021, 10, 1, tzinfo=UTC)).alias("available_at"),
        pl.lit("ACT").alias("status"),
    )
    actual = layer.attach_availability(panel, pl.concat([events, future]), injuries, schedule)
    assert actual.equals(expected)


def test_same_day_conflicting_states_are_unknown_not_retired():
    panel, events, injuries, schedule = fixtures()
    conflicting = events.filter(pl.col("player_id") == "retired").with_columns(
        pl.lit("ACT").alias("status")
    )
    row = (
        layer.attach_availability(panel, pl.concat([events, conflicting]), injuries, schedule)
        .filter(pl.col("player_id") == "retired")
        .row(0, named=True)
    )
    assert row["availability_eligible"]
    assert row["availability_status"] == "UNKNOWN"
    assert row["availability__conflict"] == 1


def test_week_one_roster_is_unavailable_preseason():
    panel, _, injuries, schedule = fixtures()
    roster = pl.DataFrame(
        {
            "gsis_id": ["retired"],
            "full_name": ["Test Player"],
            "season": [2021],
            "week": [1],
            "team": ["NO"],
            "status": ["RET"],
        }
    )
    identity = pl.DataFrame(
        {"gsis_id": ["retired"], "display_name": ["Test Player"], "football_name": ["Test Player"]}
    )
    events, _ = layer.build_events(pl.DataFrame(), roster, identity, schedule)
    before = layer.attach_availability(panel, events, injuries, schedule)
    after = layer.attach_availability(
        panel.with_columns(pl.lit(2).alias("origin")), events, injuries, schedule
    )
    assert before["availability_eligible"].all()
    assert not after.filter(pl.col("player_id") == "retired")["availability_eligible"][0]


def test_modified_after_cutoff_injury_is_withheld():
    panel, events, _, schedule = fixtures(2)
    injuries = pl.DataFrame(
        {
            "gsis_id": ["injured", "unknown"],
            "season": [2021, 2021],
            "report_status": ["Out", "Out"],
            "date_modified": [datetime(2021, 9, 18, tzinfo=UTC), datetime(2021, 9, 22, tzinfo=UTC)],
        }
    )
    result = layer.attach_availability(panel, events, injuries, schedule)
    assert result.filter(pl.col("player_id") == "injured")["availability__injury_out"][0] == 1
    assert result.filter(pl.col("player_id") == "unknown")["availability__injury_out"][0] is None


def test_ambiguous_names_are_not_assigned_from_team_or_future_career():
    identity = pl.DataFrame(
        {
            "gsis_id": ["one", "two"],
            "display_name": ["John Doe", "John Doe Jr."],
            "football_name": ["John Doe", "John Doe"],
        }
    )
    roster = pl.DataFrame({"gsis_id": ["one"], "full_name": ["John Doe"]})
    names, ambiguous = layer.name_index(identity, roster)
    assert "john doe" not in names
    assert ambiguous["john doe"] == ["one", "two"]


@pytest.mark.parametrize(
    "category,description,expected",
    [
        ("reserve-list", "Reserve/Retired", "RET"),
        ("other", "Reinstated from Reserve/Retired", "ACT"),
        ("reserve-list", "Reserve/Injured from Waived/Injured", "RES"),
        ("signings", "Practice Squad", "DEV"),
        ("terminations", "Terminated, Vested Veteran, all contracts", "CUT"),
        ("reserve-list", "Reserve/Suspended By Commissioner", "SUS"),
    ],
)
def test_transaction_destinations(category, description, expected):
    assert layer.transaction_status(category, description) == expected


def test_report_actions_do_not_spread_retirement_to_other_players():
    reports = [
        dict(
            date="2021-06-11T07:00Z",
            team="NO",
            source_url="https://example.com",
            description=(
                "Placed QB Test Player on reserve/retired list. Signed QB Other Player. "
                "Placed WR Third Player on injured reserve."
            ),
        )
    ]
    frame, rejected = espn.parse_reports(
        reports, {"test player": "one", "other player": "two", "third player": "three"}
    )
    assert dict(frame.select("player_id", "status").iter_rows()) == {
        "one": "RET",
        "two": "ACT",
        "three": "RES",
    }
    assert not rejected


def test_designated_to_return_is_not_activated():
    assert (
        espn.state("designated", "Designated Test Player to return from injured reserve") == "RES"
    )
    assert espn.state("activated", "Activated Test Player from injured reserve") == "ACT"


def test_retirement_announcement_is_not_a_formal_roster_exclusion():
    assert espn.state("announced", "Announced the retirement of QB Test Player") == "RET_ANNOUNCED"
    panel, events, injuries, schedule = fixtures()
    events = events.with_columns(
        pl.when(pl.col("player_id") == "retired")
        .then(pl.lit("RET_ANNOUNCED"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    row = (
        layer.attach_availability(panel, events, injuries, schedule)
        .filter(pl.col("player_id") == "retired")
        .row(0, named=True)
    )
    assert row["availability_eligible"]
    assert row["availability__status_ret_announced"] == 1


def test_unknown_next_sentence_does_not_inherit_previous_action():
    reports = [
        dict(
            date="2024-04-01T07:00Z",
            team="DEN",
            source_url="https://example.com",
            description=(
                "Announced the retirement of CB Test Player. "
                "Declined the fifth-year option on QB Other Player."
            ),
        )
    ]
    frame, _ = espn.parse_reports(reports, {"test player": "one", "other player": "two"})
    assert frame["player_id"].to_list() == ["one"]
    assert frame["status"].to_list() == ["RET_ANNOUNCED"]


def test_independent_audit_rejects_forged_clock_and_state():
    from experiments.future_player_lab.campaigns.availability.compare import audit_states

    panel, events, injuries, schedule = fixtures()
    audit = layer.attach_availability(panel, events, injuries, schedule)
    assert audit_states(audit, events, schedule)["latest_states_verified"]
    with pytest.raises(ValueError, match="calendar"):
        audit_states(
            audit.with_columns(pl.lit("2021-12-01T12:00:00+00:00").alias("availability_as_of")),
            events,
            schedule,
        )
    with pytest.raises(ValueError, match="Latest source state"):
        audit_states(
            audit.with_columns(pl.lit("ACT").alias("availability_status")), events, schedule
        )
