"""Unit tests for snapshot construction and its type coercion.

espn-api's field types are not uniform across player kinds, and the coercion here is
what keeps that irregularity from reaching the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from patron.espn.sync import (
    LeagueSnapshot,
    PlayerState,
    TeamState,
    TransactionState,
    _fetch_draft_market,
    _number,
    _player_state,
    _text,
)


class FakePlayer:
    def __init__(self, **attributes) -> None:
        self.__dict__.update(attributes)


class FakeTeam:
    def __init__(self, team_id: int, team_name: str) -> None:
        self.team_id = team_id
        self.team_name = team_name


class TestCoercion:
    def test_a_defense_s_empty_list_injury_status_becomes_none(self) -> None:
        """The bug this guards: espn-api returns `[]` rather than a string for a
        defense's injuryStatus, which was enough to make the whole player frame fail
        to build with a schema error."""
        assert _text([]) is None

    @pytest.mark.parametrize("value", [[], {}, (), None])
    def test_non_scalars_and_none_become_none(self, value: object) -> None:
        assert _text(value) is None

    def test_empty_and_whitespace_strings_become_none(self) -> None:
        assert _text("") is None
        assert _text("   ") is None

    def test_real_values_survive(self) -> None:
        assert _text("QUESTIONABLE") == "QUESTIONABLE"
        assert _number(12) == pytest.approx(12.0)
        assert _number("3.5") == pytest.approx(3.5)

    def test_unparseable_numbers_become_none_not_zero(self) -> None:
        assert _number("n/a") is None
        assert _number(None) is None
        assert _number(True) is None, "a bool is not a measurement"


class TestPlayerState:
    def test_a_rostered_player(self) -> None:
        state = _player_state(
            FakePlayer(
                playerId=3117251,
                name="Christian McCaffrey",
                position="RB",
                proTeam="SF",
                injuryStatus="QUESTIONABLE",
                percent_owned=99.87,
            ),
            FakeTeam(3, "Patron Saints"),
        )

        assert state.espn_id == 3117251
        assert state.injury_status == "QUESTIONABLE"
        assert state.owner_team_id == 3
        assert state.percent_owned == pytest.approx(99.87)

    def test_market_fields_are_kept_separate_from_projections(self) -> None:
        state = _player_state(
            FakePlayer(playerId=1, name="Rookie", position="WR", proTeam="NYJ"),
            draft_rank=12,
            position_rank=5,
            adp=14.7,
        )

        assert state.espn_draft_rank == 12
        assert state.espn_position_rank == 5
        assert state.espn_adp == pytest.approx(14.7)

    def test_a_free_agent_has_no_owner(self) -> None:
        state = _player_state(FakePlayer(playerId=1, name="Free", position="WR", proTeam="SEA"))
        assert state.owner_team_id is None
        assert state.owner_team_name is None

    def test_an_unsigned_player_has_no_team(self) -> None:
        """ESPN writes the literal string "None" in proTeam for an unsigned player.
        Left alone it renders as a team called "None" and compares unequal to every
        real team code, faking a trade."""
        state = _player_state(
            FakePlayer(playerId=1, name="Kareem Hunt", position="RB", proTeam="None")
        )
        assert state.espn_team is None

    def test_espn_team_is_translated_to_nflverse_vocabulary(self) -> None:
        state = _player_state(FakePlayer(playerId=1, name="X", position="WR", proTeam="LAR"))
        assert state.espn_team == "LA"

    def test_a_defense_survives_construction(self) -> None:
        state = _player_state(
            FakePlayer(
                playerId=-16003, name="Bears D/ST", position="D/ST", proTeam="CHI", injuryStatus=[]
            ),
            FakeTeam(3, "Patron Saints"),
        )
        assert state.injury_status is None
        assert state.position == "D/ST"


class TestSnapshotFrame:
    def _snapshot(self, players: list[PlayerState]) -> LeagueSnapshot:
        return LeagueSnapshot(
            captured_at="2026-08-29T00:00:00+00:00",
            league_id=1,
            league_name="Sweaty Plays",
            season=2026,
            week=1,
            my_team_id=3,
            players=players,
        )

    def test_an_empty_snapshot_still_yields_a_typed_frame(self) -> None:
        frame = self._snapshot([]).to_frame()
        assert frame.height == 0
        assert frame.schema["espn_id"] == pl.Int64

    def test_all_null_columns_keep_their_declared_type(self) -> None:
        """Inferred schemas turn an all-null column into Null dtype, which then fails
        to join. The schema is declared for exactly this reason."""
        frame = self._snapshot(
            [
                PlayerState(1, "A", "WR", None, None, None, None, None, None, None, None),
                PlayerState(2, "B", "RB", None, None, None, None, None, None, None, None),
            ]
        ).to_frame()

        assert frame.schema["injury_status"] == pl.String
        assert frame.schema["percent_owned"] == pl.Float64
        assert frame.schema["espn_draft_rank"] == pl.Int64
        assert frame.schema["espn_adp"] == pl.Float64

    def test_observed_adp_is_archived_once_per_date(self, tmp_path: Path) -> None:
        snapshot = self._snapshot(
            [
                PlayerState(
                    1,
                    "Market Player",
                    "WR",
                    "NYJ",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    espn_draft_rank=20,
                    espn_position_rank=8,
                    espn_adp=22.4,
                )
            ]
        )

        path = snapshot.write_adp_snapshot(tmp_path)

        assert path == tmp_path / "espn_adp_2026_2026-08-29.parquet"
        row = pl.read_parquet(path).to_dicts()[0]
        assert row["espn_id"] == 1
        assert row["espn_adp"] == pytest.approx(22.4)

    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        original = self._snapshot(
            [PlayerState(1, "A", "WR", "SF", "ACTIVE", 50.0, 10.0, 8.0, 3, "Mine", "BE")]
        )
        original.teams = [
            TeamState(
                3,
                "Mine",
                "me",
                1,
                0,
                120,
                division_id=2,
                division_name="NFC",
            )
        ]
        original.transactions = [TransactionState("2026-08-01", "WAIVER", "Mine", "A", 17)]

        path = original.write(tmp_path / "snap.json")
        restored = LeagueSnapshot.read(path)

        assert restored.league_name == "Sweaty Plays"
        assert restored.players[0].player_display_name == "A"
        assert restored.teams[0].faab_remaining == 120
        assert restored.teams[0].division_id == 2
        assert restored.teams[0].division_name == "NFC"
        assert restored.transactions[0].bid_amount == 17

    def test_an_old_snapshot_without_divisions_still_loads(self, tmp_path: Path) -> None:
        original = self._snapshot([])
        original.teams = [TeamState(3, "Mine", "me", 1, 0, 120)]
        path = original.write(tmp_path / "snap.json")

        raw = path.read_text().replace(
            ',\n      "division_id": null,\n      "division_name": null',
            "",
        )
        path.write_text(raw)
        restored = LeagueSnapshot.read(path)

        assert restored.teams[0].division_id is None
        assert restored.teams[0].division_name is None


class TestDraftMarket:
    def test_ppr_response_order_becomes_skill_and_position_rank(self) -> None:
        class Request:
            def league_get(self, **_: object) -> dict[str, object]:
                def player(player_id: int, position_id: int, adp: float) -> dict[str, object]:
                    return {
                        "player": {
                            "id": player_id,
                            "defaultPositionId": position_id,
                            "ownership": {"averageDraftPosition": adp},
                        }
                    }

                return {
                    "players": [
                        player(11, 2, 1.3),
                        player(-16001, 16, 2.0),  # D/ST is not on Patron's board.
                        player(12, 3, 4.8),
                        player(13, 2, 5.1),
                    ]
                }

        league = FakePlayer(espn_request=Request(), current_week=1)
        market = _fetch_draft_market(league)

        assert market[11] == (1, 1, pytest.approx(1.3))
        assert market[12] == (2, 1, pytest.approx(4.8))
        assert market[13] == (3, 2, pytest.approx(5.1))
        assert -16001 not in market


class TestTimestamps:
    """ESPN's activity feed reports times as epoch milliseconds. Passed through as a
    string it renders as "1787929226212" in the UI — technically the data, and useless
    to read."""

    def test_epoch_milliseconds_become_iso(self) -> None:
        from patron.espn.sync import _epoch_ms_to_iso

        assert _epoch_ms_to_iso(1787929226212) == "2026-08-28T15:00:26.212000+00:00"

    def test_a_numeric_string_is_accepted(self) -> None:
        from patron.espn.sync import _epoch_ms_to_iso

        assert _epoch_ms_to_iso("1787929226212").startswith("2026-08-28")

    def test_seconds_are_not_mistaken_for_milliseconds(self) -> None:
        """Dividing a seconds value by 1000 would silently produce a date in 1970."""
        from patron.espn.sync import _epoch_ms_to_iso

        assert _epoch_ms_to_iso(1787929226).startswith("2026-08-28")

    def test_none_stays_none(self) -> None:
        from patron.espn.sync import _epoch_ms_to_iso

        assert _epoch_ms_to_iso(None) is None

    def test_an_already_formatted_string_passes_through(self) -> None:
        from patron.espn.sync import _epoch_ms_to_iso

        assert _epoch_ms_to_iso("2026-08-28T15:00:00Z") == "2026-08-28T15:00:00Z"


def test_snapshot_round_trips_the_draft_recap(tmp_path) -> None:
    from patron.espn.sync import DraftPick, LeagueSnapshot

    snapshot = LeagueSnapshot(
        captured_at="2026-08-29T00:00:00+00:00",
        league_id=1,
        league_name="Test",
        season=2026,
        week=1,
        my_team_id=3,
        draft=[DraftPick(1, 1, 1, 3, "Mine", 100, "Star", bid_amount=None, keeper=False)],
    )
    path = snapshot.write(tmp_path / "snapshot.json")
    restored = LeagueSnapshot.read(path)
    assert restored.draft[0] == snapshot.draft[0]
    assert LeagueSnapshot.read(path).draft[0].overall == 1
