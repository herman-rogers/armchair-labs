"""Unit tests for the ESPN-to-nflverse join.

Every test here corresponds to a bug found against the live league. The join is where
§8's "name joining is the tax" gets paid, and its failure mode is silence — a wrong
answer that still renders — so the regression guards matter more than usual.
"""

from __future__ import annotations

import polars as pl
import pytest

from engine.espn import teams
from engine.espn.crosswalk import (
    AVAILABILITY,
    CHANGED_TEAM,
    FREE_AGENT,
    IS_FREE_AGENT,
    IS_MINE,
    ROSTERED,
    UNKNOWN,
    JoinReport,
    append_espn_fallbacks,
    attach_ownership,
    board_lookups,
    resolve_player_ids,
)


def board(*rows: tuple[str, str, str, str]) -> pl.DataFrame:
    """(player_id, name, position, team)"""
    return pl.DataFrame(
        {
            "player_id": [r[0] for r in rows],
            "player_display_name": [r[1] for r in rows],
            "position": [r[2] for r in rows],
            "team": [r[3] for r in rows],
        }
    )


def espn(*rows: tuple[int, str, str, str | None, int | None]) -> pl.DataFrame:
    """(espn_id, name, position, espn_team, owner_team_id)"""
    return pl.DataFrame(
        {
            "espn_id": [r[0] for r in rows],
            "player_display_name": [r[1] for r in rows],
            "position": [r[2] for r in rows],
            "espn_team": [r[3] for r in rows],
            "owner_team_id": [r[4] for r in rows],
            "owner_team_name": [f"Team {r[4]}" if r[4] else None for r in rows],
        },
        schema_overrides={"espn_team": pl.String, "owner_team_id": pl.Int64},
    )


class TestTeamAbbreviations:
    @pytest.mark.parametrize(
        ("nflverse", "espn_code"), [("LA", "LAR"), ("WAS", "WSH"), ("KC", "KC")]
    )
    def test_known_aliases_reconcile(self, nflverse: str, espn_code: str) -> None:
        assert teams.same_team(nflverse, espn_code)

    def test_a_real_move_is_still_detected(self) -> None:
        assert not teams.same_team("JAX", "NO")

    def test_unsigned_markers_are_not_teams(self) -> None:
        """ESPN writes the literal string "None" in proTeam for an unsigned player.
        Left alone it renders as a team called None and compares unequal to everything.
        """
        for marker in ("None", "FA", "", "-"):
            assert teams.to_nflverse(marker) is None
            assert teams.same_team("SF", marker), "unknown is not evidence of a trade"

    def test_a_missing_team_is_not_a_trade(self) -> None:
        assert teams.same_team(None, "SF")
        assert teams.same_team("SF", None)

    def test_the_two_vocabularies_fully_reconcile(self) -> None:
        """The whole NFL, both ways. If ESPN or nflverse renames a club, this fails
        here rather than silently reporting a squad's worth of phantom trades."""
        nflverse_teams = {
            "ARI",
            "ATL",
            "BAL",
            "BUF",
            "CAR",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "HOU",
            "IND",
            "JAX",
            "KC",
            "LA",
            "LAC",
            "LV",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "PIT",
            "SEA",
            "SF",
            "TB",
            "TEN",
            "WAS",
        }
        espn_teams = {
            "ARI",
            "ATL",
            "BAL",
            "BUF",
            "CAR",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "HOU",
            "IND",
            "JAX",
            "KC",
            "LAR",
            "LAC",
            "LV",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "PIT",
            "SEA",
            "SF",
            "TB",
            "TEN",
            "WSH",
        }
        nflverse_only, espn_only = teams.reconcile(nflverse_teams, espn_teams)

        assert nflverse_only == set(), f"nflverse codes with no ESPN match: {nflverse_only}"
        assert espn_only == set(), f"ESPN codes with no nflverse match: {espn_only}"


class TestResolution:
    def test_crosswalk_id_is_the_primary_route(self) -> None:
        frame, report = resolve_player_ids(
            espn((3117251, "Christian McCaffrey", "RB", "SF", 3)),
            {"00-0034844"},
            {},
            espn_to_gsis={3117251: "00-0034844"},
        )

        assert frame["player_id"][0] == "00-0034844"
        assert report.matched_by_id == 1
        assert report.matched_by_name == 0

    def test_name_fallback_catches_a_crosswalk_gap(self) -> None:
        """Adds nothing against the live league today, but it is the difference between
        a missing row and a wrong one when the crosswalk lags a call-up."""
        _, names = board_lookups(board(("00-0001", "D.J. Moore", "WR", "CHI")))
        frame, report = resolve_player_ids(
            espn((999, "DJ Moore", "WR", "BUF", None)), {"00-0001"}, names, espn_to_gsis={}
        )

        assert frame["player_id"][0] == "00-0001"
        assert report.matched_by_name == 1

    def test_a_rookie_is_unmatched_not_mismatched(self) -> None:
        frame, report = resolve_player_ids(
            espn((888, "Jeremiyah Love", "RB", "ARI", None)), set(), {}, espn_to_gsis={}
        )

        assert frame["player_id"][0] is None
        assert len(report.unmatched) == 1
        assert report.unmatched[0][0] == "Jeremiyah Love"

    def test_kickers_and_defenses_do_not_count_as_failures(self) -> None:
        """They are never on the board by design. Counting them dragged a genuine 82%
        match rate down to a misleading 70%, which would hide a real regression."""
        _, report = resolve_player_ids(
            espn(
                (1, "Will Reichard", "K", "MIN", 3),
                (2, "Bears D/ST", "D/ST", "CHI", 3),
            ),
            set(),
            {},
            espn_to_gsis={},
        )

        assert report.not_ranked == 2
        assert report.unmatched == []
        assert report.total == 0

    def test_summary_reports_both_categories(self) -> None:
        report = JoinReport(matched_by_id=342, unmatched=[("x", "WR", "SF")] * 75, not_ranked=73)
        summary = report.summary()

        assert "342" in summary
        assert "82%" in summary
        assert "73 K/DST not ranked" in summary


class TestOwnership:
    def _tagged(self) -> pl.DataFrame:
        rows = board(
            ("g1", "Owned Guy", "RB", "SF"),
            ("g2", "Free Guy", "WR", "SEA"),
            ("g3", "My Guy", "TE", "ARI"),
            ("g4", "Never Mentioned", "WR", "NYJ"),
        )
        players = espn(
            (1, "Owned Guy", "RB", "SF", 7),
            (2, "Free Guy", "WR", "SEA", None),
            (3, "My Guy", "TE", "ARI", 3),
        ).with_columns(pl.Series("player_id", ["g1", "g2", "g3"]))
        return attach_ownership(rows, players, my_team_id=3)

    def test_ownership_is_tagged(self) -> None:
        by_name = {r["player_display_name"]: r for r in self._tagged().iter_rows(named=True)}

        assert by_name["Owned Guy"][AVAILABILITY] == ROSTERED
        assert by_name["Free Guy"][AVAILABILITY] == FREE_AGENT
        assert by_name["My Guy"][IS_MINE] is True
        assert by_name["Owned Guy"][IS_MINE] is False

    def test_a_player_espn_never_mentioned_is_unknown_not_available(self) -> None:
        """The bug this guards: treating absence from the snapshot as availability put
        267 unverified players onto a 472-row wire — more than half the list."""
        row = next(
            r
            for r in self._tagged().iter_rows(named=True)
            if r["player_display_name"] == "Never Mentioned"
        )

        assert row[AVAILABILITY] == UNKNOWN
        assert row[IS_FREE_AGENT] is False, "unknown must never be claimable"

    def test_only_confirmed_free_agents_are_claimable(self) -> None:
        tagged = self._tagged()
        assert tagged.filter(pl.col(IS_FREE_AGENT)).height == 1


class TestEspnFallbacks:
    def test_unresolved_rookie_is_inserted_without_inventing_model_metrics(self) -> None:
        rows = board(("g1", "Veteran", "WR", "SEA")).with_columns(
            pl.lit(1).alias("rank"),
            pl.lit(12.0).alias("ppg"),
            pl.lit(4.0).alias("adj_vor"),
            pl.lit("").alias("flags"),
        )
        players = espn(
            (1, "Veteran", "WR", "SEA", 3),
            (888, "Jeremiyah Love", "RB", "ARI", None),
        ).with_columns(
            pl.Series("player_id", ["g1", None], dtype=pl.String),
            pl.Series("espn_draft_rank", [20, 1], dtype=pl.Int64),
            pl.Series("espn_position_rank", [8, 1], dtype=pl.Int64),
            pl.Series("espn_adp", [22.0, 1.5], dtype=pl.Float64),
        )

        tagged = attach_ownership(rows, players, my_team_id=3)
        combined = append_espn_fallbacks(tagged, players, my_team_id=3)
        rookie = combined.filter(pl.col("player_id") == "espn:888").row(0, named=True)

        assert combined.height == 2
        assert rookie["rank"] == 1
        assert rookie["rank_source"] == "espn_ppr"
        assert rookie["espn_fallback"] is True
        assert rookie["flags"] == "ESPN-only"
        assert rookie["ppg"] is None
        assert rookie["adj_vor"] is None
        assert rookie[AVAILABILITY] == FREE_AGENT
        assert "No prior NFL production" in rookie["projection_reason"]

    def test_model_order_is_preserved_around_fallbacks(self) -> None:
        rows = board(
            ("g1", "First", "WR", "SEA"),
            ("g2", "Second", "WR", "SF"),
            ("g3", "Third", "RB", "ARI"),
        ).with_columns(pl.Series("rank", [1, 2, 3]), pl.lit("").alias("flags"))
        players = espn((888, "Rookie", "RB", "ARI", None)).with_columns(
            pl.Series("player_id", [None], dtype=pl.String),
            pl.Series("espn_draft_rank", [2], dtype=pl.Int64),
            pl.Series("espn_position_rank", [1], dtype=pl.Int64),
            pl.Series("espn_adp", [2.3], dtype=pl.Float64),
        )

        tagged = attach_ownership(rows, players)
        combined = append_espn_fallbacks(tagged, players)

        assert combined["player_id"].to_list() == ["g1", "espn:888", "g2", "g3"]
        assert combined["rank"].to_list() == [1, 2, 3, 4]


class TestChangedTeam:
    def _flag(self, board_team: str, espn_team: str | None) -> bool:
        rows = board(("g1", "Guy", "WR", board_team))
        players = espn((1, "Guy", "WR", espn_team, None)).with_columns(
            pl.Series("player_id", ["g1"])
        )
        return attach_ownership(rows, players)[CHANGED_TEAM][0]

    def test_a_real_move_flags(self) -> None:
        assert self._flag("JAX", "NO") is True

    @pytest.mark.parametrize(("board_team", "espn_team"), [("LA", "LAR"), ("WAS", "WSH")])
    def test_an_abbreviation_difference_does_not_flag(
        self, board_team: str, espn_team: str
    ) -> None:
        """The phantom-trade bug: raw string comparison reported 19 players as having
        changed teams when none had, poisoning a signal that decides whether prior
        situational stats are still meaningful."""
        assert self._flag(board_team, espn_team) is False

    def test_same_team_does_not_flag(self) -> None:
        assert self._flag("SF", "SF") is False
