"""Unit tests for lineup optimisation.

The point of this module is that total roster value is a misleading comparison: you
cannot start five running backs. These tests pin the slot-filling rules that make the
comparison honest.
"""

from __future__ import annotations

import polars as pl
import pytest

from engine.espn.lineup import LineupSlot, best_lineup, value_column_for

SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "RB/WR/TE": 1, "K": 1, "D/ST": 1}


def roster(*players: tuple[str, str, float | None]) -> pl.DataFrame:
    """(name, position, value)"""
    return pl.DataFrame(
        {
            "player_id": [f"g{i}" for i in range(len(players))],
            "player_display_name": [p[0] for p in players],
            "position": [p[1] for p in players],
            "value": [p[2] for p in players],
        },
        schema_overrides={"value": pl.Float64},
    )


def names(starters: list[LineupSlot], slot: str) -> list[str | None]:
    return [s.player_name for s in starters if s.slot == slot]


class TestSlotFilling:
    def test_the_best_player_fills_a_dedicated_slot(self) -> None:
        lineup = best_lineup(
            roster(("Good QB", "QB", 10.0), ("Bad QB", "QB", 2.0)), {"QB": 1}, "value"
        )
        assert names(lineup.starters, "QB") == ["Good QB"]

    def test_slot_counts_are_respected(self) -> None:
        lineup = best_lineup(
            roster(("RB1", "RB", 9.0), ("RB2", "RB", 8.0), ("RB3", "RB", 7.0)),
            {"RB": 2},
            "value",
        )
        assert names(lineup.starters, "RB") == ["RB1", "RB2"]
        assert [b.player_name for b in lineup.bench] == ["RB3"]

    def test_you_cannot_start_five_running_backs(self) -> None:
        """The whole reason this module exists. Total roster value would rank this
        team far above one with a balanced lineup it can actually field."""
        stacked = best_lineup(roster(*[(f"RB{i}", "RB", 10.0) for i in range(5)]), SLOTS, "value")
        # 2 RB slots + 1 flex = 30, not 50.
        assert stacked.total == pytest.approx(30.0)


class TestFlex:
    def test_flex_accepts_any_named_position(self) -> None:
        for position in ("RB", "WR", "TE"):
            lineup = best_lineup(roster(("Guy", position, 5.0)), {"RB/WR/TE": 1}, "value")
            assert names(lineup.starters, "RB/WR/TE") == ["Guy"]

    def test_flex_rejects_a_position_not_in_its_name(self) -> None:
        lineup = best_lineup(roster(("Passer", "QB", 20.0)), {"RB/WR/TE": 1}, "value")
        assert names(lineup.starters, "RB/WR/TE") == [None]

    def test_dedicated_slots_are_filled_before_flex(self) -> None:
        """If the flex went first it would take the best back and leave the dedicated
        RB slot to a worse one — a lineup no manager would ever set. The total is the
        same either way, which is exactly why this needs asserting on the assignment
        rather than the sum."""
        lineup = best_lineup(
            roster(("Best RB", "RB", 9.0), ("Ok RB", "RB", 5.0), ("Ok WR", "WR", 4.0)),
            {"RB": 1, "RB/WR/TE": 1},
            "value",
        )
        assert names(lineup.starters, "RB") == ["Best RB"]
        assert names(lineup.starters, "RB/WR/TE") == ["Ok RB"]

    def test_flex_takes_the_best_leftover_across_positions(self) -> None:
        lineup = best_lineup(
            roster(("RB1", "RB", 9.0), ("WR1", "WR", 8.0), ("TE1", "TE", 7.0)),
            {"RB": 1, "RB/WR/TE": 1},
            "value",
        )
        assert names(lineup.starters, "RB/WR/TE") == ["WR1"]


class TestGaps:
    def test_an_unfillable_slot_is_kept_and_scores_zero(self) -> None:
        """A team with no tight end starts a zero there. Silently dropping the slot
        would flatter them in a comparison."""
        lineup = best_lineup(roster(("RB1", "RB", 9.0)), {"RB": 1, "TE": 1}, "value")
        tight_end = next(s for s in lineup.starters if s.slot == "TE")

        assert tight_end.empty
        assert tight_end.value == 0.0
        assert lineup.total == pytest.approx(9.0)

    def test_unrankable_players_are_counted_not_started(self) -> None:
        """A rookie has no prior tape and so no value. He cannot be scored, but a
        roster leaning on him should say so rather than just looking thin."""
        lineup = best_lineup(
            roster(("Veteran", "WR", 8.0), ("Rookie", "WR", None)), {"WR": 2}, "value"
        )

        assert lineup.unranked_starters == 1
        assert names(lineup.starters, "WR") == ["Veteran", None]


class TestTotals:
    def test_total_sums_only_starters(self) -> None:
        lineup = best_lineup(
            roster(("RB1", "RB", 9.0), ("RB2", "RB", 8.0), ("Benched", "RB", 7.0)),
            {"RB": 2},
            "value",
        )
        assert lineup.total == pytest.approx(17.0)

    def test_by_position_groups_flex_under_the_real_position(self) -> None:
        lineup = best_lineup(
            roster(("RB1", "RB", 9.0), ("WR1", "WR", 8.0)), {"RB": 1, "RB/WR/TE": 1}, "value"
        )
        assert lineup.by_position == {"RB": pytest.approx(9.0), "WR": pytest.approx(8.0)}


class TestValueColumn:
    def test_a_projection_is_preferred_over_last_season(self) -> None:
        """A schedule scan asks about weeks that have not happened yet."""
        board = pl.DataFrame({"adj_vor": [1.0], "adj_proj_vor": [2.0], "ppg": [3.0]})
        assert value_column_for(board) == "adj_proj_vor"

    def test_falls_back_through_the_chain(self) -> None:
        assert value_column_for(pl.DataFrame({"adj_vor": [1.0], "ppg": [3.0]})) == "adj_vor"
        assert value_column_for(pl.DataFrame({"ppg": [3.0]})) == "ppg"

    def test_a_board_with_no_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="no usable strength column"):
            value_column_for(pl.DataFrame({"player_id": ["x"]}))


class TestRankableSlots:
    """Kickers and defenses are tiered by this league rather than ranked, so they
    carry no comparable value. Leaving their slots in made every team show the same
    two empty rows — noise, and a total that looked like it was missing something."""

    def test_unrankable_slots_are_dropped(self) -> None:
        from engine.espn.lineup import rankable_slots

        assert rankable_slots(SLOTS) == {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "RB/WR/TE": 1,
        }

    def test_a_flex_survives_only_if_every_position_it_accepts_is_ranked(self) -> None:
        from engine.espn.lineup import rankable_slots

        assert "RB/WR/TE" in rankable_slots({"RB/WR/TE": 1})
        assert "QB/RB/WR/TE/K" not in rankable_slots({"QB/RB/WR/TE/K": 1})

    def test_the_lineup_has_no_kicker_row(self) -> None:
        lineup = best_lineup(roster(("RB1", "RB", 9.0)), SLOTS, "value")
        assert all(slot.slot not in {"K", "D/ST"} for slot in lineup.starters)


class TestUnrankedCount:
    def test_the_caller_supplies_the_count(self) -> None:
        """An unrankable player is absent from the board entirely, not present with a
        null value, so this cannot be derived from the roster frame — deriving it
        reported zero for a roster full of rookies."""
        lineup = best_lineup(roster(("Veteran", "WR", 8.0)), {"WR": 2}, "value", unranked_count=3)
        assert lineup.unranked_starters == 3

    def test_it_falls_back_to_counting_nulls(self) -> None:
        lineup = best_lineup(
            roster(("Veteran", "WR", 8.0), ("Rookie", "WR", None)), {"WR": 2}, "value"
        )
        assert lineup.unranked_starters == 1
