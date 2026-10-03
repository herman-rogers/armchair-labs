"""Unit tests for the manual override system.

The behaviour that matters most here is the failure mode: an override that matches
nothing must be loud. A typo silently means a player carrying an Achilles tear keeps
his full ranking, which is precisely what the file exists to prevent.
"""

from __future__ import annotations

import polars as pl
import pytest

from engine.board.overrides import (
    ADJUSTED_VOR,
    OVERRIDE_DELTA,
    OVERRIDE_REASON,
    BlindSpot,
    Override,
    OverrideSet,
    Removal,
    UnmatchedOverrideError,
    apply_overrides,
)
from engine.config.settings import load_overrides_config


def board(*players: tuple[str, str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_display_name": [name for name, _, _ in players],
            "position": [position for _, position, _ in players],
            "vor": [vor for _, _, vor in players],
        }
    )


class TestDeltas:
    def test_a_delta_lowers_adjusted_vor(self) -> None:
        result = apply_overrides(
            board(("George Kittle", "TE", 4.1)),
            OverrideSet(overrides=[Override("George Kittle", "TE", -2.5, "Achilles")]),
        )

        assert result[ADJUSTED_VOR][0] == pytest.approx(1.6)
        assert result[OVERRIDE_DELTA][0] == pytest.approx(-2.5)
        assert result[OVERRIDE_REASON][0] == "Achilles"

    def test_untouched_players_keep_their_vor(self) -> None:
        result = apply_overrides(
            board(("George Kittle", "TE", 4.1), ("Puka Nacua", "WR", 12.4)),
            OverrideSet(overrides=[Override("George Kittle", "TE", -2.5, "Achilles")]),
        )
        nacua = result.filter(pl.col("player_display_name") == "Puka Nacua").to_dicts()[0]

        assert nacua[ADJUSTED_VOR] == pytest.approx(12.4)
        assert nacua[OVERRIDE_DELTA] == pytest.approx(0.0)
        assert nacua[OVERRIDE_REASON] is None

    def test_a_positive_delta_promotes(self) -> None:
        """Overrides cut both ways — a receiver walking into a better target tree is
        the same mechanism as an injury, with the sign flipped."""
        result = apply_overrides(
            board(("DJ Moore", "WR", -1.1)),
            OverrideSet(overrides=[Override("DJ Moore", "WR", 1.5, "better target tree")]),
        )
        assert result[ADJUSTED_VOR][0] == pytest.approx(0.4)

    def test_deltas_reorder_the_board(self) -> None:
        result = apply_overrides(
            board(("George Kittle", "TE", 4.1), ("Nico Collins", "WR", 4.2)),
            OverrideSet(overrides=[Override("George Kittle", "TE", -2.5, "Achilles")]),
        ).sort(ADJUSTED_VOR, descending=True)

        assert result["player_display_name"].to_list() == ["Nico Collins", "George Kittle"]


class TestMatching:
    @pytest.mark.parametrize("board_name", ["D.J. Moore", "DJ Moore", "dj moore", "D.J. Moore Jr."])
    def test_names_match_through_normalization(self, board_name: str) -> None:
        result = apply_overrides(
            board((board_name, "WR", 0.0)),
            OverrideSet(overrides=[Override("DJ Moore", "WR", -1.0, "reason")]),
        )
        assert result[ADJUSTED_VOR][0] == pytest.approx(-1.0)

    def test_position_disambiguates_a_shared_name(self) -> None:
        result = apply_overrides(
            board(("Josh Allen", "QB", 5.7), ("Josh Allen", "RB", 1.0)),
            OverrideSet(overrides=[Override("Josh Allen", "QB", -2.0, "knee")]),
        )
        by_position = {row["position"]: row[ADJUSTED_VOR] for row in result.to_dicts()}

        assert by_position == {"QB": pytest.approx(3.7), "RB": pytest.approx(1.0)}


class TestUnmatchedOverrides:
    def test_an_unmatched_override_raises_by_default(self) -> None:
        with pytest.raises(UnmatchedOverrideError) as excinfo:
            apply_overrides(
                board(("Puka Nacua", "WR", 12.4)),
                OverrideSet(overrides=[Override("Georg Kittle", "TE", -2.5, "Achilles")]),
            )

        message = str(excinfo.value)
        assert "Georg Kittle" in message
        assert "Achilles" in message, "the reason should be shown so the entry is findable"
        assert "overrides.yaml" in message, "the error should say where to fix it"

    def test_a_wrong_position_counts_as_unmatched(self) -> None:
        with pytest.raises(UnmatchedOverrideError):
            apply_overrides(
                board(("George Kittle", "TE", 4.1)),
                OverrideSet(overrides=[Override("George Kittle", "WR", -2.5, "Achilles")]),
            )

    def test_strict_can_be_relaxed_for_a_partial_board(self) -> None:
        result = apply_overrides(
            board(("Puka Nacua", "WR", 12.4)),
            OverrideSet(overrides=[Override("George Kittle", "TE", -2.5, "Achilles")]),
            strict=False,
        )
        assert result[ADJUSTED_VOR][0] == pytest.approx(12.4)


class TestRemovals:
    def test_a_removed_player_leaves_the_board(self) -> None:
        result = apply_overrides(
            board(("Ricky Pearsall", "WR", 1.0), ("Puka Nacua", "WR", 12.4)),
            OverrideSet(removals=[Removal("Ricky Pearsall", "WR", "out for 2026")]),
        )
        assert result["player_display_name"].to_list() == ["Puka Nacua"]

    def test_removing_an_absent_player_is_not_an_error(self) -> None:
        """The entry asked for a state that already holds. Worth a log line, not a
        failure — unlike an unmatched delta, nothing is silently wrong."""
        result = apply_overrides(
            board(("Puka Nacua", "WR", 12.4)),
            OverrideSet(removals=[Removal("Ricky Pearsall", "WR", "out for 2026")]),
        )
        assert result.height == 1

    def test_a_removed_player_cannot_also_be_adjusted(self) -> None:
        with pytest.raises(UnmatchedOverrideError):
            apply_overrides(
                board(("Ricky Pearsall", "WR", 1.0)),
                OverrideSet(
                    removals=[Removal("Ricky Pearsall", "WR", "out")],
                    overrides=[Override("Ricky Pearsall", "WR", -1.0, "also hurt")],
                ),
            )


class TestShippedConfig:
    """The overrides.yaml that ships with the repo must stay loadable and coherent."""

    def test_it_parses(self) -> None:
        override_set = OverrideSet.from_config()

        assert override_set.overrides, "expected the six seeded injury overrides"
        assert all(isinstance(entry, Override) for entry in override_set.overrides)
        assert all(isinstance(entry, BlindSpot) for entry in override_set.blind_spots)

    def test_every_entry_carries_a_reason(self) -> None:
        """An override without a why is a magic number, and it will outlive the
        situation that justified it."""
        for entry in OverrideSet.from_config().overrides:
            assert entry.reason.strip(), f"{entry.player} has no reason"

    def test_the_seeded_overrides_are_the_six_injury_rows(self) -> None:
        names = {entry.player for entry in OverrideSet.from_config().overrides}
        assert names == {
            "George Kittle",
            "Tucker Kraft",
            "Ashton Jeanty",
            "Patrick Mahomes",
            "Daniel Jones",
            "DeVonta Smith",
        }

    def test_no_duplicate_keys(self) -> None:
        overrides = OverrideSet.from_config().overrides
        keys = [entry.key for entry in overrides]
        assert len(keys) == len(set(keys)), "a duplicated key would silently win or lose"

    def test_raw_config_has_the_expected_sections(self) -> None:
        config = load_overrides_config()
        assert set(config) >= {"overrides", "blind_spots", "removed"}
