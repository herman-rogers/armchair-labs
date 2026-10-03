"""Unit tests for value over replacement."""

from __future__ import annotations

import polars as pl
import pytest

from engine.metrics.vor import (
    REPLACEMENT_PPG,
    VOR,
    ReplacementLevelError,
    add_vor,
    replacement_levels,
)


def pool(position: str, ppgs: list[float], games: int | list[int] = 17) -> pl.DataFrame:
    game_counts = [games] * len(ppgs) if isinstance(games, int) else games
    return pl.DataFrame(
        {
            "player_id": [f"{position}{i}" for i in range(len(ppgs))],
            "position": [position] * len(ppgs),
            "ppg": ppgs,
            "games": game_counts,
        }
    )


class TestReplacementLevels:
    def test_baseline_is_the_nth_ranked_player(self) -> None:
        # Descending PPG; the 3rd-best is 12.0.
        levels = replacement_levels(pool("RB", [20.0, 15.0, 12.0, 9.0]), {"RB": 3})
        assert levels["RB"] == pytest.approx(12.0)

    def test_input_order_does_not_matter(self) -> None:
        levels = replacement_levels(pool("RB", [9.0, 20.0, 12.0, 15.0]), {"RB": 3})
        assert levels["RB"] == pytest.approx(12.0)

    def test_a_short_pool_falls_back_to_the_worst_eligible_player(self) -> None:
        """Fewer eligible players than the baseline rank means the position is thinner
        than the league is deep. The worst of them is then the honest replacement."""
        levels = replacement_levels(pool("TE", [11.0, 9.0]), {"TE": 12})
        assert levels["TE"] == pytest.approx(9.0)

    def test_positions_are_computed_independently(self) -> None:
        frame = pl.concat([pool("RB", [20.0, 12.0]), pool("WR", [18.0, 8.0])])
        levels = replacement_levels(frame, {"RB": 2, "WR": 2})

        assert levels == {"RB": pytest.approx(12.0), "WR": pytest.approx(8.0)}

    def test_an_empty_position_raises_rather_than_returning_zero(self) -> None:
        with pytest.raises(ReplacementLevelError, match="no QB met"):
            replacement_levels(pool("RB", [20.0]), {"QB": 12})


class TestGamesEligibility:
    def test_small_samples_cannot_set_the_baseline(self) -> None:
        """A three-game cameo at 30 PPG must not define replacement level; it would
        inflate the bar and deflate every VOR at the position at once."""
        frame = pool("RB", [30.0, 20.0, 12.0], games=[3, 17, 17])

        levels = replacement_levels(frame, {"RB": 2}, min_games=8)
        assert levels["RB"] == pytest.approx(12.0), "the 3-game player should be excluded"

        unfiltered = replacement_levels(frame, {"RB": 2}, min_games=0)
        assert unfiltered["RB"] == pytest.approx(20.0)

    def test_the_threshold_is_inclusive(self) -> None:
        frame = pool("RB", [20.0, 12.0], games=[8, 17])
        assert replacement_levels(frame, {"RB": 1}, min_games=8)["RB"] == pytest.approx(20.0)


class TestAddVor:
    def test_vor_is_ppg_minus_replacement(self) -> None:
        scored = add_vor(pool("RB", [20.0, 12.0]), {"RB": 12.3})

        assert scored[REPLACEMENT_PPG].to_list() == [pytest.approx(12.3)] * 2
        assert scored[VOR].to_list() == [pytest.approx(7.7), pytest.approx(-0.3)]

    def test_scarcity_can_outrank_raw_points(self) -> None:
        """The whole reason the board sorts on VOR, taken straight from the 2026 board:
        McBride (TE, 18.6 PPG) ranks #9 and St. Brown (WR, 19.1 PPG) ranks #10. The
        tight end scores less and is worth more, because replacement tight end is 10.6
        and replacement receiver is 11.3."""
        frame = pl.concat([pool("TE", [18.6]), pool("WR", [19.1])])
        scored = add_vor(frame, {"TE": 10.6, "WR": 11.3}).sort(VOR, descending=True)

        assert scored["position"].to_list() == ["TE", "WR"]
        assert scored[VOR].to_list() == [pytest.approx(8.0), pytest.approx(7.8)]

    def test_positions_without_a_baseline_get_a_null_vor(self) -> None:
        """Kickers and defenses are tiered, not VOR-ranked; they must drop out of the
        sort rather than sorting as if replacement were zero."""
        scored = add_vor(pl.concat([pool("RB", [20.0]), pool("K", [9.0])]), {"RB": 12.3})
        kicker = scored.filter(pl.col("position") == "K")

        assert kicker[VOR][0] is None
        assert kicker[REPLACEMENT_PPG][0] is None
