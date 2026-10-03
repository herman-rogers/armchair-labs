"""Unit tests for fixture parsing and the staged comparison.

These run offline against the committed board file — no nflverse access — so the
validation machinery itself is verified independently of whether a live build agrees
with it.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from engine.validation import (
    _spearman,
    compare_to_fixture,
    fixture_replacement_levels,
    load_fixture_board,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "static" / "2026_draft_list.md"


@pytest.fixture(scope="module")
def fixture_board() -> pl.DataFrame:
    return load_fixture_board(FIXTURE)


class TestFixtureParsing:
    def test_every_ranked_row_is_parsed(self, fixture_board: pl.DataFrame) -> None:
        assert fixture_board.height == 160

    def test_ranks_are_complete_and_contiguous(self, fixture_board: pl.DataFrame) -> None:
        assert fixture_board["fixture_rank"].to_list() == list(range(1, 161))

    def test_the_top_of_the_board(self, fixture_board: pl.DataFrame) -> None:
        top = fixture_board.head(3).to_dicts()

        assert top[0]["player"] == "Puka Nacua"
        assert top[0]["fixture_vor"] == pytest.approx(12.4)
        assert top[0]["fixture_ppg"] == pytest.approx(23.7)
        assert top[1]["player"] == "Christian McCaffrey"

    def test_the_anchors_named_in_the_design_doc(self, fixture_board: pl.DataFrame) -> None:
        """The doc cites CMC #2, McBride #9, Stafford #33. The first two hold; Stafford
        is actually #31 in the published file, which is why the file is treated as
        authoritative over the doc."""
        by_name = {row["player"]: row["fixture_rank"] for row in fixture_board.to_dicts()}

        assert by_name["Christian McCaffrey"] == 2
        assert by_name["Trey McBride"] == 9
        assert by_name["Matthew Stafford"] == 31

    def test_flags_are_captured(self, fixture_board: pl.DataFrame) -> None:
        by_name = {row["player"]: row["fixture_flags"] for row in fixture_board.to_dicts()}

        assert by_name["CeeDee Lamb"] == "BUY"
        assert by_name["Jahmyr Gibbs"] == "TD-luck"
        assert by_name["Jonathan Taylor"] == "TD-luck/age"
        assert by_name["Puka Nacua"] == ""

    def test_notes_survive(self, fixture_board: pl.DataFrame) -> None:
        kittle = fixture_board.filter(pl.col("player") == "George Kittle").to_dicts()[0]
        assert "ACHILLES" in kittle["notes"]

    def test_negative_vor_parses(self, fixture_board: pl.DataFrame) -> None:
        assert fixture_board["fixture_vor"].min() < 0

    def test_an_unparseable_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.md"
        empty.write_text("# No table here\n")
        with pytest.raises(ValueError, match="no board rows parsed"):
            load_fixture_board(empty)


class TestRecoveredBaselines:
    def test_baselines_come_back_exactly(self, fixture_board: pl.DataFrame) -> None:
        """VOR is `ppg - replacement`, so the baselines the published board used are
        recoverable from the artifact itself. This is what makes the baseline stage the
        sharpest check in the suite: the expected value is derived, not assumed."""
        levels = fixture_replacement_levels(fixture_board)

        assert levels["QB"] == pytest.approx(17.8, abs=0.05)
        assert levels["RB"] == pytest.approx(12.3, abs=0.05)
        assert levels["WR"] == pytest.approx(11.3, abs=0.05)
        assert levels["TE"] == pytest.approx(10.6, abs=0.05)

    def test_the_identity_holds_on_individual_rows(self, fixture_board: pl.DataFrame) -> None:
        """Tolerance is 0.11 because the source rounds both PPG and VOR to one decimal
        independently, so their difference can drift a full 0.1 from the true baseline
        before anything is actually wrong. Taking the median across a position, as
        `fixture_replacement_levels` does, is what absorbs that."""
        levels = fixture_replacement_levels(fixture_board)
        for row in fixture_board.head(20).iter_rows(named=True):
            implied = row["fixture_ppg"] - row["fixture_vor"]
            assert implied == pytest.approx(levels[row["position"]], abs=0.11)


class TestSpearman:
    def test_perfect_agreement(self) -> None:
        assert _spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        assert _spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_monotone_rescaling_does_not_matter(self) -> None:
        assert _spearman([1.0, 2.0, 3.0], [10.0, 200.0, 3000.0]) == pytest.approx(1.0)

    def test_ties_are_averaged(self) -> None:
        assert _spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)

    def test_a_single_point_is_undefined(self) -> None:
        import math

        assert math.isnan(_spearman([1.0], [1.0]))


class TestComparison:
    def _board_from(self, fixture: pl.DataFrame) -> pl.DataFrame:
        """A synthetic computed board that reproduces the fixture exactly."""
        return fixture.select(
            pl.col("player").alias("player_display_name"),
            "position",
            pl.col("fixture_ppg").alias("ppg"),
            pl.col("fixture_vor").alias("vor"),
            pl.col("fixture_flags").alias("flags"),
            pl.col("fixture_rank").alias("rank"),
        )

    def test_a_perfect_rebuild_passes_every_stage(self, fixture_board: pl.DataFrame) -> None:
        report = compare_to_fixture(
            self._board_from(fixture_board),
            fixture_board,
            replacement_levels=fixture_replacement_levels(fixture_board),
        )

        assert report.passed, report.render()
        assert {stage.name for stage in report.stages} >= {
            "replacement baselines",
            "coverage",
            "per-player PPG",
            "per-player VOR",
            "ranking",
            "flags",
        }

    def test_a_wrong_baseline_fails_the_baseline_stage(self, fixture_board: pl.DataFrame) -> None:
        levels = fixture_replacement_levels(fixture_board)
        report = compare_to_fixture(
            self._board_from(fixture_board), fixture_board, {**levels, "RB": 15.0}
        )
        baseline = next(s for s in report.stages if s.name == "replacement baselines")

        assert not baseline.passed
        assert any("OUT OF TOLERANCE" in row for row in baseline.rows)

    def test_shifted_scoring_fails_ppg_and_names_the_worst(
        self, fixture_board: pl.DataFrame
    ) -> None:
        board = self._board_from(fixture_board).with_columns(pl.col("ppg") + 1.0)
        report = compare_to_fixture(board, fixture_board, fixture_replacement_levels(fixture_board))
        ppg_stage = next(s for s in report.stages if s.name == "per-player PPG")

        assert not ppg_stage.passed
        assert ppg_stage.rows, "the worst offenders should be listed, not just counted"
        assert not report.passed

    def test_missing_players_show_up_as_coverage_loss(self, fixture_board: pl.DataFrame) -> None:
        """A silent join failure is the §8 failure mode. It must reduce coverage
        visibly rather than quietly shrinking the sample later stages average over."""
        board = self._board_from(fixture_board).head(100)
        report = compare_to_fixture(board, fixture_board, fixture_replacement_levels(fixture_board))
        coverage = next(s for s in report.stages if s.name == "coverage")

        assert not coverage.passed
        assert "100/160" in coverage.detail
        assert coverage.rows

    def test_flag_disagreement_is_reported_but_not_fatal(self, fixture_board: pl.DataFrame) -> None:
        """Flags are thresholds on a continuous metric; a player a hair either side of
        the line flips without anything being wrong."""
        board = self._board_from(fixture_board).with_columns(pl.lit("").alias("flags"))
        report = compare_to_fixture(board, fixture_board, fixture_replacement_levels(fixture_board))
        flags = next(s for s in report.stages if s.name == "flags")

        assert flags.passed, "flag disagreement must stay advisory"
        assert flags.rows, "but it must still be reported"
        assert report.passed

    def test_names_join_through_normalization(self, fixture_board: pl.DataFrame) -> None:
        """The published board writes "Harold Fannin Jr."; nflverse may not."""
        board = self._board_from(fixture_board).with_columns(
            pl.col("player_display_name").str.replace_all(r" (Jr\.|Sr\.|III|II)$", "")
        )
        report = compare_to_fixture(board, fixture_board, fixture_replacement_levels(fixture_board))
        coverage = next(s for s in report.stages if s.name == "coverage")

        assert coverage.passed
        assert "160/160" in coverage.detail

    def test_report_renders_readably(self, fixture_board: pl.DataFrame) -> None:
        report = compare_to_fixture(
            self._board_from(fixture_board),
            fixture_board,
            fixture_replacement_levels(fixture_board),
        )
        rendered = report.render()

        assert "[PASS] replacement baselines" in rendered
        assert "all stages passed" in rendered
