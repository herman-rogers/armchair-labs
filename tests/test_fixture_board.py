"""End-to-end validation: a live rebuild against the published draft board.

This is Phase 1's actual deliverable expressed as a test. It builds the board from
current nflverse data and runs the staged comparison in `engine.validation` against
`data/static/2026_draft_list.md`.

Network-marked and slow (it pulls three seasons of play-by-play on a cold cache).
Run with `just test-all`.

The tolerances are bands, not targets. The published board was hand-finished on
Aug 27 2026 against a play-by-play snapshot that cannot be pinned, and it rounds every
number to one decimal. Chasing an exact match would mean tuning the model to noise.
"""

from __future__ import annotations

import pytest

from engine.board import pipeline
from engine.config.league import get_league
from engine.config.settings import get_settings
from engine.validation import compare_to_fixture, load_fixture_board

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def built():
    return pipeline.build(config=get_league(), settings=get_settings())


@pytest.fixture(scope="module")
def fixture():
    return load_fixture_board(get_settings().static_dir / "2026_draft_list.md")


@pytest.fixture(scope="module")
def report(built, fixture):
    return compare_to_fixture(built.board, fixture, replacement_levels=built.replacement_levels)


class TestStagedComparison:
    def test_replacement_baselines_match(self, report) -> None:
        """The sharpest check in the suite. It isolates scoring from ranking: if the
        baselines are right, points per game are right across the whole pool."""
        stage = next(s for s in report.stages if s.name == "replacement baselines")
        assert stage.passed, stage.render()

    def test_every_published_player_is_found(self, report) -> None:
        """A silent join failure is the §8 tax. Anything under full coverage means the
        rebuild is quietly omitting players the published board ranked."""
        stage = next(s for s in report.stages if s.name == "coverage")
        assert stage.passed, stage.render()

    def test_per_player_ppg_is_within_tolerance(self, report) -> None:
        stage = next(s for s in report.stages if s.name == "per-player PPG")
        assert stage.passed, stage.render()

    def test_per_player_vor_is_within_tolerance(self, report) -> None:
        stage = next(s for s in report.stages if s.name == "per-player VOR")
        assert stage.passed, stage.render()

    def test_the_ranking_reproduces(self, report) -> None:
        stage = next(s for s in report.stages if s.name == "ranking")
        assert stage.passed, stage.render()

    def test_the_whole_report_passes(self, report) -> None:
        assert report.passed, report.render()


class TestNamedAnchors:
    """Spot checks the design doc calls out by name, asserted directly rather than
    through an aggregate, so a regression says which player moved."""

    def _rank_of(self, board, name: str) -> int:
        rows = board.filter(board["player_display_name"] == name)
        assert rows.height == 1, f"expected exactly one {name} on the board"
        return int(rows["rank"][0])

    def test_the_top_of_the_board(self, built) -> None:
        top = built.board.head(2)["player_display_name"].to_list()
        assert top == ["Puka Nacua", "Christian McCaffrey"]

    def test_mcbride_ranks_ninth(self, built) -> None:
        """The positional-scarcity result: a tight end in the top ten, which is the
        whole reason the board sorts on VOR rather than points."""
        assert self._rank_of(built.board, "Trey McBride") == 9

    def test_the_bonus_edge_lifts_stafford(self, built) -> None:
        """The league's private edge, made visible: big-play bonuses are why a
        deep-ball quarterback outranks his consensus price here."""
        assert self._rank_of(built.board, "Matthew Stafford") <= 35


class TestOverridesReorderTheBoard:
    def test_kittle_is_demoted_by_the_achilles_override(self, built) -> None:
        """Kittle's raw VOR puts him around 22nd. The published board has him 53rd,
        because a torn Achilles is not in the 2025 tape. That gap is the override file
        doing its job."""
        kittle = built.board.filter(
            built.board["player_display_name"] == "George Kittle"
        ).to_dicts()[0]

        assert kittle["override_delta"] < 0
        assert "Achilles" in kittle["override_reason"]
        assert kittle["adj_vor"] < kittle["vor"]
        assert kittle["rank"] > 40, "the override should have pushed him well down"

    def test_removed_players_are_absent(self, built) -> None:
        names = set(built.board["player_display_name"].to_list())
        assert "Ricky Pearsall" not in names


class TestBonusIntegrity:
    def test_the_bonus_reconciliation_balances(self, built) -> None:
        """Every bonus point earned on the field reached a player. If this fails,
        points are being lost to null player ids and the deep threats this league
        rewards are being undervalued."""
        if built.bonus_audit is None:
            pytest.skip("bonuses came from cache; the audit runs when they are built")
        assert built.bonus_audit.is_balanced, built.bonus_audit.summary()
        assert built.bonus_audit.dropped_credits == 0


class TestSupportingBoards:
    def test_kickers_are_ranked_and_spread(self, built) -> None:
        """The distance brackets are supposed to create a real gap at a position the
        room treats as random."""
        kickers = built.kickers.filter(built.kickers["games"] >= 8)
        assert kickers.height > 20

        spread = float(kickers["ppg"].max()) - float(kickers["ppg"].min())
        assert spread > 2.0, f"expected a meaningful kicker spread, got {spread:.1f}"

    def test_every_defense_is_scored(self, built) -> None:
        assert built.defenses.height == 32
        assert built.defenses["avg_pts_allowed"].null_count() == 0


def test_position_overrides_reclassify_two_way_players(league_config) -> None:
    """A receiver nflverse files as CB reaches the board when league.yaml says so."""
    import polars as pl

    from engine.board.builder import build_player_seasons

    weeks = pl.DataFrame(
        {
            "player_id": ["00-0040718", "00-0040718", "other"],
            "player_display_name": ["Travis Hunter", "Travis Hunter", "Some WR"],
            "position": ["CB", "CB", "WR"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 1],
            "team": ["JAX", "JAX", "SEA"],
            "targets": [8, 6, 4],
            "receptions": [5, 4, 3],
            "receiving_yards": [33, 22, 40],
            "receiving_tds": [0, 1, 0],
            "carries": [0, 0, 0],
            "rushing_yards": [0, 0, 0],
            "rushing_tds": [0, 0, 0],
            "attempts": [0, 0, 0],
            "passing_yards": [0, 0, 0],
            "passing_tds": [0, 0, 0],
            "passing_interceptions": [0, 0, 0],
        }
    )
    bonuses = pl.DataFrame(
        {"season": [2025], "week": [1], "player_id": ["other"], "bonus_pts": [0.0]}
    )
    plain = build_player_seasons(
        weeks, bonuses, league_config.model_copy(update={"position_overrides": {}})
    )
    fixed = build_player_seasons(
        weeks,
        bonuses,
        league_config.model_copy(update={"position_overrides": {"00-0040718": "WR"}}),
    )
    assert "00-0040718" not in plain["player_id"].to_list()
    hunter = fixed.filter(pl.col("player_id") == "00-0040718").to_dicts()[0]
    assert hunter["position"] == "WR" and hunter["games"] == 2 and hunter["targets"] == 14
