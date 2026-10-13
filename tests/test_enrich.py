"""DuckDB enrichment: cache resolution, league scoring in SQL, position repair, bonuses."""

import duckdb
import polars as pl
import pytest

from engine.data.enrich import build_observations, register_raw
from engine.data.enrich.scoring_sql import component_points_sql, touchdown_bonus_sql
from engine.data.enrich.sources import DATASETS, RawSources, dataset_key
from engine.scoring.engine import score_components

STAT_COLUMNS = {
    "completions": 0,
    "attempts": 0,
    "passing_yards": 0,
    "passing_tds": 0,
    "passing_interceptions": 0,
    "passing_2pt_conversions": 0,
    "sack_fumbles_lost": 0,
    "carries": 0,
    "rushing_yards": 0,
    "rushing_tds": 0,
    "rushing_fumbles_lost": 0,
    "rushing_2pt_conversions": 0,
    "receptions": 0,
    "targets": 0,
    "receiving_yards": 0,
    "receiving_tds": 0,
    "receiving_fumbles_lost": 0,
    "receiving_2pt_conversions": 0,
    "special_teams_tds": 0,
    "fantasy_points_ppr": 0.0,
}


def week(player_id, name, position, season, week, team="SF", **stats):
    return {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "season": season,
        "week": week,
        "season_type": "REG",
        "team": team,
        **{**STAT_COLUMNS, **stats},
    }


@pytest.fixture
def snapshot(tmp_path):
    """A raw snapshot whose nflverse files sit where nflreadpy's cache would put them."""
    data = tmp_path / "data"
    cache = data / "raw/objects"
    cache.mkdir(parents=True)
    assets = {}

    def preserve(dataset, frame, season=None):
        name = f"history/cache/nflverse/{dataset_key(dataset, season)}.parquet"
        path = cache / name.rsplit("/", 1)[1]
        frame.write_parquet(path)
        assets[name] = {"object": str(path.relative_to(data)), "sha256": "x", "bytes": 1}

    stats = pl.DataFrame(
        [
            week(
                "a",
                "Back",
                "HB",
                2025,
                1,
                rushing_yards=100,
                rushing_tds=1,
                receptions=2,
                receiving_yards=20,
            ),
            week("a", "Back", "HB", 2025, 2, rushing_yards=50),
            week(
                "q",
                "Passer",
                "QB",
                2025,
                1,
                passing_yards=300,
                passing_tds=2,
                passing_interceptions=1,
            ),
            week("w", "Wide", "WR", 2025, 1, receptions=5, receiving_yards=120, receiving_tds=1),
            week("h", "Two Way", "CB", 2025, 1, receptions=3, receiving_yards=40),
            week("a", "Back", "HB", 2026, 1, rushing_yards=999),  # current season: excluded
        ]
    )
    preserve("stats_player_week", stats.filter(pl.col("season") == 2025), 2025)
    preserve("stats_player_week", stats.filter(pl.col("season") == 2026), 2026)
    plays = pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 1, 2],
            "season_type": ["REG", "REG", "POST", "REG"],
            "touchdown": [1, 1, 1, 1],
            "pass_touchdown": [1, 0, 1, 0],
            "rush_touchdown": [0, 1, 0, 1],
            "yards_gained": [55, 45, 70, 12],
            "passer_player_id": ["q", None, "q", None],
            "receiver_player_id": ["w", None, "w", None],
            "rusher_player_id": [None, "a", None, "a"],
        }
    )
    preserve("pbp", plays, 2025)
    rosters = pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 2, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "gsis_id": ["a", "a", "a", "w"],
            "position": ["RB", "RB", "WR", "WR"],
        }
    )
    preserve("roster_weekly", rosters, 2025)
    for season in (2025, 2026):
        preserve(
            "snap_counts",
            pl.DataFrame(
                {
                    "season": [season],
                    "week": [1],
                    "game_id": ["g"],
                    "pfr_player_id": ["p"],
                    "position": ["FB"],
                }
            ),
            season,
        )
        preserve(
            "injuries",
            pl.DataFrame(
                {
                    "season": [season],
                    "week": [1],
                    "team": ["SF"],
                    "gsis_id": ["a"],
                    "report_status": ["Questionable"],
                }
            ),
            season,
        )
    preserve(
        "players",
        pl.DataFrame(
            {"gsis_id": ["a", "q", "w", "h"], "display_name": ["Back", "Passer", "Wide", "Two Way"]}
        ),
    )
    preserve(
        "schedules",
        pl.DataFrame(
            {
                "game_id": ["g"],
                "season": [2025],
                "week": [1],
                "game_type": ["REG"],
                "gameday": ["2025-09-07"],
                "home_team": ["SF"],
                "away_team": ["LA"],
            }
        ),
    )
    preserve(
        "ff_playerids",
        pl.DataFrame({"fantasypros_id": ["100"], "gsis_id": ["a"], "espn_id": [1]}),
    )
    manifest = {"assets": assets, "current_observations": {"season": 2026, "through_week": 1}}
    return data, manifest


def test_cache_keys_follow_nflreadpy_and_discover_seasons(snapshot):
    data, manifest = snapshot
    sources = RawSources(data, manifest)
    assert sources.seasons("stats_player_week") == [2025, 2026]
    assert sources.seasons("pbp") == [2025]
    assert sources.nflverse("players").exists()
    assert dataset_key("players") != dataset_key("schedules")
    with pytest.raises(ValueError):
        dataset_key("players", 2025)
    assert set(DATASETS) >= {"stats_player_week", "pbp", "roster_weekly", "players"}


def test_sql_scoring_matches_the_polars_rules_and_bonuses_credit_both_players(
    snapshot, monkeypatch
):
    data, manifest = snapshot
    from engine.config import league

    monkeypatch.setattr(
        league,
        "get_league",
        lambda: league.LeagueConfig.model_validate(
            {**league.load_league_config(), "position_overrides": {"h": "WR"}}
        ),
    )
    import engine.data.enrich as enrich

    monkeypatch.setattr(enrich, "get_league", league.get_league)
    db = duckdb.connect()
    coverage = register_raw(db, RawSources(data, manifest))
    assert coverage["stats_player_week"] == 2
    audit = build_observations(db, current_season=2026)
    weeks = db.execute("SELECT * FROM nfl_player_weeks ORDER BY player_id, week").pl()
    assert weeks["season"].max() == 2025, "the current season belongs to the weekly capture"
    # Scoring: SQL from scoring.yaml equals the Polars rules path, plus the bonus.
    expected = score_components(weeks.select(list(STAT_COLUMNS)))["component_pts"]
    assert weeks["component_pts"].to_list() == pytest.approx(expected.to_list())
    by = {(r["player_id"], r["week"]): r for r in weeks.to_dicts()}
    assert by[("q", 1)]["bonus_pts"] == 3.0 and by[("w", 1)]["bonus_pts"] == 3.0  # 55-yard pass
    assert by[("a", 1)]["bonus_pts"] == 2.0  # 45-yard rush; the 12-yard score pays nothing
    assert by[("a", 2)]["bonus_pts"] == 0.0
    assert by[("q", 1)]["league_points"] == pytest.approx(300 * 0.04 + 8 - 2 + 3)
    assert audit["balanced"] and audit["scoring_plays"] == 2 and audit["dropped_credits"] == 0
    # Positions: FB/HB canonical, weekly roster wins, conflicts stay with the stat feed.
    assert by[("a", 1)]["position"] == "RB" and by[("a", 1)]["position_source"] == "weekly_roster"
    assert (
        by[("a", 2)]["position"] == "RB" and by[("a", 2)]["position_source"] == "stat_feed_fallback"
    )
    assert by[("h", 1)]["position"] == "WR" and by[("h", 1)]["position_source"] == "league_override"
    assert by[("a", 1)]["raw_position"] == "RB"
    seasons = {
        r["player_id"]: r for r in db.execute("SELECT * FROM nfl_player_seasons").pl().to_dicts()
    }
    assert seasons["a"]["games"] == 2 and seasons["a"]["rushing_yards"] == 150
    assert seasons["a"]["total_pts"] == pytest.approx(
        by[("a", 1)]["league_points"] + by[("a", 2)]["league_points"]
    )
    assert seasons["a"]["ppg"] == pytest.approx(seasons["a"]["total_pts"] / 2)
    # Participation and injury history stop at completed seasons like the player weeks.
    assert db.execute("SELECT position, season FROM nfl_snap_counts").fetchall() == [("RB", 2025)]
    assert db.execute("SELECT season FROM nfl_injuries").fetchall() == [(2025,)]


def test_sql_expressions_are_generated_from_the_rules():
    assert "coalesce(passing_yards, 0) * 0.04" in component_points_sql()
    bonus = touchdown_bonus_sql()
    assert bonus.index("yards_gained >= 50") < bonus.index(
        "yards_gained >= 40 AND yards_gained <= 49"
    )
