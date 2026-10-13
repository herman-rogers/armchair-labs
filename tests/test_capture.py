"""Provider capture: carry-forward by hash, completed-week detection, scored current tables."""

import json

import duckdb
import polars as pl
import pytest

from engine.data import capture
from engine.data.enrich.sources import DATASETS, RawSources, asset_name


def frame_bytes(frame: pl.DataFrame) -> bytes:
    import io

    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def stats(season, weeks, player="a", team="SF", position="RB"):
    return pl.DataFrame(
        [
            {
                "player_id": player,
                "player_display_name": "Back",
                "position": position,
                "season": season,
                "week": week,
                "season_type": "REG",
                "team": team,
                "attempts": 0,
                "completions": 0,
                "passing_yards": 0,
                "passing_tds": 0,
                "passing_interceptions": 0,
                "passing_2pt_conversions": 0,
                "sack_fumbles_lost": 0,
                "carries": 10,
                "rushing_yards": 50,
                "rushing_tds": 1,
                "rushing_fumbles_lost": 0,
                "rushing_2pt_conversions": 0,
                "receptions": 2,
                "targets": 3,
                "receiving_yards": 20,
                "receiving_tds": 0,
                "receiving_fumbles_lost": 0,
                "receiving_2pt_conversions": 0,
                "special_teams_tds": 0,
                "fantasy_points_ppr": 0.0,
            }
            for week in weeks
        ]
    )


def schedule(season, finished_weeks, scheduled_weeks):
    rows = []
    for week in scheduled_weeks:
        done = week in finished_weeks
        rows.append(
            {
                "game_id": f"{season}_{week}_LA_SF",
                "season": season,
                "week": week,
                "game_type": "REG",
                "gameday": f"{season}-09-{week:02d}",
                "home_team": "SF",
                "away_team": "LA",
                "home_score": 20 if done else None,
                "away_score": 17 if done else None,
            }
        )
    return pl.DataFrame(rows)


def provider(season, through):
    """Fake provider: every dataset URL answers with a small parquet or csv payload."""
    weeks = list(range(1, through + 1))
    both = pl.concat([stats(season, weeks), stats(season, weeks, player="b", team="LA")])
    files = {
        "stats_player_week": lambda year: frame_bytes(
            both if year == season else stats(year, range(1, 18))
        ),
        "pbp": lambda year: frame_bytes(
            pl.DataFrame(
                {
                    "season": [year],
                    "week": [1],
                    "season_type": ["REG"],
                    "touchdown": [1],
                    "pass_touchdown": [0],
                    "rush_touchdown": [1],
                    "yards_gained": [45],
                    "passer_player_id": [None],
                    "receiver_player_id": [None],
                    "rusher_player_id": ["a"],
                }
            )
        ),
        "roster_weekly": lambda year: frame_bytes(
            pl.DataFrame(
                {
                    "season": [year],
                    "week": [1],
                    "game_type": ["REG"],
                    "gsis_id": ["a"],
                    "position": ["RB"],
                }
            )
        ),
        "snap_counts": lambda year: frame_bytes(
            pl.DataFrame(
                {
                    "season": [year, year],
                    "week": [1, 1],
                    "game_type": ["REG", "REG"],
                    "game_id": ["g", "g"],
                    "pfr_player_id": ["pa", "pb"],
                    "position": ["RB", "RB"],
                    "offense_snaps": [40.0, 30.0],
                    "offense_pct": [0.8, 0.6],
                }
            )
        ),
        "injuries": lambda year: frame_bytes(
            pl.DataFrame(
                {
                    "season": [year],
                    "week": [1],
                    "team": ["SF"],
                    "gsis_id": ["a"],
                    "report_status": ["Out"],
                }
            )
        ),
        "players": lambda _: frame_bytes(
            pl.DataFrame(
                {
                    "gsis_id": ["a", "b", "r"],
                    "display_name": ["Back", "Other", "Rookie"],
                    "position": ["RB", "RB", "WR"],
                    "pfr_id": ["pa", "pb", None],
                    "espn_id": ["1", "2", "3"],
                    "rookie_season": [2020, 2021, season],
                    "draft_year": [2020, 2021, season],
                    "draft_pick": [5, 40, 12],
                    "latest_team": ["SF", "LA", "DAL"],
                }
            )
        ),
        "schedules": lambda _: frame_bytes(
            pl.concat(
                [
                    schedule(season, range(1, through + 1), range(1, 18)),
                    schedule(season - 1, range(1, 18), range(1, 18)),
                ]
            )
        ),
        "ff_playerids": lambda _: b"fantasypros_id,gsis_id,espn_id\n100,a,1\n",
        "ff_rankings": lambda _: frame_bytes(
            pl.DataFrame(
                {
                    "scrape_date": [f"{season}-09-10"],
                    "page_type": ["redraft-overall"],
                    "ecr_type": ["ro"],
                    "id": ["100"],
                    "player": ["Back"],
                    "pos": ["RB"],
                    "team": ["SF"],
                    "ecr": [3.5],
                    "sd": [1.0],
                    "best": [1.0],
                    "worst": [8.0],
                    "rank_delta": [0.0],
                }
            )
        ),
    }
    calls = []

    def fetch(url):
        calls.append(url)
        for dataset, (template, _) in DATASETS.items():
            prefix = template.split("{season}")[0] if "{season}" in template else template
            if url.startswith(prefix) and ("{season}" in template or url == template):
                year = int(url[len(prefix) :].split(".")[0]) if "{season}" in template else None
                return files[dataset](year)
        raise AssertionError(f"Unexpected download {url}")

    return fetch, calls


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    static = tmp_path / "repo/data/static"
    static.mkdir(parents=True)
    (static / "historical_absences.json").write_text(json.dumps({"events": []}))
    (static / "college_identity_overrides.json").write_text("[]")
    monkeypatch.setattr(capture, "REPO_ROOT", tmp_path / "repo")
    college = data / "research/college_src"
    (college / "raw").mkdir(parents=True)
    (college / "raw/players_2025.parquet").write_bytes(frame_bytes(pl.DataFrame({"x": [1]})))
    from engine.data.releases import digest

    json.dump(
        {
            "status": "captured",
            "kind": "college_source",
            "files": [
                {
                    "filename": "players_2025.parquet",
                    "sha256": digest(college / "raw/players_2025.parquet"),
                }
            ],
        },
        (college / "manifest.json").open("w"),
    )
    monkeypatch.setattr(capture, "SEASON_START", {k: 2025 for k in capture.SEASON_START})
    return data


def test_capture_scores_current_weeks_and_carries_history_forward(workspace):
    data = workspace
    fetch, calls = provider(2026, through=3)
    first = capture.capture_raw(data, "snap_a", 2026, "college_src", fetcher=fetch)
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["current_observations"] == {
        "season": 2026,
        "through_week": 3,
        "saved_at": manifest["current_observations"]["saved_at"],
    }
    assert manifest["carried_forward_assets"] == 0
    sources = RawSources(data, manifest)
    assert sources.seasons("stats_player_week") == [2025, 2026]
    weeks = pl.read_parquet(sources.current("weeks.parquet"))
    assert weeks["week"].max() == 3 and set(weeks["player_id"]) == {"a", "b"}
    assert weeks.filter((pl.col("player_id") == "a") & (pl.col("week") == 1))["bonus_pts"][0] == 2.0
    assert weeks["league_points"].dtype == pl.Float64
    snaps = pl.read_parquet(sources.current("snaps.parquet"))
    assert snaps["player_id"].to_list() == ["a", "b"]
    players = pl.read_parquet(sources.current("players.parquet"))
    assert players["player_id"].to_list() == ["r"] and players["espn_id"][0] == 3
    assert asset_name("ff_playerids").endswith(".csv") and sources.has("ff_rankings")
    # Second capture a week later: completed seasons are carried forward, not re-fetched.
    fetch2, calls2 = provider(2026, through=4)
    second = capture.capture_raw(
        data, "snap_b", 2026, "college_src", previous=manifest, fetcher=fetch2
    )
    again = json.loads((second / "manifest.json").read_text())
    assert again["carried_forward_assets"] == 5  # one 2025 file per per-season dataset
    assert again["current_observations"]["through_week"] == 4
    assert again["previous_snapshot"] == "snap_a"
    assert not any("2025" in url for url in calls2)
    assert capture.latest_snapshot(data)["version"] == "snap_b"


def test_completed_cutoff_requires_scores_and_stat_coverage():
    db = duckdb.connect()
    sched = schedule(2026, finished_weeks=[1, 2, 3], scheduled_weeks=range(1, 6))
    db.register("raw_schedules", sched.to_arrow())
    # Week 3 is final but LA has no stat rows, so the cutoff stops at week 2.
    both = pl.concat([stats(2026, [1, 2, 3]), stats(2026, [1, 2], player="b", team="LA")])
    db.register("raw_stats_player_week", both.to_arrow())
    assert capture.completed_cutoff(db, 2026) == 2


def test_completed_week_uses_only_the_two_small_provider_files():
    fetch, calls = provider(2026, through=2)
    assert capture.completed_week(2026, fetcher=fetch) == 2
    assert len(calls) == 2
