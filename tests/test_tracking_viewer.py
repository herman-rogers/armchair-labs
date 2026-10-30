"""Game engine replay data: both archive adapters and the API routes over them."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.api import tracking_routes
from engine.data import tracking_viewer
from engine.data.releases import TableRelease

GOLD_COLUMNS = {
    "speed": pl.Float64,
    "acceleration": pl.Float64,
    "orientation": pl.Float64,
    "direction": pl.Float64,
    "event": pl.Utf8,
}


def _release(root: Path, tables: dict[str, pl.DataFrame], roles: dict[str, str] | None = None):
    root.mkdir(parents=True)
    manifest: dict = {"version": root.name, "tables": {}, "files": {}}
    for name, frame in tables.items():
        path = root / f"{name}.parquet"
        frame.write_parquet(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["tables"][name] = {
            "path": path.name,
            "sha256": digest,
            "role": (roles or {}).get(name, "x"),
        }
        manifest["files"][path.name] = digest
    (root / "manifest.json").write_text(json.dumps(manifest))
    return TableRelease(root=root, manifest=manifest)


def _frames(rows: list[dict], **defaults) -> pl.DataFrame:
    base = {
        "speed": None,
        "acceleration": None,
        "orientation": None,
        "direction": None,
        "event": None,
        "entity_type": "player",
        "play_direction": "right",
    }
    return pl.DataFrame(
        [{**base, **defaults, **r} for r in rows],
        schema_overrides=GOLD_COLUMNS,
    )


@pytest.fixture
def sumer(tmp_path):
    games = pl.DataFrame(
        {
            "gameId": [2022090800],
            "season": [2022],
            "week": [1],
            "gameDate": ["09/08/2022"],
            "gameTimeEastern": ["20:20:00"],
            "homeTeamAbbr": ["LA"],
            "visitorTeamAbbr": ["BUF"],
            "homeFinalScore": [10],
            "visitorFinalScore": [31],
        }
    )
    plays = pl.DataFrame(
        {
            "gameId": [2022090800, 2022090800],
            "playId": [80, 56],
            "quarter": [1, 1],
            "gameClock": ["14:29", "15:00"],
            "down": [2, 1],
            "yardsToGo": [4, 10],
            "possessionTeam": ["BUF", "BUF"],
            "defensiveTeam": ["LA", "LA"],
            "yardlineSide": ["BUF", "BUF"],
            "yardlineNumber": [31, 25],
            "playDescription": ["scramble", "pass to Diggs"],
            "playResult": [7, 6],
            "passResult": ["R", "C"],
            "absoluteYardlineNumber": [41, 35],
            "preSnapHomeScore": [0, 0],
            "preSnapVisitorScore": [0, 0],
            "offenseFormation": ["SHOTGUN", "SHOTGUN"],
            "defendersInTheBox": [6, 6],
            "expectedPointsAdded": [0.5, 0.1],
            "ballCarrierDisplayName": ["Josh Allen", "S. Diggs"],
        }
    )
    players = pl.DataFrame(
        {
            "nflId": [1, 2],
            "displayName": ["Josh Allen", "Jalen Ramsey"],
            "position": ["QB", "CB"],
        }
    )

    def entity(play, nfl_id, team, xs, etype="player", events=(None, None, None)):
        return [
            {
                "game_id": "2022090800",
                "play_id": play,
                "nfl_id": nfl_id,
                "frame_id": i + 1,
                "x": x,
                "y": 20.0 + i,
                "team": team,
                "entity_type": etype,
                "event": events[i],
                "speed": 2.0 * i,
                "acceleration": 1.0,
                "orientation": 90.0,
                "direction": 45.0,
            }
            for i, x in enumerate(xs)
        ]

    tracking = _frames(
        entity("56", "1", "BUF", [34.0, 35.0, 36.0])
        + entity("56", "2", "LA", [40.0, 39.0, 38.0])
        + entity("56", None, "football", [35.2, 36.0, 37.0], "ball", (None, "ball_snap", "tackle"))
        + entity("80", "1", "BUF", [41.0, 42.0, 43.0])
        + entity("80", "2", "LA", [45.0, 44.0, 43.0])
    )
    release = _release(
        tmp_path / "gold/releases/sumer_r1",
        {
            "2024_games": games,
            "2024_plays": plays,
            "2024_players": players,
            "2024_tracking_week_1": tracking,
        },
    )
    return tracking_viewer.SumerSportsAdapter(release)


def test_sumer_games_and_play_order(sumer):
    games = sumer.games()
    assert games == [
        {
            "game_id": "2022090800",
            "season": 2022,
            "week": 1,
            "date": "09/08/2022",
            "home": "LA",
            "away": "BUF",
            "home_score": 10,
            "away_score": 31,
            "plays": 2,
        }
    ]
    plays = sumer.plays("2022090800")
    assert [p["play_id"] for p in plays] == ["56", "80"], "game clock order, not play id order"
    assert plays[0]["play_type"] == "pass" and plays[1]["play_type"] == "run"
    assert plays[0]["events"] == ["ball_snap", "tackle"]
    assert plays[0]["frames"] == 3


def test_sumer_play_payload(sumer):
    play = sumer.play("2022090800", "56")
    assert play["frames"] == 3 and play["frame_rate"] == 10.0
    assert play["line_of_scrimmage"] == 36.0, "ball position at the snap wins over the yardline"
    assert play["first_down_line"] == 46.0
    assert play["events"] == [{"frame": 1, "name": "ball_snap"}, {"frame": 2, "name": "tackle"}]
    by_id = {e["id"]: e for e in play["entities"]}
    assert set(by_id) == {"1", "2", "ball"}
    assert by_id["1"]["side"] == "offense" and by_id["2"]["side"] == "defense"
    assert by_id["1"]["name"] == "Josh Allen" and by_id["2"]["position"] == "CB"
    assert by_id["ball"]["kind"] == "ball" and by_id["ball"]["x"] == [35.2, 36.0, 37.0]
    assert by_id["1"]["s"] == [0.0, 2.0, 4.0]
    assert play["context"]["description"] == "pass to Diggs"
    assert play["teams"] == {"home": "LA", "away": "BUF", "offense": "BUF", "defense": "LA"}
    with pytest.raises(KeyError):
        sumer.play("2022090800", "999")


@pytest.fixture
def bdb2026(tmp_path):
    def rows(nfl_id, xs, phase):
        return [
            {
                "game_id": "2023090700",
                "play_id": "101",
                "nfl_id": nfl_id,
                "frame_id": i + 1,
                "x": x,
                "y": 10.0,
                "team": "Offense" if nfl_id == "11" else "Defense",
                "speed": 1.0 if phase == "input" else None,
                "acceleration": 0.5,
                "orientation": 180.0,
                "direction": 0.0,
            }
            for i, x in enumerate(xs)
        ]

    gold = _release(
        tmp_path / "gold/releases/bdb26_r1",
        {
            "train_input_2023_w01": _frames(
                rows("11", [50.0, 51.0], "input")
                + rows("22", [55.0, 54.0], "input")
                + rows("33", [60.0, 60.0], "input")
            ),
            "train_output_2023_w01": _frames(
                rows("11", [52.0, 53.0, 54.0], "output") + rows("22", [53.0, 53.0, 53.0], "output")
            ),
        },
        {
            "train_input_2023_w01": "tracking_observation",
            "train_output_2023_w01": "tracking_labels",
        },
    )
    enriched = _release(
        tmp_path / "enriched/releases/bdb26_r1",
        {
            "train_input_2023_w01": pl.DataFrame(
                {
                    "game_id": [2023090700] * 3,
                    "play_id": [101] * 3,
                    "nfl_id": [11, 22, 33],
                    "frame_id": [1, 1, 1],
                    "player_name": ["Amon-Ra St. Brown", "Trent McDuffie", "Jared Goff"],
                    "player_position": ["WR", "CB", "QB"],
                    "player_side": ["Offense", "Defense", "Offense"],
                    "player_role": ["Targeted Receiver", "Defensive Coverage", "Passer"],
                    "player_to_predict": [True, True, False],
                    "ball_land_x": [63.26] * 3,
                    "ball_land_y": [-0.22] * 3,
                    "absolute_yardline_number": [42] * 3,
                }
            )
        },
    )
    supplementary = pl.DataFrame(
        {
            "game_id": [2023090700, 2023090700],
            "season": [2023, 2023],
            "week": [1, 1],
            "game_date": ["09/07/2023"] * 2,
            "home_team_abbr": ["KC"] * 2,
            "visitor_team_abbr": ["DET"] * 2,
            "play_id": [101, 999],
            "quarter": [1, 1],
            "game_clock": ["14:25", "13:00"],
            "down": [3, 1],
            "yards_to_go": [3, 10],
            "possession_team": ["DET"] * 2,
            "defensive_team": ["KC"] * 2,
            "yardline_side": ["DET"] * 2,
            "yardline_number": [32, 40],
            "play_description": ["deep right", "x"],
            "yards_gained": [0, 5],
            "pass_result": ["I", "C"],
            "pre_snap_home_score": [0, 0],
            "pre_snap_visitor_score": [0, 0],
            "offense_formation": ["SHOTGUN"] * 2,
            "defenders_in_the_box": [5, 6],
            "expected_points_added": [-0.9, 0.3],
            "team_coverage_type": ["COVER_2_ZONE"] * 2,
            "route_of_targeted_receiver": ["GO", "OUT"],
        }
    )
    return tracking_viewer.Bdb2026Adapter(gold, enriched, supplementary)


def test_bdb2026_games_plays_and_stitched_play(bdb2026):
    assert bdb2026.games() == [
        {
            "game_id": "2023090700",
            "season": 2023,
            "week": 1,
            "date": "09/07/2023",
            "home": "KC",
            "away": "DET",
            "home_score": None,
            "away_score": None,
            "plays": 2,
        }
    ]
    plays = bdb2026.plays("2023090700")
    assert [p["play_id"] for p in plays] == ["101"], "plays without tracking are left out"
    assert plays[0]["frames"] == 5 and plays[0]["events"] == [
        "ball_snap",
        "pass_forward",
        "ball_land",
    ]

    play = bdb2026.play("2023090700", "101")
    assert play["frames"] == 5
    assert play["events"] == [
        {"frame": 0, "name": "ball_snap"},
        {"frame": 1, "name": "pass_forward"},
        {"frame": 4, "name": "ball_land"},
    ]
    by_id = {e["id"]: e for e in play["entities"]}
    assert by_id["11"]["x"] == [50.0, 51.0, 52.0, 53.0, 54.0], "flight frames follow the throw"
    assert by_id["11"]["s"] == [1.0, 1.0, None, None, None], "no observed speed in flight"
    assert by_id["33"]["x"] == [60.0, 60.0, None, None, None], "passer is not tracked in flight"
    assert by_id["33"]["tracked_in_flight"] is False and by_id["11"]["tracked_in_flight"] is True
    assert by_id["22"]["side"] == "defense" and by_id["22"]["team"] == "KC"
    assert by_id["11"]["team"] == "DET" and by_id["11"]["role"] == "Targeted Receiver"
    assert play["ball_land"] == {"x": 63.26, "y": -0.22}
    assert play["line_of_scrimmage"] == 42.0 and play["first_down_line"] == 45.0
    assert play["context"]["coverage"] == "COVER_2_ZONE" and play["context"]["route"] == "GO"
    with pytest.raises(KeyError):
        bdb2026.play("2023090700", "999")


@pytest.fixture
def client(monkeypatch, tmp_path):
    calls = []

    def fake_games(data_dir, source):
        calls.append(("games", source))
        return [{"game_id": "1"}]

    def fake_plays(data_dir, source, game):
        if game == "404":
            raise KeyError("Unknown game: 404")
        return [{"play_id": "5"}]

    def fake_play(data_dir, source, game, play):
        if play == "409":
            raise ValueError("Source table changed")
        return {"play_id": play, "entities": []}

    monkeypatch.setattr(tracking_routes, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(tracking_viewer, "games", fake_games)
    monkeypatch.setattr(tracking_viewer, "plays", fake_plays)
    monkeypatch.setattr(tracking_viewer, "play", fake_play)
    monkeypatch.setattr(
        tracking_viewer, "sources", lambda data_dir: [{"id": "bdb_2024_sumersports"}]
    )
    app = FastAPI()
    app.include_router(tracking_routes.router)
    return TestClient(app)


def test_routes(client):
    assert client.get("/api/tracking/sources").json() == {
        "sources": [{"id": "bdb_2024_sumersports"}]
    }
    assert client.get("/api/tracking/bdb_2024_sumersports/games").json() == {
        "games": [{"game_id": "1"}]
    }
    assert client.get("/api/tracking/nope/games").status_code == 404
    assert client.get("/api/tracking/bdb_2024_sumersports/games/1/plays").json() == {
        "plays": [{"play_id": "5"}]
    }
    assert client.get("/api/tracking/bdb_2024_sumersports/games/404/plays").status_code == 404
    assert client.get("/api/tracking/bdb_2024_sumersports/games/abc/plays").status_code == 404
    assert client.get("/api/tracking/bdb_2024_sumersports/games/1/plays/5").json()["play_id"] == "5"
    assert client.get("/api/tracking/bdb_2024_sumersports/games/1/plays/409").status_code == 409


def test_play_order_helper():
    assert tracking_viewer._play_order(1, "15:00") < tracking_viewer._play_order(1, "14:29")
    assert tracking_viewer._play_order(1, "0:05") < tracking_viewer._play_order(2, "15:00")
    assert tracking_viewer._play_order(None, None) == 3601
