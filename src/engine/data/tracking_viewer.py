"""Play replay data for the dashboard game engine.

Serves verified tracking releases as per-play entity trajectories plus the play
context each archive supplies. Observed frames only: no models, no ghosts, no
predictions. Each archive is adapted to one response shape so the frontend renders
them identically and labels what each one does not track.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from engine.data.releases import TableRelease
from engine.data.tracking import load_source

FRAME_RATE = 10.0
FIELD_LENGTH = 120.0
FIELD_WIDTH = 160 / 3


@dataclass(frozen=True)
class ViewerSource:
    id: str
    label: str
    season: int
    tracked: str  # "full" or "coverage"
    description: str


SOURCES: dict[str, ViewerSource] = {
    "bdb_2024_sumersports": ViewerSource(
        "bdb_2024_sumersports",
        "2022 season, weeks 1–9",
        2022,
        "full",
        "All 22 players and the ball from before the snap to the tackle. Completed "
        "passes and runs only; incompletions are not in this archive.",
    ),
    "bdb_2026_prediction": ViewerSource(
        "bdb_2026_prediction",
        "2023 regular season pass plays",
        2023,
        "coverage",
        "Passer, route runners and coverage defenders from the snap to the throw, then "
        "the targeted receiver and nearby defenders while the ball is in the air. "
        "Linemen and the ball are not tracked; the ball landing spot is marked.",
    ),
}


def _clock_seconds(clock: str | None) -> int:
    if not clock or ":" not in clock:
        return -1
    minutes, seconds = clock.split(":", 1)
    try:
        return int(minutes or 0) * 60 + int(seconds)
    except ValueError:
        return -1


def _play_order(quarter: int | None, clock: str | None) -> int:
    """Game time as a sortable integer: later quarters, then less clock remaining."""
    return (quarter or 0) * 10_000 + (3600 - _clock_seconds(clock))


def _none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rounded(values: list[Any], digits: int) -> list[float | None]:
    return [None if v is None or not math.isfinite(v) else round(v, digits) for v in values]


def _series(frame: pl.DataFrame, column: str, index: dict[int, int], size: int, digits: int):
    """Column values placed at their global frame index; absent frames stay null."""
    out: list[float | None] = [None] * size
    for frame_id, value in zip(frame["frame_id"].to_list(), frame[column].to_list(), strict=True):
        position = index.get(frame_id)
        if position is not None and value is not None and math.isfinite(value):
            out[position] = round(value, digits)
    return out


class SumerSportsAdapter:
    """2022 tackling archive: every player and the ball on each play."""

    def __init__(self, release: TableRelease):
        self.release = release

    def games(self) -> list[dict]:
        games = self.release.read("2024_games")
        counts = (
            self.release.read("2024_plays", columns=["gameId"])
            .group_by("gameId")
            .len()
            .rename({"len": "plays"})
        )
        rows = games.join(counts, on="gameId", how="left").sort(["week", "gameDate", "gameId"])
        return [
            {
                "game_id": str(r["gameId"]),
                "season": r["season"],
                "week": r["week"],
                "date": r["gameDate"],
                "home": r["homeTeamAbbr"],
                "away": r["visitorTeamAbbr"],
                "home_score": r["homeFinalScore"],
                "away_score": r["visitorFinalScore"],
                "plays": r["plays"] or 0,
            }
            for r in rows.to_dicts()
        ]

    def _game(self, game_id: str) -> dict:
        rows = self.release.read("2024_games").filter(pl.col("gameId") == int(game_id))
        if rows.is_empty():
            raise KeyError(f"Unknown game: {game_id}")
        return rows.to_dicts()[0]

    def _tracking(self, game: dict) -> pl.LazyFrame:
        table = f"2024_tracking_week_{game['week']}"
        return pl.scan_parquet(self.release.path(table)).filter(
            pl.col("game_id") == str(game["gameId"])
        )

    def plays(self, game_id: str) -> list[dict]:
        game = self._game(game_id)
        plays = self.release.read("2024_plays").filter(pl.col("gameId") == int(game_id))
        frames = (
            self._tracking(game)
            .group_by("play_id")
            .agg(
                pl.col("frame_id").max().alias("frames"),
                pl.col("event").drop_nulls().unique().alias("events"),
            )
            .collect()
        )
        tracked = {r["play_id"]: r for r in frames.to_dicts()}
        out = []
        for r in plays.to_dicts():
            track = tracked.get(str(r["playId"]))
            if track is None:
                continue
            out.append(
                {
                    "play_id": str(r["playId"]),
                    "order": _play_order(r["quarter"], r["gameClock"]),
                    "quarter": r["quarter"],
                    "clock": r["gameClock"],
                    "down": r["down"],
                    "distance": r["yardsToGo"],
                    "possession": r["possessionTeam"],
                    "defense": r["defensiveTeam"],
                    "yardline_side": r["yardlineSide"],
                    "yardline_number": r["yardlineNumber"],
                    "description": r["playDescription"],
                    "result_yards": r["playResult"],
                    "pass_result": r["passResult"],
                    "play_type": "pass" if r["passResult"] in ("C", "I", "IN", "S") else "run",
                    "home_score": r["preSnapHomeScore"],
                    "away_score": r["preSnapVisitorScore"],
                    "frames": track["frames"],
                    "events": sorted(track["events"]),
                }
            )
        out.sort(key=lambda p: (p["order"], int(p["play_id"])))
        return out

    def play(self, game_id: str, play_id: str) -> dict:
        game = self._game(game_id)
        plays = self.release.read("2024_plays").filter(
            (pl.col("gameId") == int(game_id)) & (pl.col("playId") == int(play_id))
        )
        if plays.is_empty():
            raise KeyError(f"Unknown play: {play_id}")
        context = plays.to_dicts()[0]
        frames = (
            self._tracking(game)
            .filter(pl.col("play_id") == str(play_id))
            .select(
                "nfl_id",
                "frame_id",
                "x",
                "y",
                "speed",
                "acceleration",
                "orientation",
                "direction",
                "team",
                "play_direction",
                "event",
                "entity_type",
            )
            .collect()
            .sort(["frame_id", "nfl_id"])
        )
        if frames.is_empty():
            raise KeyError(f"No tracking for play: {play_id}")
        frame_ids = sorted(frames["frame_id"].unique().to_list())
        index = {fid: i for i, fid in enumerate(frame_ids)}
        size = len(frame_ids)
        players = self.release.read("2024_players").with_columns(pl.col("nflId").cast(pl.Utf8))
        names = {r["nflId"]: r for r in players.to_dicts()}
        possession = context["possessionTeam"]
        entities = []
        for (entity_id,), group in frames.group_by(["nfl_id"], maintain_order=True):
            if entity_id is None:
                entities.append(
                    {
                        "id": "ball",
                        "kind": "ball",
                        "name": "Football",
                        "position": None,
                        "team": None,
                        "side": "ball",
                        "x": _series(group, "x", index, size, 2),
                        "y": _series(group, "y", index, size, 2),
                        "s": _series(group, "speed", index, size, 2),
                        "a": _series(group, "acceleration", index, size, 2),
                        "o": [None] * size,
                        "dir": [None] * size,
                    }
                )
                continue
            team = group["team"][0]
            player = names.get(str(entity_id), {})
            entities.append(
                {
                    "id": str(entity_id),
                    "kind": "player",
                    "name": player.get("displayName") or f"Player {entity_id}",
                    "position": player.get("position"),
                    "team": team,
                    "side": "offense" if team == possession else "defense",
                    "x": _series(group, "x", index, size, 2),
                    "y": _series(group, "y", index, size, 2),
                    "s": _series(group, "speed", index, size, 2),
                    "a": _series(group, "acceleration", index, size, 2),
                    "o": _series(group, "orientation", index, size, 1),
                    "dir": _series(group, "direction", index, size, 1),
                }
            )
        events = (
            frames.filter(pl.col("event").is_not_null())
            .select("frame_id", "event")
            .unique()
            .sort("frame_id")
        )
        direction = frames["play_direction"][0]
        los = float(context["absoluteYardlineNumber"])
        snap = events.filter(pl.col("event") == "ball_snap")
        if not snap.is_empty():
            ball_at_snap = frames.filter(
                (pl.col("entity_type") == "ball") & (pl.col("frame_id") == snap["frame_id"][0])
            )
            if not ball_at_snap.is_empty() and ball_at_snap["x"][0] is not None:
                los = float(ball_at_snap["x"][0])
        yards_to_go = context["yardsToGo"] or 0
        first_down = los + yards_to_go if direction == "right" else los - yards_to_go
        return {
            "source": "bdb_2024_sumersports",
            "game_id": str(game_id),
            "play_id": str(play_id),
            "frame_rate": FRAME_RATE,
            "frames": size,
            "play_direction": direction,
            "line_of_scrimmage": round(los, 2),
            "first_down_line": round(first_down, 2),
            "ball_land": None,
            "teams": {
                "home": game["homeTeamAbbr"],
                "away": game["visitorTeamAbbr"],
                "offense": possession,
                "defense": context["defensiveTeam"],
            },
            "context": {
                "quarter": context["quarter"],
                "clock": context["gameClock"],
                "down": context["down"],
                "distance": yards_to_go,
                "yardline_side": context["yardlineSide"],
                "yardline_number": context["yardlineNumber"],
                "description": context["playDescription"],
                "result_yards": context["playResult"],
                "pass_result": context["passResult"],
                "formation": context["offenseFormation"],
                "defenders_in_box": context["defendersInTheBox"],
                "home_score": context["preSnapHomeScore"],
                "away_score": context["preSnapVisitorScore"],
                "epa": _none(context["expectedPointsAdded"]),
                "ball_carrier": context["ballCarrierDisplayName"],
                "coverage": None,
                "route": None,
            },
            "events": [
                {"frame": index[r["frame_id"]], "name": r["event"]} for r in events.to_dicts()
            ],
            "entities": entities,
        }


class Bdb2026Adapter:
    """2023 pass plays: the passing game's players before the throw and in flight."""

    def __init__(self, release: TableRelease, enriched: TableRelease, supplementary: pl.DataFrame):
        self.release = release
        self.enriched = enriched
        self.supplementary = supplementary

    def _input_table(self, week: int) -> str | None:
        name = f"train_input_2023_w{week:02d}"
        return name if name in self.release.manifest["tables"] else None

    def games(self) -> list[dict]:
        rows = (
            self.supplementary.group_by(
                "game_id", "season", "week", "game_date", "home_team_abbr", "visitor_team_abbr"
            )
            .len()
            .rename({"len": "plays"})
            .sort(["week", "game_date", "game_id"])
        )
        return [
            {
                "game_id": str(r["game_id"]),
                "season": r["season"],
                "week": r["week"],
                "date": r["game_date"],
                "home": r["home_team_abbr"],
                "away": r["visitor_team_abbr"],
                "home_score": None,
                "away_score": None,
                "plays": r["plays"],
            }
            for r in rows.to_dicts()
            if r["season"] == 2023 and self._input_table(r["week"])
        ]

    def _game_rows(self, game_id: str) -> pl.DataFrame:
        rows = self.supplementary.filter(pl.col("game_id") == int(game_id))
        if rows.is_empty():
            raise KeyError(f"Unknown game: {game_id}")
        return rows

    def _scan(self, kind: str, week: int, game_id: str) -> pl.LazyFrame:
        name = f"train_{kind}_2023_w{week:02d}"
        return pl.scan_parquet(self.release.path(name)).filter(pl.col("game_id") == str(game_id))

    def plays(self, game_id: str) -> list[dict]:
        rows = self._game_rows(game_id)
        week = rows["week"][0]
        if not self._input_table(week):
            raise KeyError(f"No tracking for game: {game_id}")
        counts = {
            kind: {
                r["play_id"]: r["frames"]
                for r in self._scan(kind, week, game_id)
                .group_by("play_id")
                .agg(pl.col("frame_id").max().alias("frames"))
                .collect()
                .to_dicts()
            }
            for kind in ("input", "output")
        }
        out = []
        for r in rows.to_dicts():
            before = counts["input"].get(str(r["play_id"]))
            if before is None:
                continue
            events = ["ball_snap", "pass_forward"]
            if counts["output"].get(str(r["play_id"])):
                events.append("ball_land")
            out.append(
                {
                    "play_id": str(r["play_id"]),
                    "order": _play_order(r["quarter"], r["game_clock"]),
                    "quarter": r["quarter"],
                    "clock": r["game_clock"],
                    "down": r["down"],
                    "distance": r["yards_to_go"],
                    "possession": r["possession_team"],
                    "defense": r["defensive_team"],
                    "yardline_side": r["yardline_side"],
                    "yardline_number": r["yardline_number"],
                    "description": r["play_description"],
                    "result_yards": r["yards_gained"],
                    "pass_result": r["pass_result"],
                    "play_type": "pass",
                    "home_score": r["pre_snap_home_score"],
                    "away_score": r["pre_snap_visitor_score"],
                    "frames": before + (counts["output"].get(str(r["play_id"])) or 0),
                    "events": events,
                }
            )
        out.sort(key=lambda p: (p["order"], int(p["play_id"])))
        return out

    def play(self, game_id: str, play_id: str) -> dict:
        rows = self._game_rows(game_id).filter(pl.col("play_id") == int(play_id))
        if rows.is_empty():
            raise KeyError(f"Unknown play: {play_id}")
        context = rows.to_dicts()[0]
        week = context["week"]
        if not self._input_table(week):
            raise KeyError(f"No tracking for game: {game_id}")
        columns = [
            "nfl_id",
            "frame_id",
            "x",
            "y",
            "speed",
            "acceleration",
            "orientation",
            "direction",
            "play_direction",
        ]
        before = (
            self._scan("input", week, game_id)
            .filter(pl.col("play_id") == str(play_id))
            .select(columns)
            .collect()
            .sort(["frame_id", "nfl_id"])
        )
        if before.is_empty():
            raise KeyError(f"No tracking for play: {play_id}")
        after = (
            self._scan("output", week, game_id)
            .filter(pl.col("play_id") == str(play_id))
            .select("nfl_id", "frame_id", "x", "y")
            .collect()
            .sort(["frame_id", "nfl_id"])
        )
        throw = int(before.select(pl.col("frame_id").max()).item())
        flight = int(after.select(pl.col("frame_id").max()).item()) if not after.is_empty() else 0
        size = throw + flight
        meta = (
            pl.scan_parquet(self.enriched.path(f"train_input_2023_w{week:02d}"))
            .filter((pl.col("game_id") == int(game_id)) & (pl.col("play_id") == int(play_id)))
            .select(
                "nfl_id",
                "player_name",
                "player_position",
                "player_side",
                "player_role",
                "player_to_predict",
                "ball_land_x",
                "ball_land_y",
                "absolute_yardline_number",
            )
            .unique(subset=["nfl_id"])
            .collect()
        )
        info = {str(r["nfl_id"]): r for r in meta.to_dicts()}
        index_before = {fid: fid - 1 for fid in range(1, throw + 1)}
        index_after = {fid: throw + fid - 1 for fid in range(1, flight + 1)}
        entities = []
        for (entity_id,), group in before.group_by(["nfl_id"], maintain_order=True):
            player = info.get(str(entity_id), {})
            tail = after.filter(pl.col("nfl_id") == entity_id)
            x = _series(group, "x", index_before, size, 2)
            y = _series(group, "y", index_before, size, 2)
            if not tail.is_empty():
                tx = _series(tail, "x", index_after, size, 2)
                ty = _series(tail, "y", index_after, size, 2)
                x = [a if a is not None else b for a, b in zip(x, tx, strict=True)]
                y = [a if a is not None else b for a, b in zip(y, ty, strict=True)]
            side = (player.get("player_side") or "").lower() or None
            entities.append(
                {
                    "id": str(entity_id),
                    "kind": "player",
                    "name": player.get("player_name") or f"Player {entity_id}",
                    "position": player.get("player_position"),
                    "team": context["possession_team"]
                    if side == "offense"
                    else context["defensive_team"],
                    "side": side,
                    "role": player.get("player_role"),
                    "tracked_in_flight": bool(player.get("player_to_predict")),
                    "x": x,
                    "y": y,
                    "s": _series(group, "speed", index_before, size, 2),
                    "a": _series(group, "acceleration", index_before, size, 2),
                    "o": _series(group, "orientation", index_before, size, 1),
                    "dir": _series(group, "direction", index_before, size, 1),
                }
            )
        direction = before["play_direction"][0]
        first = meta.to_dicts()[0] if not meta.is_empty() else {}
        los = first.get("absolute_yardline_number")
        los = float(los) if los is not None else None
        yards_to_go = context["yards_to_go"] or 0
        first_down = None
        if los is not None:
            first_down = los + yards_to_go if direction == "right" else los - yards_to_go
        ball_land = None
        if first.get("ball_land_x") is not None and first.get("ball_land_y") is not None:
            ball_land = {
                "x": round(float(first["ball_land_x"]), 2),
                "y": round(float(first["ball_land_y"]), 2),
            }
        events = [{"frame": 0, "name": "ball_snap"}, {"frame": throw - 1, "name": "pass_forward"}]
        if flight:
            events.append({"frame": size - 1, "name": "ball_land"})
        return {
            "source": "bdb_2026_prediction",
            "game_id": str(game_id),
            "play_id": str(play_id),
            "frame_rate": FRAME_RATE,
            "frames": size,
            "play_direction": direction,
            "line_of_scrimmage": los,
            "first_down_line": None if first_down is None else round(first_down, 2),
            "ball_land": ball_land,
            "teams": {
                "home": context["home_team_abbr"],
                "away": context["visitor_team_abbr"],
                "offense": context["possession_team"],
                "defense": context["defensive_team"],
            },
            "context": {
                "quarter": context["quarter"],
                "clock": context["game_clock"],
                "down": context["down"],
                "distance": yards_to_go,
                "yardline_side": context["yardline_side"],
                "yardline_number": context["yardline_number"],
                "description": context["play_description"],
                "result_yards": context["yards_gained"],
                "pass_result": context["pass_result"],
                "formation": context["offense_formation"],
                "defenders_in_box": context["defenders_in_the_box"],
                "home_score": context["pre_snap_home_score"],
                "away_score": context["pre_snap_visitor_score"],
                "epa": _none(context["expected_points_added"]),
                "ball_carrier": None,
                "coverage": context["team_coverage_type"],
                "route": context["route_of_targeted_receiver"],
            },
            "events": events,
            "entities": entities,
        }


def _enriched_release(data_dir: Path, version: str) -> TableRelease:
    root = data_dir / "enriched/releases" / version
    manifest = json.loads((root / "manifest.json").read_text())
    return TableRelease(root=root, manifest=manifest)


@lru_cache(maxsize=8)
def adapter(data_dir: str, source_id: str):
    """One verified adapter per source. Missing releases raise KeyError/ValueError."""
    if source_id not in SOURCES:
        raise KeyError(f"Unknown tracking source: {source_id}")
    data = Path(data_dir)
    release = load_source(data, source_id)
    if source_id == "bdb_2024_sumersports":
        return SumerSportsAdapter(release)
    analytics = load_source(data, "bdb_2026_analytics")
    supplementary_table = next(
        name for name in analytics.manifest["tables"] if "supplementary" in name
    )
    supplementary = analytics.read(supplementary_table)
    return Bdb2026Adapter(
        release, _enriched_release(data, release.manifest["version"]), supplementary
    )


def sources(data_dir: Path) -> list[dict]:
    out = []
    for source in SOURCES.values():
        entry: dict[str, Any] = {
            "id": source.id,
            "label": source.label,
            "season": source.season,
            "tracked": source.tracked,
            "description": source.description,
            "available": True,
            "games": 0,
        }
        try:
            entry["games"] = len(games(data_dir, source.id))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            entry["available"] = False
            entry["issue"] = str(exc)
        out.append(entry)
    return out


@lru_cache(maxsize=16)
def _games(data_dir: str, source_id: str) -> list[dict]:
    return adapter(data_dir, source_id).games()


@lru_cache(maxsize=64)
def _plays(data_dir: str, source_id: str, game_id: str) -> list[dict]:
    return adapter(data_dir, source_id).plays(game_id)


@lru_cache(maxsize=512)
def _play(data_dir: str, source_id: str, game_id: str, play_id: str) -> dict:
    return adapter(data_dir, source_id).play(game_id, play_id)


def games(data_dir: Path, source_id: str) -> list[dict]:
    return _games(str(data_dir), source_id)


def plays(data_dir: Path, source_id: str, game_id: str) -> list[dict]:
    return _plays(str(data_dir), source_id, game_id)


def play(data_dir: Path, source_id: str, game_id: str, play_id: str) -> dict:
    return _play(str(data_dir), source_id, game_id, play_id)


def clear_cache() -> None:
    for cached in (adapter, _games, _plays, _play):
        cached.cache_clear()
