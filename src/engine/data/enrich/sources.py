"""Resolve raw-snapshot assets to files DuckDB can read directly.

A raw snapshot preserves the nflreadpy filesystem cache byte for byte. nflreadpy
names each cached file by the MD5 of ``f"{url}_{sorted(kwargs)}"``; reproducing that
key here lets the builder address datasets by name and season without importing
nflreadpy or touching the network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from engine.data.releases import inside

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
DYNASTYPROCESS = "https://github.com/dynastyprocess/data/raw/master/files/"

# dataset -> (url template, cache kwargs template). ``{season}`` is substituted when a
# season is given; datasets without it are single files.
DATASETS: dict[str, tuple[str, dict[str, object]]] = {
    "stats_player_week": (
        NFLVERSE + "stats_player/stats_player_week_{season}.parquet",
        {"season": "{season}", "summary_level": "week", "stat_type": "player"},
    ),
    "pbp": (NFLVERSE + "pbp/play_by_play_{season}.parquet", {"season": "{season}"}),
    "roster_weekly": (
        NFLVERSE + "weekly_rosters/roster_weekly_{season}.parquet",
        {"season": "{season}"},
    ),
    "snap_counts": (NFLVERSE + "snap_counts/snap_counts_{season}.parquet", {"season": "{season}"}),
    "injuries": (NFLVERSE + "injuries/injuries_{season}.parquet", {"season": "{season}"}),
    "players": (NFLVERSE + "players/players.parquet", {}),
    "schedules": (NFLVERSE + "schedules/games.parquet", {}),
    "ff_playerids": (DYNASTYPROCESS + "db_playerids.csv", {}),
}
SEASON_RANGE = range(1999, 2040)


def cache_key(url: str, **kwargs: object) -> str:
    return hashlib.md5(f"{url}_{str(sorted(kwargs.items()))}".encode()).hexdigest()


def dataset_key(dataset: str, season: int | None = None) -> str:
    template, kwargs = DATASETS[dataset]
    if ("{season}" in template) != (season is not None):
        raise ValueError(f"{dataset} {'requires' if season is None else 'takes no'} season")
    url = template.format(season=season)
    resolved = {
        name: (int(value.format(season=season)) if value == "{season}" else value)
        for name, value in kwargs.items()
    }
    return cache_key(url, **resolved)


@dataclass(frozen=True)
class RawSources:
    """Files of one raw snapshot, addressed by dataset and season."""

    data_dir: Path
    manifest: dict

    def _asset(self, name: str) -> Path:
        asset = self.manifest["assets"].get(name)
        if asset is None:
            raise FileNotFoundError(f"Raw snapshot has no asset {name}")
        return inside(self.data_dir, asset["object"])

    def nflverse(self, dataset: str, season: int | None = None) -> Path:
        return self._asset(f"history/cache/nflverse/{dataset_key(dataset, season)}.parquet")

    def seasons(self, dataset: str) -> list[int]:
        """Every season of a dataset preserved in the snapshot."""
        names = self.manifest["assets"]
        return [
            season
            for season in SEASON_RANGE
            if f"history/cache/nflverse/{dataset_key(dataset, season)}.parquet" in names
        ]

    def nflverse_files(self, dataset: str) -> list[Path]:
        return [self.nflverse(dataset, season) for season in self.seasons(dataset)]

    def static(self, name: str) -> Path:
        return self._asset(f"history/static/{name}")

    def current(self, name: str) -> Path:
        return self._asset(f"current/{name}")

    def college(self, name: str) -> Path:
        return self._asset(f"college/{name}")

    def college_manifest(self) -> dict:
        import json

        return json.loads(self.college("manifest.json").read_text())
