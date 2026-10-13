"""Resolve raw-snapshot assets to files DuckDB can read directly.

New snapshots name provider files ``nflverse/<dataset>/<season>.<ext>``. Snapshots
captured before October 2026 preserved nflreadpy's filesystem cache instead, whose
files are named by the MD5 of ``f"{url}_{sorted(kwargs)}"``; those are still resolved
so older raw releases can be re-enriched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from engine.data.releases import inside

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
DYNASTYPROCESS = "https://github.com/dynastyprocess/data/raw/master/files/"

# dataset -> (url template, nflreadpy cache kwargs). ``{season}`` marks per-season files.
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
    "ff_rankings": (DYNASTYPROCESS + "db_fpecr.parquet", {}),
}
# First season each per-season provider file exists.
SEASON_START = {
    "stats_player_week": 2001,
    "pbp": 2001,
    "roster_weekly": 2004,
    "snap_counts": 2013,
    "injuries": 2009,
}
SEASON_RANGE = range(1999, 2040)


def asset_name(dataset: str, season: int | None = None) -> str:
    template, _ = DATASETS[dataset]
    if ("{season}" in template) != (season is not None):
        raise ValueError(f"{dataset} {'requires' if season is None else 'takes no'} season")
    suffix = Path(template).suffix
    return f"nflverse/{dataset}/{season if season is not None else 'latest'}{suffix}"


def cache_key(url: str, **kwargs: object) -> str:
    return hashlib.md5(f"{url}_{str(sorted(kwargs.items()))}".encode()).hexdigest()


def dataset_key(dataset: str, season: int | None = None) -> str:
    """nflreadpy's cache file stem for a dataset; used for pre-October-2026 snapshots."""
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

    def _names(self, dataset: str, season: int | None) -> tuple[str, str]:
        return (
            asset_name(dataset, season),
            f"history/cache/nflverse/{dataset_key(dataset, season)}.parquet",
        )

    def has(self, dataset: str, season: int | None = None) -> bool:
        return any(name in self.manifest["assets"] for name in self._names(dataset, season))

    def resolve(self, dataset: str, season: int | None = None) -> tuple[Path, str]:
        """The preserved file and the asset name it was registered under."""
        for name in self._names(dataset, season):
            if name in self.manifest["assets"]:
                return self._asset(name), name
        raise FileNotFoundError(f"Raw snapshot has no {dataset} {season or ''} file".strip())

    def nflverse(self, dataset: str, season: int | None = None) -> Path:
        return self.resolve(dataset, season)[0]

    def seasons(self, dataset: str) -> list[int]:
        """Every season of a dataset preserved in the snapshot."""
        return [season for season in SEASON_RANGE if self.has(dataset, season)]

    def nflverse_files(self, dataset: str, seasons: list[int] | None = None) -> list[Path]:
        return [self.nflverse(dataset, season) for season in seasons or self.seasons(dataset)]

    def static(self, name: str) -> Path:
        for candidate in (f"static/{name}", f"history/static/{name}"):
            if candidate in self.manifest["assets"]:
                return self._asset(candidate)
        raise FileNotFoundError(f"Raw snapshot has no static input {name}")

    def current(self, name: str) -> Path:
        return self._asset(f"current/{name}")

    def college(self, name: str) -> Path:
        return self._asset(f"college/{name}")

    def college_manifest(self) -> dict:
        import json

        return json.loads(self.college("manifest.json").read_text())
