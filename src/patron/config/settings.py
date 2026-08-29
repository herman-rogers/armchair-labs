"""Runtime configuration: filesystem paths, secrets, and the parsed YAML configs.

Everything that varies by environment lands here. The YAML files next to this module
carry league rules that are the same wherever the engine runs; `.env` carries the
things that are not (secrets, machine-local paths).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).parent
PACKAGE_ROOT = CONFIG_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent


class Settings(BaseSettings):
    """Environment-driven settings, read from `.env` and the process environment."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=REPO_ROOT / "data")

    # ESPN private-league cookies. Unused in Phase 1; required from Phase 2 on.
    # These grant full league access — they belong in .env and nowhere else.
    espn_league_id: int | None = None
    espn_team_id: int | None = None
    espn_s2: str | None = None
    espn_swid: str | None = None

    # nflverse is a free community resource. Cache aggressively and stay a quiet
    # houseguest: one day is longer than the data changes during the week, and the
    # nightly refresh (Phase 3) busts it deliberately when new games land.
    nflverse_cache_duration: int = 86_400

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def nflverse_cache_dir(self) -> Path:
        return self.cache_dir / "nflverse"

    @property
    def derived_cache_dir(self) -> Path:
        return self.cache_dir / "derived"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def static_dir(self) -> Path:
        return self.data_dir / "static"

    def ensure_dirs(self) -> None:
        """Create the generated-data directories. Safe to call repeatedly."""
        for path in (
            self.nflverse_cache_dir,
            self.derived_cache_dir,
            self.outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def has_espn_credentials(self) -> bool:
        return all((self.espn_league_id, self.espn_s2, self.espn_swid))


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open() as handle:
        return yaml.safe_load(handle)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def load_league_config() -> dict[str, Any]:
    """Raw league.yaml. Prefer `patron.config.league.get_league()` for a typed view."""
    return _load_yaml("league.yaml")


@functools.lru_cache(maxsize=1)
def load_scoring_config() -> dict[str, Any]:
    """Raw scoring.yaml. Prefer `patron.scoring.rules.get_rules()` for a typed view."""
    return _load_yaml("scoring.yaml")


@functools.lru_cache(maxsize=1)
def load_overrides_config() -> dict[str, Any]:
    return _load_yaml("overrides.yaml")
