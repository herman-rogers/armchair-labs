"""ESPN credential storage.

Private-league access needs two cookies, `espn_s2` and `SWID`, which together grant
full access to the league under the user's account. They are treated accordingly:
written only to `.env`, which is gitignored, with file permissions tightened on write.

The `.env` file is edited rather than rewritten. It is the user's file — it may carry
unrelated settings and comments, and an auth flow has no business flattening them.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ESPN_S2_KEY = "ESPN_S2"
ESPN_SWID_KEY = "ESPN_SWID"
ESPN_LEAGUE_ID_KEY = "ESPN_LEAGUE_ID"
ESPN_TEAM_ID_KEY = "ESPN_TEAM_ID"


@dataclass(frozen=True)
class EspnCredentials:
    """The two cookies plus the league they address."""

    espn_s2: str
    swid: str
    league_id: int | None = None
    team_id: int | None = None

    def masked(self) -> str:
        """A form safe to print in a terminal or paste into an issue."""
        return (
            f"espn_s2={self.espn_s2[:6]}…{self.espn_s2[-4:]} "
            f"({len(self.espn_s2)} chars), SWID={self.swid[:10]}…"
        )

    def as_env(self) -> dict[str, str]:
        values = {ESPN_S2_KEY: self.espn_s2, ESPN_SWID_KEY: self.swid}
        if self.league_id is not None:
            values[ESPN_LEAGUE_ID_KEY] = str(self.league_id)
        if self.team_id is not None:
            values[ESPN_TEAM_ID_KEY] = str(self.team_id)
        return values


def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def read(env_path: Path) -> EspnCredentials | None:
    """Read credentials from `.env`, then the process environment.

    Environment variables win, so a homelab deployment can inject them without a file.
    """
    values = _parse_env(env_path.read_text()) if env_path.exists() else {}
    for key in (ESPN_S2_KEY, ESPN_SWID_KEY, ESPN_LEAGUE_ID_KEY, ESPN_TEAM_ID_KEY):
        if os.environ.get(key):
            values[key] = os.environ[key]

    espn_s2 = values.get(ESPN_S2_KEY)
    swid = values.get(ESPN_SWID_KEY)
    if not espn_s2 or not swid:
        return None

    def as_int(key: str) -> int | None:
        raw = values.get(key)
        try:
            return int(raw) if raw else None
        except ValueError:
            logger.warning("ignoring non-numeric %s=%r in .env", key, raw)
            return None

    return EspnCredentials(
        espn_s2=espn_s2,
        swid=swid,
        league_id=as_int(ESPN_LEAGUE_ID_KEY),
        team_id=as_int(ESPN_TEAM_ID_KEY),
    )


def write(credentials: EspnCredentials, env_path: Path) -> None:
    """Merge credentials into `.env`, preserving everything already there.

    Existing keys are replaced in place so their surrounding comments survive; new keys
    are appended. The file is then chmod 600 — it holds full league access, and a
    world-readable one on a shared machine is a real exposure.
    """
    updates = credentials.as_env()
    lines = env_path.read_text().splitlines() if env_path.exists() else []

    for key, value in updates.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        replaced = False
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines).rstrip("\n") + "\n")
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    logger.info("wrote %s to %s (mode 600)", ", ".join(updates), env_path)


def clear(env_path: Path) -> int:
    """Remove credential keys from `.env`. Returns how many were removed."""
    if not env_path.exists():
        return 0

    managed = {ESPN_S2_KEY, ESPN_SWID_KEY}
    kept, removed = [], 0
    for line in env_path.read_text().splitlines():
        key = line.partition("=")[0].strip()
        if key in managed and "=" in line:
            removed += 1
            continue
        kept.append(line)

    env_path.write_text("\n".join(kept).rstrip("\n") + "\n")
    return removed
