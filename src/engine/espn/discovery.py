"""League discovery and credential verification.

Once the browser flow has cookies, the league ID can be discovered rather than asked
for. ESPN's fan API returns the leagues an account manages, which saves digging a
number out of a URL and — more usefully — catches the case where the cookies are valid
but belong to an account that is not in the league being configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from engine.espn.credentials import EspnCredentials

logger = logging.getLogger(__name__)

FAN_API = "https://fan.api.espn.com/apis/v2/fans/{swid}"

#: ESPN's preference type for "fantasy league manager" entries.
FANTASY_LEAGUE_TYPE_ID = 9

#: Fantasy football. ESPN calls it FFL; other sports use the same fan API shape.
FOOTBALL_SLUG = "ffl"

TIMEOUT_SECONDS = 20


class DiscoveryError(RuntimeError):
    """Raised when ESPN could not be queried for the account's leagues."""


@dataclass(frozen=True)
class DiscoveredLeague:
    league_id: int
    name: str
    season: int
    team_id: int | None
    team_name: str | None

    def describe(self) -> str:
        team = f" — your team: {self.team_name}" if self.team_name else ""
        return f"{self.name} ({self.season}, id {self.league_id}){team}"


def _cookie_jar(credentials: EspnCredentials) -> dict[str, str]:
    return {"espn_s2": credentials.espn_s2, "SWID": credentials.swid}


def discover_leagues(credentials: EspnCredentials) -> list[DiscoveredLeague]:
    """List the fantasy football leagues this account manages.

    Best-effort. The fan API is as unofficial as the rest of ESPN's fantasy surface, so
    a shape change here should degrade to "ask for the league id", never break auth.
    """
    swid = credentials.swid.strip()
    url = FAN_API.format(swid=swid)
    params = {
        "displayEvents": "false",
        "displayNow": "true",
        "displayRecs": "false",
        "featureFlags": "expandAthlete",
        "showAirings": "false",
        "source": "ESPN.com+-+FAN+API",
        "lang": "en",
        "region": "us",
    }

    try:
        response = requests.get(
            url,
            params=params,
            cookies=_cookie_jar(credentials),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise DiscoveryError(f"Could not reach ESPN's fan API: {error}") from error

    if response.status_code == 401:
        raise DiscoveryError("ESPN rejected the credentials (401). They may have expired.")
    if not response.ok:
        raise DiscoveryError(f"ESPN's fan API returned {response.status_code}.")

    try:
        preferences = response.json().get("preferences") or []
    except ValueError as error:
        raise DiscoveryError("ESPN's fan API returned a response that was not JSON.") from error

    leagues: list[DiscoveredLeague] = []
    for preference in preferences:
        if preference.get("typeId") != FANTASY_LEAGUE_TYPE_ID:
            continue
        entry = preference.get("metaData", {}).get("entry")
        if not entry:
            continue

        groups = entry.get("groups") or [{}]
        for group in groups:
            league_id = group.get("groupId")
            if league_id is None:
                continue
            if entry.get("abbrev", "").lower() not in {FOOTBALL_SLUG, ""}:
                continue
            leagues.append(
                DiscoveredLeague(
                    league_id=int(league_id),
                    name=group.get("groupName") or "Unnamed league",
                    season=int(entry.get("seasonId") or 0),
                    team_id=entry.get("entryId"),
                    team_name=entry.get("name"),
                )
            )

    leagues.sort(key=lambda league: (-league.season, league.name))
    logger.info("discovered %s fantasy league(s)", len(leagues))
    return leagues


def verify(credentials: EspnCredentials, league_id: int, season: int) -> tuple[bool, str]:
    """Check whether the credentials can actually read the league.

    Returns:
        `(ok, message)`. The message is written for a terminal — it should say what to
        do next, not just what failed.
    """
    from espn_api.football import League
    from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague

    try:
        league = League(
            league_id=league_id,
            year=season,
            espn_s2=credentials.espn_s2,
            swid=credentials.swid,
        )
    except ESPNAccessDenied:
        return False, (
            "ESPN denied access. The cookies are probably expired — run `engine auth login` again."
        )
    except ESPNInvalidLeague:
        return False, (
            f"League {league_id} does not exist for the {season} season. "
            "Check the league id, or run `engine auth login` to rediscover it."
        )
    except Exception as error:  # noqa: BLE001 - unofficial API, unbounded failure modes
        return False, f"Could not read league {league_id}: {type(error).__name__}: {error}"

    return True, (
        f"Connected to '{league.settings.name}' — "
        f"{len(league.teams)} teams, {season} season, week {league.current_week}."
    )
