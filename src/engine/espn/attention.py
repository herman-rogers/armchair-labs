"""Current league attention flags describe captured evidence, never medical clearance."""

import json
from collections import Counter
from pathlib import Path

from engine.data.releases import load_manifest, reference
from engine.espn.sync import LeagueSnapshot

RESERVE = {"BE", "BN", "IR", "ER", ""}
CLEAR_TAGS = {"ACTIVE", "NORMAL", "HEALTHY"}
URGENT_TAGS = {
    "OUT",
    "O",
    "IR",
    "INJURY_RESERVE",
    "INJURED_RESERVE",
    "SUSPENSION",
    "SUSPENDED",
    "SSPD",
    "PUP",
    "DOUBTFUL",
    "D",
}


def reviewed_news(data_dir: Path, season: int) -> tuple[list[dict], str | None]:
    """Read the latest accepted injury capture, verifying its preserved sources."""
    candidates = []
    for path in (data_dir / "raw/snapshots").glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (
            manifest.get("kind") == "current_injury_capture"
            and manifest.get("status") == "accepted"
            and manifest.get("current_observations", {}).get("season") == season
        ):
            candidates.append((manifest.get("registered_at", ""), path.parent))
    if not candidates:
        return [], "No reviewed injury-news capture is available for this season."
    try:
        _, root = max(candidates)
        _, manifest = load_manifest(data_dir, "raw", reference(root))
        asset = manifest["assets"]["injuries/observations.json"]
        news = json.loads((data_dir / asset["object"]).read_text())["observations"]
        from engine.tables.application import active
        from engine.tables.league import query_documents

        if active():
            news = query_documents(
                data_dir / "league/news" / str(season), "league_reviewed_news", news
            )
        return [n for n in news if n.get("season") == season], None
    except (ValueError, OSError, KeyError):
        return (
            [],
            "Reviewed injury news could not be verified. ESPN status tags are shown separately.",
        )


def flags_for(player: dict, *, on_bye: bool = False, news=()) -> list[dict]:
    flags = []
    status = str(player.get("injury_status") or "").strip().upper().replace(" ", "_")
    if status not in CLEAR_TAGS:
        flags.append(
            {
                "kind": "injury" if status else "unknown_status",
                "severity": "urgent" if status in URGENT_TAGS else "watch",
                "label": f"ESPN: {status.replace('_', ' ')}" if status else "Status not reported",
            }
        )
    if on_bye:
        flags.append(
            {
                "kind": "bye",
                "severity": "urgent"
                if player.get("lineup_slot") not in RESERVE | {None}
                else "watch",
                "label": "Bye this week",
            }
        )
    if news:
        flags.append({"kind": "news", "severity": "watch", "label": "Review injury report"})
    if not player.get("player_id") and player.get("position") in {"QB", "RB", "WR", "TE"}:
        flags.append({"kind": "profile", "severity": "watch", "label": "Profile link unavailable"})
    return flags


def annotate_players(players: list[dict], snapshot: LeagueSnapshot, news: list[dict]) -> list[dict]:
    byes = {
        p.espn_id
        for game in snapshot.week_lineups
        if game.week == snapshot.week
        for p in game.home_lineup + game.away_lineup
        if p.on_bye
    }
    for player in players:
        notes = [n for n in news if n.get("espn_id") == player["espn_id"]]
        player["injury_news"] = notes
        player["attention"] = flags_for(player, on_bye=player["espn_id"] in byes, news=notes)
    return players


def roster_attention(snapshot: LeagueSnapshot) -> list[dict]:
    notices = []
    if snapshot.my_team_id is None:
        return [
            {
                "label": "Your team is not identified; select a roster to review player alerts.",
                "severity": "watch",
            }
        ]
    roster = [p for p in snapshot.players if p.owner_team_id == snapshot.my_team_id]
    if not roster:
        return [
            {
                "label": "Your roster was not captured; lineup completeness cannot be checked.",
                "severity": "watch",
            }
        ]
    slots = Counter(p.lineup_slot for p in roster)
    for slot, required in snapshot.roster_slots.items():
        if slot not in RESERVE and required > slots[slot]:
            notices.append(
                {
                    "label": (
                        f"{required - slots[slot]} unfilled {slot} starting slot(s) "
                        "in the saved roster. Review your lineup in ESPN."
                    ),
                    "severity": "urgent",
                }
            )
    team = next((t for t in snapshot.teams if t.team_id == snapshot.my_team_id), None)
    if team and team.faab_remaining == 0:
        notices.append(
            {"label": "No FAAB remaining in the captured league budget.", "severity": "watch"}
        )
    return notices
