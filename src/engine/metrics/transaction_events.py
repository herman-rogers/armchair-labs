"""Conservative, player-local interpretation of dated transaction announcements.

An announcing club is NOT a trade destination. Missing destinations and conflicting
same-day actions are deliberately unresolved, never decided by input row order.
"""

from __future__ import annotations

import re
from typing import Any

import polars as pl

_CLUBS = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}
_ALIASES = {code: {name.lower(), name.lower().rsplit(" ", 1)[-1]} for code, name in _CLUBS.items()}
for _code, _name in _CLUBS.items():
    # A city without its nickname is safe except for the two shared cities.
    _city = _name.rsplit(" ", 1)[0].lower()
    if _city not in {"new york", "los angeles"}:
        _ALIASES[_code].add(_city)
_ALIASES["WAS"].update(
    {"washington", "washington redskins", "redskins", "washington football team"}
)
_ALIASES["LV"].update({"oakland", "oakland raiders"})
_ALIASES["LA"].update({"st louis", "st. louis", "st. louis rams"})
_ALIASES["LAC"].update({"san diego", "san diego chargers"})

_VERBS = (
    r"(?:activated|reinstated|removed|returned|placed|waived|released|terminated|"
    r"signed|re-signed|resigned|acquired|claimed|traded|trade|retired|suspended|"
    r"designated|elevated|promoted|agreed|reached|exercised|declined|drafted|selected|extended)\b"
)


def _canonical_verbs(text: str) -> str:
    forms = {
        "sign": "signed",
        "signs": "signed",
        "re-sign": "re-signed",
        "re-signs": "re-signed",
        "waive": "waived",
        "waives": "waived",
        "release": "released",
        "releases": "released",
        "terminate": "terminated",
        "terminates": "terminated",
        "activate": "activated",
        "activates": "activated",
        "place": "placed",
        "places": "placed",
        "acquire": "acquired",
        "acquires": "acquired",
        "claim": "claimed",
        "claims": "claimed",
        "agree": "agreed",
        "agrees": "agreed",
        "select": "selected",
        "selects": "selected",
        "extend": "extended",
        "extends": "extended",
        "reinstate": "reinstated",
        "reinstates": "reinstated",
    }
    return re.sub(
        r"\b(?:" + "|".join(re.escape(word) for word in forms) + r")\b",
        lambda match: forms[match[0].lower()],
        text,
        flags=re.I,
    )


def normalize_transaction_sources(frame: pl.DataFrame) -> pl.DataFrame:
    """Migrate v1 raw caches without rewriting them; retain all source provenance."""
    if "source_team" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("source_team"))
    prose = pl.col("category").is_in(["official", "espn"])
    return frame.with_columns(
        pl.when(prose)
        .then(pl.coalesce("source_team", "to_team"))
        .otherwise(pl.col("source_team"))
        .alias("source_team"),
        pl.when(prose).then(None).otherwise(pl.col("to_team")).alias("to_team"),
    )


def player_clause(description: str, aliases: set[str], normalize: Any) -> str:
    # Initials and Jr./Sr. are not sentence boundaries.
    text = re.sub(r"\b(Jr|Sr|[A-Z])\.(?=\s|[A-Z]\.)", r"\1", description)
    text = _canonical_verbs(text)
    text = re.sub(r"\bNo\.\s*(?=\d)", "No ", text)
    text = re.sub(rf"\.(?={_VERBS})", ". ", text, flags=re.IGNORECASE)
    clauses = re.split(r"[;]|\.\s+", text)
    for sentence in clauses:
        # Preserve lists sharing one verb, split only at a new action.
        pieces = re.split(
            rf"(?:,\s*(?:and\s+)?|\s+(?:and|&)\s+)(?=(?:the team\s+)?{_VERBS})",
            sentence,
            flags=re.IGNORECASE,
        )
        for piece in pieces:
            # A shared 'placed' verb can introduce several DIFFERENT reserve lists.
            # Split only when each entry supplies its own 'on ...' mechanism; lists
            # sharing a single trailing mechanism must remain together.
            if len(re.findall(r"\bon\s+(?:reserve|active|exempt|injured)", piece, re.I)) > 1:
                members = re.split(
                    r",\s*(?:and\s+)?(?=(?:QB|RB|FB|HB|WR|TE|OL|OT|OG|G|C|T|DL|DT|DE|LB|OLB|ILB|CB|DB|S|K|P|LS)\s)",
                    piece,
                )
                for member in members:
                    normalized = f" {normalize(member)} "
                    if any(f" {alias} " in normalized for alias in aliases):
                        return member if re.search(_VERBS, member, re.I) else "Placed " + member
            normalized = f" {normalize(piece)} "
            if any(f" {alias} " in normalized for alias in aliases):
                return piece
    return description


def _destination(text: str, preposition: str) -> str | None:
    # Require the direction word adjacent to a club name, not merely somewhere in
    # prose (which may also mention the announcing team or a draft-pick partner).
    for code, names in _ALIASES.items():
        for name in sorted(names, key=len, reverse=True):
            if re.search(rf"\b{preposition}\s+(?:the\s+)?{re.escape(name)}\b", text):
                return code
    return None


def interpret_event(
    event: dict,
    clause: str,
    current_team: str | None,
    *,
    current_status: str = "unknown",
    matched_aliases: set[str] | None = None,
) -> dict:
    text = _canonical_verbs(clause).lower()
    category = str(event.get("category") or "")
    source = event.get("source_team")
    destination = event.get("to_team")
    team = destination or source or event.get("from_team") or current_team
    status, action = "unknown", "unparsed"
    if re.search(r"\b(retired|retirement)\b", text):
        team, status, action = None, "off", "retire"
    elif any(
        term in text
        for term in (
            "activated",
            "reinstated",
            "removed from",
            "returned to active",
            "suspension lifted",
            "exemption lifted",
            "taken off ir",
            "promoted",
            "elevated",
        )
    ):
        status, action = "active", "activate"
    elif category == "trades" or re.search(r"\b(trade|traded|acquired)\b", text):
        # 'Traded picks to ATL for WR Julio Jones' is TEN acquiring the player.
        # Direction must refer to this player, not the draft-pick consideration.
        after_for = re.split(r"\bfor\b", text, maxsplit=1)[-1] if "for" in text else ""
        normalized_for = (
            " "
            + " ".join(
                re.sub(r"[^a-z0-9 ]", " ", after_for.replace("'", "").replace(".", "")).split()
            )
            + " "
        )
        player_received = bool(matched_aliases) and any(
            f" {alias} " in normalized_for for alias in (matched_aliases or set())
        )
        if "acquired" in text:
            # Acquiring a pick *for* the player is an outgoing trade, not a signing.
            outgoing = bool(re.search(r"acquired.*(?:selection|pick).*\bfor\b", text))
            team = _destination(text, "(?:with|from)") if outgoing else (destination or source)
        elif player_received:
            team = destination or source
        else:
            team = destination or _destination(text, "to")
        status, action = ("active" if team else "unknown"), "trade"
    elif (
        re.search(r"\b(waived|released|terminated|cut)\b", text)
        or "declared free agent" in text
        or category == "terminations"
        or category == "waivers"
        and "claim" not in text
    ):
        team, status, action = None, "off", "release"
    elif (
        any(
            term in text
            for term in (
                "injured reserve",
                "reserve/injured",
                "reserve/injury",
                "physically unable",
                "pup list",
                "reserve/pup",
                "active/pup",
                "non-football",
                "nfi list",
                "covid",
                "suspend",
                "exempt",
                "not with club",
                "did not report",
            )
        )
        or re.search(r"\bopt[ -]?out\b|\bpup\b|\bnfi\b", text)
        or category == "reserve-list"
        or re.search(r"\b(?:placed|on)\s+(?:on\s+)?ir\b", text)
    ):
        status, action = "reserve", "reserve"
    elif re.search(r"signed.*\b(?:from|off)\b.*practice squad", text):
        status, action = "active", "sign"
    elif "practice squad" in text:
        status, action = "practice", "practice"
    elif re.search(r"\b(extension|extended|exercised|new deal)\b", text) or (
        "contract" in text and "reached" in text
    ):
        # An extension establishes the club, but does not lift an existing absence.
        status, action = current_status, "contract"
    elif re.search(r"\b(drafted|selected)\b", text):
        status, action = current_status, "draft"
    elif "passed physical" in text:
        status, action = current_status, "physical"
    elif (
        re.search(r"\b(signed|re-signed|claimed|promoted|elevated)\b", text)
        or category
        in {
            "signings",
            "waivers",
        }
        or "agreed to terms" in text
        or "signing" in text
        or "awarded from waivers" in text
    ):
        status, action = "active", "sign"
    return {
        "team": team,
        "status": status,
        "action": action,
        "clause": clause,
        "category": category,
        "source_url": event.get("source_url"),
        "source_team": source,
        "transaction_date": event["transaction_date"],
    }


def availability_class(category: str, description: str, status: str) -> str:
    """Preserve the kind of dated availability event instead of one reserve bit.

    Transaction prose cannot tell us a medical prognosis, but it can distinguish an
    injured-reserve move from PUP/NFI, suspension, and other reserve mechanisms.  The
    fitted games model is allowed to learn their different historical consequences;
    none of the labels use a later roster or game outcome.
    """
    text = description.lower()
    if status == "active":
        return "active"
    if status == "practice":
        return "practice"
    if status == "off":
        return "off"
    if "suspend" in text:
        return "suspended"
    if re.search(r"\bopt[ -]?out\b", text):
        return "opt_out"
    if "covid" in text:
        return "covid"
    if "exempt" in text or "not with club" in text:
        return "exempt"
    if re.search(r"\bpup\b|\bnfi\b", text) or any(
        term in text
        for term in (
            "physically unable",
            "pup list",
            "reserve/pup",
            "active/pup",
            "non-football injury",
            "non-football illness",
            "nfi list",
        )
    ):
        return "pup_nfi"
    if any(
        term in text for term in ("injured reserve", "reserve/injured", "reserve/injury")
    ) or re.search(r"\bon ir\b", text):
        return "injured_reserve"
    if status == "reserve" or category == "reserve-list":
        return "reserve_other"
    return status


def resolve_day(events: list[dict]) -> tuple[dict, str]:
    """Resolve duplicate announcements, never invent intra-day chronology."""
    # Official authority is local to this player's day, not an entire team-season.
    official = [row for row in events if row["category"] == "official"]
    candidates = official or events
    parsed = [row for row in candidates if row["action"] != "unparsed"]
    candidates = parsed or candidates
    # An outgoing trade with no destination is complemented by its acquiring club.
    known_trades = [row for row in candidates if row["action"] == "trade" and row["team"]]
    if known_trades and all(row["action"] == "trade" for row in candidates):
        candidates = known_trades
    states = {
        (
            row["team"],
            row["status"],
            availability_class(row["category"], row["clause"], row["status"]),
        )
        for row in candidates
    }
    representative = sorted(candidates, key=lambda row: (str(row["source_url"]), row["clause"]))[0]
    if len(states) != 1:
        teams = {row["team"] for row in candidates}
        return {
            **representative,
            "team": next(iter(teams)) if len(teams) == 1 else None,
            "status": "unknown",
        }, "conflicting_same_day"
    return representative, (
        "observed"
        if representative["status"] != "unknown"
        else "team_observed_status_unknown"
        if representative["action"] in {"contract", "draft", "physical"} and representative["team"]
        else "unresolved_action"
    )
