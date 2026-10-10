"""Normalize dated evidence and answer as-of queries without outcome filtering."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from engine.metrics.transaction_events import player_clause

TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "OAK": "LV",
    "SD": "LAC",
    "SDG": "LAC",
    "SL": "LA",
    "STL": "LA",
    "LAR": "LA",
    "WSH": "WAS",
}

ACTION = re.compile(
    r"(?<![/\w-])(?:activated|reinstated|promoted|elevated|waived|released|terminated|"
    r"placed|signed|re-signed|resigned|claimed|traded|acquired|announced|designated|"
    r"returned|reverted|extended|exercised|declined)\b",
    re.I,
)

SCHEMA = {
    "event_id": pl.String,
    "player_id": pl.String,
    "player_name": pl.String,
    "dimension": pl.String,
    "status": pl.String,
    "team": pl.String,
    "event_date": pl.Date,
    "usable_at": pl.Datetime("us", "UTC"),
    "published_at": pl.Datetime("us", "UTC"),
    "collected_at": pl.Datetime("us", "UTC"),
    "source": pl.String,
    "source_url": pl.String,
    "capture_key": pl.String,
    "raw_sha256": pl.String,
    "raw_record_id": pl.String,
    "identity_basis": pl.String,
    "identity_evidence_id": pl.String,
    "date_basis": pl.String,
    "season": pl.Int32,
    "week": pl.Int32,
    "description": pl.String,
}


def normalize(value: str) -> str:
    value = re.sub(r"['’]s\b", "", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9 ]", "", value).split()
    if words and words[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        words.pop()
    return " ".join(words)


def stamp(value) -> datetime | None:
    if value is None:
        return None
    result = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    # A timezone-free modification time does not establish an absolute cutoff.
    return result.astimezone(UTC) if result.tzinfo else None


def next_day(value: date) -> datetime:
    return datetime.combine(value + timedelta(days=1), time(12), UTC)


def record_id(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()


def event(row: dict, **values) -> dict:
    result = {key: None for key in SCHEMA}
    for key in ("source", "source_url", "capture_key", "raw_sha256", "description"):
        result[key] = row.get(key)
    result.update(
        collected_at=stamp(row["collected_at"]),
        published_at=stamp(row.get("published_at")),
        raw_record_id=row.get("raw_record_id") or record_id(row),
    )
    result.update(values)
    result["team"] = TEAM_ALIASES.get(result["team"], result["team"])
    if result["published_at"]:
        result["usable_at"] = max(result["usable_at"], result["published_at"])
    result["event_id"] = record_id(result)
    return result


def identity_index(players: pl.DataFrame, rosters: pl.DataFrame):
    names: dict[str, set[str]] = {}
    display = {}
    for frame, columns in (
        (players, ["display_name", "football_name"]),
        (rosters, ["full_name", "football_name"]),
    ):
        for col in columns:
            if col not in frame.columns:
                continue
            for pid, name in frame.select("gsis_id", col).drop_nulls().unique().iter_rows():
                names.setdefault(normalize(name), set()).add(pid)
                display.setdefault(pid, name)
    return names, display


def classify(clause: str) -> tuple[str, str] | None:
    """Do not turn announcements, extensions, or return windows into activations."""
    text = clause.lower()
    if len(list(ACTION.finditer(text))) > 1:
        return None
    if re.search(r"\bremains?\b.*(?:reserve|suspend|inactive)", text):
        return None
    if re.search(r"\b(designated|cleared)\b.*\breturn\b|return.*\bwindow\b", text):
        return "return_notice", "RETURN_WINDOW"
    if re.search(r"\b(activated|reinstated|promoted|elevated)\b|returned to.*active", text):
        return "roster", "ACT"
    if re.search(r"\b(waived|released|terminated)\b", text):
        return "roster", "CUT"
    if "retir" in text:
        if re.search(r"\bplaced\b.*reserve[ /-]+retired", text):
            return "roster", "RET"
        return "retirement_notice", "RET_ANNOUNCED"
    if re.search(r"\b(extended|extension|exercised|declined|drafted|selected)\b", text):
        return None
    if "suspend" in text:
        return "roster", "SUS"
    if re.search(r"\bplaced\b", text):
        if re.search(r"physically unable|\bpup\b", text):
            return "roster", "PUP"
        if "non-football" in text or re.search(r"\bnfi\b", text):
            return "roster", "NFI"
        if "exempt" in text:
            return "roster", "EXE"
        if "reserve" in text or re.search(r"\bir\b", text):
            return "roster", "RES"
    if "practice squad" in text:
        if re.search(r"signed.*\b(from|off)\b.*practice squad", text):
            return "roster", "ACT"
        if re.search(r"\b(signed|re-signed|returned|reverted)\b", text):
            return "roster", "DEV"
    if re.search(r"\b(signed|re-signed|resigned|claimed)\b", text):
        return "roster", "FUTURE" if "future" in text else "ROSTERED"
    # Trade prose may name two players and two directions. Retain for review.
    return None


def prior_roster_index(rows: list[dict]) -> dict:
    index = {}
    for row in rows:
        if row["source"] == "weekly_roster" and row["team"]:
            key = row["player_id"], row["team"]
            index.setdefault(key, []).append(
                (row["usable_at"], row["event_id"], row["status"], row["collected_at"])
            )
    return {key: sorted(values) for key, values in index.items()}


def transaction_events(reports: list[dict], names, display, prior_rosters=None):
    lengths = sorted({len(n.split()) for n in names if len(n.split()) >= 2}, reverse=True)
    rows, audit = [], []
    for report in reports:
        rid = report["raw_record_id"]
        day = date.fromisoformat(str(report["event_date"])[:10])
        clock = next_day(day)
        words = normalize(report["description"]).split()
        matched, occupied = set(), set()
        for length in lengths:
            for start in range(len(words) - length + 1):
                span = set(range(start, start + length))
                if span & occupied:
                    continue
                name = " ".join(words[start : start + length])
                if name in names:
                    matched.add(name)
                    occupied.update(span)
        if not matched:
            audit.append(dict(raw_record_id=rid, reason="no_identity_match", name=None))
            continue
        pids, identity_evidence = {}, {}
        for name in matched:
            if len(names[name]) != 1:
                supported = {}
                team = TEAM_ALIASES.get(report.get("team"), report.get("team"))
                for pid in names[name]:
                    history = (prior_rosters or {}).get((pid, team), [])
                    offset = bisect_right(history, (clock, "~")) - 1
                    tied_statuses = set()
                    if offset >= 0:
                        left = bisect_left(history, (history[offset][0], ""))
                        tied_statuses = {h[2] for h in history[left : offset + 1]}
                    if (
                        offset >= 0
                        and history[offset][0] >= clock - timedelta(days=365)
                        and len(tied_statuses) == 1
                        and history[offset][2] not in {"RET", "CUT", "UNKNOWN"}
                    ):
                        supported[pid] = history[offset]
                if len(supported) == 1:
                    pid, evidence = next(iter(supported.items()))
                    pids.setdefault(pid, set()).add(name)
                    identity_evidence[pid] = evidence
                else:
                    audit.append(dict(raw_record_id=rid, reason="ambiguous_identity", name=name))
            else:
                pids.setdefault(next(iter(names[name])), set()).add(name)
        for pid, aliases in pids.items():
            prose = report["description"].replace("|", "; ")
            prose = re.sub(
                r"(?<=[a-z)])(?=(?:Placed|Released|Signed|Waived|Terminated|Activated|"
                r"Reinstated|Reserve/|Traded)\b)",
                ". ",
                prose,
            )
            clause = player_clause(prose, aliases, normalize)
            # A verb between two named players is not a safe shared action.
            action = next(ACTION.finditer(clause), None)
            if action:
                before = " " + normalize(clause[: action.start()]) + " "
                after = " " + normalize(clause[action.end() :]) + " "
                if any(" " + name + " " in before for name in matched) and any(
                    " " + name + " " in after for name in matched
                ):
                    audit.append(
                        dict(raw_record_id=rid, reason="mixed_player_actions", name=display[pid])
                    )
                    continue
            # A player repeated in multiple action clauses needs explicit review.
            normalized = " " + normalize(report["description"]) + " "
            if any(normalized.count(" " + a + " ") > 1 for a in aliases):
                audit.append(
                    dict(raw_record_id=rid, reason="repeated_player_actions", name=display[pid])
                )
                continue
            # An explicitly later date embedded in a retrospective report must not
            # inherit the earlier archive date. Keep the record, withhold inference.
            if re.search(r"\bon\s+\d{1,2}/\d{1,2}|\bthen\b", clause, re.I):
                audit.append(
                    dict(raw_record_id=rid, reason="embedded_chronology", name=display[pid])
                )
                continue
            state = classify(clause)
            if state is None:
                audit.append(dict(raw_record_id=rid, reason="unparsed_action", name=display[pid]))
                continue
            rows.append(
                event(
                    report,
                    player_id=pid,
                    player_name=display[pid],
                    dimension=state[0],
                    status=state[1],
                    team=None if state[1] == "CUT" else report.get("team"),
                    event_date=day,
                    usable_at=next_day(day),
                    date_basis="event_date_plus_next_day_noon",
                    identity_basis=(
                        "exact_name_prior_team_roster"
                        if pid in identity_evidence
                        else "exact_unique_name"
                    ),
                    identity_evidence_id=(
                        identity_evidence[pid][1] if pid in identity_evidence else None
                    ),
                    collected_at=max(stamp(report["collected_at"]), identity_evidence[pid][3])
                    if pid in identity_evidence
                    else stamp(report["collected_at"]),
                    description=clause,
                )
            )
    return rows, audit


def as_of(
    events: pl.DataFrame,
    cutoff: datetime,
    *,
    mode: str = "observed",
    max_age_days=550,
    injury_max_age_days=14,
    return_notice_max_age_days=30,
    player_ids: list[str] | None = None,
) -> pl.DataFrame:
    """One latest state per player and dimension, with evidence and conflicts.

    Observed mode requires that our archive actually held the record at cutoff.
    Reconstructed mode uses dated retrospective evidence and explicitly says so.
    Missing players get UNKNOWN roster rows when a candidate list is supplied.
    This function makes no hard eligibility decision or health inference.
    """
    if mode not in {"observed", "reconstructed"} or cutoff.tzinfo is None:
        raise ValueError("Use a timezone-aware cutoff and observed/reconstructed mode")
    if min(max_age_days, injury_max_age_days, return_notice_max_age_days) <= 0:
        raise ValueError("Evidence age limits must be positive")
    known = events.filter(
        (pl.col("usable_at") <= cutoff)
        & (pl.col("usable_at") >= cutoff - timedelta(days=max_age_days))
    )
    known = known.filter(
        (pl.col("dimension") != "injury_report")
        | (pl.col("usable_at") >= cutoff - timedelta(days=injury_max_age_days))
    )
    known = known.filter(
        (pl.col("dimension") != "return_notice")
        | (pl.col("usable_at") >= cutoff - timedelta(days=return_notice_max_age_days))
    )
    if mode == "observed":
        known = known.filter(pl.col("collected_at") <= cutoff)
    if player_ids is not None:
        known = known.filter(pl.col("player_id").is_in(player_ids))
    # Keep all evidence at the latest timestamp; input row order cannot resolve a tie.
    last = known.filter(
        pl.col("usable_at") == pl.col("usable_at").max().over("player_id", "dimension")
    )
    result = (
        last.group_by("player_id", "dimension")
        .agg(
            pl.col("status").unique().sort().alias("statuses"),
            pl.col("team").drop_nulls().unique().sort().alias("teams"),
            pl.col("usable_at").max(),
            pl.col("event_id").sort().alias("evidence_ids"),
            pl.col("source").unique().sort().alias("sources"),
            pl.col("date_basis").unique().sort().alias("date_bases"),
            pl.col("season").drop_nulls().unique().sort().alias("seasons"),
            pl.col("week").drop_nulls().unique().sort().alias("weeks"),
        )
        .with_columns(
            ((pl.col("statuses").list.len() > 1) | (pl.col("teams").list.len() > 1)).alias(
                "conflict"
            ),
        )
        .with_columns(
            pl.when(pl.col("conflict"))
            .then(pl.lit("UNKNOWN"))
            .otherwise(pl.col("statuses").list.first())
            .alias("status"),
            pl.when(pl.col("teams").list.len() == 1)
            .then(pl.col("teams").list.first())
            .otherwise(None)
            .alias("team"),
            pl.lit(mode).alias("vintage_mode"),
            ((pl.lit(cutoff) - pl.col("usable_at")).dt.total_seconds() / 86400).alias("age_days"),
        )
    )
    if player_ids is not None:
        present = set(result.filter(pl.col("dimension") == "roster")["player_id"])
        missing = [
            dict(
                player_id=p,
                dimension="roster",
                status="UNKNOWN",
                conflict=False,
                vintage_mode=mode,
                statuses=[],
                teams=[],
                evidence_ids=[],
                sources=[],
            )
            for p in sorted(set(player_ids) - present)
        ]
        if missing:
            result = pl.concat(
                [result, pl.DataFrame(missing, schema=result.schema)], how="vertical"
            )
    return result.sort("player_id", "dimension")
