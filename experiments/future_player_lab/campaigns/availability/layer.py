"""Point-in-time availability states; future participation never determines eligibility."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, date, datetime, time, timedelta

import polars as pl

STATUSES = (
    "ACT",
    "RET",
    "RET_ANNOUNCED",
    "RES",
    "PUP",
    "SUS",
    "DEV",
    "CUT",
    "INA",
    "EXE",
    "UNKNOWN",
)
INJURIES = ("Out", "Doubtful", "Questionable", "Probable")
EVENT_SCHEMA = {
    "player_id": pl.String,
    "available_at": pl.Datetime("us", "UTC"),
    "status": pl.String,
    "team": pl.String,
    "source": pl.String,
    "source_url": pl.String,
    "description": pl.String,
}


def following_day(value):
    """Date-grain observations become usable the following day at noon UTC."""
    if isinstance(value, str):
        value = date.fromisoformat(value[:10])
    return datetime.combine(value + timedelta(days=1), time(12), UTC)


def forecast_cutoff(schedule, year, origin):
    season = schedule.filter(pl.col("season") == year)
    if origin:
        days = season.filter(pl.col("week") <= origin)["gameday"].cast(pl.Date)
        if days.is_empty():
            raise ValueError("Missing completed-week calendar")
        return following_day(days.max())
    # Opening-day morning, strictly before any regular-season game.
    return datetime.combine(season["gameday"].cast(pl.Date).min(), time(12), UTC)


def normalize_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9 ]", "", name).split()
    if words and words[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        words.pop()
    return " ".join(words)


def name_index(identity, rosters):
    aliases = {}
    for frame, id_col, name_cols in (
        (identity, "gsis_id", ["display_name", "football_name"]),
        (rosters, "gsis_id", ["full_name"]),
    ):
        for column in name_cols:
            for player_id, name in frame.select(id_col, column).drop_nulls().unique().iter_rows():
                key = normalize_name(name)
                aliases.setdefault(key, set()).add(player_id)
    # Ambiguous normalized names are never assigned using eventual production.
    return {k: next(iter(v)) for k, v in aliases.items() if len(v) == 1}, {
        k: sorted(v) for k, v in aliases.items() if len(v) > 1
    }


def transaction_status(category, description):
    text = description.lower()
    # Destination precedes 'from'; reinstatements must supersede retirements.
    destination = text.split(" from ")[0]
    if "reinstat" in destination or "activated" in destination:
        return "ACT"
    if category in {"terminations", "waivers"}:
        return "CUT"
    if "retired" in destination or "retirement" in destination:
        return "RET"
    if "physically unable" in destination:
        return "PUP"
    if "suspend" in destination:
        return "SUS"
    if "reserve/" in destination:
        return "RES"
    if "practice squad" in destination:
        return "DEV"
    if "exempt" in destination:
        return "EXE"
    if category in {"signings", "trades"} or "activated" in text:
        return "ACT"
    return "UNKNOWN"


def build_events(transactions, rosters, identity, schedule):
    aliases, ambiguous = name_index(identity, rosters)
    rows, unmatched, unknown = [], [], []
    for row in transactions.iter_rows(named=True):
        player_id = row.get("player_id") or aliases.get(normalize_name(row["player_name"]))
        if not player_id:
            unmatched.append(row)
            continue
        status = row.get("status") or transaction_status(row["category"], row["description"])
        if status == "UNKNOWN":
            unknown.append(row)
        rows.append(
            dict(
                player_id=player_id,
                available_at=following_day(row["event_date"]),
                status=status,
                team=row["team"] or row.get("from_team"),
                source="transaction",
                source_url=row["source_url"],
                description=row["description"],
            )
        )
    week_dates = schedule.group_by("season", "week").agg(
        pl.col("gameday").cast(pl.Date).max().alias("week_end")
    )
    weekly = rosters.filter(pl.col("week") > 0).join(week_dates, on=["season", "week"], how="inner")
    for row in (
        weekly.select("gsis_id", "season", "week", "week_end", "team", "status")
        .drop_nulls("gsis_id")
        .unique()
        .iter_rows(named=True)
    ):
        status = row["status"]
        status = {
            "UFA": "CUT",
            "RFA": "CUT",
            "NWT": "CUT",
            "TRC": "CUT",
            "TRD": "CUT",
            "TRT": "CUT",
            "RSN": "RES",
            "RSR": "CUT",
        }.get(status, status)
        rows.append(
            dict(
                player_id=row["gsis_id"],
                available_at=following_day(row["week_end"]),
                status=status if status in STATUSES else "UNKNOWN",
                team=row["team"],
                source="weekly_roster",
                source_url=f"https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{row['season']}.parquet",
                description=(
                    f"season={row['season']}; completed_week={row['week']}; status={row['status']}"
                ),
            )
        )
    events = (
        pl.DataFrame(rows, schema=EVENT_SCHEMA).unique().sort("available_at", "player_id", "source")
    )
    return events, dict(
        matched_transaction_rows=len(transactions) - len(unmatched),
        unmatched_transactions=unmatched,
        ambiguous_names=ambiguous,
        unknown_transaction_states=unknown,
        matching="Unique normalized identity/roster name only; ambiguous names withheld",
    )


def attach_availability(panel, events, injuries, schedule):
    """Add a fixed feature vocabulary and auditable eligibility metadata.

    Retirements alone are a hard exclusion. Injury, suspension, reserve status,
    release, absence, or missing coverage do not imply zero rest-of-season value.
    State conflicts at identical timestamps stay unknown, including eligibility.
    """
    pieces = []
    for (year, origin), block in panel.partition_by(["year", "origin"], as_dict=True).items():
        cutoff = forecast_cutoff(schedule, year, origin)
        known = events.filter(
            (pl.col("available_at") <= cutoff)
            & pl.col("player_id").is_in(block["player_id"].implode())
        )
        # Only the current and preceding calendar years can describe the current
        # roster state. Older retirements are not assumed to persist indefinitely.
        known = known.filter(pl.col("available_at") >= datetime(year - 1, 1, 1, tzinfo=UTC))
        recent_counts = dict(
            known.filter(
                (pl.col("source") == "transaction")
                & (pl.col("available_at") >= cutoff - timedelta(days=30))
            )
            .group_by("player_id")
            .len()
            .iter_rows()
        )
        latest = {}
        for (player_id,), group in known.partition_by("player_id", as_dict=True).items():
            stamp = group["available_at"].max()
            last = group.filter(pl.col("available_at") == stamp)
            states, teams = (
                last["status"].unique().to_list(),
                last["team"].drop_nulls().unique().to_list(),
            )
            latest[player_id] = dict(
                status=states[0] if len(states) == 1 else "UNKNOWN",
                team=teams[0] if len(teams) == 1 else None,
                stamp=stamp,
                conflict=len(states) > 1 or len(teams) > 1,
                source="|".join(sorted(last["source"].unique())),
                evidence=" | ".join(sorted(last["source_url"].unique())),
                description=" | ".join(sorted(last["description"].unique())),
            )
        injury_by_id = {}
        if injuries.height:
            eligible_injuries = injuries.filter(
                (pl.col("season") == year)
                & pl.col("date_modified").is_not_null()
                & (pl.col("date_modified") <= cutoff)
            )
            for (player_id,), group in eligible_injuries.partition_by(
                "gsis_id", as_dict=True
            ).items():
                stamp = group["date_modified"].max()
                statuses = (
                    group.filter(pl.col("date_modified") == stamp)["report_status"]
                    .unique()
                    .to_list()
                )
                injury_by_id[player_id] = (statuses[0] if len(statuses) == 1 else None, stamp)
        additions = []
        for player in block.iter_rows(named=True):
            state = latest.get(player["player_id"])
            injury = injury_by_id.get(player["player_id"])
            status = state["status"] if state else "UNKNOWN"
            additions.append(
                {
                    "availability_status": status,
                    "availability_team": state["team"] if state else None,
                    "availability_as_of": cutoff.isoformat(),
                    "availability_observed_at": state["stamp"].isoformat() if state else None,
                    "availability_source": state["source"] if state else None,
                    "availability_evidence": state["evidence"] if state else None,
                    "availability_description": state["description"] if state else None,
                    "availability_eligible": status != "RET",
                    "availability__known": float(state is not None and status != "UNKNOWN"),
                    "availability__conflict": float(state["conflict"]) if state else 0.0,
                    "availability__current_year": float(state["stamp"].year == year)
                    if state
                    else 0.0,
                    "availability__days_since_observed": (cutoff - state["stamp"]).total_seconds()
                    / 86400
                    if state
                    else None,
                    "availability__team_changed": float(state["team"] != player["team"])
                    if state and state["team"]
                    else None,
                    "availability__transactions_30d": float(
                        recent_counts.get(player["player_id"], 0)
                    ),
                    **{f"availability__status_{s.lower()}": float(status == s) for s in STATUSES},
                    "availability__injury_report_known": float(injury is not None),
                    "availability__injury_report_age_days": (cutoff - injury[1]).total_seconds()
                    / 86400
                    if injury
                    else None,
                    **{
                        f"availability__injury_{s.lower()}": float(injury[0] == s)
                        if injury
                        else None
                        for s in INJURIES
                    },
                }
            )
        extra = pl.DataFrame(additions, infer_schema_length=None).with_columns(
            pl.col(
                "availability__days_since_observed",
                "availability__team_changed",
                "availability__injury_report_age_days",
                *[f"availability__injury_{s.lower()}" for s in INJURIES],
            ).cast(pl.Float64)
        )
        pieces.append(block.hstack(extra))
    return pl.concat(pieces, how="diagonal_relaxed").sort("year", "player_id")
