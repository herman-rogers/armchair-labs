"""Cutoff-aware, report-only features for additive projection challengers.

Nothing in this module is wired into the live board.  It constructs one row per
returning player and forecast season so the walk-forward report can decide whether
roster status, vacated opportunity, snaps, draft capital, or contracts add anything
beyond the production model.  Historical weekly rosters and depth charts have no
publication timestamps, so their Week 1 rows remain explicitly labelled proxies.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import date
from typing import cast

import polars as pl

from patron.metrics.enrichment import depth_chart_as_of
from patron.metrics.positions import canonical_positions
from patron.metrics.transaction_events import (
    availability_class as _availability_class,
)
from patron.metrics.transaction_events import (
    interpret_event,
    normalize_transaction_sources,
    player_clause,
    resolve_day,
)
from patron.scoring.columns import require_columns

_SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
_ROSTERED_STATUSES = ("ACT", "INA", "RES")


def _normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("'", "").replace(".", "")).split()
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return " ".join(tokens)


def _transaction_aliases(players: pl.DataFrame) -> dict[str, set[str]]:
    """Full-name nflverse aliases keyed by GSIS id, excluding short-name aliases."""
    aliases: dict[str, set[str]] = {}
    available = set(players.columns)
    for row in players.to_dicts():
        player_id = str(row.get("gsis_id") or "")
        if not player_id:
            continue
        values = [row.get("display_name")]
        first = row.get("first_name")
        last = row.get("last_name")
        if first and last:
            values.append(f"{first} {last}")
        common = row.get("common_first_name")
        if common and last:
            values.append(f"{common} {last}")
        football = row.get("football_name")
        if football and last:
            values.append(f"{football} {last}")
        aliases[player_id] = {
            normalized
            for value in values
            if value is not None and (normalized := _normalized_name(value))
        }
    # Keep the schema check explicit even though aliases tolerate optional columns.
    require_columns(list(available), ("gsis_id", "display_name"), "transaction identity input")
    return aliases


def build_transaction_features(
    transactions: pl.DataFrame,
    player_seasons: pl.DataFrame,
    players: pl.DataFrame,
    forecast_seasons: Iterable[int],
    cutoff: str,
    *,
    cutoff_by_season: Mapping[int, date] | None = None,
    trusted_sources_only: bool = False,
    forecast_candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Reconstruct candidates' states at the historical draft cutoff.

    The state begins with the last source-season team and changes only when a dated
    transaction on or before the cutoff says the player moved, was cut, joined a
    reserve list, or signed. Names from ESPN's team-level descriptions are resolved
    within the returner and optional rookie/market pools; ambiguous aliases are accepted
    only when the transaction team selects exactly one candidate. Ambiguity is
    checked against all known identities, including players outside the fold.
    """
    require_columns(
        transactions.columns,
        (
            "transaction_date",
            "transaction_year",
            "category",
            "from_team",
            "to_team",
            "player_name",
            "description",
        ),
        "NFL transaction input",
    )
    require_columns(
        player_seasons.columns,
        ("season", "player_id", "player_display_name", "team", "position"),
        "player season input",
    )
    transactions = normalize_transaction_sources(transactions)
    if trusted_sources_only:
        # ESPN's historical event dates have known errors. Keep the raw archive for
        # diagnostic replay, but do not let an unverified date establish cutoff state.
        transactions = transactions.filter(pl.col("category") != "espn")
    aliases = _transaction_aliases(players)
    identities: dict[str, set[str]] = {}
    for player_id, names in aliases.items():
        for name in names:
            identities.setdefault(name, set()).add(player_id)
    cutoff_month, cutoff_day = (int(part) for part in cutoff.split("-"))
    output: list[dict[str, object]] = []
    for forecast_season in forecast_seasons:
        source_rows = (
            player_seasons.filter(
                (pl.col("season") == forecast_season - 1)
                & pl.col("position").is_in(_SKILL_POSITIONS)
            )
            .sort("player_id")
            .unique(subset=["player_id"], keep="last")
            .to_dicts()
        )
        prior_ids = {row["player_id"] for row in source_rows}
        if forecast_candidates is not None:
            extras = (
                forecast_candidates.filter(
                    (pl.col("forecast_season") == forecast_season)
                    & ~pl.col("player_id").is_in(prior_ids)
                )
                .unique("player_id")
                .to_dicts()
            )
            source_rows.extend({**row, "team": None} for row in extras)
        if not source_rows:
            continue
        state: dict[str, dict[str, object]] = {}
        by_alias: dict[str, set[str]] = {}
        for row in source_rows:
            player_id = str(row["player_id"])
            player_aliases = set(aliases.get(player_id, set()))
            player_aliases.add(_normalized_name(row.get("player_display_name")))
            for alias in player_aliases:
                if alias:
                    by_alias.setdefault(alias, set()).add(player_id)
            state[player_id] = {
                "team": row.get("team"),
                "status": "unknown",
                "availability_class": "unknown",
                "resolution": "inferred_prior_team" if player_id in prior_ids else "no_prior_team",
                "source_url": None,
                "clause": None,
                "evidence": [],
                "transaction_count": 0,
                "last_transaction_date": None,
                "matched": 0.0,
            }
        cutoff_date = (cutoff_by_season or {}).get(
            forecast_season, date(forecast_season, cutoff_month, cutoff_day)
        )
        events = (
            transactions.filter(
                (pl.col("transaction_year") == forecast_season)
                & (pl.col("transaction_date") <= cutoff_date)
            )
            .sort("transaction_date")
            .to_dicts()
        )
        by_day: dict[date, list[dict]] = {}
        for event in events:
            by_day.setdefault(event["transaction_date"], []).append(event)
        for event_date, daily_events in sorted(by_day.items()):
            pending: dict[str, list[dict]] = {}
            for event in daily_events:
                event_teams = {
                    event.get("source_team"),
                    event.get("from_team"),
                    event.get("to_team"),
                } - {None}
                raw_name = _normalized_name(event.get("player_name"))
                if raw_name:
                    candidates = set(by_alias.get(raw_name, set()))
                else:
                    normalized_description = _normalized_name(event.get("description"))
                    candidates = set()
                    for alias, player_ids in by_alias.items():
                        if (
                            len(alias.split()) < 2
                            or f" {alias} " not in f" {normalized_description} "
                        ):
                            continue
                        # A name can also belong to a defender or a player outside
                        # this fold. Single-candidate folds do not establish identity.
                        if len(identities.get(alias, player_ids)) == 1:
                            candidates.update(player_ids)
                            continue
                        team_matches = {
                            player_id
                            for player_id in player_ids
                            if state[player_id].get("team") in event_teams
                        }
                        if len(team_matches) == 1:
                            candidates.update(team_matches)
                if raw_name and len(identities.get(raw_name, candidates)) > 1:
                    candidates = {
                        player_id
                        for player_id in candidates
                        if state[player_id].get("team") in event_teams
                    }
                for player_id in sorted(candidates):
                    names = set(aliases.get(player_id, set()))
                    names.update(alias for alias, ids in by_alias.items() if player_id in ids)
                    clause = player_clause(
                        str(event.get("description") or ""), names, _normalized_name
                    )
                    pending.setdefault(player_id, []).append(
                        interpret_event(
                            event,
                            clause,
                            cast(str | None, state[player_id].get("team")),
                            current_status=str(state[player_id]["status"]),
                            matched_aliases=names,
                        )
                    )
            for player_id, daily_player_events in pending.items():
                resolved, resolution = resolve_day(daily_player_events)
                current = state[player_id]
                current.update(
                    team=resolved["team"],
                    status=resolved["status"],
                    availability_class=current["availability_class"]
                    if resolved["action"] == "contract"
                    else _availability_class(
                        resolved["category"], resolved["clause"], resolved["status"]
                    )
                    if resolution == "observed"
                    else "unknown",
                    transaction_count=cast(int, current["transaction_count"]) + 1,
                    last_transaction_date=event_date,
                    matched=1.0,
                    resolution=resolution,
                    source_url=resolved["source_url"],
                    clause=resolved["clause"],
                    evidence=sorted(
                        {
                            str(row["source_url"]) + " | " + row["clause"]
                            for row in daily_player_events
                        }
                    ),
                )
        for player_id, current in state.items():
            status = str(current["status"])
            availability_class = str(current["availability_class"])
            last_date = current["last_transaction_date"]
            recency_days = (
                float((cutoff_date - last_date).days) if isinstance(last_date, date) else None
            )
            recent_event_score = (
                max(0.0, 1.0 - min(recency_days, 365.0) / 365.0)
                if recency_days is not None
                else 0.0
            )
            output.append(
                {
                    "forecast_season": forecast_season,
                    "forecast_cutoff_date": cutoff_date,
                    "player_id": player_id,
                    "cutoff_preseason_team": current.get("team"),
                    "cutoff_preseason_status": status,
                    "cutoff_preseason_status_score": {
                        "active": 1.0,
                        "reserve": 0.25,
                        "practice": 0.1,
                        "off": 0.0,
                    }.get(status),
                    "cutoff_state_resolution": current["resolution"],
                    "cutoff_state_observed": current["resolution"] == "observed",
                    "cutoff_source_url": current["source_url"],
                    "cutoff_evidence_clause": current["clause"],
                    "cutoff_evidence": current["evidence"],
                    "cutoff_preseason_rostered": float(status in {"active", "reserve"})
                    if status != "unknown"
                    else None,
                    "cutoff_preseason_reserve": float(status == "reserve")
                    if status != "unknown"
                    else None,
                    "cutoff_availability_class": availability_class,
                    "cutoff_injured_reserve": float(availability_class == "injured_reserve")
                    if status != "unknown"
                    else None,
                    "cutoff_pup_nfi": float(availability_class == "pup_nfi")
                    if status != "unknown"
                    else None,
                    "cutoff_suspended": float(availability_class == "suspended")
                    if status != "unknown"
                    else None,
                    "cutoff_reserve_other": float(
                        availability_class in {"reserve_other", "exempt", "covid", "opt_out"}
                    )
                    if status != "unknown"
                    else None,
                    "cutoff_transaction_recency_days": recency_days,
                    "cutoff_recent_event_score": recent_event_score,
                    "cutoff_transaction_matched": current["matched"],
                    "cutoff_transaction_count": current["transaction_count"],
                    "cutoff_last_transaction_date": current["last_transaction_date"],
                }
            )
    if not output:
        return pl.DataFrame()
    return pl.DataFrame(output, infer_schema_length=None).with_columns(
        pl.col("forecast_season").cast(pl.Int32),
        pl.col("forecast_cutoff_date").cast(pl.Date),
        pl.col("cutoff_transaction_count").cast(pl.Int32),
        pl.col("cutoff_last_transaction_date").cast(pl.Date),
    )


def build_snap_features(snap_counts: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Aggregate source-season offensive role and its late-season direction.

    PFR snap rows use PFR ids, while the projection uses GSIS ids. ``load_players``
    supplies the audited crosswalk. The late window is the final four regular-season
    weeks in each season, not a hard-coded week number (17- and 18-game eras differ).
    """
    require_columns(
        snap_counts.columns,
        (
            "game_id",
            "season",
            "game_type",
            "week",
            "pfr_player_id",
            "position",
            "offense_snaps",
            "offense_pct",
        ),
        "snap count input",
    )
    require_columns(players.columns, ("gsis_id", "pfr_id"), "player identity input")
    if snap_counts.height == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.String,
                "season": pl.Int32,
                "source_offense_snaps_pg": pl.Float64,
                "source_offense_snap_pct": pl.Float64,
                "late_offense_snap_pct": pl.Float64,
                "offense_snap_pct_trend": pl.Float64,
            }
        )

    identity = (
        players.select(
            pl.col("pfr_id").alias("pfr_player_id"), pl.col("gsis_id").alias("player_id")
        )
        .drop_nulls()
        .filter((pl.col("pfr_player_id") != "") & (pl.col("player_id") != ""))
        .unique(subset=["pfr_player_id"], keep="first")
    )
    regular = (
        snap_counts.filter(
            (pl.col("game_type") == "REG")
            & pl.col("position").str.to_uppercase().is_in(_SKILL_POSITIONS)
        )
        .join(identity, on="pfr_player_id", how="inner")
        .with_columns(
            pl.col("season").cast(pl.Int32),
            pl.col("offense_snaps").cast(pl.Float64, strict=False),
            pl.col("offense_pct").cast(pl.Float64, strict=False),
        )
    )
    season_end = regular.group_by("season").agg(pl.col("week").max().alias("_last_week"))
    regular = regular.join(season_end, on="season", how="left")
    full = regular.group_by(["player_id", "season"]).agg(
        pl.col("game_id").n_unique().alias("_snap_games"),
        pl.col("offense_snaps").fill_null(0).sum().alias("_offense_snaps"),
        pl.col("offense_pct").mean().alias("source_offense_snap_pct"),
    )
    late = (
        regular.filter(pl.col("week") >= pl.col("_last_week") - 3)
        .group_by(["player_id", "season"])
        .agg(pl.col("offense_pct").mean().alias("late_offense_snap_pct"))
    )
    return (
        full.join(late, on=["player_id", "season"], how="left")
        .with_columns(
            (pl.col("_offense_snaps") / pl.col("_snap_games")).alias("source_offense_snaps_pg")
        )
        .with_columns(
            (pl.col("late_offense_snap_pct") - pl.col("source_offense_snap_pct")).alias(
                "offense_snap_pct_trend"
            )
        )
        .drop("_snap_games", "_offense_snaps")
        .sort(["player_id", "season"])
    )


def build_player_background(players: pl.DataFrame) -> pl.DataFrame:
    """Static, outcome-free draft information keyed to the projection player id."""
    require_columns(
        players.columns,
        ("gsis_id", "draft_year", "draft_round", "draft_pick"),
        "player background input",
    )
    pick = pl.col("draft_pick").cast(pl.Float64, strict=False)
    return (
        players.filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("draft_year").cast(pl.Int32, strict=False),
            pl.col("draft_round").cast(pl.Float64, strict=False),
            pick.alias("draft_pick"),
            pl.when(pick.is_not_null()).then(1.0).otherwise(0.0).alias("was_drafted"),
            pl.when(pick > 0).then(-pick.log()).otherwise(None).alias("draft_capital_score"),
        )
        .unique(subset=["player_id"], keep="first")
    )


def build_combine_features(combine: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Small, interpretable athletic profile joined from PFR to GSIS identity.

    Speed score adjusts forty time for body weight; burst combines vertical and broad
    jump; agility is negated so higher remains better.  Raw missing tests stay null and
    are imputed inside each historical training fold rather than treated as poor tests.
    """
    require_columns(
        combine.columns,
        ("pfr_id", "pos", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle"),
        "combine input",
    )
    require_columns(players.columns, ("gsis_id", "pfr_id"), "player identity input")
    identity = (
        players.select(
            pl.col("pfr_id"),
            pl.col("gsis_id").alias("player_id"),
        )
        .drop_nulls()
        .unique(subset=["pfr_id"], keep="first")
    )
    weight = pl.col("wt").cast(pl.Float64, strict=False)
    forty = pl.col("forty").cast(pl.Float64, strict=False)
    vertical = pl.col("vertical").cast(pl.Float64, strict=False)
    broad = pl.col("broad_jump").cast(pl.Float64, strict=False)
    cone = pl.col("cone").cast(pl.Float64, strict=False)
    shuttle = pl.col("shuttle").cast(pl.Float64, strict=False)
    return (
        canonical_positions(combine, "pos")
        .filter(pl.col("pos").is_in(_SKILL_POSITIONS))
        .join(identity, on="pfr_id", how="inner")
        .select(
            "player_id",
            weight.alias("combine_weight"),
            pl.when((weight > 0) & (forty > 0))
            .then(weight * 200.0 / forty.pow(4))
            .otherwise(None)
            .alias("combine_speed_score"),
            pl.when(vertical.is_not_null() & broad.is_not_null())
            .then(vertical + broad / 10.0)
            .otherwise(None)
            .alias("combine_burst_score"),
            pl.when(cone.is_not_null() & shuttle.is_not_null())
            .then(-(cone + shuttle))
            .otherwise(None)
            .alias("combine_agility_score"),
        )
        .unique(subset=["player_id"], keep="first")
    )


def build_rookie_features(
    players: pl.DataFrame,
    combine_features: pl.DataFrame,
    forecast_seasons: Iterable[int],
    *,
    market_rankings: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Preseason-only feature rows for the distinct rookie population.

    Returning-player history is structurally unavailable for rookies, so they should
    never be mean-imputed into that model.  This universe starts from nflverse player
    identity and rookie/draft metadata, then adds combine measurements.  Draft capital
    is the primary football signal; conference is retained as one deliberately coarse
    college-context flag rather than a large collection of noisy school dummies.
    """
    require_columns(
        players.columns,
        (
            "gsis_id",
            "display_name",
            "position",
            "birth_date",
            "college_conference",
            "rookie_season",
            "draft_year",
            "draft_round",
            "draft_pick",
        ),
        "rookie player input",
    )
    require_columns(
        combine_features.columns,
        (
            "player_id",
            "combine_weight",
            "combine_speed_score",
            "combine_burst_score",
            "combine_agility_score",
        ),
        "rookie combine input",
    )
    seasons = [int(season) for season in forecast_seasons]
    if not seasons:
        return pl.DataFrame()

    if market_rankings is not None and market_rankings.height:
        # The dated rookie-year market is stronger evidence of historical fantasy
        # eligibility than today's player metadata after a later position switch.
        players = (
            players.with_columns(
                pl.coalesce("rookie_season", "draft_year").cast(pl.Int32).alias("_market_season")
            )
            .join(
                market_rankings.select(
                    pl.col("player_id").alias("gsis_id"),
                    pl.col("forecast_season").cast(pl.Int32).alias("_market_season"),
                    pl.col("market_position").alias("_market_position"),
                ).unique(subset=["gsis_id", "_market_season"]),
                on=["gsis_id", "_market_season"],
                how="left",
                validate="m:1",
            )
            .with_columns(pl.coalesce("_market_position", "position").alias("position"))
        )

    position = (
        pl.col("position")
        .str.to_uppercase()
        .replace_strict({"HB": "RB", "FB": "RB"}, default=pl.col("position").str.to_uppercase())
    )
    rookie_season = pl.coalesce(
        pl.col("rookie_season").cast(pl.Int32, strict=False),
        pl.col("draft_year").cast(pl.Int32, strict=False),
    )
    pick = pl.col("draft_pick").cast(pl.Float64, strict=False)
    conference = pl.col("college_conference").fill_null("").str.to_uppercase()
    # Historical names are included because the input spans more than one conference era.
    major_conferences = ("ACC", "BIG 10", "BIG TEN", "BIG 12", "PAC-10", "PAC-12", "SEC")
    base = (
        players.filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
        .with_columns(
            rookie_season.alias("forecast_season"),
            position.alias("position"),
            pl.col("birth_date").cast(pl.String).str.to_date(strict=False).alias("_birth_date"),
        )
        .filter(
            pl.col("forecast_season").is_in(seasons) & pl.col("position").is_in(_SKILL_POSITIONS)
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("player_display_name"),
            "position",
            "forecast_season",
            pl.lit(1.0).alias("rookie_indicator"),
            pl.lit("rookie").alias("player_population"),
            pl.col("draft_round").cast(pl.Float64, strict=False).alias("rookie_draft_round"),
            pick.alias("rookie_draft_pick"),
            pl.when(pick.is_not_null()).then(1.0).otherwise(0.0).alias("rookie_was_drafted"),
            pl.when(pick > 0)
            .then(-pick.log())
            # Keep UDFAs inside the exact same evaluation universe as drafted
            # rookies. A null here lets the draft-capital baseline silently drop
            # them and inflates its apparent hit rate on an easier player pool.
            .otherwise(-pl.lit(300.0).log())
            .alias("rookie_draft_capital_score"),
            pl.when(pl.col("draft_round").cast(pl.Int32, strict=False) == 1)
            .then(1.0)
            .otherwise(0.0)
            .alias("rookie_round_one"),
            pl.when(pl.col("draft_round").cast(pl.Int32, strict=False).is_between(2, 3))
            .then(1.0)
            .otherwise(0.0)
            .alias("rookie_day_two"),
            pl.when(conference.is_in(major_conferences))
            .then(1.0)
            .otherwise(0.0)
            .alias("college_major_conference"),
            (
                (pl.date(pl.col("forecast_season"), 9, 1) - pl.col("_birth_date")).dt.total_days()
                / 365.2425
            ).alias("rookie_age"),
        )
        .unique(subset=["forecast_season", "player_id"], keep="first")
    )
    return base.join(combine_features, on="player_id", how="left").sort(
        ["forecast_season", "position", "rookie_draft_pick", "player_id"]
    )


def build_contract_features(
    contracts: pl.DataFrame,
    forecast_seasons: Iterable[int],
    *,
    cutoff_by_season: Mapping[int, date] | None = None,
    cutoff: str = "08-31",
) -> pl.DataFrame:
    """Separate usable point-in-time terms from undated historical proxies.

    A signing year is not a signing date or the first effective contract year.
    Undated same-year deals are excluded. Prior-year undated money/assumed terms
    remain diagnostic proxies only; canonical model inputs require an agreement
    date AND source availability date. Remaining years additionally requires an
    explicit effective_start_year. Current is_active is never used.
    """
    require_columns(
        contracts.columns,
        ("gsis_id", "year_signed", "years", "apy_cap_pct", "guaranteed"),
        "contract input",
    )
    base = (
        contracts.select(
            "gsis_id",
            "year_signed",
            "years",
            "apy_cap_pct",
            "guaranteed",
            *(
                name
                for name in ("signing_date", "available_date", "effective_start_year")
                if name in contracts.columns
            ),
        )
        .with_columns(
            *(
                pl.lit(None, dtype=pl.Date).alias(name)
                for name in ("signing_date", "available_date")
                if name not in contracts.columns
            ),
            *(
                pl.lit(None, dtype=pl.Int32).alias(name)
                for name in ("effective_start_year",)
                if name not in contracts.columns
            ),
        )
        .with_columns(
            pl.col("signing_date", "available_date").cast(pl.Date, strict=False),
            pl.col("year_signed", "years", "effective_start_year").cast(pl.Int32, strict=False),
        )
        .filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != "") & (pl.col("years") > 0))
    )
    month, day = map(int, cutoff.split("-"))
    frames = []
    for season in forecast_seasons:
        as_of = (cutoff_by_season or {}).get(season, date(season, month, day))
        eligible = (
            base.filter(
                # An explicit date always controls eligibility, even if the year field is stale.
                pl.when(pl.col("signing_date").is_not_null())
                .then(pl.col("signing_date") <= as_of)
                .otherwise(pl.col("year_signed") < season)
                & (pl.col("available_date").is_null() | (pl.col("available_date") <= as_of))
            )
            .with_columns(
                (
                    pl.col("signing_date").is_not_null() & pl.col("available_date").is_not_null()
                ).alias("contract_point_in_time_verified"),
                pl.coalesce("effective_start_year", "year_signed").alias("_start"),
            )
            .filter(pl.col("_start") + pl.col("years") > season)
        )
        # Tied signing years without dates cannot establish which deal was known last.
        eligible = (
            eligible.with_columns(
                pl.coalesce("signing_date", pl.date(pl.col("year_signed"), 1, 1)).alias("_order")
            )
            .filter(pl.col("_order") == pl.col("_order").max().over("gsis_id"))
            .unique()
        )
        eligible = (
            eligible.with_columns((pl.len().over("gsis_id") > 1).alias("contract_ambiguous"))
            .sort("gsis_id", "year_signed", "years", "apy_cap_pct", "guaranteed")
            .unique(subset=["gsis_id"], keep="first")
        )
        trusted = pl.col("contract_point_in_time_verified") & ~pl.col("contract_ambiguous")
        frames.append(
            eligible.select(
                pl.col("gsis_id").alias("player_id"),
                pl.lit(season, dtype=pl.Int32).alias("forecast_season"),
                pl.col("contract_point_in_time_verified"),
                pl.col("contract_ambiguous"),
                pl.col("signing_date").alias("contract_signing_date"),
                pl.col("available_date").alias("contract_available_date"),
                pl.when(trusted)
                .then(pl.col("apy_cap_pct"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("contract_apy_cap_pct"),
                pl.when(trusted)
                .then(pl.col("guaranteed").clip(lower_bound=0).log1p())
                .otherwise(None)
                .cast(pl.Float64)
                .alias("contract_guaranteed_log"),
                pl.when(trusted & pl.col("effective_start_year").is_not_null())
                .then(
                    (pl.col("effective_start_year") + pl.col("years") - season).clip(
                        upper_bound=pl.col("years")
                    )
                )
                .otherwise(None)
                .cast(pl.Float64)
                .alias("contract_years_remaining"),
                pl.col("apy_cap_pct").cast(pl.Float64).alias("contract_apy_cap_pct_proxy"),
                (pl.col("_start") + pl.col("years") - season)
                .cast(pl.Float64)
                .alias("contract_years_remaining_proxy"),
                pl.col("guaranteed")
                .clip(lower_bound=0)
                .log1p()
                .alias("contract_guaranteed_log_proxy"),
            )
        )
    return (
        pl.concat(frames, how="diagonal_relaxed").sort("forecast_season", "player_id")
        if frames
        else pl.DataFrame()
    )


def _week_one_rosters(rosters: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        rosters.columns,
        (
            "season",
            "week",
            "game_type",
            "team",
            "position",
            "gsis_id",
            "status",
            "years_exp",
        ),
        "weekly roster input",
    )
    status = pl.col("status").fill_null("").str.to_uppercase()
    score = (
        pl.when(status == "ACT")
        .then(1.0)
        .when(status == "INA")
        .then(0.8)
        .when(status == "RES")
        .then(0.25)
        .when(status == "DEV")
        .then(0.1)
        .otherwise(0.0)
    )
    return (
        rosters.filter(
            (pl.col("game_type") == "REG")
            & (pl.col("week") == 1)
            & pl.col("position").str.to_uppercase().is_in(_SKILL_POSITIONS)
            & pl.col("gsis_id").is_not_null()
            & (pl.col("gsis_id") != "")
        )
        .with_columns(score.alias("preseason_status_score"), status.alias("preseason_status"))
        .sort(["season", "gsis_id", "preseason_status_score"], descending=[False, False, True])
        .unique(subset=["season", "gsis_id"], keep="first")
        .select(
            pl.col("season").cast(pl.Int32).alias("forecast_season"),
            pl.col("gsis_id").alias("player_id"),
            pl.col("team").alias("preseason_team"),
            "preseason_status",
            "preseason_status_score",
            status.is_in(_ROSTERED_STATUSES).cast(pl.Float64).alias("preseason_rostered"),
            (status == "RES").cast(pl.Float64).alias("preseason_reserve"),
            pl.col("years_exp").cast(pl.Float64, strict=False).alias("preseason_years_exp"),
        )
    )


def build_forecast_features(
    rosters: pl.DataFrame,
    player_seasons: pl.DataFrame,
    forecast_seasons: Iterable[int],
    cutoff: str,
    *,
    depth_charts: pl.DataFrame | None = None,
    snap_features: pl.DataFrame | None = None,
    player_background: pl.DataFrame | None = None,
    contract_features: pl.DataFrame | None = None,
    combine_features: pl.DataFrame | None = None,
    transaction_features: pl.DataFrame | None = None,
    cutoff_by_season: Mapping[int, date] | None = None,
) -> pl.DataFrame:
    """Build every additive experimental input on the forecast-season grain."""
    missing_stats = [
        name
        for name in ("games", "targets", "carries", "attempts")
        if name not in player_seasons.columns
    ]
    if missing_stats:
        player_seasons = player_seasons.with_columns(
            *(pl.lit(0.0).alias(name) for name in missing_stats)
        )
    snapshots = _week_one_rosters(rosters)
    frames: list[pl.DataFrame] = []
    for forecast_season in forecast_seasons:
        source_season = forecast_season - 1
        source = player_seasons.filter(pl.col("season") == source_season).select(
            "player_id",
            "position",
            pl.col("team").alias("source_team"),
            pl.col("games").cast(pl.Float64).fill_null(0.0).alias("_source_games"),
            pl.col("targets").cast(pl.Float64).fill_null(0.0).alias("_source_targets"),
            pl.col("carries").cast(pl.Float64).fill_null(0.0).alias("_source_carries"),
        )
        if source.height == 0:
            continue
        roster = (
            snapshots.filter(pl.col("forecast_season") == forecast_season)
            .drop("forecast_season")
            .rename(
                {
                    "preseason_team": "week1_proxy_team",
                    "preseason_status": "week1_proxy_status",
                    "preseason_status_score": "week1_proxy_status_score",
                    "preseason_rostered": "week1_proxy_rostered",
                    "preseason_reserve": "week1_proxy_reserve",
                }
            )
        )
        frame = source.join(roster, on="player_id", how="left").with_columns(
            pl.lit(forecast_season).cast(pl.Int32).alias("forecast_season")
        )
        if transaction_features is not None and transaction_features.height:
            transaction = transaction_features.filter(
                pl.col("forecast_season") == forecast_season
            ).drop("forecast_season")
            frame = frame.join(transaction, on="player_id", how="left").with_columns(
                pl.col("cutoff_preseason_team").alias("preseason_team"),
                pl.col("cutoff_preseason_status").alias("preseason_status"),
                pl.col("cutoff_preseason_rostered").alias("preseason_rostered"),
                pl.col("cutoff_preseason_reserve").alias("preseason_reserve"),
                pl.col("cutoff_preseason_status_score").alias("preseason_status_score"),
            )
        else:
            frame = frame.with_columns(
                pl.col("week1_proxy_team").alias("preseason_team"),
                pl.col("week1_proxy_status").alias("preseason_status"),
                pl.col("week1_proxy_rostered").fill_null(0.0).alias("preseason_rostered"),
                pl.col("week1_proxy_reserve").fill_null(0.0).alias("preseason_reserve"),
                pl.col("week1_proxy_status_score").fill_null(0.0).alias("preseason_status_score"),
            )
        frame = frame.with_columns(
            pl.when(pl.col("preseason_rostered").is_not_null())
            .then(
                (pl.col("preseason_rostered") > 0)
                & pl.col("preseason_team").is_not_null()
                & (pl.col("preseason_team") != pl.col("source_team"))
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("team_changed"),
            (
                pl.col("preseason_rostered").is_not_null()
                & (
                    (pl.col("preseason_rostered") <= 0)
                    | pl.col("preseason_team").is_null()
                    | (pl.col("preseason_team") != pl.col("source_team"))
                )
            ).alias("_departed"),
        )
        vacated = (
            frame.group_by("source_team")
            .agg(
                pl.col("preseason_rostered")
                .is_not_null()
                .mean()
                .alias("team_context_observed_share"),
                pl.col("_source_targets").sum().alias("_team_targets"),
                pl.col("_source_carries").sum().alias("_team_carries"),
                pl.col("_source_targets")
                .filter(pl.col("_departed"))
                .sum()
                .alias("_vacated_targets"),
                pl.col("_source_carries")
                .filter(pl.col("_departed"))
                .sum()
                .alias("_vacated_carries"),
            )
            .with_columns(
                pl.when(pl.col("_team_targets") > 0)
                .then(pl.col("_vacated_targets") / pl.col("_team_targets"))
                .otherwise(0.0)
                .alias("team_vacated_target_share"),
                pl.when(pl.col("_team_carries") > 0)
                .then(pl.col("_vacated_carries") / pl.col("_team_carries"))
                .otherwise(0.0)
                .alias("team_vacated_carry_share"),
            )
            .with_columns(
                pl.col("team_vacated_target_share").alias("known_vacated_target_share"),
                pl.col("team_vacated_carry_share").alias("known_vacated_carry_share"),
            )
            .with_columns(
                *(
                    pl.when(pl.col("team_context_observed_share") == 1.0)
                    .then(pl.col(name))
                    .otherwise(None)
                    .alias(name)
                    for name in ("team_vacated_target_share", "team_vacated_carry_share")
                )
            )
            .select(
                pl.col("source_team").alias("preseason_team"),
                "team_vacated_target_share",
                "team_vacated_carry_share",
                "known_vacated_target_share",
                "known_vacated_carry_share",
                "team_context_observed_share",
            )
        )
        frame = frame.join(vacated, on="preseason_team", how="left")

        cutoff_date = (cutoff_by_season or {}).get(forecast_season)
        season_cutoff = cutoff_date.strftime("%m-%d") if cutoff_date is not None else cutoff
        depth = depth_chart_as_of(depth_charts, forecast_season, season_cutoff)
        if depth is not None and depth.height:
            frame = frame.join(
                depth.select("player_id", "depth_chart_rank"), on="player_id", how="left"
            )
        else:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Int32).alias("depth_chart_rank"))
        frame = frame.with_columns(
            pl.when(pl.col("depth_chart_rank") > 0)
            .then(1.0 / pl.col("depth_chart_rank").cast(pl.Float64))
            .otherwise(0.0)
            .alias("_role_access"),
        ).with_columns(
            (pl.col("team_vacated_target_share") * pl.col("_role_access")).alias(
                "vacated_target_opportunity"
            ),
            (pl.col("team_vacated_carry_share") * pl.col("_role_access")).alias(
                "vacated_carry_opportunity"
            ),
        )

        if snap_features is not None and snap_features.height:
            frame = frame.join(
                snap_features.filter(pl.col("season") == source_season).drop("season"),
                on="player_id",
                how="left",
            )
        if player_background is not None and player_background.height:
            frame = frame.join(player_background, on="player_id", how="left")
        career = (
            player_seasons.filter(pl.col("season") <= source_season)
            .group_by("player_id")
            .agg(
                pl.col("season").n_unique().cast(pl.Float64).alias("career_seasons_observed"),
                pl.col("games").cast(pl.Float64).fill_null(0.0).sum().alias("career_games"),
                (
                    pl.col("carries").cast(pl.Float64).fill_null(0.0)
                    + pl.col("targets").cast(pl.Float64).fill_null(0.0)
                )
                .sum()
                .alias("career_opportunities"),
                pl.col("attempts")
                .cast(pl.Float64)
                .fill_null(0.0)
                .sum()
                .alias("career_pass_attempts"),
            )
        )
        frame = frame.join(career, on="player_id", how="left")
        if combine_features is not None and combine_features.height:
            frame = frame.join(combine_features, on="player_id", how="left")
        frame = frame.with_columns(
            pl.when(pl.col("preseason_years_exp").is_not_null())
            .then(pl.col("preseason_years_exp"))
            .when(pl.col("draft_year").is_not_null())
            .then((pl.lit(forecast_season) - pl.col("draft_year")).cast(pl.Float64))
            .otherwise(None)
            .alias("player_experience"),
        )
        if contract_features is not None and contract_features.height:
            frame = frame.join(
                contract_features.filter(pl.col("forecast_season") == forecast_season).drop(
                    "forecast_season"
                ),
                on="player_id",
                how="left",
            )
        frames.append(
            frame.drop(
                "position",
                "source_team",
                "preseason_status",
                "_source_targets",
                "_source_carries",
                "_source_games",
                "_departed",
                "_role_access",
                "depth_chart_rank",
                "draft_year",
                strict=False,
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["forecast_season", "player_id"])
