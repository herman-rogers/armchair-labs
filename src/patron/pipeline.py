"""Pipeline orchestration: the one place that loads, computes, and writes.

Everything below `patron.board` and `patron.metrics` is pure. This module is where the
impure edges meet — nflverse downloads on one side, parquet and JSON on the other —
so the pure core stays testable offline and the I/O stays in one auditable place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from patron.board.builder import build_board, build_player_seasons
from patron.config.league import LeagueConfig, get_league
from patron.config.settings import Settings, get_settings
from patron.data import nflverse
from patron.data.derived import cached_frame
from patron.scoring.bonuses import BonusAudit, extract_touchdown_bonuses
from patron.scoring.dst import build_dst_proxy
from patron.scoring.kickers import aggregate_kicker_seasons, score_kicker_weeks

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Everything one board build produced, plus how the build went."""

    board: pl.DataFrame
    player_seasons: pl.DataFrame
    kickers: pl.DataFrame
    defenses: pl.DataFrame
    bonus_audit: BonusAudit | None
    replacement_levels: dict[str, float]


def load_bonuses(
    config: LeagueConfig,
    force: bool = False,
) -> tuple[pl.DataFrame, BonusAudit | None]:
    """Per-player, per-week big-play bonuses, cached as a derived artifact.

    Reducing multi-season play-by-play is the expensive step in the whole pipeline, and
    it produces a few thousand rows from the largest table nflverse publishes. That is
    exactly the shape worth persisting.

    The audit is only available when the artifact is actually built; a cache hit
    returns None for it, since the reconciliation was done when the file was written.
    """
    audit: BonusAudit | None = None

    def build() -> pl.DataFrame:
        nonlocal audit
        plays = nflverse.load_touchdown_plays(config.seasons)
        bonuses, audit = extract_touchdown_bonuses(plays)
        logger.info(audit.summary())
        if not audit.is_balanced:
            logger.warning(
                "bonus points earned and credited disagree; %s credit(s) were dropped "
                "to null player ids",
                audit.dropped_credits,
            )
        return bonuses

    frame = cached_frame("bonuses", config.seasons, build, force=force)
    return frame, audit


def build(
    config: LeagueConfig | None = None,
    settings: Settings | None = None,
    force: bool = False,
    strict_overrides: bool = True,
) -> BuildResult:
    """Run the full Phase 1 pipeline and return every artifact it produced."""
    config = config or get_league()
    settings = settings or get_settings()
    settings.ensure_dirs()

    weeks = nflverse.load_player_weeks(config.seasons)
    bonuses, audit = load_bonuses(config, force=force)

    player_seasons = build_player_seasons(weeks, bonuses, config)
    birth_dates = nflverse.load_birth_dates(config.board_season)
    board = build_board(
        player_seasons,
        birth_dates,
        config=config,
        strict_overrides=strict_overrides,
    )

    board_weeks = weeks.filter(pl.col("season") == config.board_season)
    kickers = aggregate_kicker_seasons(
        score_kicker_weeks(board_weeks.filter(pl.col("position") == "K"))
    )
    defenses = build_dst_proxy(board_weeks, nflverse.load_schedules(config.board_season))

    levels = dict(
        board.filter(pl.col("repl_ppg").is_not_null())
        .group_by("position")
        .agg(pl.col("repl_ppg").first())
        .iter_rows()
    )

    return BuildResult(
        board=board,
        player_seasons=player_seasons,
        kickers=kickers,
        defenses=defenses,
        bonus_audit=audit,
        replacement_levels=levels,
    )


#: Board columns written to JSON, in display order.
BOARD_EXPORT_COLUMNS: tuple[str, ...] = (
    "rank",
    "player_display_name",
    "position",
    "team",
    "adj_vor",
    "vor",
    "ppg",
    "flags",
    "games",
    "season_pts",
    "bonus_pts",
    "floor",
    "volatility",
    "wtd_opp",
    "target_share",
    "air_yards_share",
    "wopr",
    "td_over_exp",
    "age_at_season",
    "repl_ppg",
    "override_delta",
    "override_reason",
    "player_id",
)


def export_board(board: pl.DataFrame, path: Path) -> Path:
    """Write the board as JSON records, rounded for display."""
    columns = [column for column in BOARD_EXPORT_COLUMNS if column in board.columns]
    export = board.select(columns).with_columns(
        pl.col(pl.Float64).round(3),
    )
    path.write_text(json.dumps(export.to_dicts(), indent=2, default=str))
    return path


def export_markdown(board: pl.DataFrame, path: Path, limit: int | None = None) -> Path:
    """Write the board in the same shape as `data/static/2026_draft_list.md`.

    Same columns and same order as the artifact this pipeline reproduces, so the two
    can be diffed by eye as well as by the fixture test.
    """
    rows = board.head(limit) if limit else board
    lines = [
        f"# Overall Draft Board — {rows.height} players",
        "",
        "Rank | Player | Pos | Team | VOR | PPG | Flags | Notes",
        "---|---|---|---|---|---|---|---",
    ]
    for row in rows.iter_rows(named=True):
        note = row.get("override_reason") or ""
        lines.append(
            f"{row['rank']} | {row['player_display_name']} | {row['position']} | "
            f"{row['team']} | {row['adj_vor']:.1f} | {row['ppg']:.1f} | "
            f"{row['flags']} | {note}"
        )
    lines.append("")
    lines.append(
        "Flags: BUY=TD under-exp w/ volume | TD-luck=over-exp, price down | "
        "age=RB 27.5+ | Xgms=small sample."
    )
    path.write_text("\n".join(lines))
    return path
