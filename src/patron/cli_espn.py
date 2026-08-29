"""`patron espn` — league sync and the reports it produces."""

from __future__ import annotations

import logging

import polars as pl
import typer

from patron.config.league import get_league
from patron.config.settings import get_settings
from patron.espn import credentials as creds
from patron.espn import crosswalk, reports, sync

app = typer.Typer(add_completion=False, help="Sync ESPN league state and rank the wire.")

SNAPSHOT_NAME = "league_snapshot.json"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


def _require_credentials() -> creds.EspnCredentials:
    stored = creds.read(get_settings().env_path)
    if stored is None:
        typer.secho("Not signed in. Run `patron auth login` first.", fg="red")
        raise typer.Exit(1)
    return stored


def _load_board() -> pl.DataFrame:
    path = get_settings().outputs_dir / "board.json"
    if not path.exists():
        typer.secho(f"No board at {path}. Run `patron board` first.", fg="red")
        raise typer.Exit(1)
    return pl.read_json(path)


def _tagged_board(refresh: bool = True) -> tuple[pl.DataFrame, sync.LeagueSnapshot]:
    """Board joined to current league state, refreshing the snapshot unless told not to."""
    settings = get_settings()
    config = get_league()
    snapshot_path = settings.outputs_dir / SNAPSHOT_NAME

    if refresh or not snapshot_path.exists():
        snapshot = sync.fetch_snapshot(_require_credentials(), config.draft_season)
        snapshot.write(snapshot_path)
    else:
        snapshot = sync.LeagueSnapshot.read(snapshot_path)

    board = _load_board()
    ids, names = crosswalk.board_lookups(board)
    espn_players, report = crosswalk.resolve_player_ids(snapshot.to_frame(), ids, names)

    # §8's instruction, followed literally: log unmatched loudly. A silent join failure
    # is how a ranked wire quietly omits the one player that matters.
    typer.echo(f"  {report.summary()}")

    return crosswalk.attach_ownership(board, espn_players, snapshot.my_team_id), snapshot


@app.command("sync")
def sync_command(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Pull a fresh snapshot of league state and write it to the outputs directory."""
    _configure_logging(verbose)
    settings = get_settings()
    config = get_league()

    snapshot = sync.fetch_snapshot(_require_credentials(), config.draft_season)
    path = snapshot.write(settings.outputs_dir / SNAPSHOT_NAME)

    typer.echo(
        f"\n{snapshot.league_name} — season {snapshot.season}, week {snapshot.week}\n"
        f"  {len(snapshot.teams)} teams, {len(snapshot.players)} players, "
        f"{len(snapshot.transactions)} transactions"
    )
    typer.echo(f"  Wrote {path}")


@app.command()
def wire(
    limit: int = typer.Option(30, "--limit", help="Rows to show."),
    position: str | None = typer.Option(None, "--position", help="Filter to one position."),
    min_vor: float = typer.Option(0.0, "--min-vor", help="Floor on wire VOR."),
    healthy: bool = typer.Option(
        False, "--healthy", help="Hide players who cannot be started this week."
    ),
    stale: bool = typer.Option(False, "--stale", help="Reuse the last snapshot."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rank the free-agent pool against what else is actually free."""
    _configure_logging(verbose)
    config = get_league()

    tagged, snapshot = _tagged_board(refresh=not stale)
    board = reports.ranked_wire(
        tagged, config.vor_baseline_rank, min_vor=min_vor, limit=None, healthy_only=healthy
    )
    if position:
        board = board.filter(pl.col("position") == position.upper())

    display = board.head(limit).select(
        "player_display_name",
        "position",
        crosswalk.ESPN_TEAM,
        pl.col(reports.WIRE_VOR).round(2).alias("wire_vor"),
        pl.col("ppg").round(1).alias("ppg"),
        "flags",
        pl.col(crosswalk.PERCENT_OWNED).round(0).alias("%rost"),
        crosswalk.INJURY_STATUS,
    )
    with pl.Config(tbl_rows=limit, tbl_hide_dataframe_shape=True, fmt_str_lengths=22):
        typer.echo(f"\n{display}")

    unrankable = reports.unrankable_players(
        crosswalk.resolve_player_ids(snapshot.to_frame(), *crosswalk.board_lookups(_load_board()))[
            0
        ].filter(pl.col(crosswalk.OWNER_TEAM_ID).is_null())
    )
    if unrankable.height:
        typer.secho(
            f"\n  {unrankable.height} free agents have no 2025 tape and cannot be ranked "
            "(rookies, returns from injury). Top by roster rate:",
            fg="yellow",
        )
        for row in unrankable.head(8).iter_rows(named=True):
            owned = row[crosswalk.PERCENT_OWNED]
            typer.echo(
                f"    {row['player_display_name']:<24} {row['position']:<3} "
                f"{row['espn_team'] or '?':<4} {owned or 0:>5.1f}% rostered"
            )


@app.command()
def roster(
    stale: bool = typer.Option(False, "--stale", help="Reuse the last snapshot."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """My roster, with the rows that need a decision first."""
    _configure_logging(verbose)

    tagged, snapshot = _tagged_board(refresh=not stale)
    mine = reports.roster_health(tagged, snapshot.my_team_id)

    display = mine.select(
        "player_display_name",
        "position",
        crosswalk.ESPN_TEAM,
        pl.col("team").alias("team_2025"),
        crosswalk.CHANGED_TEAM,
        crosswalk.INJURY_STATUS,
        pl.col("adj_vor").round(2).alias("vor"),
        pl.col("ppg").round(1).alias("ppg"),
        "flags",
    )
    with pl.Config(tbl_rows=40, tbl_hide_dataframe_shape=True, fmt_str_lengths=22):
        typer.echo(f"\n{display}")


@app.command()
def opponents(
    stale: bool = typer.Option(False, "--stale", help="Reuse the last snapshot."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Each rival's thinnest position — where a trade offer or a blocking claim lands."""
    _configure_logging(verbose)
    config = get_league()

    tagged, _ = _tagged_board(refresh=not stale)
    weaknesses = reports.opponent_weaknesses(tagged, config.vor_baseline_rank)

    for team in weaknesses[crosswalk.OWNER_TEAM_NAME].unique().sort():
        rows = weaknesses.filter(pl.col(crosswalk.OWNER_TEAM_NAME) == team).head(2)
        thin = ", ".join(
            f"{row['position']} (best: {row['best_player']} {row['best_vor']:+.1f})"
            for row in rows.iter_rows(named=True)
        )
        typer.echo(f"  {str(team)[:30]:<32} thin at {thin}")
