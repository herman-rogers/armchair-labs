"""`patron auth` — browser-based ESPN sign-in.

ESPN has no token flow for its fantasy API, so access rides on two session cookies.
This subcommand opens a real browser, waits for a normal sign-in, and stores what comes
out, instead of asking anyone to go spelunking in devtools.
"""

from __future__ import annotations

import logging

import typer

from patron.config.league import get_league
from patron.config.settings import get_settings
from patron.espn import auth, discovery
from patron.espn import credentials as creds

app = typer.Typer(add_completion=False, help="Sign in to ESPN and manage stored credentials.")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


def _pick_league(leagues: list[discovery.DiscoveredLeague]) -> discovery.DiscoveredLeague | None:
    if not leagues:
        return None
    if len(leagues) == 1:
        typer.echo(f"  Found one league: {leagues[0].describe()}")
        return leagues[0]

    typer.echo("\n  Leagues on this account:")
    for index, league in enumerate(leagues, start=1):
        typer.echo(f"    {index}. {league.describe()}")

    choice = typer.prompt("\n  Which one?", default="1")
    try:
        return leagues[int(choice) - 1]
    except (ValueError, IndexError):
        typer.secho("  Not a listed option; skipping league selection.", fg="yellow")
        return None


@app.command()
def login(
    timeout: int = typer.Option(
        auth.DEFAULT_TIMEOUT_SECONDS, "--timeout", help="Seconds to wait for sign-in."
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Ignore the saved browser profile and sign in from scratch."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Open a browser, sign in to ESPN, and save the session."""
    _configure_logging(verbose)
    settings = get_settings()
    config = get_league()

    if fresh and settings.browser_profile_dir.exists():
        import shutil

        shutil.rmtree(settings.browser_profile_dir)
        typer.echo("Cleared the saved browser profile.")

    typer.echo("Opening Chrome…")
    try:
        result = auth.login(
            profile_dir=settings.browser_profile_dir,
            timeout_seconds=timeout,
            on_status=lambda message: typer.echo(f"  {message}"),
        )
    except auth.BrowserUnavailableError as error:
        typer.secho(f"\n{error}", fg="red")
        raise typer.Exit(1) from error
    except auth.LoginTimeoutError as error:
        typer.secho(f"\n{error}", fg="red")
        raise typer.Exit(1) from error

    found = result.credentials
    typer.secho(f"\n  Captured: {found.masked()}", fg="green")

    # Discovery is a convenience, not a requirement. A shape change in ESPN's fan API
    # must not turn a successful login into a failure.
    league_id, team_id = None, None
    try:
        typer.echo("\nLooking up your leagues…")
        chosen = _pick_league(discovery.discover_leagues(found))
        if chosen:
            league_id, team_id = chosen.league_id, chosen.team_id
    except discovery.DiscoveryError as error:
        typer.secho(f"  Could not list leagues: {error}", fg="yellow")
        typer.echo("  Set ESPN_LEAGUE_ID in .env by hand — the sign-in itself worked.")

    stored = creds.EspnCredentials(
        espn_s2=found.espn_s2, swid=found.swid, league_id=league_id, team_id=team_id
    )
    creds.write(stored, settings.env_path)
    typer.secho(f"\nSaved to {settings.env_path} (mode 600).", fg="green")

    if league_id:
        typer.echo("\nVerifying…")
        ok, message = discovery.verify(stored, league_id, config.draft_season)
        typer.secho(f"  {message}", fg="green" if ok else "red")
        if not ok:
            raise typer.Exit(1)


@app.command()
def status(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Show whether stored credentials exist and still work."""
    _configure_logging(verbose)
    settings = get_settings()
    config = get_league()

    stored = creds.read(settings.env_path)
    if stored is None:
        typer.secho("Not signed in.", fg="yellow")
        typer.echo("Run `patron auth login` to open a browser and sign in.")
        raise typer.Exit(1)

    typer.echo(f"Stored:    {stored.masked()}")
    typer.echo(f"League id: {stored.league_id or '(not set)'}")
    typer.echo(f"Team id:   {stored.team_id or '(not set)'}")

    if not stored.league_id:
        typer.secho(
            "\nNo league id stored, so the credentials cannot be verified. "
            "Set ESPN_LEAGUE_ID in .env or re-run `patron auth login`.",
            fg="yellow",
        )
        raise typer.Exit(1)

    typer.echo("\nChecking with ESPN…")
    ok, message = discovery.verify(stored, stored.league_id, config.draft_season)
    typer.secho(f"  {message}", fg="green" if ok else "red")
    raise typer.Exit(0 if ok else 1)


@app.command()
def logout(
    keep_profile: bool = typer.Option(
        False, "--keep-profile", help="Leave the browser profile in place."
    ),
) -> None:
    """Remove stored credentials and the saved browser session."""
    settings = get_settings()

    removed = creds.clear(settings.env_path)
    typer.echo(f"Removed {removed} credential key(s) from {settings.env_path}.")

    if not keep_profile and settings.browser_profile_dir.exists():
        import shutil

        shutil.rmtree(settings.browser_profile_dir)
        typer.echo("Deleted the saved browser profile.")
