"""Command-line entry points — system operations only.

The CLI does the things a person or a scheduler does *to* the system: sign in,
ingest observations, run the server, clear caches. It deliberately does not read
league data. Pulling ESPN state and rendering it belongs to the server and the
frontend, which is where it can be cached, shared between viewers, and refreshed
without anyone remembering to run a command.
"""

from __future__ import annotations

import logging

import typer

from engine.cli_auth import app as auth_app
from engine.cli_campaigns import app as campaigns_app
from engine.cli_models import app as models_app
from engine.cli_tables import app as tables_app
from engine.config.settings import PACKAGE_ROOT, get_settings

app = typer.Typer(
    add_completion=False,
    help="Armchair Labs data, statistics, forecasting, and fantasy analysis engine.",
)
app.add_typer(auth_app, name="auth")
app.add_typer(tables_app, name="tables")
app.add_typer(models_app, name="models")
app.add_typer(campaigns_app, name="campaigns")


@app.command(
    "data",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def data_pipeline(ctx: typer.Context):
    """Generate sources and products; refresh and upload the queryable table catalog."""
    from engine.data.pipeline import main

    main(ctx.args)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes."),
) -> None:
    """Run the API server, which owns ESPN syncing and serves the frontend's data."""
    import uvicorn

    typer.echo(f"Serving on http://{host}:{port}  (frontend dev server: just web)")
    uvicorn.run(
        "engine.api.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(PACKAGE_ROOT)] if reload else None,
    )


@app.command()
def dev(
    restart: bool = typer.Option(False, "--restart", help="Stop existing services first."),
    api_port: int = typer.Option(8000, "--api-port"),
    web_port: int = typer.Option(5173, "--web-port"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
    setup: bool = typer.Option(True, "--setup/--no-setup", help="Prepare dependencies and data."),
    auth: bool = typer.Option(
        True, "--auth/--no-auth", help="Verify ESPN login and load league data."
    ),
    rescore: bool = typer.Option(
        True, "--rescore/--no-rescore", help="Add this week's campaign forecasts first."
    ),
) -> None:
    """Prepare data, tables, campaign forecasts and ESPN, then start the API and frontend."""
    from engine.dev import DevAlreadyRunningError, run

    try:
        exit_code = run(
            restart=restart,
            api_port=api_port,
            web_port=web_port,
            reload=reload,
            prepare=setup,
            auth=auth,
            rescore=rescore,
        )
    except (DevAlreadyRunningError, RuntimeError, OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    raise typer.Exit(exit_code)


@app.command()
def setup(
    auth: bool = typer.Option(True, "--auth/--no-auth", help="Verify ESPN and load league data."),
    rescore: bool = typer.Option(
        True, "--rescore/--no-rescore", help="Add this week's campaign forecasts."
    ),
) -> None:
    """Prepare dependencies, tables, campaign forecasts, and ESPN without starting servers."""
    import subprocess

    from engine.startup import prepare

    try:
        prepare(auth=auth, rescore=rescore)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        typer.echo(f"Setup failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("dev-stop")
def dev_stop() -> None:
    """Stop recorded services and reclaim project-owned legacy port orphans."""
    from engine.dev import DevAlreadyRunningError, stop

    try:
        stopped = stop(include_legacy=True)
    except DevAlreadyRunningError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(
        "Armchair Labs dev services stopped."
        if stopped
        else "No Armchair Labs dev services running."
    )


@app.command("dev-status")
def dev_status() -> None:
    """Show whether the supervised API and frontend are running."""
    from engine.dev import status

    raise typer.Exit(0 if status() else 1)


@app.command("clear-cache")
@app.command("refresh", hidden=True)
def refresh(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Drop cached nflverse downloads so the next capture pulls fresh."""
    _configure_logging(verbose)
    import shutil

    cache = get_settings().nflverse_cache_dir
    shutil.rmtree(cache, ignore_errors=True)
    typer.echo("Cleared the nflverse research cache. Observation captures download fresh anyway.")


if __name__ == "__main__":
    app()
