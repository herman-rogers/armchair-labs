"""Prepare the local application: query tables, campaign bundles, frontend, and ESPN."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engine.config.settings import REPO_ROOT, get_settings
from engine.data import shared
from engine.tables import transport
from engine.tables.application import catalog_state, query_scope
from engine.tables.query import connect
from engine.tables.storage import current


def say(message: str) -> None:
    print(message, flush=True)


def cloud_login() -> None:
    """Use ADC or an existing gcloud login, prompting only when neither works."""
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request

    from engine.data.shared_gcs import GCSStore

    store = GCSStore(transport.DEFAULT_STORE)
    try:
        store.credentials.refresh(Request())
    except GoogleAuthError as error:
        gcloud = shutil.which("gcloud")
        if not gcloud:
            raise RuntimeError(
                "Data download needs Google Cloud access. Install gcloud and rerun "
                "this command, or configure application-default credentials."
            ) from error
        say("Sign in to Google Cloud to download the private data release.")
        subprocess.run([gcloud, "auth", "login"], check=True)


def prepare_data(data: Path) -> None:
    """Fetch the published table catalog when missing, then verify real reads."""
    data = data.resolve()
    if data.name != "data":
        raise ValueError("Startup requires a data directory named 'data' under the project root.")
    say("Checking query tables…")
    try:
        catalog = current(data)
    except FileNotFoundError:
        catalog = {}
    if not catalog.get("tables"):
        reference = REPO_ROOT / "data/releases/tables/current.json"
        try:
            transport.fetch(data, reference, offline=True)
        except FileNotFoundError:
            cloud_login()
            transport.fetch(data, reference)
        catalog = current(data)
    with query_scope(data):
        catalog_state()
    with connect(data) as db:
        players = db.execute("SELECT count(*) FROM analytics.players").fetchone()[0]
        season, week = db.execute(
            "SELECT max(season), max(week) FROM analytics.current_weeks"
        ).fetchone()
    say(
        f"Tables ready: {len(catalog['tables'])} tables; {players:,} players; "
        f"observations through {season} week {week}."
    )
    from engine.chairstats import CampaignStore

    campaigns = CampaignStore(data / "campaigns").campaigns()
    if campaigns:
        say("Campaigns: " + ", ".join(c["campaign_id"] for c in campaigns))
    else:
        say("No campaign bundles under data/campaigns; model views will be empty.")


def prepare_frontend(root: Path) -> None:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("Install Node.js/npm, then rerun this command.")
    web = root / "web"
    fingerprint = shared.sha256(web / "package-lock.json")
    marker = web / "node_modules/.engine-lock-sha256"
    if not (web / "node_modules/.bin/vite").exists() or (
        not marker.exists() or marker.read_text().strip() != fingerprint
    ):
        say("Installing frontend dependencies from package-lock.json…")
        subprocess.run([npm, "ci"], cwd=web, check=True)
        marker.write_text(fingerprint + "\n")


def prepare_auth() -> None:
    from engine.cli_auth import login
    from engine.config.league import get_league
    from engine.espn import credentials, discovery
    from engine.espn.service import LeagueService
    from engine.tables.league import query_snapshot

    settings = get_settings()
    config = get_league()
    stored = credentials.read(settings.env_path)
    ok = False
    if stored and stored.league_id:
        say("Checking ESPN session…")
        try:
            ok, message = discovery.verify(
                stored, stored.league_id, config.draft_season, raise_unavailable=True
            )
            say(message)
        except discovery.DiscoveryError:
            # Network failures do not mean the cookies expired. The service may
            # still be able to restore a captured snapshot or retry the request.
            say("ESPN verification unavailable; keeping saved credentials and checking cache.")
            ok = True
    if not ok:
        say("Sign in to ESPN in the browser to enable league data.")
        login(timeout=600, fresh=False, verbose=False)
        get_settings.cache_clear()
        settings = get_settings()
    say("Loading league observations…")
    snapshot, stale, _ = LeagueService(settings=settings).observations()
    query_snapshot(snapshot, settings.data_dir)
    say(
        f"League ready: {snapshot.season}, week {snapshot.week}"
        + (" (cached; ESPN unavailable)" if stale else "")
    )


def prepare(*, auth: bool = True) -> None:
    """One repeatable preparation path for setup, dev, and restart."""
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv, then rerun this command.")
    # Keep GCS dependencies optional for library/research users. uv installs the
    # declared extra for the child; no bespoke dependency installer or downloader.
    with shared.lock(get_settings().runtime_dir / "startup"):
        prepare_frontend(REPO_ROOT)
        subprocess.run(
            [uv, "run", "--extra", "shared-data", "python", "-m", "engine.startup"],
            cwd=REPO_ROOT,
            check=True,
        )
        if auth:
            prepare_auth()
        else:
            say("ESPN skipped (--no-auth); league features require sign-in.")


if __name__ == "__main__":
    try:
        prepare_data(get_settings().data_dir)
    except Exception as error:
        raise SystemExit(f"Data setup failed: {error}") from error
