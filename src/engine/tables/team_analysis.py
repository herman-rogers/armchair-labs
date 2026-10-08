"""Pinned SQL source for Team analysis, shared with notebook/table consumers."""

from contextlib import contextmanager
from pathlib import Path

from engine.data.releases import current_catalog
from engine.data.verification import verified
from engine.tables.query import connect
from engine.tables.storage import content_hash, current, load_table

TABLE = "team_player_games"
REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "team",
    "player_id",
    "name",
    "position",
    "starting_qb_id",
    "starter_game",
    "league_points",
}


class TableVersionChanged(ValueError):
    """A client or in-flight request is pinned to a superseded table version."""


def check_source(data_dir: Path, manifest: dict) -> None:
    # Only compare the small app catalog's reference. Query replicas need no raw,
    # enriched, or source batch files; integrity comes from the sealed table bundle.
    app_catalog = current_catalog(data_dir)
    source = manifest.get("source_release", manifest.get("gold"))
    if app_catalog and source != app_catalog["gold"]:
        raise ValueError(
            "Team observations are behind the selected data release; "
            "refresh team_player_games before serving this analysis"
        )


def coverage_report(data_dir: Path, catalog: dict, ref: dict) -> dict:
    """Validate coverage once per loaded snapshot, including its file dependencies."""

    def load():
        _, manifest = load_table(data_dir, TABLE, ref)
        cutoff = manifest["observation_cutoff"]
        with connect(data_dir, catalog=catalog) as db:
            earliest, latest, latest_week = db.execute(
                "SELECT min(season), max(season), "
                "max(week) FILTER (WHERE season = ?) FROM analytics.team_player_games",
                [cutoff["season"]],
            ).fetchone()
            if (
                earliest is None
                or latest > cutoff["season"]
                or (latest_week is not None and latest_week > cutoff["through_week"])
            ):
                raise ValueError("Team observations disagree with their published cutoff")
            report = dict(
                table="analytics.team_player_games",
                table_version=ref["sha256"],
                version=manifest["version"],
                source_version=(manifest.get("source_release", manifest.get("gold")) or {}).get(
                    "version"
                ),
                built_at=manifest["built_at"],
                season=cutoff["season"],
                through_week=cutoff["through_week"],
                earliest_season=earliest,
                scoring="League points",
                evidence="historical",
                season_type="REG",
            )
        return report

    return verified(("team-coverage", str(data_dir.resolve()), content_hash(catalog)), load)


@contextmanager
def team_session(data_dir: Path, expected_version: str | None = None):
    catalog = current(data_dir)
    ref = catalog["tables"].get(TABLE)
    if ref is None:
        raise ValueError("The team_player_games table has not been published")
    if expected_version is not None and expected_version != ref["sha256"]:
        raise TableVersionChanged("Team observations changed; refreshing the table catalog")
    _, manifest = load_table(data_dir, TABLE, ref)
    if not manifest["columns"].keys() >= REQUIRED_COLUMNS:
        raise ValueError("The team_player_games table is missing required analysis columns")
    check_source(data_dir, manifest)
    report = coverage_report(data_dir, catalog, ref)
    with connect(data_dir, catalog=catalog) as db:
        yield db, report
    if current(data_dir)["tables"].get(TABLE) != ref:
        raise TableVersionChanged("Team observations changed during the request; refresh and retry")
    check_source(data_dir, manifest)
