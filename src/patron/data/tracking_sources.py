"""Source inventory for spatial tracking and complementary public observations.

Competition year is not data season. No source here implies complete NFL coverage.
"""

from dataclasses import asdict, dataclass

import requests


@dataclass(frozen=True)
class Source:
    id: str
    kind: str
    provider: str
    page: str
    coverage: str
    license: str
    adapter: str
    release_tag: str | None = None
    competition: str | None = None

    def metadata(self):
        return asdict(self)


SOURCES = {
    "bdb_2024_sumersports": Source(
        "bdb_2024_sumersports",
        "tracking_frames",
        "NFL competition data republished by SumerSports for reproducibility",
        "https://github.com/SumerSports/SportsTrackingTransformer/releases/tag/data-v1.0",
        "2022 Weeks 1–9 tackling archive; public mirror, not official Kaggle acquisition",
        "NFL competition terms apply; noncommercial research; mirror grants no new license",
        "bdb",
    ),
    "nflverse_rosters": Source(
        "nflverse_rosters",
        "identity_context",
        "nflverse",
        "https://nflreadr.nflverse.com/reference/load_rosters.html",
        "2016 onward roster history; gsis_it_id bridges NGS tracking IDs to GSIS",
        "CC-BY-4.0; attribute nflverse",
        "tabular",
        "rosters",
    ),
    "bdb_2019_sample": Source(
        "bdb_2019_sample",
        "tracking_frames",
        "NFL Football Operations",
        "https://github.com/nfl-football-ops/Big-Data-Bowl",
        "One 2017 game of tracking; accompanying metadata covers a larger sample",
        "NFL competition terms; no general redistribution license asserted",
        "bdb",
    ),
    "nflverse_ngs": Source(
        "nflverse_ngs",
        "tracking_summaries",
        "NFL Next Gen Stats via nflverse",
        "https://nflreadr.nflverse.com/reference/load_nextgen_stats.html",
        "2016 onward; minimum-attempt thresholds; week 0 is season aggregate",
        "nflverse CC-BY-4.0; retain provider attribution",
        "ngs",
        "nextgen_stats",
    ),
    "nflverse_participation": Source(
        "nflverse_participation",
        "play_participation",
        "NFL NGS / FTN via nflverse",
        "https://nflreadr.nflverse.com/reference/load_participation.html",
        "2016 onward; FTN from 2023, released after postseason; not live coordinates",
        "CC-BY-SA-4.0; NFL NGS attribution through 2022, FTN from 2023",
        "tabular",
        "pbp_participation",
    ),
    "nflverse_ftn": Source(
        "nflverse_ftn",
        "manual_charting",
        "FTN Data via nflverse",
        "https://nflreadr.nflverse.com/reference/load_ftn_charting.html",
        "2022 onward; selected play charting, not trajectories",
        "CC-BY-SA-4.0; attribute FTN Data via nflverse",
        "tabular",
        "ftn_charting",
    ),
    "nflverse_players": Source(
        "nflverse_players",
        "identity",
        "nflverse",
        "https://github.com/nflverse/nflverse-data/releases/tag/players",
        "Player identifier crosswalk; retrospective identity only",
        "CC-BY-4.0; attribute nflverse",
        "tabular",
        "players",
    ),
}

for year, coverage, adapter in [
    (2020, "2017–19 rushing handoff snapshots in current archive; not frame sequences", "handoff"),
    (2021, "2018 passing plays; selected positions, not all 22 on every frame", "bdb"),
    (2022, "2018–20 special teams plays", "bdb"),
    (2023, "2021 Weeks 1–8 dropbacks, snap to pass release", "bdb"),
    (2024, "2022 Weeks 1–9 tackling sample", "bdb"),
    (2025, "Pre-snap tendencies; determine actual seasons/weeks from downloaded games", "bdb"),
]:
    slug = f"nfl-big-data-bowl-{year}"
    key = f"bdb_{year}"
    SOURCES[key] = Source(
        key,
        "tracking_snapshot" if year == 2020 else "tracking_frames",
        "NFL / Kaggle",
        f"https://www.kaggle.com/competitions/{slug}/data",
        coverage,
        "Subject to competition rules; authenticated access may be required",
        adapter,
        competition=slug,
    )

for track in ("prediction", "analytics"):
    key = f"bdb_2026_{track}"
    slug = f"nfl-big-data-bowl-2026-{track}"
    SOURCES[key] = Source(
        key,
        "tracking_frames",
        "NFL / Kaggle",
        f"https://www.kaggle.com/competitions/{slug}/data",
        "NFL describes 2023–24 training; input/output clocks restart; verify file coverage",
        "Subject to competition rules; do not double-count shared competition plays",
        "bdb",
        competition=slug,
    )

for key, slug in [
    ("nfl_impact", "nfl-impact-detection"),
    ("nfl_helmets", "nfl-health-and-safety-helmet-assignment"),
    ("nfl_contact", "nfl-player-contact-detection"),
]:
    SOURCES[key] = Source(
        key,
        "video_and_sensor_tracking",
        "NFL / Kaggle",
        f"https://www.kaggle.com/competitions/{slug}/data",
        "Selected labeled clips and sensor data; safety sample is not season-representative",
        "Subject to competition rules",
        "tabular",
        competition=slug,
    )


def public_assets(source: Source, session: requests.Session) -> list[dict]:
    """Discover official files; preserve metadata separately from observation bytes."""
    if source.id == "bdb_2019_sample":
        base = "https://api.github.com/repos/nfl-football-ops/Big-Data-Bowl"
        response = session.get(f"{base}/commits/master", timeout=60)
        response.raise_for_status()
        revision = response.json()["sha"]
        response = session.get(f"{base}/contents/Data", params={"ref": revision}, timeout=60)
        response.raise_for_status()
        return [
            {
                "name": a["name"],
                "url": a["download_url"],
                "size": a["size"],
                "source_revision": revision,
                "source_updated_at": None,
            }
            for a in response.json()
            if a["name"].endswith(".csv")
        ]
    if source.release_tag:
        url = f"https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{source.release_tag}"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        assets = []
        for a in response.json()["assets"]:
            name = a["name"]
            if not name.endswith(".parquet") or "_old_" in name:
                continue
            if source.id == "nflverse_rosters" and (
                not name.startswith("roster_") or int(name[7:11]) < 2016
            ):
                continue
            assets.append(
                {
                    "name": name,
                    "url": a["browser_download_url"],
                    "size": a["size"],
                    "source_revision": str(a["id"]),
                    "source_updated_at": a["updated_at"],
                    "provider_digest": a.get("digest"),
                }
            )
        if not assets:
            raise ValueError(f"No Parquet assets discovered for {source.id}")
        return assets
    raise ValueError(f"{source.id} requires Kaggle download or a local import")
