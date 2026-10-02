import gzip
import json
from datetime import date

import polars as pl
import pytest

from patron.data.historical_evidence import (
    BACKFILL_FILE,
    BACKFILL_MANIFEST,
    archive_rows,
    article_content,
    expand_dated_revisions,
    load_transaction_backfill,
    sha256,
)
from patron.metrics.experimental import build_transaction_features
from patron.metrics.roster_evidence import attach_roster_evidence, supplement_identity_names


def test_crosswalk_adds_only_unambiguous_identity_names():
    players = pl.DataFrame(
        {"gsis_id": ["existing"], "display_name": ["Existing Player"], "team": ["CLE"]}
    )
    crosswalk = pl.DataFrame(
        {
            "gsis_id": ["existing", "new", "ambiguous", "ambiguous"],
            "name": ["Wrong Replacement", "New Player", "First Name", "Different Name"],
            "team": ["DAL", "NYJ", "GB", "GB"],
        }
    )
    result = supplement_identity_names(players, crosswalk).sort("gsis_id")
    assert result["display_name"].to_list() == ["Existing Player", "New Player"]
    assert result["team"].to_list() == ["CLE", None]


def test_wrong_year_and_redirected_archives_are_not_evidence():
    record = {"status_code": 200, "final_url": "https://club.test/team/transactions/2025"}
    assert archive_rows("", record, 2010, "KC")[1] == "redirected_year"
    record["final_url"] = "https://club.test/team/transactions/2010"
    html = '<link rel="canonical" href="https://club.test/team/transactions/2025">'
    assert archive_rows(html, record, 2010, "KC")[1] == "unverified_year"
    html = html.replace("2025", "2010") + "<option selected>2025</option>"
    assert archive_rows(html, record, 2010, "KC")[1] == "wrong_selected_year"


def test_article_excludes_navigation_and_retains_distinct_dates():
    html = """<script>{"datePublished":"2019-03-15", "dateModified":"2023-02-28"}</script>
    <nav><p>Unrelated suspension</p></nav>
    <div class="story-part-rich-text-editor-wrapper"><p>First <a href="#">eight</a> games.</p></div>
    <aside><p>Another player</p></aside>
    <div class="story-part-rich-text-editor-wrapper"><p>Second paragraph.</p></div>"""
    parsed = article_content(html)
    assert parsed["paragraphs"] == ["First eight games.", "Second paragraph."]
    assert "Unrelated" not in parsed["text"]
    assert parsed["published_at"] != parsed["modified_at"]


def test_embedded_reactivation_is_dated_when_it_took_effect():
    frame = pl.DataFrame(
        {
            "transaction_date": [date(2022, 1, 3)],
            "transaction_year": [2022],
            "description": [
                "Placed TE Josh Oliver on Exempt/COVID-19, "
                "then reverted to the active roster on 1/6."
            ],
            "date_basis": ["official_archive_event_date"],
        }
    )
    result = expand_dated_revisions(frame)
    assert result.height == 2
    assert result["transaction_date"].to_list() == [date(2022, 1, 3), date(2022, 1, 6)]
    assert result["description"][1].startswith("Activated TE Josh Oliver")


def test_export_fails_if_source_capture_changes(tmp_path):
    static = tmp_path / "static"
    cache = tmp_path / "cache/historical_evidence/v1"
    static.mkdir()
    cache.mkdir(parents=True)
    path = static / BACKFILL_FILE
    pl.DataFrame(
        {"source_capture_file": ["raw.html.gz"], "source_sha256": [sha256(b"source")]}
    ).write_parquet(path)
    (cache / "raw.html.gz").write_bytes(gzip.compress(b"source"))
    (static / BACKFILL_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha256": sha256(path.read_bytes()),
                "captures": {"raw.html.gz": sha256(b"source")},
            }
        )
    )
    assert load_transaction_backfill(tmp_path).height == 1
    (cache / "raw.html.gz").write_bytes(gzip.compress(b"changed"))
    with pytest.raises(ValueError, match="capture changed"):
        load_transaction_backfill(tmp_path)


def test_rookies_and_market_candidates_use_dated_events_without_current_team_metadata():
    players = pl.DataFrame(
        {
            "gsis_id": ["r", "m"],
            "display_name": ["Rookie Player", "Market Player"],
            "latest_team": ["NYJ", "DAL"],
        }
    )
    prior = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "player_id": pl.String,
            "player_display_name": pl.String,
            "team": pl.String,
            "position": pl.String,
        }
    )
    candidates = pl.DataFrame(
        {
            "player_id": ["r", "m"],
            "player_display_name": ["Rookie Player", None],
            "forecast_season": [2019, 2019],
            "position": ["RB", "WR"],
        }
    )
    events = pl.DataFrame(
        [
            {
                "transaction_date": date(2019, month, 1),
                "transaction_year": 2019,
                "category": "official",
                "from_team": None,
                "to_team": "CLE",
                "player_name": None,
                "description": f"Signed {name}.",
                "source_url": "fixture",
            }
            for month, name in [(5, "Rookie Player"), (8, "Market Player")]
        ]
    )
    features = build_transaction_features(
        events,
        prior,
        players,
        [2019],
        "07-17",
        forecast_candidates=candidates,
        trusted_sources_only=True,
    )
    result = attach_roster_evidence(candidates, features, players).sort("player_id")
    market, rookie = result.to_dicts()
    assert market["player_display_name"] == "Market Player"
    assert market["cutoff_state_resolution"] == "no_prior_team"
    assert market["cutoff_preseason_team"] is None
    assert not market["cutoff_team_observed"]
    assert rookie["cutoff_preseason_team"] == "CLE"
    assert rookie["cutoff_team_observed"]
