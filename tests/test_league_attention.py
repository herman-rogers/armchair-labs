from dataclasses import asdict

from patron.espn.attention import annotate_players, flags_for, reviewed_news, roster_attention
from patron.espn.sync import PlayerState
from tests.test_league_observations import snapshot


def player(status="ACTIVE", slot="QB"):
    return PlayerState(11, "Player", "QB", "NYG", status, 100.0, 100.0, 20.0, 1, "Mine", slot)


def test_injury_tags_news_and_unknown_are_distinct():
    p = {**asdict(player()), "player_id": "verified"}
    assert flags_for(p) == []
    assert flags_for(p, news=[{"summary": "Review"}])[0]["kind"] == "news"
    p["injury_status"] = "OUT"
    assert flags_for(p)[0]["severity"] == "urgent"
    p["injury_status"] = "QUESTIONABLE"
    assert flags_for(p)[0]["severity"] == "watch"
    p["injury_status"] = None
    assert flags_for(p)[0]["kind"] == "unknown_status"
    assert flags_for(p, on_bye=True)[1]["severity"] == "urgent"
    p["lineup_slot"] = "BE"
    assert flags_for(p, on_bye=True)[1]["severity"] == "watch"


def test_old_bye_does_not_flag_current_roster_and_news_keeps_provenance():
    snap = snapshot()
    snap.week_lineups[0].home_lineup[0].on_bye = True
    p = {**asdict(player()), "player_id": "verified"}
    note = {
        "espn_id": 11,
        "known_on": "2026-09-23",
        "source_url": "https://example.test",
        "summary": "Reported expectation",
    }
    rows = annotate_players([p], snap, [note])
    assert [f["kind"] for f in rows[0]["attention"]] == ["news"]
    assert rows[0]["injury_news"] == [note]
    snap.week_lineups[0].week = snap.week
    assert "bye" in [f["kind"] for f in annotate_players([p], snap, [])[0]["attention"]]


def test_empty_slots_are_not_claimed_when_roster_is_missing():
    snap = snapshot()
    assert "not captured" in roster_attention(snap)[0]["label"]
    snap.players = [player(slot="BE")]
    assert "unfilled QB" in roster_attention(snap)[0]["label"]
    snap.players[0].lineup_slot = "QB"
    assert roster_attention(snap) == []


def test_no_news_capture_is_explicit(tmp_path):
    notes, warning = reviewed_news(tmp_path, 2026)
    assert notes == []
    assert "No reviewed" in warning
