"""Historical evidence clocks, conservative parsing, and acquisition integrity."""

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from engine.data.availability_history import archive, ledger
from engine.data.availability_history.build import injury_events, roster_events


def report(text, day="2021-06-11"):
    row = dict(
        description=text,
        event_date=day,
        team="NO",
        source="official_club",
        source_url="https://example.test/transactions",
        raw_sha256="raw",
        capture_key="test",
        collected_at="2026-10-10T12:00:00+00:00",
        published_at=None,
    )
    return {**row, "raw_record_id": ledger.record_id(row)}


def events(*reports):
    names = {"drew brees": {"brees"}, "tom brady": {"brady"}, "mike williams": {"mike1", "mike2"}}
    display = {"brees": "Drew Brees", "brady": "Tom Brady"}
    rows, audit = ledger.transaction_events(list(reports), names, display)
    return pl.DataFrame(rows, schema=ledger.SCHEMA), audit


def test_retirement_and_return_are_cutoff_local_and_future_invariant():
    old = report("Placed QB Drew Brees on reserve/retired.")
    new = report("Reinstated QB Drew Brees to the active roster.", "2021-10-01")
    all_events, _ = events(old, new)
    first, _ = events(old)
    cutoff = datetime(2021, 9, 9, 12, tzinfo=UTC)
    a = ledger.as_of(first, cutoff, mode="reconstructed")
    b = ledger.as_of(all_events, cutoff, mode="reconstructed")
    assert a.equals(b)
    assert a["status"].to_list() == ["RET"]
    assert ledger.as_of(all_events, datetime(2021, 10, 3, tzinfo=UTC), mode="reconstructed")[
        "status"
    ].to_list() == ["ACT"]


def test_observed_mode_does_not_backdate_later_capture():
    frame, _ = events(report("Placed QB Drew Brees on reserve/retired."))
    out = ledger.as_of(frame, datetime(2021, 9, 9, tzinfo=UTC), player_ids=["brees", "newcomer"])
    assert out["status"].to_list() == ["UNKNOWN", "UNKNOWN"]


def test_notice_does_not_replace_roster_evidence():
    frame, _ = events(
        report("Signed QB Tom Brady."), report("Announced QB Tom Brady's retirement.", "2021-07-01")
    )
    result = ledger.as_of(frame, datetime(2021, 9, 9, tzinfo=UTC), mode="reconstructed")
    assert dict(result.select("dimension", "status").iter_rows()) == {
        "retirement_notice": "RET_ANNOUNCED",
        "roster": "ROSTERED",
    }


def test_same_day_conflict_is_unknown_and_order_invariant():
    frame, _ = events(
        report("Placed QB Drew Brees on reserve/retired."), report("Activated QB Drew Brees.")
    )
    cutoff = datetime(2021, 9, 9, tzinfo=UTC)
    a = ledger.as_of(frame, cutoff, mode="reconstructed")
    assert a.equals(ledger.as_of(frame.reverse(), cutoff, mode="reconstructed"))
    assert a["status"].to_list() == ["UNKNOWN"]
    assert a["conflict"].to_list() == [True]


def test_ambiguous_identity_and_embedded_future_actions_are_withheld():
    frame, audit = events(
        report("Signed WR Mike Williams."),
        report("Placed QB Drew Brees on reserve, then activated him on 10/1."),
    )
    assert frame.is_empty()
    assert {r["reason"] for r in audit} == {"ambiguous_identity", "embedded_chronology"}


def test_retirement_does_not_spill_into_an_unrelated_sentence():
    frame, audit = events(
        report("Placed QB Drew Brees on reserve/retired. Declined QB Tom Brady's option.")
    )
    assert frame.select("player_id", "status").rows() == [("brees", "RET")]
    assert audit[0]["reason"] == "unparsed_action"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Designated Drew Brees to return from injured reserve.",
            ("return_notice", "RETURN_WINDOW"),
        ),
        (
            "Signed Tom Brady to a one-day contract to retire.",
            ("retirement_notice", "RET_ANNOUNCED"),
        ),
        ("Signed Tom Brady from the practice squad.", ("roster", "ACT")),
        ("Signed Tom Brady to a future contract.", ("roster", "FUTURE")),
        ("Extended Tom Brady's contract.", None),
        ("Traded Tom Brady for Drew Brees.", None),
    ],
)
def test_conservative_transaction_types(text, expected):
    assert ledger.classify(text) == expected


def receipt():
    return dict(
        url="https://example.test",
        key="example",
        sha256="raw",
        collected_at="2026-10-10T12:00:00+00:00",
    )


def test_roster_week_cannot_supply_preseason_state():
    frame = pl.DataFrame(
        [
            dict(
                season=2021,
                week=1,
                game_type="REG",
                gsis_id="brees",
                full_name="Drew Brees",
                status="ACT",
                team="NO",
            )
        ]
    )
    rows, _ = roster_events(frame, receipt(), {(2021, 1): date(2021, 9, 13)})
    ev = pl.DataFrame(rows, schema=ledger.SCHEMA)
    assert ledger.as_of(ev, datetime(2021, 9, 9, tzinfo=UTC), mode="reconstructed").is_empty()
    assert rows[0]["usable_at"] == datetime(2021, 9, 14, 12, tzinfo=UTC)


def test_undated_injuries_are_retained_as_coverage_gaps_not_healthy():
    frame = pl.DataFrame(
        [
            dict(
                season=2026,
                week=1,
                gsis_id="brees",
                full_name="Drew Brees",
                report_status="Out",
                team="NO",
            )
        ]
    )
    rows, counts = injury_events(frame, receipt())
    assert rows == []
    assert counts["undated"] == 1


def test_late_modification_is_not_available_earlier():
    frame = pl.DataFrame(
        [
            dict(
                season=2021,
                week=1,
                gsis_id="brees",
                full_name="Drew Brees",
                report_status="Out",
                team="NO",
                date_modified=datetime(2021, 9, 20, tzinfo=UTC),
            )
        ]
    )
    rows, _ = injury_events(frame, receipt())
    out = ledger.as_of(
        pl.DataFrame(rows, schema=ledger.SCHEMA),
        datetime(2021, 9, 14, tzinfo=UTC),
        mode="reconstructed",
    )
    assert out.is_empty()


def test_incomplete_attempt_is_not_mixed_with_fallback(monkeypatch, tmp_path):
    def attempt(root, year, family, limit):
        if family == "site":
            return [{"partial": True}], dict(status="fetch_failed")
        return [{"complete": True}], dict(status="all_reported_pages")

    monkeypatch.setattr(archive, "_espn_attempt", attempt)
    rows, status = archive.espn_year(tmp_path, 2003)
    assert rows == [{"complete": True}]
    assert len(status["attempts"]) == 2


def test_repeated_api_page_is_rejected(monkeypatch, tmp_path):
    def capture(root, key, url):
        page = int(url.split("page=")[-1])
        path = root / f"{key}.bin"
        path.write_text(
            json.dumps(
                dict(
                    count=2,
                    pageCount=2,
                    pageIndex=page,
                    transactions=[dict(date="2003-01-01", description="Signed QB Drew Brees.")],
                )
            )
        )
        return dict(
            key=key,
            file=path.name,
            http_status=200,
            sha256="raw",
            collected_at="2026-10-10T12:00:00Z",
        )

    monkeypatch.setattr(archive, "capture", capture)
    with pytest.raises(ValueError, match="Repeated ESPN page"):
        archive._espn_attempt(tmp_path, 2003, "site", 500)


def test_capture_tampering_is_rejected(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"changed")
    (tmp_path / "x.json").write_text(
        json.dumps(dict(url="https://example.test", sha256="expected"))
    )
    with pytest.raises(ValueError, match="Capture changed"):
        archive.capture(tmp_path, "x", "https://example.test")


def test_week_grain_injury_has_explicit_conservative_clock():
    frame = pl.DataFrame(
        [
            dict(
                season=2026,
                week=1,
                game_type="REG",
                gsis_id="p",
                full_name="Player",
                report_status="Out",
                team="OAK",
            )
        ]
    )
    rows, counts = injury_events(frame, receipt(), {(2026, 1): date(2026, 9, 14)})
    assert counts["week_proxy"] == 1
    assert rows[0]["usable_at"] == datetime(2026, 9, 15, 12, tzinfo=UTC)
    assert rows[0]["published_at"] is None
    assert rows[0]["team"] == "LV"
    assert "proxy" in rows[0]["date_basis"]
    ev = pl.DataFrame(rows, schema=ledger.SCHEMA)
    assert ledger.as_of(ev, datetime(2026, 9, 15, 11, tzinfo=UTC), mode="reconstructed").is_empty()
    assert ledger.as_of(ev, datetime(2026, 10, 1, tzinfo=UTC), mode="reconstructed").is_empty()


def test_ambiguous_identity_can_use_only_prior_team_evidence():
    names = {"mike williams": {"old", "new"}}
    display = {"old": "Mike Williams", "new": "Mike Williams"}
    raw = report("Signed WR Mike Williams.")
    previous = ledger.event(
        raw,
        player_id="old",
        dimension="roster",
        status="ACT",
        team="NO",
        source="weekly_roster",
        usable_at=datetime(2021, 1, 5, tzinfo=UTC),
        collected_at=datetime(2026, 10, 11, tzinfo=UTC),
    )
    future = ledger.event(
        raw,
        player_id="new",
        dimension="roster",
        status="ACT",
        team="NO",
        source="weekly_roster",
        usable_at=datetime(2021, 9, 15, tzinfo=UTC),
    )
    rows, _ = ledger.transaction_events(
        [raw], names, display, ledger.prior_roster_index([previous, future])
    )
    assert len(rows) == 1
    assert rows[0]["player_id"] == "old"
    assert rows[0]["identity_evidence_id"] == previous["event_id"]
    assert rows[0]["collected_at"] == previous["collected_at"]
    rows, audit = ledger.transaction_events(
        [raw], names, display, ledger.prior_roster_index([future])
    )
    assert not rows
    assert audit[0]["reason"] == "ambiguous_identity"


@pytest.mark.parametrize(
    "description",
    [
        "Drew Brees (QB) activated from injured reserve, Tom Brady (QB) placed on injured reserve",
        "Placed QB Drew Brees on injured reserve Activated QB Tom Brady to the active roster",
        "QB Drew Brees Waived injured – QB Tom Brady",
    ],
)
def test_bundled_actions_do_not_assign_one_players_move_to_another(description):
    frame, audit = events(report(description))
    assert frame.is_empty()
    assert len(audit) == 2


def test_pipe_delimited_actions_can_be_separated():
    frame, _ = events(report("Waived: QB Drew Brees | Activated: QB Tom Brady"))
    assert dict(frame.select("player_id", "status").iter_rows()) == {"brees": "CUT", "brady": "ACT"}


def test_activation_does_not_imply_lifting_a_different_reserve_status():
    assert (
        ledger.classify("Activated Tom Brady from COVID reserve but he remains on injured reserve")
        is None
    )
