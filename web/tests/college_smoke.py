"""Offline browser checks for college lineage, nulls, research labels and failures."""

import argparse
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright
from ui_smoke import fixtures

from engine.metrics.player_profile import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    args = parser.parse_args()
    status, _, _, _ = fixtures()
    player = {
        "player_id": "n1",
        "player_display_name": "Alpha Receiver",
        "position": "WR",
        "forecast_year": 2026,
        "college_ids": ["c1"],
        "college_team": "College A",
        "eligible": True,
        "link_methods": ["provider_id_name_chronology"],
        "points_college_only": 40,
        "points_draft_only": 80,
        "points_college_plus_draft": 85,
        "nfl_year1_points": None,
    }
    identity = {
        "college_id": "c1",
        "college_name": "Alpha Receiver",
        "player_id": "n1",
        "nfl_name": "Alpha Receiver",
        "first_college_season": 2023,
        "last_college_season": 2025,
        "status": "linked",
        "method": "provider_id_name_chronology",
        "evidence": None,
    }
    study = {
        "version": "college_fixture",
        "college_start": 2004,
        "college_end": 2025,
        "current_forecast_year": 2026,
        "nfl_complete_through": 2025,
        "identity": {
            "college_players": 2,
            "linked_nfl_players": 1,
            "statuses": [],
            "identifier_conflicts": [],
        },
        "cohorts": [
            {"forecast_year": 2026, "position": "WR", "candidates": 2, "linked": 1, "eligible": 1}
        ],
        "nfl_backtest": [
            {
                "position": "WR",
                "target": "nfl_year1_points",
                "era": "modern_2018_plus",
                "n": 300,
                "mae": {"draft_only": 25, "college_plus_draft": 24},
                "folds": [],
            }
        ],
        "college_backtest": [],
        "limitations": ["Unlinked does not mean zero NFL production."],
    }
    career = {
        "version": "profiles_fixture",
        "identity": {
            "player_id": "n1",
            "player_display_name": "Alpha Receiver",
            "position": "WR",
            "rookie_season": 2026,
        },
        "cutoff": {"season": 2026, "week": 2},
        "available_through": {"season": 2026, "week": 2},
        "sources": {
            "history": "history_fixture",
            "college": "college_fixture",
            "outlook": "outlook_fixture",
            "observations_saved_at": "2026-09-22T00:00:00Z",
        },
        "links": [identity],
        "college_seasons": [],
        "nfl_seasons": [],
        "career": {},
        "features": {},
        "transitions": [],
        "role_history": [],
        "role_periods": [],
        "injury_reports": [],
        "preseason_forecasts": [],
        "college_forecasts": [],
        "college_development_forecasts": [],
        "current_outlook": {
            "forecast_next4": 43,
            "cutoff_week": 2,
            "usage_baseline_next4": 40,
            "recent_snap_share": 0.7,
            "role_evidence": "Observed",
            "opportunity_trend": "Insufficient history",
            "recent_targets_pg": 5,
            "recent_carries_pg": 0,
        },
        "missing_evidence": [],
        "metrics": METRICS,
        "limits": ["Separate forecast horizons."],
    }
    state = {"broken": False}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            path = urlparse(route.request.url).path
            if path == "/api/status":
                route.fulfill(json=status)
            elif path == "/api/research/college":
                if state["broken"]:
                    route.fulfill(status=409, json={"detail": "College artifact changed: fixture"})
                else:
                    route.fulfill(json=study)
            elif path == "/api/research/college/players":
                route.fulfill(json={"total": 1, "players": [player]})
            elif path == "/api/profiles/player":
                route.fulfill(json=career)
            elif path == "/api/research/college/identities":
                route.fulfill(json={"total": 1, "players": [identity]})
            else:
                route.fulfill(status=503, json={"detail": "Offline fixture: unrelated endpoint"})

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url)
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        page.get_by_label("Population", exact=True).select_option("college")
        page.get_by_label("Player view").select_option("players")
        expect(page.get_by_text("Alpha Receiver", exact=True)).to_be_visible()
        expect(page.get_by_text("Research only.", exact=False)).to_be_visible()
        page.get_by_role("button", name="Show details for Alpha Receiver").click()
        page.get_by_role("button", name="Sources & gaps", exact=True).click()
        expect(page.get_by_text("College ESPN c1", exact=False)).to_be_visible()
        page.get_by_role("button", name="Overview", exact=True).click()
        expect(
            page.get_by_text("Next four weeks: 43.0 expected points", exact=False)
        ).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.get_by_role("tab", name="Research", exact=True).click()
        page.get_by_label("Research topic").select_option("college-identities")
        expect(page.get_by_role("button", name="Show details for Alpha Receiver")).to_be_visible()
        page.get_by_label("Research topic").select_option("college-evidence")
        expect(page.get_by_text("Added-college error", exact=True)).to_be_visible()
        state["broken"] = True
        page.reload()
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        page.get_by_label("Population", exact=True).select_option("college")
        page.get_by_label("Player view").select_option("players")
        expect(page.get_by_text("College artifact changed: fixture", exact=True)).to_be_visible()
        assert not errors, errors
        browser.close()
    print(
        "PASS: college lineage, separate horizons, evidence, identity audit, "
        "mobile, fail-closed errors"
    )


if __name__ == "__main__":
    main()
