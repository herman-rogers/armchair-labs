"""Current NextGen desktop/mobile and archive-boundary checks against local releases."""

import argparse
import csv
import io
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from test_league_observations import snapshot
from engine.espn.observations import weekly_matchups

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5199")
    parser.add_argument("--api", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    expect.set_options(timeout=30000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            parsed = urlparse(route.request.url)
            requests.append(parsed.path)
            if parsed.path == "/api/nextgen/league/matchups":
                week = int(parse_qs(parsed.query).get("week", [2])[0])
                route.fulfill(status=200, json={**weekly_matchups(snapshot(), week), "stale": False, "age_seconds": 0})
            elif parsed.path == "/api/nextgen/league/refresh":
                route.fulfill(status=200, json={"stale": False})
            elif parsed.path == "/api/nextgen/league":
                # This browser test must not refresh ESPN; snapshot service is unit-tested.
                route.fulfill(
                    status=200,
                    json=dict(
                        version="test-observation-fixture",
                        season=2026,
                        week=2,
                        captured_at="2026-09-23T00:00:00Z", my_team_id=1,
                        stale=False,
                        league_name="Sweaty Plays",
                        teams=[],
                        players=[],
                        draft=[],
                        transactions=[],
                        decision_status="No approved trade, waiver or weekly-lineup model.",
                    ),
                )
            else:
                response = route.fetch(
                    url=args.api + parsed.path + ("?" + parsed.query if parsed.query else "")
                )
                route.fulfill(response=response)

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url)
        expect(page).to_have_url(re.compile(r"/intelligence/rankings$"))
        main_tabs = page.get_by_role("navigation", name="Main sections")
        expect(main_tabs.get_by_role("link")).to_have_text(["Intelligence", "League", "Research"])
        tabs = page.get_by_role("navigation", name="Intelligence views")
        expect(tabs.get_by_role("link")).to_have_text(["NextGen rankings", "QB passing", "Rookies"])
        expect(tabs.get_by_role("link", name="NextGen rankings", exact=True)).to_have_attribute("aria-current", "page")
        page.get_by_label("Find ranked player", exact=True).fill("Lamar Jackson")
        expect(page.get_by_role("columnheader", name="Our rank · baseline", exact=False)).to_have_count(0)
        expect(page.get_by_role("columnheader", name="ECR · overall", exact=False)).to_be_visible()
        expect(page.get_by_role("columnheader", name="ECR · position", exact=False)).to_be_visible()
        page.locator("table").get_by_role("link", name="Lamar Jackson", exact=True).first.click()
        expect(page.get_by_role("heading", name="Lamar Jackson", exact=True)).to_be_visible()
        expect(page).to_have_url(re.compile(r"/players/[^/?]+$"))
        page.get_by_role("link", name="Season history", exact=True).click()
        expect(page).to_have_url(re.compile(r"/players/[^/?]+/history$"))
        expect(page.get_by_text("Louisville", exact=True).first).to_be_visible()
        page.get_by_role("link", name="Forecasts", exact=True).click()
        expect(page.get_by_role("heading", name="Preseason reference forecasts")).to_be_visible()
        expect(page.get_by_text("Next-gen comparison", exact=True)).to_have_count(0)
        page.get_by_label("Profile history through").select_option("2018")
        expect(page).to_have_url(re.compile(r"/forecasts\?season=2018$"))
        expect(
            page.get_by_text("NFL observations through 2018, Week 18.", exact=False)
        ).to_be_visible()
        page.get_by_role("link", name="Next Gen Stats", exact=True).click()
        # Section links keep the profile's view params.
        expect(page).to_have_url(re.compile(r"/tracking\?season=2018$"))
        expect(page.get_by_role("heading", name="NFL Next Gen Stats")).to_be_visible()
        page.go_back()
        expect(page).to_have_url(re.compile(r"/forecasts\?season=2018$"))
        page.go_forward()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/nextgen-profile-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("link", name="← Dashboard", exact=True).click()
        expect(page).to_have_url(re.compile(r"/intelligence/rankings$"))
        page.get_by_label("Find ranked player", exact=True).fill("Michael Wilson")
        page.locator("table").get_by_role("link", name="Michael Wilson", exact=True).first.click()
        expect(page.locator(".player-stats").get_by_text("Median weekly points", exact=True)).to_be_visible()
        expect(page.locator(".player-stats table")).to_have_count(0)
        page.screenshot(path="/tmp/nextgen-measurements-desktop.png", full_page=True)
        # Back returns to the filtered rankings, not a fresh dashboard.
        page.go_back()
        expect(page).to_have_url(re.compile(r"/intelligence/rankings\?q=Michael\+Wilson$"))
        expect(page.get_by_label("Find ranked player", exact=True)).to_have_value("Michael Wilson")
        main_tabs.get_by_role("link", name="Research", exact=True).click()
        page.get_by_role("navigation", name="Research views").get_by_role("link", name="Reference forecasts", exact=True).click()
        expect(page).to_have_url(re.compile(r"/research/forecasts$"))
        page.get_by_label("Outcome", exact=True).select_option("receiving_yards")
        expect(
            page.get_by_role("columnheader", name="Reference source", exact=False)
        ).to_be_visible()
        with page.expect_download() as download:
            page.get_by_role("button", name="Export eligible estimates").click()
        contents = Path(download.value.path()).read_text()
        exported = list(csv.DictReader(io.StringIO(contents)))
        assert exported and {row["model"] for row in exported} == {"baseline"}
        assert {row["serving_status"] for row in exported} == {"baseline"}
        main_tabs.get_by_role("link", name="Research", exact=True).click()
        page.get_by_role("navigation", name="Research views").get_by_role("link", name="Evidence", exact=True).click()
        expect(page).to_have_url(re.compile(r"/research/evidence$"))
        page.get_by_label("Outcome", exact=True).select_option("season_appearance")
        expect(page.get_by_text("Brier score", exact=True).first).to_be_visible()
        page.get_by_role("button", name=re.compile("Show details")).first.click()
        expect(
            page.get_by_role("heading", name="Appearance probability calibration")
        ).to_be_visible()
        main_tabs.get_by_role("link", name="Intelligence", exact=True).click()
        expect(page).to_have_url(re.compile(r"/intelligence/rankings$"))
        expect(page.get_by_label("Player view", exact=True)).to_have_count(0)
        page.get_by_label("Find ranked player", exact=True).fill("Lamar Jackson")
        page.locator("table").get_by_role("link", name="Lamar Jackson", exact=True).first.click()
        page.get_by_role("link", name="Similar careers", exact=True).click()
        expect(page.get_by_role("heading", name="Statistically similar careers")).to_be_visible()
        expect(page.get_by_text("Cam Newton", exact=True)).to_be_visible()
        page.get_by_role("button", name="Show details for Cam Newton comparison", exact=True).click()
        expect(page.get_by_role("columnheader", name="Pass yards / covered week", exact=False)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/similar-careers-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("link", name="← Dashboard", exact=True).click()
        tabs.get_by_role("link", name="Rookies", exact=True).click()
        expect(page).to_have_url(re.compile(r"/intelligence/rookies$"))
        page.get_by_role("button", name="Clear filters", exact=True).click()
        expect(page.get_by_role("heading", name="2026 rookie class")).to_be_visible()
        expect(page.get_by_role("columnheader", name="NFL draft pick", exact=False)).to_be_visible()
        main_tabs.get_by_role("link", name="League", exact=True).click()
        expect(page).to_have_url(re.compile(r"/league/overview$"))
        expect(page.get_by_role("heading", name="Week 2 matchups")).to_be_visible()
        page.get_by_role("button", name=re.compile(r": Show lineups$")).first.click()
        expect(page.get_by_text("Historical starter", exact=True)).to_be_visible()
        expect(page.get_by_text("Historical bench", exact=True)).to_be_visible()
        page.get_by_role("tablist", name="Week", exact=True).get_by_role("tab", name="4", exact=True).click()
        expect(page.get_by_role("heading", name="Week 4 matchups")).to_be_visible()
        expect(page).to_have_url(re.compile(r"/league/overview\?week=4$"))
        page.get_by_role("button", name=re.compile(r": Show lineups$")).first.click()
        expect(page.get_by_text("No lineup captured for this week. Current rosters are available under Rosters.").first).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/league-restored-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        league_tabs = page.get_by_role("navigation", name="League views")
        for label, path in [("League overview", "overview"), ("Rosters", "rosters"), ("Free agents", "free-agents"),
                            ("Transactions", "transactions"), ("Draft recap", "draft")]:
            league_tabs.get_by_role("link", name=label, exact=True).click()
            expect(page).to_have_url(re.compile(rf"/league/{path}$"))
        page.get_by_role("button", name="Refresh league", exact=True).click()
        assert all(
            p.startswith(("/api/nextgen/", "/api/profiles"))
            or p in {"/api/status", "/api/research/data-catalog"}
            for p in requests
        ), requests
        main_tabs.get_by_role("link", name="Research", exact=True).click()
        page.get_by_role("navigation", name="Research views").get_by_role("link", name="Research archive", exact=True).click()
        expect(page).to_have_url(re.compile(r"/research/archive$"))
        expect(
            page.get_by_text("Research archive — excluded from everyday analysis", exact=True)
        ).to_be_visible()
        page.get_by_label("Entry type").select_option("data_release")
        page.get_by_label("Search research", exact=True).fill("canonical_20260923_r5")
        expect(page.get_by_text("quarantined", exact=True)).to_be_visible()
        expect(
            page.get_by_text(re.compile("known invalid receiving-target counts")).first
        ).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/nextgen-archive-mobile.png", full_page=True)
        assert page.request.get(args.api + "/api/board").status == 409
        response = page.request.get(args.api + "/api/board?scope=research&limit=1")
        assert response.status == 200
        assert response.headers["x-artifact-disposition"] == "archived_not_for_analysis"
        assert (
            page.request.get(args.api + "/api/nextgen/forecasts?model=ridge&format=csv").status
            == 409
        )
        assert not errors, errors
        page.wait_for_load_state("networkidle")
        page.unroute_all(behavior="ignoreErrors")
        browser.close()
        print(f"NextGen browser checks passed; {len(exported)} eligible estimates exported.")


if __name__ == "__main__":
    main()
