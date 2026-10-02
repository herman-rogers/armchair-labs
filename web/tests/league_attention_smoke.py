"""League attention on desktop/mobile; fixtures never refresh ESPN."""
import re
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playwright.sync_api import expect, sync_playwright
from patron.espn.attention import annotate_players, roster_attention
from patron.espn.observations import weekly_matchups
from tests.test_league_attention import player
from tests.test_league_observations import snapshot


def main():
    snap = snapshot()
    snap.players = [player(status="OUT")]
    current = deepcopy(snap.week_lineups[0])
    current.week = snap.week
    current.home_lineup[0].on_bye = True
    snap.week_lineups.append(current)
    rows = [{**asdict(snap.players[0]), "player_id": None, "is_mine": True, "availability": "rostered"}]
    notes = [{"espn_id": 11, "known_on": "2026-09-23", "source_url": "https://example.test/report", "summary": "Fixture reported concern", "evidence_status": "reported_expectation"}]
    overview = dict(version="fixture", season=2026, week=3, captured_at=snap.captured_at, stale=True, my_team_id=1,
                    league_name="Attention fixture", teams=[], transactions=[], draft=[],
                    players=annotate_players(rows, snap, notes), attention_notices=roster_attention(snap), injury_news_warning=None)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        def route_api(route):
            url = urlparse(route.request.url)
            if url.path == "/api/nextgen/league":
                route.fulfill(status=200, json=overview)
            elif url.path == "/api/nextgen/league/matchups":
                week = int(parse_qs(url.query).get("week", [3])[0])
                route.fulfill(status=200, json={**weekly_matchups(snap, week), "stale": True, "age_seconds": 1200})
            else:
                route.continue_()
        page.route(re.compile(r".*/api/nextgen/league(?:\?.*)?$|.*/api/nextgen/league/matchups.*"), route_api)
        page.goto("http://127.0.0.1:5173")
        page.get_by_role("navigation", name="Main sections").get_by_role("link", name="League", exact=True).click()
        expect(page).to_have_url(re.compile(r"/league/overview$"))
        attention = page.get_by_role("region", name="Needs attention", exact=True)
        attention.locator("summary").first.click()
        expect(attention.get_by_text("ESPN: OUT", exact=True)).to_be_visible()
        expect(attention.get_by_text("Bye this week", exact=True)).to_be_visible()
        attention.locator("details.injury-notes > summary").first.click()
        expect(attention.get_by_text("Fixture reported concern", exact=False)).to_be_visible()
        expect(attention.get_by_text("Saved league data is stale.", exact=False)).to_be_visible()
        matchup = page.get_by_role("region", name="Weekly matchups", exact=True)
        matchup.locator(".matchup.mine .matchup-row").first.click()
        expect(matchup.get_by_role("cell", name="OUT", exact=True)).to_be_visible()
        page.get_by_role("tablist", name="Week", exact=True).get_by_role("tab", name="2", exact=True).click()
        expect(page).to_have_url(re.compile(r"/league/overview\?week=2$"))
        matchup.locator(".matchup.mine .matchup-row").first.click()
        expect(matchup.locator(".matchup.mine .league-lineups")).to_be_visible()
        expect(matchup.get_by_role("cell", name="OUT", exact=True)).to_have_count(0)
        page.get_by_role("link", name="Review players", exact=True).click()
        expect(page).to_have_url(re.compile(r"/league/rosters\?attention=1$"))
        expect(page.get_by_label("Needs attention only")).to_be_checked()
        expect(page.get_by_role("cell", name="OUT", exact=True)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/league-attention-mobile.png", full_page=True)
        assert not errors, errors
        browser.close()
    print("League attention browser checks passed")


if __name__ == "__main__":
    main()
