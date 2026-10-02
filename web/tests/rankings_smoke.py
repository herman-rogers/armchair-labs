"""Exercise published rankings in the browser without refreshing ESPN."""

import argparse
import csv
import io
import re
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5199")
    parser.add_argument("--api", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    expect.set_options(timeout=30000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        response = page.request.get(args.api + "/api/nextgen/rankings?position=QB&limit=1000")
        assert response.ok, response.text()
        published = response.json()
        quarterbacks = {r["player_display_name"]: r for r in published["rankings"]}
        baker, stafford, dart = (
            quarterbacks[n] for n in ("Baker Mayfield", "Matthew Stafford", "Jaxson Dart")
        )
        players = []
        for index, r in enumerate((baker, stafford, dart)):
            mine = index != 0
            players.append(
                dict(
                    player_id=r["player_id"],
                    espn_id=index + 1,
                    player_display_name=r["player_display_name"],
                    position="QB",
                    owner_team_id=3 if mine else None,
                    owner_team_name="My team" if mine else None,
                    lineup_slot="QB" if mine else None,
                    espn_team=r["team"],
                    availability="rostered" if mine else "free_agent",
                    injury_status="OUT" if index == 2 else "ACTIVE",
                    is_mine=mine,
                    attention=[],
                    injury_news=[],
                )
            )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            parsed = urlparse(route.request.url)
            if parsed.path == "/api/nextgen/league":
                route.fulfill(
                    status=200,
                    json=dict(
                        version="ranking-test-snapshot",
                        season=2026,
                        week=3,
                        captured_at="2026-09-23T00:00:00Z",
                        my_team_id=3,
                        stale=False,
                        age_seconds=0,
                        league_name="Test league",
                        regular_season_weeks=14,
                        teams=[
                            dict(
                                team_id=3,
                                team_name="My team",
                                wins=1,
                                losses=1,
                                is_mine=True,
                                faab_remaining=145,
                                division_name=None,
                            )
                        ],
                        players=players,
                        transactions=[],
                        draft=[],
                        decision_status="Observations",
                    ),
                )
            elif parsed.path == "/api/nextgen/league/matchups":
                route.fulfill(
                    status=200,
                    json=dict(
                        season=2026,
                        current_week=3,
                        requested_week=3,
                        available_weeks=[1, 2, 3],
                        captured_at="2026-09-23T00:00:00Z",
                        stale=False,
                        age_seconds=0,
                        source="fixture",
                        matchups=[],
                    ),
                )
            else:
                route.fulfill(
                    response=route.fetch(
                        url=args.api + parsed.path + ("?" + parsed.query if parsed.query else "")
                    )
                )

        context.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url)
        expect(page).to_have_url(re.compile(r"/intelligence/rankings$"))
        views = page.get_by_role("navigation", name="Intelligence views")
        expect(views.get_by_role("link", name="NextGen rankings", exact=True)).to_have_attribute(
            "aria-current", "page"
        )
        expect(
            page.get_by_role("columnheader", name="NextGen · position", exact=False)
        ).to_be_visible()
        expect(views.get_by_role("link", name="Players", exact=True)).to_have_count(0)
        page.get_by_label("Ranking position").select_option("QB")
        page.get_by_label("Find ranked player", exact=True).fill("Baker Mayfield")
        expect(page).to_have_url(re.compile(r"q=Baker\+Mayfield"))
        expect(
            page.get_by_role("cell", name=f"QB{baker['position_rank']}", exact=True)
        ).to_be_visible()
        expect(
            page.get_by_role("cell", name=f"{baker['prediction']:.1f}", exact=True)
        ).to_be_visible()
        page.get_by_label("Ranking pool").select_option("free")
        expect(page).to_have_url(re.compile(r"/intelligence/rankings\?position=QB&q=Baker\+Mayfield&pool=free$"))
        expect(
            page.get_by_role("cell", name=f"QB{baker['position_rank']}", exact=True)
        ).to_be_visible()
        with page.expect_download() as download:
            page.get_by_role("button", name="Export rankings · all ownership", exact=True).click()
        exported = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text())))
        assert len(exported) == 1 and exported[0]["player_id"] == baker["player_id"]
        assert int(exported[0]["position_rank"]) == baker["position_rank"]
        assert exported[0]["analysis_version"] == published["version"]
        profile_link = page.locator("table").get_by_role("link", name="Baker Mayfield", exact=True)
        profile_url = profile_link.get_attribute("href")
        assert profile_url == f"/players/{baker['player_id']}"
        other = page.context.new_page()
        other.goto(args.url + profile_url)
        expect(other.get_by_role("heading", name="Baker Mayfield", exact=True)).to_be_visible()
        expect(other.locator(".player-stats")).to_be_visible()
        other.reload()
        expect(other.get_by_role("heading", name="Baker Mayfield", exact=True)).to_be_visible()
        other.close()
        profile_link.click()
        expect(page.get_by_role("heading", name="Baker Mayfield", exact=True)).to_be_visible()
        stats = page.get_by_role("region", name="Baker Mayfield player stats", exact=True)
        for period in ("prior", "recent3", "career", "current"):
            response = page.request.get(args.api + f"/api/nextgen/players?period={period}&search=Baker%20Mayfield")
            assert response.ok, response.text()
            observed = next(row for row in response.json()["players"] if row["player_id"] == baker["player_id"])
            stats.get_by_label("Observation period").select_option(period)
            value = stats.locator("dl > div").filter(has=page.locator("dt", has_text="Passing yards")).first.locator("dd")
            expect(value).to_have_text(f"{observed['passing_yards']:.1f}")
            expect(stats.get_by_text(f"{observed['observed_weeks']} observed weeks", exact=False)).to_be_visible()
            expect(stats.locator("table")).to_have_count(0)
        for heading in ("Production", "Opportunity", "Efficiency", "Consistency"):
            expect(stats.get_by_role("heading", name=heading, exact=True)).to_be_visible()
        expect(page.locator(".player-stat-definitions dd").first).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/nextgen-player-stats-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("link", name="Forecasts", exact=True).click()
        expect(page).to_have_url(re.compile(rf"/players/{re.escape(baker['player_id'])}/forecasts\?period=current$"))
        expect(
            page.get_by_role("heading", name="NextGen · rest of season", exact=True)
        ).to_be_visible()
        page.get_by_label("Profile history through").select_option("2025")
        expect(
            page.get_by_role("heading", name="NextGen · rest of season", exact=True)
        ).to_have_count(0)
        page.reload()
        expect(page.get_by_label("Profile history through")).to_have_value("2025")
        # Browser Back walks the profile's history entries back to the filtered rankings.
        for _ in range(10):
            if urlparse(page.url).path == "/intelligence/rankings":
                break
            page.go_back()
        expect(page).to_have_url(re.compile(r"/intelligence/rankings\?position=QB&q=Baker\+Mayfield&pool=free$"))
        expect(page.get_by_label("Find ranked player", exact=True)).to_have_value("Baker Mayfield")
        expect(page.get_by_label("Ranking pool")).to_have_value("free")
        expect(page.locator("tbody tr.expanded-row")).to_have_count(0)
        page.get_by_label("Ranking horizon").select_option("next4")
        expect(
            page.get_by_role("columnheader", name="Next 4 weeks · points", exact=False)
        ).to_be_visible()
        page.get_by_text("Ranking evidence and model decisions", exact=True).click()
        expect(
            page.get_by_role("heading", name=re.compile(r"^QB · (Reference|Validated) forecast$"))
        ).to_be_visible()
        page.get_by_label("Ranking pool").select_option("mine")
        page.get_by_label("Find ranked player", exact=True).fill("Jaxson Dart")
        expect(page).to_have_url(re.compile(r"q=Jaxson\+Dart"))
        expect(page.get_by_role("cell", name="Unranked", exact=True)).to_be_visible()
        expect(page.get_by_text("Reported season-ending absence", exact=False)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/nextgen-rankings-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("navigation", name="Main sections").get_by_role(
            "link", name="League", exact=True
        ).click()
        page.get_by_role("navigation", name="League views").get_by_role(
            "link", name="Free agents", exact=True
        ).click()
        expect(page).to_have_url(re.compile(r"/league/free-agents$"))
        expect(
            page.get_by_role("cell", name=f"QB{baker['position_rank']}", exact=True)
        ).to_be_visible()
        page.get_by_role("navigation", name="League views").get_by_role("link", name="Rosters", exact=True).click()
        expect(page).to_have_url(re.compile(r"/league/rosters$"))
        expect(
            page.get_by_role("cell", name=f"QB{stafford['position_rank']}", exact=True)
        ).to_be_visible()
        assert not errors, errors
        page.wait_for_load_state("networkidle")
        context.unroute_all(behavior="ignoreErrors")
        browser.close()
        print(
            "Ranking browser checks passed: horizons, filters, stable ranks, CSV, profiles, "
            "historical cutoff, availability, mobile and league integration."
        )


if __name__ == "__main__":
    main()
