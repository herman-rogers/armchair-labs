"""Live dashboard checks for team selection, covariance risk and responsive layout."""

import argparse
import re
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


def check_chart_edge_cases(browser, base_url, payload):
    """Signed bars, unavailable values and negative observations survive rendering."""
    data = deepcopy(payload)
    data['risk'].update(independent_variance=100, covariance_effect=-30, variance=70)
    pair = next(p for p in data['pairs'] if p['a'] in data['risk']['player_ids'] and p['b'] in data['risk']['player_ids'])
    pair['points'] = [dict(game_id=f'fixture-{i}', season=2026, week=i+1, a=x, b=y) for i,(x,y) in enumerate([(-3,-2),(0,0),(8,12)])]
    page = browser.new_page(viewport={'width':390,'height':844})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('**/api/nextgen/team-analysis?*', lambda route: route.fulfill(json=data))
    page.goto(base_url + '/intelligence/teams')
    bars = page.locator('.ta-variance .recharts-bar-rectangle path')
    expect(bars).to_have_count(3)
    expect(page.locator('.ta-variance')).to_contain_text('-30.0')
    positive, negative = bars.nth(0).bounding_box(), bars.nth(1).bounding_box()
    assert negative['x'] < positive['x']
    assert abs(negative['x'] + negative['width'] - positive['x']) < 2
    scatter = page.locator('.ta-pair-detail .chart-frame')
    expect(scatter.locator('.recharts-scatter-symbol')).to_have_count(3)
    scatter.scroll_into_view_if_needed()
    scatter.locator('.recharts-scatter-symbol').first.hover()
    expect(scatter.locator('.chart-tooltip')).to_contain_text('2026 · Week 1')
    expect(scatter.locator('.chart-tooltip')).to_contain_text('-3.0 pts')
    expect(scatter.locator('.chart-tooltip')).to_contain_text('-2.0 pts')
    page.wait_for_function('document.documentElement.scrollWidth <= innerWidth + 1')
    page.screenshot(path='/tmp/team-analysis-scatter-mobile.png')
    data['risk'].update(independent_variance=0, covariance_effect=0, variance=0)
    page.reload()
    expect(page.locator('.ta-variance')).to_contain_text('0.0')
    data['risk'].update(independent_variance=None, covariance_effect=None, variance=None)
    pair['points'] = []
    page.reload()
    expect(page.get_by_text('Variance decomposition unavailable for this selection.', exact=True)).to_be_visible()
    expect(page.get_by_text('No shared scoring observations to plot.', exact=True)).to_be_visible()
    expect(page.locator('.ta-variance .recharts-wrapper')).to_have_count(0)
    assert not errors, errors
    page.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--output", default="data/outputs/team_analysis_browser")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    expect.set_options(timeout=30000)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 1100})
        page.clock.install()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def settled():
            expect(page.locator(".ta-risk")).to_have_attribute("aria-busy", "false")

        catalog_response = page.request.get(args.url + "/api/nextgen/team-analysis/catalog")
        assert catalog_response.status == 200
        catalog = catalog_response.json()
        assert catalog["table"] == "analytics.team_player_games"
        assert catalog["earliest_season"] == 2013
        catalog_requests = []
        page.on("request", lambda request: catalog_requests.append(request.url)
                if "/team-analysis/catalog" in request.url else None)
        page.goto(args.url + "/intelligence/teams")
        settled()
        pinned = page.request.get(
            args.url + "/api/nextgen/team-analysis?table_version=" + catalog["table_version"]
        )
        assert pinned.status == 200
        assert pinned.json()["report"]["table_version"] == catalog["table_version"]
        check_chart_edge_cases(browser, args.url, pinned.json())
        expect(page.get_by_text("Player analysis", exact=True)).to_have_count(0)
        expect(page.get_by_role("link", name="Team analysis", exact=True)).to_have_attribute(
            "aria-current", "page"
        )
        expect(page.get_by_label("Select Matthew Stafford")).to_be_checked()
        expect(page.get_by_label("Select Puka Nacua")).to_be_checked()
        expect(page.get_by_label("Select Davante Adams")).to_be_checked()
        assert page.locator(".ta-matrix button").count() > 200
        page.screenshot(path=str(output / "desktop.png"), full_page=True)

        page.get_by_label("Select Davante Adams").uncheck()
        settled()
        assert "players=" in page.url
        page.reload()
        settled()
        expect(page.get_by_label("Select Davante Adams")).not_to_be_checked()
        page.go_back()
        settled()
        expect(page.get_by_label("Select Davante Adams")).to_be_checked()
        page.get_by_role("button", name="Clear", exact=True).click()
        settled()
        expect(page.locator(".ta-risk")).to_contain_text("Select starters")
        page.get_by_label("Select Matthew Stafford").check()
        settled()
        expect(page.locator(".ta-kpis")).to_contain_text("0.0%")
        page.get_by_label("Select Cooper Kupp").check()
        settled()
        page.get_by_label("Select Davante Adams").check()
        settled()
        expect(page.locator(".ta-risk")).to_contain_text("Insufficient overlap · 0 shared games")

        page.get_by_label("NFL team", exact=True).select_option("BUF")
        settled()
        expect(page.get_by_label("Select Josh Allen")).to_be_checked()
        assert "players=" not in page.url
        page.get_by_label("NFL team", exact=True).select_option("LA")
        settled()
        page.get_by_label("Variance basis", exact=True).select_option("raw")
        settled()
        page.get_by_label("Participation", exact=True).select_option("active")
        settled()
        page.get_by_label("From season", exact=True).select_option("2023")
        settled()
        page.get_by_label("Through season", exact=True).select_option("2024")
        settled()
        expect(page.get_by_label("Select Davante Adams")).to_have_count(0)
        page.get_by_label("Starting quarterback", exact=True).select_option("all")
        settled()
        page.get_by_role("button", name=re.compile("Matthew Stafford and Puka Nacua:")).click()
        expect(page.locator(".ta-pair-detail h4")).to_have_text("Matthew Stafford ↔ Puka Nacua")
        page.get_by_label("Minimum shared games", exact=True).select_option("25")
        page.get_by_role("button", name="Switch to dark mode", exact=True).click()
        page.screenshot(path=str(output / "dark.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        assert page.locator(".ta-matrix-scroll").evaluate("e => e.scrollWidth > e.clientWidth")
        invalid = page.request.get(args.url + "/api/nextgen/team-analysis?basis=invalid")
        assert invalid.status == 422
        stale = page.request.get(
            args.url + "/api/nextgen/team-analysis", headers={"X-Data-Catalog": "outdated"}
        )
        assert stale.status == 409 and stale.headers.get("x-data-catalog-stale") == "true"
        stale_table = page.request.get(
            args.url + "/api/nextgen/team-analysis?table_version=outdated"
        )
        assert stale_table.status == 409
        assert stale_table.headers.get("x-table-catalog-stale") == "true"
        page.get_by_label("From season", exact=True).select_option("2013")
        settled()
        page.get_by_label("Through season", exact=True).select_option("2013")
        settled()
        expect(page.locator(".ta-sample-line")).to_contain_text("2013–2013")
        historical = page.request.get(
            args.url + "/api/nextgen/team-analysis?team=LA&start=2013&end=2013&qb=all"
        )
        assert historical.status == 200 and historical.json()["team_games"] > 0

        # Simulate a table-only publication while the app's gold token stays fixed.
        # No real data or pointers are changed by this browser cache check.
        replacement = "browser-test-new-table-version"
        requests = []
        published_week = catalog["through_week"] + 1

        def changed_table(route):
            params = parse_qs(urlparse(route.request.url).query, keep_blank_values=True)
            assert "table_version" not in params
            requests.append(params)
            response = route.fetch()
            assert response.status == 200
            payload = response.json()
            payload["report"].update(table_version=replacement, through_week=published_week)
            route.fulfill(response=response, json=payload)

        page.route("**/api/nextgen/team-analysis?*", changed_table)
        page.get_by_label("Variance basis", exact=True).select_option("season")
        settled()
        expect(page.get_by_label("Through season", exact=True)).to_contain_text(
            f"Latest · {catalog['season']} W{published_week}"
        )
        # A subsequent table-only publication is picked up by the normal data poll.
        published_week += 1
        page.clock.fast_forward(61_000)
        expect(page.get_by_label("Through season", exact=True)).to_contain_text(
            f"Latest · {catalog['season']} W{published_week}"
        )
        assert len(requests) >= 2
        assert not catalog_requests, catalog_requests
        assert not errors, errors
        browser.close()
    print(
        "Team analysis browser checks passed: selection, overlap, URL history, filters, "
        "themes, mobile, validation and release isolation."
    )


if __name__ == "__main__":
    main()
