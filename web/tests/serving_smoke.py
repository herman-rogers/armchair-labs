"""Exercise real serving APIs and frontend query reuse, without pulling live ESPN."""

import argparse
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    args = parser.parse_args()
    expect.set_options(timeout=30000)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, api_responses = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "response",
            lambda response: api_responses.append((urlparse(response.url).path, response.status)),
        )
        page.route(
            "**/api/nextgen/league**",
            lambda route: route.fulfill(
                status=503, json={"detail": "Live ESPN intentionally excluded from serving test"}
            ),
        )
        page.route(
            "**/api/league**",
            lambda route: route.fulfill(
                status=503, json={"detail": "Live ESPN intentionally excluded from serving test"}
            ),
        )
        page.goto(args.url + "/intelligence/rankings")
        expect(page.get_by_role("link", name="Josh Allen", exact=True)).to_be_visible()
        ranking_count = api_responses.count(("/api/nextgen/rankings", 200))
        page.get_by_label("Find ranked player").fill("Josh Allen")
        expect(
            page.locator("section[aria-label='NextGen rankings'] table").first.locator("tbody tr")
        ).to_have_count(1)
        assert api_responses.count(("/api/nextgen/rankings", 200)) == ranking_count
        page.get_by_role("link", name="Josh Allen", exact=True).click()
        expect(page.get_by_role("heading", name="Josh Allen", exact=True)).to_be_visible()
        expect(page.locator("#profile-similar")).to_contain_text("Comparing the first")
        expect(page.locator("#profile-forecasts")).to_contain_text("remaining points")
        assert api_responses.count(("/api/nextgen/rankings", 200)) == ranking_count
        page.get_by_label("Profile history through").select_option("2024")
        expect(page.locator(".player-profile")).to_contain_text(
            "NFL observations through 2024, Week 18"
        )
        page.get_by_label("Profile history through").select_option("")
        expect(page.locator(".player-profile")).to_contain_text("NFL observations through 2026")
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.goto(args.url + "/research/archive/intelligence/players")
        expect(page.get_by_role("heading", name="Career history", exact=True)).to_be_visible()
        expect(page.get_by_text(re.compile(r"[0-9,]+ matching players"))).to_be_visible()
        directory_count = api_responses.count(("/api/profiles/directory", 200))
        page.get_by_label("Search players", exact=True).fill("Josh Allen")
        expect(page.get_by_text("1 matching players", exact=False)).to_be_visible()
        assert api_responses.count(("/api/profiles/directory", 200)) == directory_count
        page.goto(args.url + "/research/archive/intelligence/players?group=college")
        expect(page.get_by_role("heading", name="College careers", exact=True)).to_be_visible()
        college = page.get_by_role("heading", name="College careers", exact=True).locator("..")
        expect(college.locator("tbody tr")).to_have_count(100)
        first_id = college.locator("tbody tr").first.locator("td").nth(1).inner_text()
        college.get_by_role("button", name="Next", exact=True).click()
        expect(page).to_have_url(re.compile(r"page=2"))
        expect(college.locator("tbody tr").first.locator("td").nth(1)).not_to_have_text(first_id)
        college.get_by_role("button", name="College player", exact=False).click()
        expect(page).not_to_have_url(re.compile(r"page=2"))
        # A stale catalog token must never retrieve a cached current response.
        response = page.request.get(
            args.url + "/api/profiles/directory", headers={"X-Data-Catalog": "obsolete"}
        )
        assert response.status == 409
        assert response.headers["x-data-catalog-stale"] == "true"
        assert not errors, errors
        unexpected = [
            (path, code)
            for path, code in api_responses
            if code >= 400
            and not path.startswith("/api/league")
            and not path.startswith("/api/nextgen/league")
            and path != "/api/profiles/directory"
        ]
        assert not unexpected, unexpected
        browser.close()
        print(
            "Serving smoke passed: local search, shared rankings, profiles, cutoffs, "
            "mobile layout, server pagination, and stale-release rejection."
        )


if __name__ == "__main__":
    main()
