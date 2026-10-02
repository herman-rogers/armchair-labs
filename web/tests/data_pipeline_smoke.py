"""Exercise the published catalog and frontend consumers without league refreshes."""

import argparse
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    expect.set_options(timeout=30000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(30000)
        errors, observed, catalog_requests = [], {}, []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            parsed = urlparse(route.request.url)
            if parsed.path.startswith("/api/league"):
                route.fulfill(status=503, json={"detail": "League unavailable during data test"})
                return
            response = route.fetch(
                url=args.api + parsed.path + ("?" + parsed.query if parsed.query else "")
            )
            if parsed.path == "/api/research/data-catalog":
                catalog_requests.append(True)
            if response.ok and parsed.path in (
                "/api/research/data-catalog",
                "/api/profiles",
                "/api/research/outlook",
                "/api/research/college",
                "/api/research/sources",
            ):
                observed[parsed.path] = response.json()
            route.fulfill(response=response)

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url)
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        panel = page.get_by_label("Current data release", exact=True)
        expect(panel).to_be_visible()
        panel.locator(":scope > summary").click()
        expect(panel.get_by_text("Coverage incomplete", exact=True)).to_be_visible()
        expect(panel.get_by_text("Forecast feature rows", exact=True)).to_be_visible()
        page.get_by_label("Search players", exact=True).fill("Lamar Jackson")
        page.get_by_role("button", name="Show details for Lamar Jackson", exact=True).click()
        expect(page.get_by_role("heading", name="Lamar Jackson", exact=True)).to_be_visible()
        catalog = observed["/api/research/data-catalog"]
        assert catalog["available"]
        assert observed["/api/profiles"]["report"]["gold"] == catalog["gold"]
        expect(
            panel.get_by_text(f"{catalog['coverage']['forecast_rows']:,}", exact=True).first
        ).to_be_visible()
        page.get_by_label("Player view", exact=True).select_option("outlook")
        page.get_by_role("button", name="Show details for Lamar Jackson", exact=True).click()
        assert observed["/api/research/outlook"]["gold"] == catalog["gold"]
        page.get_by_label("Population", exact=True).select_option("college")
        expect(
            page.get_by_role("heading", name="NFL translation forecasts", exact=True)
        ).to_be_visible()
        assert observed["/api/research/college"]["gold"] == catalog["gold"]
        page.get_by_role("tab", name="Research", exact=True).click()
        expect(page.get_by_label("Research dataset", exact=True)).to_have_value(
            catalog["historical_research_reference"]["version"]
        )
        expect(page.get_by_text("These are archived model results.", exact=False)).to_be_visible()
        page.get_by_role("tab", name="Players", exact=True).click()
        page.get_by_label("Population", exact=True).select_option("all")
        page.get_by_label("Player view", exact=True).select_option("profiles")
        page.set_viewport_size({"width": 390, "height": 844})
        panel.get_by_text("Included tables and remaining gaps", exact=True).click()
        expect(panel.get_by_text(catalog["gold"]["version"], exact=True)).to_be_visible()
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        assert panel.evaluate("el => el.scrollWidth <= el.clientWidth")
        page.screenshot(path="/tmp/canonical-data-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.screenshot(path="/tmp/canonical-data-desktop.png", full_page=True)
        # The refresh event must refetch the catalog. Server-side stale tokens are
        # also rejected without mutating the catalog or serving another release.
        before = len(catalog_requests)
        with page.expect_response(lambda r: r.url.endswith("/api/research/data-catalog")):
            page.evaluate("window.dispatchEvent(new Event('data-catalog-changed'))")
        assert len(catalog_requests) > before
        stale = page.request.get(args.api + "/api/profiles", headers={"X-Data-Catalog": "old"})
        assert stale.status == 409 and stale.headers["x-data-catalog-stale"] == "true"
        assert not errors, errors
        browser.close()
    print(
        "PASS: canonical catalog, matching profile/outlook/college data, archived models, "
        "catalog refresh, stale token rejection, desktop/mobile"
    )


if __name__ == "__main__":
    main()
