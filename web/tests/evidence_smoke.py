"""Verify evidence units, baseline presentation, workload slices and archive boundaries."""

import argparse
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    expect.set_options(timeout=30000)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            parsed = urlparse(route.request.url)
            if parsed.path.startswith("/api/nextgen/league"):
                route.fulfill(status=503, json={"detail": "No league refresh during evidence test"})
            else:
                response = route.fetch(
                    url=args.api + parsed.path + ("?" + parsed.query if parsed.query else "")
                )
                route.fulfill(response=response)

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url + "/?section=research&research=evidence")
        expect(page).to_have_url(re.compile(r"/research/evidence$"))
        page.get_by_role("navigation", name="Main sections").get_by_role(
            "link", name="Research", exact=True
        ).click()
        tabs = page.get_by_role("navigation", name="Research views")
        tabs.get_by_role("link", name="Evidence", exact=True).click()
        page.get_by_label("Outcome", exact=True).select_option("passing_yards")
        expect(page).to_have_url(re.compile(r"/research/evidence\?outcome=passing_yards$"))
        expect(page.get_by_text("season passing yards", exact=True)).to_be_visible()
        expect(page.get_by_text("Prior-season reference", exact=True)).to_be_visible()
        page.get_by_role("button", name=re.compile("Show details")).first.click()
        expect(page.get_by_text("Average absolute error:", exact=False)).to_be_visible()
        expect(
            page.get_by_role("heading", name="Passing-yard error by prior workload")
        ).to_be_visible()
        expect(page.get_by_text("300+ prior-season pass attempts", exact=True)).to_be_visible()
        expect(page.get_by_text("95% season-bootstrap interval", exact=False)).to_have_count(0)
        expect(page.get_by_role("columnheader", name="Model error", exact=False)).to_have_count(0)
        page.screenshot(path="/tmp/passing-yard-evidence-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/passing-yard-evidence-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_label("Evaluation window", exact=True).select_option("modern")
        page.get_by_role("button", name=re.compile("Show details")).first.click()
        expect(
            page.locator(".player-details").get_by_text("2019–2025", exact=False)
        ).to_be_visible()
        expect(page).to_have_url(re.compile(r"/research/evidence\?outcome=passing_yards&window=modern$"))
        tabs.get_by_role("link", name="Research archive", exact=True).click()
        # Filters are per page: the archive starts from its own defaults.
        expect(page).to_have_url(re.compile(r"/research/archive$"))
        page.get_by_label("Outcome", exact=True).select_option("passing_yards")
        row = page.get_by_role("row").filter(has=page.get_by_role("cell", name="boost", exact=True))
        row.get_by_role("button", name=re.compile("Show details")).click()
        expect(page.get_by_text("95% season-bootstrap interval", exact=False)).to_be_visible()
        page.go_back()
        page.go_back()
        expect(page).to_have_url(re.compile(r"/research/evidence\?outcome=passing_yards&window=modern$"))
        expect(page.get_by_label("Evaluation window", exact=True)).to_have_value("modern")
        expect(page.get_by_role("cell", name="boost", exact=True)).to_have_count(0)
        page.get_by_label("Outcome", exact=True).select_option("season_appearance")
        expect(
            page.get_by_text("Historical appearance frequency", exact=True).first
        ).to_be_visible()
        page.get_by_role("button", name=re.compile("Show details")).first.click()
        expect(
            page.get_by_role("heading", name="Appearance probability calibration")
        ).to_be_visible()
        assert not errors, errors
        page.unroute_all(behavior="wait")
        browser.close()
    print(
        "PASS: evidence units, baseline comparisons, workload groups, "
        "research boundary, desktop/mobile"
    )


if __name__ == "__main__":
    main()
