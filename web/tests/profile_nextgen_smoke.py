"""Verify published gold profiles, tracking and approved forecasts in the browser."""

import argparse
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    expect.set_options(timeout=45000)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            parsed = urlparse(route.request.url)
            if parsed.path.startswith(("/api/league", "/api/nextgen/league")):
                route.fulfill(status=503, json={"detail": "No league refresh during profile test"})
                return
            response = route.fetch(
                url=args.api + parsed.path + ("?" + parsed.query if parsed.query else ""),
                timeout=45000,
            )
            route.fulfill(response=response)

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url)
        expect(page.get_by_role("tab", name="Players", exact=True)).to_be_visible()
        search = page.get_by_label("Search players", exact=True)
        search.fill("Lamar Jackson")
        page.get_by_role("button", name="Show details for Lamar Jackson", exact=True).click()
        profile = page.get_by_role("region", name="Lamar Jackson player profile")
        expect(profile.get_by_text("Completed NFL seasons observed", exact=True)).to_be_visible()
        profile.get_by_role("button", name="Next Gen Stats", exact=True).click()
        expect(
            profile.get_by_role("heading", name="NFL Next Gen Stats", exact=True)
        ).to_be_visible()
        expect(
            profile.get_by_role("columnheader", name="Completion percentage over expected")
        ).to_be_visible()
        expect(
            profile.get_by_text("Tracking sample sizes are not retained", exact=False)
        ).to_be_visible()
        assert "2026" not in profile.locator("tbody").inner_text()
        profile.get_by_role("button", name="Forecasts", exact=True).click()
        expect(profile.get_by_text("Season fantasy points · baseline", exact=True)).to_be_visible()
        assert "ridge" not in profile.locator("tbody").inner_text().lower()
        assert "Saved preseason forecasts" not in profile.inner_text()
        profile.get_by_label("Profile history through").select_option("2018")
        expect(
            profile.get_by_text("NFL observations through 2018, Week 18.", exact=False)
        ).to_be_visible()
        profile.get_by_role("button", name="Next Gen Stats", exact=True).click()
        expect(profile.locator("tbody")).to_contain_text("2018")
        assert "2019" not in profile.locator("tbody").inner_text()
        profile.get_by_label("Profile history through").select_option("")
        page.set_viewport_size({"width": 390, "height": 844})
        for tab in [
            "Next Gen Stats",
            "Forecasts",
            "Season history",
            "Role & participation",
            "Sources & gaps",
            "Overview",
        ]:
            profile.get_by_role("button", name=tab, exact=True).click()
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), tab
            assert profile.evaluate("el => el.scrollWidth <= el.clientWidth"), tab
        page.screenshot(path="/tmp/nextgen-profile-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        for name, metric in [
            ("Derrick Henry", "Rushing yards over expected"),
            ("Justin Jefferson", "Average separation"),
            ("Travis Kelce", "Yards after catch over expected"),
        ]:
            search.fill(name)
            page.get_by_role("button", name=f"Show details for {name}", exact=True).click()
            profile = page.get_by_role("region", name=f"{name} player profile")
            profile.get_by_role("button", name="Next Gen Stats", exact=True).click()
            expect(profile.get_by_role("columnheader", name=metric)).to_be_visible()
        page.screenshot(path="/tmp/nextgen-profile-desktop.png", full_page=True)
        assert not errors, errors
        browser.close()
    print("PASS: gold profiles, QB/RB/WR/TE tracking, approved forecasts, cutoffs, desktop/mobile")


if __name__ == "__main__":
    main()
