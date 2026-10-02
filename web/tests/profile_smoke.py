"""Browser verification against local profile artifacts; no league refreshes."""

import argparse
import re
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    expect.set_options(timeout=30000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5199")
    parser.add_argument("--api", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            parsed = urlparse(route.request.url)
            if (
                parsed.path.startswith("/api/league")
                or parsed.path == "/api/research/league-impact"
            ):
                route.fulfill(status=503, json={"detail": "League unavailable during profile test"})
            else:
                response = route.fetch(
                    url=args.api + parsed.path + ("?" + parsed.query if parsed.query else "")
                )
                route.fulfill(response=response)

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url)
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        expect(
            page.get_by_role("tablist", name="Intelligence views").get_by_role("tab")
        ).to_have_text(["Players", "League impact", "Research"])
        expect(page.get_by_role("tab", name="Players", exact=True)).to_have_attribute(
            "aria-selected", "true"
        )
        page.get_by_label("Search players", exact=True).fill("Lamar Jackson")
        page.get_by_role("button", name="Show details for Lamar Jackson", exact=True).click()
        expect(page.get_by_role("heading", name="Lamar Jackson", exact=True)).to_be_visible()
        expect(page.get_by_text("Completed NFL seasons observed", exact=True)).to_be_visible()
        page.get_by_role("button", name="Season history", exact=True).click()
        expect(page.get_by_role("heading", name="College seasons and transfers")).to_be_visible()
        expect(page.get_by_text("Louisville", exact=True).first).to_be_visible()
        page.get_by_role("button", name="Role & participation", exact=True).click()
        expect(page.get_by_text("High participation", exact=True).first).to_be_visible()
        expect(page.get_by_text("Limited participation", exact=True).first).to_be_visible()
        page.get_by_label("Profile history through").select_option("2018")
        expect(
            page.get_by_text("NFL observations through 2018, Week 18.", exact=False)
        ).to_be_visible()
        page.get_by_role("button", name="Overview", exact=True).click()
        expect(page.get_by_text("No current outlook at this cutoff.", exact=True)).to_be_visible()
        page.get_by_label("Profile history through").select_option("")
        expect(page.get_by_text("Next four weeks:", exact=False)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        profile = page.get_by_role("region", name="Lamar Jackson player profile")
        assert profile.evaluate("el => el.getBoundingClientRect().width <= innerWidth")
        assert profile.evaluate("el => el.scrollWidth <= el.clientWidth")
        expect(page.get_by_label("Research model", exact=True)).to_have_count(0)
        for tab in [
            "Season history",
            "Role & participation",
            "Forecasts",
            "Sources & gaps",
            "Overview",
        ]:
            page.get_by_role("button", name=tab, exact=True).click()
            assert profile.evaluate("el => el.scrollWidth <= el.clientWidth"), tab
        page.screenshot(path="/tmp/player-profile-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.screenshot(path="/tmp/player-profile-desktop.png", full_page=True)
        page.get_by_label("Player view").select_option("outlook")
        expect(page.get_by_label("Search players", exact=True)).to_have_value("Lamar Jackson")
        page.get_by_role("button", name="Show details for Lamar Jackson", exact=True).click()
        expect(page.get_by_role("heading", name="Lamar Jackson", exact=True)).to_be_visible()
        page.get_by_label("Player view").select_option("profiles")
        expect(page.get_by_label("Search players", exact=True)).to_have_value("Lamar Jackson")
        page.get_by_label("Population", exact=True).select_option("college")
        expect(page.get_by_label("Search players", exact=True)).to_have_value("Lamar Jackson")
        page.get_by_label("Player view").select_option("players")
        page.get_by_label("NFL entry class").select_option("all")
        page.get_by_label("Search players", exact=True).fill("Lamar Jackson")
        page.get_by_role("button", name="Show details for Lamar Jackson", exact=True).click()
        expect(page.get_by_role("heading", name="Lamar Jackson", exact=True)).to_be_visible()
        assert not errors, errors
        browser.close()
    print(
        "PASS: career profile, role context, college integration, "
        "historical cutoffs, desktop/mobile"
    )


if __name__ == "__main__":
    main()
