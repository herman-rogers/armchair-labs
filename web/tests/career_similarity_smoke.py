"""Exercise career comparison coverage against the running local API and UI."""

import argparse
import re

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    args = parser.parse_args()
    expect.set_options(timeout=60000)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        # Profiles do not need live league data or any external refresh.
        page.route(
            re.compile(r"/api/(?:league|nextgen/league)"),
            lambda route: route.fulfill(status=503, json={"detail": "Offline league"}),
        )
        for player_id, name in [
            ("00-0034857", "Josh Allen"),
            ("00-0038555", "Tank Bigsby"),
            ("00-0036358", "CeeDee Lamb"),
            ("00-0034753", "Mark Andrews"),
        ]:
            page.goto(f"{args.url}/players/{player_id}/similar")
            expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
            region = page.get_by_role("region", name="Similar careers", exact=True)
            expect(region.get_by_text("Comparing the first", exact=False)).to_be_visible()
            expect(region.get_by_role("columnheader", name="Weeks covered")).to_be_visible()
            region.get_by_role(
                "button", name=re.compile("Show details for .+ comparison")
            ).first.click()
            expect(region.get_by_role("columnheader", name="Omitted weeks")).to_be_visible()
            if name == "Josh Allen":
                expect(region.get_by_role("link", name="2024, Week 18 source")).to_be_visible()
                expect(region.get_by_role("link", name="2025, Week 18 source")).to_be_visible()
            else:
                expect(region.get_by_text("Some seasons have partial", exact=False)).to_be_visible()
            region.get_by_text("Selected player’s statistical coverage", exact=True).click()
            expect(region.get_by_role("columnheader", name="Omitted weeks").last).to_be_visible()
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 1000})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), name
            print(f"PASS: {name} comparisons and coverage, desktop/mobile", flush=True)

        page.goto(f"{args.url}/players/00-0033280/similar")
        region = page.get_by_role("region", name="Similar careers", exact=True)
        expect(region.get_by_text("Career statistics are present", exact=False)).to_be_visible()
        region.get_by_text("Selected player’s statistical coverage", exact=True).click()
        expect(region.get_by_role("columnheader", name="Weeks covered")).to_be_visible()
        assert not errors, errors
        print("PASS: insufficient coverage explains affected seasons", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
