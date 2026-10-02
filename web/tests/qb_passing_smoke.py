"""Exercise published QB forecasts, historical evidence and research boundaries."""

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
                route.fulfill(status=503, json={"detail": "League refresh disabled in QB smoke"})
            else:
                route.continue_(
                    url=args.api + parsed.path + ("?" + parsed.query if parsed.query else "")
                )

        page.route(re.compile(r"^https?://[^/]+/api/"), intercept)
        page.goto(args.url + "/?section=intelligence&intelligence=qb-passing")
        expect(page).to_have_url(re.compile(r"/intelligence/qb-passing$"))
        expect(
            page.get_by_role("heading", name="QB passing: opportunity and production")
        ).to_be_visible()
        expect(
            page.get_by_role("cell", name="Adaptive reference policy", exact=True)
        ).to_be_visible()
        page.get_by_role("button", name="Show details for reference", exact=True).click()
        expect(page.get_by_text("Mean squared error reduction against", exact=False)).to_have_count(
            0
        )
        page.get_by_role("button", name="Hide details for reference", exact=True).click()
        expect(
            page.get_by_role("cell", name="Conditional boosting · research only", exact=True)
        ).to_have_count(0)
        page.get_by_label("Find a quarterback").fill("Bryce Young")
        expect(page.get_by_role("link", name="Bryce Young", exact=True)).to_be_visible()
        row = page.get_by_role("row").filter(
            has=page.get_by_role("link", name="Bryce Young", exact=True)
        )
        row.get_by_role("button", name=re.compile("Show details")).click()
        expect(page.get_by_text("Evidence about passing production:", exact=False)).to_be_visible()
        page.screenshot(path="/tmp/qb-passing-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/qb-passing-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_label("Passing forecast horizon").select_option("next_game")
        page.get_by_label("Passing evaluation window").select_option("modern")
        # URL-backed checkbox: the router commits the change a tick after the click.
        page.get_by_label("Inspect research challengers").click()
        expect(page.get_by_label("Inspect research challengers")).to_be_checked()
        expect(page).to_have_url(re.compile(r"/intelligence/qb-passing\?q=Bryce\+Young&horizon=next_game&evidence_window=modern&challengers=1$"))
        expect(
            page.get_by_role("cell", name="Conditional boosting · research only", exact=True)
        ).to_be_visible()
        expect(
            page.get_by_role("heading", name="Opportunity and execution are tested separately")
        ).to_be_visible()
        # Back unchecks the challengers view (one history entry per change).
        page.go_back()
        expect(page.get_by_label("Inspect research challengers")).not_to_be_checked()
        expect(
            page.get_by_role("cell", name="Conditional boosting · research only", exact=True)
        ).to_have_count(0)
        page.get_by_label("Find a quarterback").fill("Jaxson Dart")
        expect(page.get_by_role("cell", name="Reported absence", exact=True)).to_be_visible()
        row = page.get_by_role("row").filter(
            has=page.get_by_role("link", name="Jaxson Dart", exact=True)
        )
        row.get_by_role("button", name=re.compile("Show details")).click()
        expect(
            page.get_by_text("The zero estimate assumes this reported absence", exact=False)
        ).to_be_visible()
        page.get_by_label("Find a quarterback").fill("Baker Mayfield")
        page.get_by_role("link", name="Baker Mayfield", exact=True).click()
        expect(
            page.get_by_role("heading", name="QB passing · remaining regular season", exact=True)
        ).to_be_visible()
        page.go_back()
        expect(page).to_have_url(re.compile(r"/intelligence/qb-passing\?q=Baker\+Mayfield&horizon=next_game&evidence_window=modern$"))
        page.goto(args.url + "/?section=research&research=qb-variations")
        expect(page).to_have_url(re.compile(r"/research/qb-experiments$"))
        expect(
            page.get_by_role("heading", name="QB passing model variations", exact=True)
        ).to_be_visible()
        expect(page.get_by_text("72 efficiency variations", exact=False)).to_be_visible()
        expect(page.get_by_role("cell", name="Research only", exact=True).first).to_be_visible()
        page.get_by_label("QB experiment history").select_option("all_history")
        page.get_by_label("Find a variation").fill("policy")
        expect(
            page.get_by_role("cell", name="Selected efficiency policy", exact=True)
        ).to_be_visible()
        page.get_by_label("QB experiment outcome").select_option("league_points")
        expect(page.get_by_role("cell", name="Selected fantasy policy", exact=True)).to_be_visible()
        page.get_by_label("QB experiment horizon").select_option("next4")
        expect(
            page.get_by_label("QB experiment horizon").get_by_role(
                "option", name="Next four calendar weeks", exact=True
            )
        ).to_have_count(1)
        page.screenshot(path="/tmp/qb-variations-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/qb-variations-mobile.png", full_page=True)
        page.wait_for_load_state("networkidle")
        assert not errors, errors
        page.unroute_all(behavior="wait")
        browser.close()
    print("PASS: QB passing desktop/mobile, horizons, research boundary, news constraint, profile")


if __name__ == "__main__":
    main()
