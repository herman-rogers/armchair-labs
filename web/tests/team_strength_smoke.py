"""Team selector, diagnostic sources, depth, legacy links, and responsive layout."""

import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

from engine.api.league_observation_routes import strength
from engine.api.nextgen_routes import league
from engine.espn.sync import LeagueSnapshot


def team_path(team):
    name = unicodedata.normalize("NFKD", team["team_name"])
    slug = re.sub(r"[^a-z0-9]+", "-", "".join(c for c in name if not unicodedata.combining(c)).lower()).strip("-") or "team"
    return f"/league/teams/{team['team_id']}/{slug}"


def main():
    snapshot = LeagueSnapshot.read(Path("data/outputs/league_snapshot.json"))
    service = SimpleNamespace(observations=lambda: (snapshot, False, 0))
    observations = league(service)
    breakdown = strength(service)
    mine = next(t for t in breakdown["teams"] if t["team_id"] == snapshot.my_team_id)
    other = next(
        t
        for t in breakdown["teams"]
        if t["team_id"] != snapshot.my_team_id and t["current"]["complete"]
    )
    expect.set_options(timeout=30000)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def intercept(route):
            path = urlparse(route.request.url).path
            if path == "/api/nextgen/league":
                value = observations
            elif path == "/api/nextgen/league/team-strength":
                value = breakdown
            else:
                return route.continue_()
            route.fulfill(status=200, json=value)

        page.route("**/api/nextgen/league**", intercept)
        origin = "http://127.0.0.1:5173"
        summary = page.get_by_label("Team summary", exact=True)
        page.goto(f"{origin}/league/teams")
        expect(page.get_by_role("heading", name="Team Strength", exact=True)).to_be_visible()
        expect(page.get_by_label("Compare with", exact=True)).to_have_count(0)
        expect(
            page.get_by_role("navigation", name="Main navigation").get_by_role(
                "link", name="Team Strength", exact=True
            )
        ).to_have_attribute("aria-current", "page")
        expect(page.get_by_label("Team", exact=True)).to_have_value("")
        expect(page.get_by_role("region", name="Team strength breakdown", exact=True)).to_have_count(0)
        expect(page).to_have_title("Team Strength · Sweaty Plays")
        page.screenshot(path="/tmp/team-strength-chooser.png", full_page=True)
        page.get_by_label("Team", exact=True).select_option(str(snapshot.my_team_id))
        expect(page).to_have_url(origin + team_path(mine))
        expect(page).to_have_title(f"{mine['team_name']} · Team Strength · Sweaty Plays")
        page.go_back()
        expect(page.get_by_label("Team", exact=True)).to_have_value("")
        expect(summary).to_have_count(0)
        page.go_forward()
        expect(page.get_by_label("Team", exact=True)).to_have_value(str(snapshot.my_team_id))
        for name in (
            "Starting-lineup strength",
            "Scoring concentration",
            "Usable depth",
            "Position breakdown",
            "Results & consistency",
            "Players & roster forecasts",
        ):
            expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
        expect(summary).to_contain_text(f"{mine['results']['ppg']:.1f}")
        expect(
            page.get_by_role("region", name="Starting-lineup strength", exact=True)
        ).to_contain_text(f"{mine['current']['total']:.1f}")
        expect(
            page.get_by_role("region", name="Scoring concentration", exact=True)
        ).to_contain_text(f"{mine['current']['top2_share'] * 100:.1f}%")
        depth = page.get_by_role("region", name="Usable depth", exact=True)
        depth.get_by_text("Replacement details & FLEX moves", exact=True).click()
        te = next(d for d in mine["depth"] if any("→ TE" in m for m in d["moves"]))
        depth.get_by_role("button", name=f"Show details for {te['starter']}", exact=True).click()
        expect(depth.get_by_text(" · ".join(te["moves"]), exact=True)).to_be_visible()
        # Roster searches must never recalculate whole-team strength.
        before = summary.inner_text()
        page.get_by_label("Find league player", exact=True).fill("Stafford")
        expect(summary).to_have_text(before, use_inner_text=True)
        expect(page).to_have_url(re.compile(r"\?q=Stafford$"))
        page.get_by_label("Team", exact=True).select_option(str(other["team_id"]))
        expect(page).to_have_url(origin + team_path(other))
        expect(page).to_have_title(f"{other['team_name']} · Team Strength · Sweaty Plays")
        expect(page.get_by_role("heading", name=other["team_name"], exact=True)).to_be_visible()
        expect(summary).to_contain_text(f"{other['results']['ppg']:.1f}")
        page.reload()
        expect(page.get_by_label("Team", exact=True)).to_have_value(str(other["team_id"]))
        page.go_back()
        expect(page.get_by_label("Team", exact=True)).to_have_value(str(snapshot.my_team_id))
        expect(page.get_by_label("Find league player", exact=True)).to_have_value("Stafford")
        page.go_forward()
        expect(page).to_have_url(origin + team_path(other))
        page.go_back()
        page.get_by_label("Find league player", exact=True).fill("")
        expect(page).to_have_url(origin + team_path(mine))
        # Opponents in scoring history link to their own named team route.
        history_link = page.get_by_role("region", name="Results and consistency", exact=True).get_by_role("link").first
        page.get_by_text("Weekly scoring history", exact=True).click()
        opponent_path = history_link.get_attribute("href")
        assert opponent_path.startswith("/league/teams/") and "?" not in opponent_path
        history_link.click()
        expect(page).to_have_url(origin + opponent_path)
        page.go_back()
        expect(page).to_have_url(origin + team_path(mine))
        adjusted = next(t for t in breakdown["teams"] if t["current"]["adjustments"] and t["current"]["weekly_total"] is not None)
        page.get_by_label("Team", exact=True).select_option(str(adjusted["team_id"]))
        expect(summary).to_contain_text(
            f"{adjusted['results']['ppg']:.1f}"
        )
        expect(summary).to_contain_text(f"{adjusted['current']['weekly_total']:.1f}")
        notice = page.get_by_label("Assumed lineup adjustments", exact=True)
        expect(notice).to_be_visible()
        for move in adjusted["current"]["adjustments"]:
            expect(notice).to_contain_text(move)
        expect(page.locator('.chart-frame')).to_have_count(5)
        page.get_by_label("Team", exact=True).select_option(str(snapshot.my_team_id))
        expect(page.locator('.chart-frame')).to_have_count(5)
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
            page.evaluate('window.scrollTo(0, 0)')
            if width == 1440:
                assert page.locator('#strength-balance').bounding_box()['y'] < 650
                assert abs(page.locator('#strength-balance').bounding_box()['y'] - page.locator('#strength-lineup').bounding_box()['y']) < 2
            footer = page.get_by_role('region', name='Team navigation', exact=True)
            for scroll in ('0', 'document.body.scrollHeight / 2', 'document.body.scrollHeight'):
                page.evaluate(f'window.scrollTo(0, {scroll})')
                box = footer.bounding_box()
                assert 0 <= box['y'] and box['y'] + box['height'] <= 1001
            footer.get_by_label('Switch team', exact=True).select_option(str(other['team_id']))
            expect(page).to_have_url(origin + team_path(other))
            expect(page.get_by_label('Team', exact=True)).to_have_value(str(other['team_id']))
            page.go_back()
            expect(footer.get_by_label('Switch team', exact=True)).to_have_value(str(snapshot.my_team_id))
            page.evaluate('window.scrollTo(0, 0)')
            page.screenshot(path=f"/tmp/team-strength-{width}.png", full_page=True)
        page.get_by_role('button', name='Switch to dark mode').click()
        page.screenshot(path='/tmp/team-strength-dark.png', full_page=True)
        page.get_by_role('button', name='Switch to light mode').click()
        page.get_by_text('Weekly player values', exact=True).click()
        expect(page.get_by_role('table', name='Actual offensive starter contributions', exact=True)).to_be_visible()
        # Main navigation always opens the unselected chooser.
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("navigation", name="Main navigation").get_by_role("link", name="Team Strength", exact=True).click()
        expect(page).to_have_url(origin + "/league/teams")
        expect(page.get_by_label("Team", exact=True)).to_have_value("")
        expect(summary).to_have_count(0)
        # Old bookmarks and missing/outdated slugs resolve by stable team ID.
        for path in (
            f"/league/rosters?team={mine['team_id']}&compare=7",
            "/league/rosters?team=mine",
            f"/league/teams/{mine['team_id']}",
            f"/league/teams/{mine['team_id']}/old-name",
        ):
            page.goto(origin + path)
            expect(page).to_have_url(origin + team_path(mine))
            expect(page).to_have_title(f"{mine['team_name']} · Team Strength · Sweaty Plays")
        page.goto(origin + "/league/rosters?compare=7")
        expect(page).to_have_url(origin + "/league/teams")
        expect(page.get_by_label("Team", exact=True)).to_have_value("")
        page.goto(origin + "/league/teams/99999/unknown")
        expect(page.get_by_role("alert").filter(has_text="Team not found")).to_be_visible()
        expect(summary).to_have_count(0)
        page.get_by_label("Team", exact=True).select_option(str(snapshot.my_team_id))
        page.get_by_role("navigation", name="Team strength sections").get_by_role(
            "link", name="Positions", exact=True
        ).click()
        expect(page).to_have_url(re.compile("#strength-positions$"))
        # Stale strength data cannot silently show against a new league snapshot.
        breakdown["captured_at"] = "2000-01-01T00:00:00Z"
        page.reload()
        expect(page.get_by_text("Team strength is waiting", exact=False)).to_be_visible()
        expect(summary).to_have_count(0)
        expect(
            page.get_by_role("heading", name="Players & roster forecasts", exact=True)
        ).to_be_visible()
        assert not errors, errors
        browser.close()
    print("Team strength browser checks passed: selector, sources, depth, filters and mobile.")


if __name__ == "__main__":
    main()
