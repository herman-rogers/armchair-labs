"""Historical correlation risk: live local data with pinned browser responses."""
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

from engine.api.league_observation_routes import strength
from engine.api.nextgen_routes import league
from engine.config.settings import get_settings
from engine.espn.sync import LeagueSnapshot


def main():
    snapshot = LeagueSnapshot.read(get_settings().outputs_dir / 'league_snapshot.json')
    service = SimpleNamespace(observations=lambda: (snapshot, False, 0))
    observations = league(service)
    breakdown = strength(service)
    mine = next(t for t in breakdown['teams'] if t['team_name'] == 'Just Say No')
    no_connections = next(t for t in breakdown['teams'] if not t['correlation_risk'].get('connections'))
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))

        def intercept(route):
            path = urlparse(route.request.url).path
            if path == '/api/nextgen/league':
                return route.fulfill(json=observations)
            if path == '/api/nextgen/league/team-strength':
                return route.fulfill(json=breakdown)
            return route.continue_()

        page.route('**/api/nextgen/league**', intercept)
        page.goto(f'http://127.0.0.1:5173/league/teams/{mine["team_id"]}/just-say-no')
        panel = page.get_by_role('region', name='Historical correlation risk', exact=True)
        expect(panel).to_be_visible(timeout=30000)
        risk = mine['correlation_risk']
        expect(panel).to_contain_text(f'{risk["observed"]["team_variance_share_pct"]:+.1f}%')
        for connection in risk['connections']:
            expect(panel).to_contain_text(f'{connection["historical"]["n"]} shared starts')
        panel.get_by_text('Season breakdown & uncertainty', exact=True).first.click()
        expect(panel.locator('tbody').first).to_contain_text('2025')
        panel.screenshot(path='/tmp/team-correlation-risk-desktop.png')
        for width in (390, 760):
            page.set_viewport_size({'width': width, 'height': 900})
            expect(panel).to_be_visible()
            if page.evaluate('document.documentElement.scrollWidth > innerWidth'):
                print(page.evaluate('Array.from(document.querySelectorAll("body *")).filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>innerWidth+1}).slice(-20).map(e=>({tag:e.tagName,cl:e.className,w:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right,text:e.textContent.slice(0,70)}))'))
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        panel.screenshot(path='/tmp/team-correlation-risk-mobile.png')
        page.get_by_label('Team', exact=True).select_option(str(no_connections['team_id']))
        expect(page.get_by_role('region', name='Historical correlation risk')).to_contain_text('No same-NFL-team')
        assert not errors, errors
        browser.close()
    print('Correlation panel: values, annual breakdown, responsive layout, empty state passed.')


if __name__ == '__main__':
    main()
