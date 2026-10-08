"""Shared breadcrumbs across every page family, including failed data requests."""
import argparse
from urllib.parse import urlparse
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    args = parser.parse_args()
    pages = {
        '/intelligence/rankings': 'Player rankings',
        '/intelligence/teams': 'Team analysis',
        '/intelligence/qb-passing': 'QB passing',
        '/intelligence/rookies': 'Rookies',
        '/league/overview': 'Overview',
        '/league/teams': 'Team Strength',
        '/league/free-agents': 'Free agents',
        '/league/transactions': 'Transactions',
        '/league/draft': 'Draft recap',
        '/research/evidence': 'Model evidence',
        '/research/qb-experiments': 'QB experiments',
        '/research/forecasts': 'Reference forecasts',
        '/research/archive': 'Research archive',
        '/league/teams/999999/missing-team': 'Team details',
        '/league/matchups/4/999999/888888': 'Matchup details',
        '/players/missing-player': 'Player profile',
        '/college/missing-player': 'College profile',
        '/players/missing-player/stats?period=career': 'Player stats',
        '/college/missing-player/history?season=2024': 'Season history',
        '/missing-page': 'Page not found',
    }
    for workspace, views in {
        'intelligence': {'players': 'Players', 'league-impact': 'League impact', 'research': 'Research'},
        'league': {'overview': 'Overview', 'matchups': 'Matchups', 'players': 'Players', 'wire': 'Waiver Wire', 'draft': 'Draft recap'},
    }.items():
        for view, label in views.items():
            pages[f'/research/archive/{workspace}/{view}'] = label
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        # The shell must remain navigable even when no page data can be loaded.
        page.route('**/api/**', lambda route: route.fulfill(status=503, content_type='application/json', body='{"detail":"Unavailable"}') if urlparse(route.request.url).path.startswith('/api/') else route.continue_())
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 900})
            for path, label in pages.items():
                page.goto(args.url + path)
                trail = page.get_by_role('navigation', name='Breadcrumb', exact=True)
                expect(trail).to_have_count(1)
                expect(trail.locator('[aria-current="page"]')).to_have_text(label)
                expect(trail.get_by_role('link', name='Home', exact=True)).to_have_count(0)
                if path != '/missing-page':
                    section = trail.locator('li').first
                    sidebar = page.locator('#page-navigation section').filter(has=page.locator('h2', has_text=section.inner_text()))
                    first_link = sidebar.locator('a').first
                    expect(section.get_by_role('link')).to_have_attribute('href', first_link.get_attribute('href'))
                assert trail.evaluate('el => el.scrollWidth <= el.clientWidth'), path
                expect(trail).to_be_in_viewport()
        page.goto(args.url + '/league/matchups/4/999999/888888')
        trail.get_by_role('link', name='Week 4 matchups').click()
        expect(page).to_have_url(args.url + '/league/overview?week=4#matchups')
        page.goto(args.url + '/players/missing-player/stats?period=career')
        trail.get_by_role('link', name='Player profile', exact=True).click()
        expect(page).to_have_url(args.url + '/players/missing-player?period=career')
        assert not errors, errors
        browser.close()
    print(f'Breadcrumbs passed for {len(pages)} routes at desktop and mobile widths, error states and parent links.')


if __name__ == '__main__':
    main()
