"""Profile URLs, native links, theme styling and dashboard return navigation."""
import argparse
import re
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    args = parser.parse_args()
    expect.set_options(timeout=30000)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.get_by_label('Ranking position').select_option('QB')
        page.get_by_label('Find ranked player').fill('Baker Mayfield')
        expect(page).to_have_url(re.compile(r'/intelligence/rankings\?position=QB&q=Baker\+Mayfield$'))
        name = page.locator('table').get_by_role('link', name='Baker Mayfield', exact=True)
        expect(name).to_be_visible()
        profile = name
        path = name.get_attribute('href')
        assert re.fullmatch(r'/players/[^/]+', path), path
        for theme in ('light', 'dark'):
            page.evaluate('(theme) => document.documentElement.dataset.theme = theme', theme)
            page.mouse.move(0, 0)
            assert name.evaluate('el => !["rgb(0, 0, 238)", "rgb(85, 26, 139)"].includes(getComputedStyle(el).color)')
            profile.focus()
            assert profile.evaluate('el => getComputedStyle(el).outlineStyle') == 'solid'
        name.click()
        expect(page.get_by_role('heading', name='Baker Mayfield', exact=True)).to_be_visible()
        stats = page.locator('.player-profile .player-stats')
        expect(stats.get_by_role('heading', name='Production', exact=True)).to_be_visible()
        expect(stats.locator('table')).to_have_count(0)
        stats.get_by_label('Observation period').select_option('current')
        expect(page).to_have_url(re.compile(re.escape(path) + r'\?period=current$'))
        page.reload()
        for section in ('overview', 'stats', 'history', 'role', 'similar', 'tracking', 'forecasts', 'evidence'):
            expect(page.locator(f'#profile-{section}')).to_be_visible()
        expect(stats.get_by_label('Observation period')).to_have_value('current')
        # Sections are paths that keep the profile's view params.
        sections = page.get_by_role('navigation', name='Player profile sections')
        sections.get_by_role('link', name='Season history', exact=True).click()
        expect(page).to_have_url(re.compile(re.escape(path) + r'/history\?period=current$'))
        expect(sections.get_by_role('link', name='Season history', exact=True)).to_have_attribute('aria-current', 'page')
        page.reload()
        expect(page.locator('#profile-history')).to_be_in_viewport()
        # Back/Forward step through section, period and rankings entries.
        page.go_back()
        expect(page).to_have_url(re.compile(re.escape(path) + r'\?period=current$'))
        page.go_back()
        expect(page).to_have_url(re.compile(re.escape(path) + r'$'))
        expect(stats.get_by_label('Observation period')).to_have_value('prior')
        page.go_back()
        expect(page.get_by_label('Find ranked player')).to_have_value('Baker Mayfield')
        expect(page.get_by_label('Ranking position')).to_have_value('QB')
        page.go_forward()
        page.go_forward()
        expect(stats.get_by_label('Observation period')).to_have_value('current')
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/tmp/player-page-mobile.png', full_page=True)
        # The breadcrumb is a plain link to the dashboard; Back is what restores filters.
        page.get_by_role('link', name='← Dashboard', exact=True).click()
        expect(page).to_have_url(re.compile(r'/intelligence/rankings$'))
        page.goto(args.url + path + '/not-a-section?period=career')
        expect(page).to_have_url(re.compile(re.escape(path) + r'\?period=career$'))
        page.goto(args.url + '/missing-page')
        expect(page.get_by_role('heading', name='Page not found')).to_be_visible()
        page.goto(args.url + '/players/not-a-player')
        expect(page.locator('.player-page .notice')).to_be_visible()
        expect(page.get_by_role('link', name='← Dashboard', exact=True)).to_be_visible()
        assert not errors, errors
        browser.close()
        print('Profile routing checks passed: themed links, keyboard focus, stats, section paths, direct URLs, refresh, Back/Forward, mobile, and missing routes.')


if __name__ == '__main__':
    main()
