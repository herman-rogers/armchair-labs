"""Published weekly movement, filtered ranks, history, profile links and mobile layout."""
import json
import re
from urllib.request import urlopen
from playwright.sync_api import expect, sync_playwright


def main():
    base = 'http://127.0.0.1:5173'
    history = json.load(urlopen('http://127.0.0.1:8000/api/nextgen/rankings/history'))
    ranks = json.load(urlopen('http://127.0.0.1:8000/api/nextgen/rankings?limit=1000'))
    previous = history['snapshots'][-1]
    by_id = {r['player_id']: r for r in previous['rankings']}
    row = next(r for r in ranks['rankings'] if r['overall_rank'] is not None and by_id.get(r['player_id'], {}).get('overall_rank') is not None and r['overall_rank'] != by_id[r['player_id']]['overall_rank'])
    change = by_id[row['player_id']]['overall_rank'] - row['overall_rank']
    movement = f"{'Up' if change > 0 else 'Down'} {abs(change)} places"
    expect.set_options(timeout=30000)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('**/api/nextgen/league**', lambda route: route.fulfill(status=503, json={'detail':'Live ESPN excluded'}))
        page.goto(base + '/intelligence/rankings')
        page.get_by_label('Find ranked player').fill(row['player_display_name'])
        expect(page.get_by_label(movement, exact=True)).to_be_visible()
        page.get_by_role('button', name=f"Show details for {row['player_display_name']}", exact=True).click()
        expect(page.get_by_role('heading', name='Weekly rank history', exact=True)).to_be_visible()
        history_table = page.locator('.rank-history')
        expect(history_table.get_by_role('row')).to_have_count(len(history['snapshots'])+2)
        page.get_by_label('Ranking position').select_option(row['position'])
        expect(page.get_by_label(movement, exact=True).first).to_be_visible()
        for width in (1440,390):
            page.set_viewport_size({'width':width,'height':1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/tmp/rank-movement-mobile.png', full_page=True)
        page.get_by_role('link', name=row['player_display_name'], exact=True).click()
        expect(page).to_have_url(re.compile(r'/players/'))
        expect(page.get_by_role('heading', name='Weekly rank history', exact=True)).to_be_visible()
        expect(page.get_by_label(movement, exact=True)).to_be_visible()
        page.goto(base + '/intelligence/rankings?horizon=next4')
        expect(page.get_by_role('columnheader', name=re.compile('Change · W'))).to_be_visible()
        assert not errors, errors
        browser.close()
    print('Rank movement browser checks passed: published comparison, filters, weekly history, profiles, horizons, mobile.')


if __name__ == '__main__':
    main()
