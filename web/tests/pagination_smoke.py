"""Regression: full-set sort/filter before display pagination, with sort and page in the URL."""
import re
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import expect, sync_playwright


def main():
    expect.set_options(timeout=30000)
    rows = [dict(player_id=f"p{i:04}", player_display_name=f"Player {i:04}", position="WR" if i % 2 else "RB",
                 population="rookie" if i % 2 else "returner", period="prior", ecr_overall=620-i, ecr_position=620-i,
                 ecr_overall_date="2026-08-28", ecr_position_date="2026-08-28", league_points=float(i),
                 observed_weeks=17, draft_pick=620-i, college_linked=True, nfl_seasons=i, rookie_season=2000+i%20)
            for i in range(620)]
    rows[0]["draft_pick"] = None
    requests = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        def intercept(route):
            u = urlparse(route.request.url)
            q = parse_qs(u.query)
            search = q.get("search", [""])[0].lower()
            position = q.get("position", ["ALL"])[0]
            matched = [r for r in rows if search in r["player_display_name"].lower()
                       and (position == "ALL" or r["position"] == position)]
            offset, limit = int(q.get("offset", [0])[0]), int(q.get("limit", [100])[0])
            requests.append((u.path, offset, search))
            route.fulfill(status=200, json=dict(total=len(matched), players=matched[offset:offset+limit], season=2026,
                through_week=2, report={"season": 2026, "through_week": 2}))
        page.route(re.compile(r'.*/api/nextgen/rookies(\?.*)?$'), intercept)
        page.goto('http://127.0.0.1:5173/?section=intelligence&intelligence=rookies')
        expect(page).to_have_url(re.compile(r'/intelligence/rookies$'))
        rookies = page.get_by_role('region', name='Rookie analysis')
        table = rookies.get_by_role('table')
        expect(table.locator('tbody > tr')).to_have_count(100)
        assert ('/api/nextgen/rookies', 0, '') in requests
        # Default sort (draft pick ascending) covers the full set; the unknown pick sorts last.
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0619')
        pick = rookies.get_by_role('columnheader', name='NFL draft pick', exact=False).get_by_role('button')
        pick.click()
        expect(page).to_have_url(re.compile(r'/intelligence/rookies\?sort=-draft_pick$'))
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0001')
        page.get_by_role('button', name='Next rookies', exact=True).click()
        expect(page).to_have_url(re.compile(r'/intelligence/rookies\?sort=-draft_pick&page=2$'))
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0101')
        # Back steps through page and sort one entry at a time; a reload keeps the view.
        page.reload()
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0101')
        page.go_back()
        expect(page).to_have_url(re.compile(r'/intelligence/rookies\?sort=-draft_pick$'))
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0001')
        page.go_back()
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0619')
        page.go_forward()
        page.go_forward()
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0101')
        # Sorting from a later page returns to the first page in the same navigation.
        rookies.get_by_role('columnheader', name='Player', exact=False).first.get_by_role('button').click()
        expect(page).to_have_url(re.compile(r'/intelligence/rookies\?sort=player_display_name$'))
        expect(table.locator('tbody > tr').first).to_contain_text('Player 0000')
        expect(page.get_by_role('button', name='Previous rookies', exact=True)).to_be_disabled()
        page.get_by_role('button', name='Next rookies', exact=True).click()
        page.get_by_label('Search players', exact=True).fill('Player 0001')
        expect(page).to_have_url(re.compile(r'q=Player\+0001'))
        expect(page).not_to_have_url(re.compile(r'page='))
        expect(table.locator('tbody > tr')).to_have_count(1)
        expect(table).to_contain_text('Player 0001')
        expect(rookies).to_contain_text('1–1 of 1')
        page.get_by_role('button', name='Clear filters', exact=True).click()
        expect(table.locator('tbody > tr')).to_have_count(100)
        page.locator('label', has_text=re.compile(r'^Position')).locator('select').select_option('WR')
        expect(rookies).to_contain_text('of 310')
        expect(page).to_have_url(re.compile(r'position=WR'))
        page.get_by_role('button', name='Clear filters', exact=True).click()
        expect(page.get_by_label('Player view', exact=True)).to_have_count(0)
        assert not errors, errors
        browser.close()
    print('Full-set sorting, URL sort/page history, later-page filtering, and rookie pagination checks passed')


if __name__ == '__main__':
    main()
