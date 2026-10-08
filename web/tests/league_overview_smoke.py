"""League briefing against saved observations and published ranks; never refresh ESPN."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playwright.sync_api import expect, sync_playwright
from engine.api.nextgen_routes import league
from engine.api.ranking_routes import history, rankings
from engine.espn.observations import weekly_matchups
from engine.espn.sync import LeagueSnapshot


def main():
    expect.set_options(timeout=30000)
    snap = LeagueSnapshot.read(Path('data/outputs/league_snapshot.json'))
    data = league(SimpleNamespace(observations=lambda: (snap, True, 3600)))
    ranks = rankings(limit=1000, offset=0)
    rank_history = history()
    while len(ranks['rankings']) < ranks['total']:
        ranks['rankings'].extend(rankings(limit=1000, offset=len(ranks['rankings']))['rankings'])
    by_id = {r['player_id']:r for r in ranks['rankings']}
    free = next(p for p in data['players'] if p['availability']=='free_agent' and by_id.get(p['player_id'],{}).get('overall_rank') is not None)
    draft = next(p for p in data['draft'] if by_id.get(p['player_id'],{}).get('overall_rank') is not None)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        def intercept(route):
            u=urlparse(route.request.url)
            q=parse_qs(u.query)
            if u.path=='/api/nextgen/league': route.fulfill(status=200,json=data)
            elif u.path=='/api/nextgen/league/matchups':
                week=int(q.get('week',[snap.week])[0])
                route.fulfill(status=200,json={**weekly_matchups(snap,week),'stale':True,'age_seconds':3600})
            elif u.path=='/api/nextgen/rankings':
                offset=int(q.get('offset',[0])[0]); limit=int(q.get('limit',[500])[0])
                route.fulfill(status=200,json={**ranks,'rankings':ranks['rankings'][offset:offset+limit]})
            elif u.path=='/api/nextgen/rankings/history': route.fulfill(status=200,json=rank_history)
            else: route.abort()
        page.route(re.compile(r'.*/api/nextgen/(league(?:/[^?]*)?|rankings(?:/history)?)(?:\?.*)?$'),intercept)
        page.goto('http://127.0.0.1:5173/?section=league')
        expect(page).to_have_url(re.compile(r'/league/overview$'))
        tabs=page.get_by_role('navigation',name='Main navigation')
        expect(tabs.get_by_role('link', name='Matchups', exact=True)).to_have_count(0)
        expect(page.get_by_role('heading',name='League outlook',exact=True)).to_be_visible()
        expect(page.get_by_role('columnheader',name='Change',exact=False)).to_be_visible()
        expect(page.get_by_label('Explore your league')).to_have_count(0)
        expect(page.get_by_role('heading',name=f'Week {snap.week} matchups')).to_be_visible()
        expect(page.get_by_role('columnheader',name='Rank',exact=False)).to_be_visible()
        expect(page.get_by_role('columnheader',name='Projected season pace',exact=False)).to_be_visible()
        expect(page.get_by_role('columnheader',name='Points / week',exact=False)).to_be_visible()
        expect(page.get_by_role('columnheader',name='Results captured',exact=False)).to_have_count(0)
        expect(page.get_by_role('columnheader',name='NextGen · average rank',exact=False)).to_have_count(0)
        expect(page.locator('.outlook-matchup')).to_have_count(len(weekly_matchups(snap,snap.week)['matchups']))
        page.set_viewport_size({'width':1920,'height':1000})
        tops=page.locator('.outlook-matchup').evaluate_all('(cards) => cards.map(c => c.getBoundingClientRect().top)')
        assert len(set(tops)) == 1, tops
        page.screenshot(path='/tmp/league-overview-wide.png',full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        team=data['teams'][0]
        page.get_by_role('button',name=f"Show details for {team['team_name']} forecast breakdown",exact=True).click()
        expect(page.get_by_role('heading',name='Weekly scoring history',exact=True)).to_be_visible()
        page.screenshot(path='/tmp/league-overview-desktop.png',full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path='/tmp/league-overview-mobile.png',full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        team_link = page.get_by_role('link',name='View player forecasts',exact=True)
        team_path = team_link.get_attribute('href')
        assert team_path.startswith(f"/league/teams/{team['team_id']}/") and '?' not in team_path
        team_link.click()
        expect(page).to_have_url(re.compile(re.escape(team_path) + '$'))
        expect(tabs.get_by_role('link',name='Team Strength',exact=True)).to_have_attribute('aria-current','page')
        expect(page.get_by_role('heading',name=team['team_name'],exact=True)).to_be_visible()
        expect(page.get_by_role('columnheader',name='NextGen · overall',exact=False)).to_be_visible()
        tabs.get_by_role('link',name='Free agents',exact=True).click()
        expect(page).to_have_url(re.compile(r'/league/free-agents$'))
        expect(page.get_by_role('heading',name='Captured free agents')).to_be_visible()
        page.get_by_label('Find league player').fill(free['player_display_name'])
        row=page.get_by_role('row').filter(has_text=free['player_display_name'])
        expect(row.get_by_role('cell',name=f"#{by_id[free['player_id']]['overall_rank']}",exact=True)).to_be_visible()
        tabs.get_by_role('link',name='Draft recap',exact=True).click()
        expect(page).to_have_url(re.compile(r'/league/draft$'))
        row=page.get_by_role('row').filter(has_text=draft['player_display_name'])
        expect(row.get_by_role('cell',name=f"#{by_id[draft['player_id']]['overall_rank']}",exact=True)).to_be_visible()
        expect(page.get_by_text('Original draft order alongside',exact=False)).to_be_visible()
        # Back steps through the league pages; the free-agent search survives.
        page.go_back()
        expect(page).to_have_url(re.compile(r'/league/free-agents\?q=[^&]+$'))
        expect(page.get_by_label('Find league player')).to_have_value(free['player_display_name'])
        # The search replaced the free-agents entry, so one more step reaches the team.
        page.go_back()
        expect(page).to_have_url(re.compile(re.escape(team_path) + '$'))
        tabs.get_by_role('link',name='Overview',exact=True).click()
        future=min(snap.week+1,snap.regular_season_weeks)
        page.get_by_label('Matchup week', exact=True).select_option(str(future))
        expect(page.get_by_role('heading',name=f'Week {future} matchups')).to_be_visible()
        expect(page).to_have_url(re.compile(rf'/league/overview\?week={future}$'))
        game_link = page.get_by_role('link', name=re.compile(': View matchup$')).first
        game_path = game_link.get_attribute('href')
        game_link.click()
        expect(page).to_have_url(re.compile(re.escape(game_path) + '$'))
        expect(page.get_by_role('region', name='Weekly matchups', exact=True)).to_have_count(0)
        expect(tabs.get_by_role('link', name='Overview', exact=True)).to_have_attribute('aria-current', 'page')
        expect(page.get_by_role('heading', name='Recorded lineups', exact=True)).to_be_visible()
        page.reload()
        expect(page.get_by_role('heading', name='Recorded lineups', exact=True)).to_be_visible()
        page.get_by_role('link', name='Next matchup →', exact=True).click()
        expect(page).not_to_have_url(re.compile(re.escape(game_path) + '$'))
        page.go_back()
        expect(page).to_have_url(re.compile(re.escape(game_path) + '$'))
        expect(page.get_by_text('No lineup captured for this week.',exact=False).first).to_be_visible()
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 844})
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path='/tmp/matchup-detail-mobile.png', full_page=True)
        page.get_by_role('link', name=f'Week {future} matchups', exact=True).click()
        expect(page.get_by_label('Matchup week', exact=True)).to_have_value(str(future))
        page.goto(f'http://127.0.0.1:5173/league/matchups/{future}/999999/888888')
        expect(page.get_by_role('heading', name='Matchup not found', exact=True)).to_be_visible()
        page.get_by_role('link', name=f'Week {future} matchups', exact=True).click()
        expect(page.get_by_label('Matchup week', exact=True)).to_have_value(str(future))
        page.goto('http://127.0.0.1:5173/league/matchups/invalid/1/2')
        expect(page.get_by_role('heading', name='Matchup not found', exact=True)).to_be_visible()
        # Historical games retain their actual lineups on a directly loadable URL.
        historical_week = next(w for w in range(1, snap.week) if any(g['home']['lineup_available'] for g in weekly_matchups(snap,w)['matchups']))
        historical_game = next(g for g in weekly_matchups(snap,historical_week)['matchups'] if g['home']['lineup_available'])
        historical_path = f"/league/matchups/{historical_week}/{historical_game['home']['team_id']}/{historical_game['away']['team_id']}"
        page.goto('http://127.0.0.1:5173' + historical_path)
        expect(page.get_by_role('heading', name='Starters', exact=True).first).to_be_visible()
        expect(page.get_by_text('Current injury tags are not applied', exact=False)).to_be_visible()
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 844})
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.set_viewport_size({'width': 1440, 'height': 1000})
        page.screenshot(path='/tmp/matchup-detail-desktop.png', full_page=True)
        page.goto(f'http://127.0.0.1:5173/league/overview?week={future}')
        expect(page).to_have_url(re.compile(rf'/league/overview\?week={future}$'))
        page.goto(f'http://127.0.0.1:5173/league/matchups?week={future}')
        expect(page).to_have_url(re.compile(rf'/league/overview\?week={future}#matchups$'))
        page.reload()
        expect(page.get_by_role('heading', name=f'Week {future} matchups')).to_be_visible()
        # Week selection survives reload and Back, while the outlook stays current.
        page.get_by_label('Matchup week', exact=True).select_option(str(historical_week))
        expect(page.get_by_text('Final scores · Select a game for lineups',exact=True)).to_be_visible()
        expect(page.get_by_role('columnheader',name=f'Week {snap.week} projected',exact=False)).to_be_visible()
        page.go_back()
        expect(page.get_by_label('Matchup week', exact=True)).to_have_value(str(future))
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 844})
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        menu = page.get_by_role('button', name='Browse pages')
        expect(tabs).to_be_hidden()
        menu.click()
        expect(tabs).to_be_visible()
        tabs.get_by_role('link', name='Transactions', exact=True).click()
        expect(page).to_have_url(re.compile(r'/league/transactions$'))
        expect(tabs).to_be_hidden()
        page.go_back()
        expect(page.get_by_role('heading', name=f'Week {future} matchups')).to_be_visible()
        page.screenshot(path='/tmp/matchups-page-mobile.png', full_page=True)
        assert not errors, errors
        page.unroute_all(behavior='wait')
        browser.close()
    print('League overview browser checks passed: page navigation, dedicated matchups, team drilldown, free-agent/draft global ranks, future weeks, mobile.')


if __name__=='__main__': main()
