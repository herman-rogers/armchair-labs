"""League briefing against saved observations and published ranks; never refresh ESPN."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playwright.sync_api import expect, sync_playwright
from patron.api.nextgen_routes import league
from patron.api.ranking_routes import rankings
from patron.espn.observations import weekly_matchups
from patron.espn.sync import LeagueSnapshot


def main():
    expect.set_options(timeout=30000)
    snap = LeagueSnapshot.read(Path('data/outputs/league_snapshot.json'))
    data = league(SimpleNamespace(observations=lambda: (snap, True, 3600)))
    ranks = rankings(limit=1000, offset=0)
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
            else: route.abort()
        page.route(re.compile(r'.*/api/nextgen/(league(?:/[^?]*)?|rankings)(?:\?.*)?$'),intercept)
        page.goto('http://127.0.0.1:5173/?section=league')
        expect(page).to_have_url(re.compile(r'/league/overview$'))
        tabs=page.get_by_role('navigation',name='League views')
        expect(tabs.get_by_role('link')).to_have_text(['League overview','Rosters','Free agents','Transactions','Draft recap'])
        expect(page.get_by_role('heading',name='Standings & roster forecasts')).to_be_visible()
        expect(page.get_by_role('heading',name=f'Week {snap.week} matchups')).to_be_visible()
        expect(page.get_by_role('columnheader',name='NextGen · team',exact=False)).to_be_visible()
        team=data['teams'][0]
        page.get_by_role('button',name=f"Show details for {team['team_name']} forecast breakdown",exact=True).click()
        expect(page.get_by_role('columnheader',name='Known remaining points',exact=False)).to_be_visible()
        page.screenshot(path='/tmp/league-overview-desktop.png',full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path='/tmp/league-overview-mobile.png',full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        page.get_by_role('link',name='View player forecasts',exact=True).click()
        expect(page).to_have_url(re.compile(rf"/league/rosters\?team={team['team_id']}$"))
        expect(tabs.get_by_role('link',name='Rosters',exact=True)).to_have_attribute('aria-current','page')
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
        # Back steps through the league tabs; the free-agent search survives.
        page.go_back()
        expect(page).to_have_url(re.compile(r'/league/free-agents\?q=[^&]+$'))
        expect(page.get_by_label('Find league player')).to_have_value(free['player_display_name'])
        # The search replaced the free-agents entry, so one more step reaches Rosters.
        page.go_back()
        expect(page).to_have_url(re.compile(rf"/league/rosters\?team={team['team_id']}$"))
        tabs.get_by_role('link',name='League overview',exact=True).click()
        future=min(snap.week+1,snap.regular_season_weeks)
        page.get_by_role('tablist',name='Week',exact=True).get_by_role('tab',name=str(future),exact=True).click()
        expect(page.get_by_role('heading',name=f'Week {future} matchups')).to_be_visible()
        expect(page).to_have_url(re.compile(rf'/league/overview\?week={future}$'))
        page.get_by_role('button', name=re.compile(': Show lineups$')).first.click()
        expect(page.get_by_text('No lineup captured for this week.',exact=False).first).to_be_visible()
        assert not errors, errors
        page.unroute_all(behavior='wait')
        browser.close()
    print('League overview browser checks passed: merged views, team drilldown, free-agent/draft global ranks, future weeks, mobile.')


if __name__=='__main__': main()
