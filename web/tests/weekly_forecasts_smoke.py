"""Model points replace provider forecasts; history never reuses current predictions."""
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright, expect
from engine.espn.sync import LeagueSnapshot
from engine.espn.observations import weekly_matchups
from engine.espn.forecast_log import accuracy
from engine.api.nextgen_routes import league
from engine.api.league_observation_routes import forecasts, forecast_evidence


def main():
    snap = LeagueSnapshot.read(Path('data/outputs/league_snapshot.json'))
    service = SimpleNamespace(observations=lambda: (snap, True, 30))
    observations = league(service)
    model = forecasts(service)
    original_model = deepcopy(model)
    report = accuracy(snap, Path('data/outputs/weekly_forecasts'))
    assert model['source'] == 'nextgen_weekly_model'
    evidence = forecast_evidence(service)
    assert all(g['home']['model_projection'] is not None and g['away']['model_projection'] is not None for g in model['matchups'])
    expect.set_options(timeout=30000)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        def intercept(route):
            u=urlparse(route.request.url)
            if u.path.endswith('/forecast-evidence'): value=evidence
            elif u.path.endswith('/forecast-accuracy'): value=report
            elif u.path.endswith('/forecasts'): value=model
            elif u.path.endswith('/matchups'):
                week=int(parse_qs(u.query).get('week',[snap.week])[0])
                value=weekly_matchups(snap,week)
                # A conspicuous provider estimate must never become the displayed forecast.
                for game in value['matchups']:
                    for side in (game['home'],game['away']): side['espn_projection']=99999
                value.update(stale=True,age_seconds=30)
            elif u.path.endswith('/league'): value=observations
            else: return route.abort()
            route.fulfill(status=200,json=value)
        page.route('**/api/nextgen/league**',intercept)
        page.goto('http://127.0.0.1:5173/league/overview')
        section=page.get_by_role('region',name='Weekly matchups',exact=True)
        expect(section.get_by_text('NextGen weekly reference points',exact=False)).to_be_visible()
        game=model['matchups'][0]
        link=section.get_by_role('link',name=f"{game['home']['team_name']} vs {game['away']['team_name']}: View matchup",exact=True)
        expect(link).to_contain_text(f"{game['home']['model_projection']:.1f}")
        expect(link).to_contain_text(f"{game['away']['model_projection']:.1f}")
        expect(link).to_have_attribute('aria-label', re.compile(': View matchup$'))
        expect(section).not_to_contain_text('99999')
        expect(page.get_by_role('heading',name='Season prediction accuracy',exact=True)).to_be_visible()
        expect(page.get_by_text('No verified pregame picks yet',exact=True)).to_be_visible()
        expect(page.get_by_role('heading',name='Weekly model evaluation',exact=True)).to_be_visible()
        expect(page.get_by_text('Weekly reference retained',exact=True)).to_have_count(4)
        page.get_by_text('This season’s reconstructed matchup test',exact=True).click()
        expect(page.get_by_text('Not a record of pregame published picks.',exact=False)).to_be_visible()
        page.get_by_text('Weekly player point forecasts',exact=True).click()
        expect(page.get_by_role('columnheader',name='NextGen weekly points',exact=False).first).to_be_visible()
        for width in (1440,390):
            page.set_viewport_size({'width':width,'height':1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.get_by_label('Matchup week',exact=True).select_option('1')
        expect(section).to_contain_text('Final scores')
        expect(section).not_to_contain_text('99999')
        page.screenshot(path='/tmp/nextgen-weekly-mobile.png',full_page=True)
        page.get_by_label('Matchup week',exact=True).select_option(str(snap.week))
        link.click()
        expect(page).to_have_url(re.compile('/league/matchups/'))
        expect(page.get_by_role('columnheader',name='NextGen weekly points',exact=False).first).to_be_visible()
        expect(page.get_by_role('columnheader',name='ESPN projection',exact=False)).to_have_count(0)
        expect(page.get_by_role('region',name='Matchup detail')).not_to_contain_text('99999')
        # Missing model coverage must not fall back to a provider total or winner.
        model['matchups'][0]['home']['model_projection'] = None
        model['matchups'][0]['home']['espn_projection'] = 99999
        page.goto('http://127.0.0.1:5173/league/overview')
        page.reload()
        expect(link).to_contain_text('No projection yet')
        expect(link.locator('.matchup-team-score strong').first).to_have_text('—')
        expect(link.get_by_label('Projected winner',exact=True)).to_have_count(0)
        # A forecast from another lineup snapshot is withheld across the overview.
        model = deepcopy(original_model)
        model['captured_at'] = '2020-01-01T00:00:00+00:00'
        page.reload()
        expect(link).to_contain_text('No projection yet')
        expect(page.locator('.outlook-kpis').nth(0)).to_contain_text('Projection unavailable')
        expect(page.get_by_text('Weekly player point forecasts',exact=True)).to_have_count(0)
        link.click()
        expect(page.locator('.matchup-scoreboard')).to_contain_text('— NextGen weekly points')
        assert not errors,errors
        page.unroute_all(behavior='wait')
        browser.close()
    print('Weekly forecasts browser checks passed: model totals, player points, historical scores, accuracy exclusions, detail pages and mobile.')


if __name__ == '__main__': main()
