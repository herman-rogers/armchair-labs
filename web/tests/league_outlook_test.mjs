import assert from 'node:assert/strict'
import { build } from 'esbuild'
const { outputFiles } = await build({entryPoints:[new URL('../src/leagueOutlook.ts',import.meta.url).pathname],bundle:true,write:false,format:'esm',platform:'node'})
const { leagueOutlook } = await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString('base64')}`)
const data = {version:'current',captured_at:'2026-10-02T20:00:00Z',season:2026,week:4,regular_season_weeks:14,players:[],teams:[],transactions:[]}
for(let id=1;id<=4;id++) {
  data.teams.push({team_id:id,team_name:`Team ${id}`,schedule:[{week:1,score:100,outcome:'L'},{week:2,score:120,outcome:'W'},{week:3,score:140,outcome:'T'},{week:4,score:999,outcome:'W'}]})
  data.players.push({player_id:`p${id}`,position:'QB',owner_team_id:id,lineup_slot:'QB'})
}
const rankings = {version:'current',horizon:'rest_of_season',report:{season:2026,through_week:3},rankings:[400,100,300,200].map((prediction,i)=>({player_id:`p${i+1}`,prediction,overall_rank:i+1}))}
const history = {version:'current',season:2026,horizon:'rest_of_season',snapshots:[{through_week:2,rankings:[400,300,200,100].map((prediction,i)=>({player_id:`p${i+1}`,prediction}))}]}
const matchups = {source:'nextgen_weekly_reference',version:'current',captured_at:data.captured_at,through_week:3,season:2026,current_week:4,requested_week:4,matchups:[{home:{team_id:2,model_projection:131.73,espn_projection:99999},away:{team_id:3,model_projection:132.72,espn_projection:1}}]}
const view=leagueOutlook(data,rankings,history,matchups), mine=view.teams[1]
assert.equal(mine.nextgen_team_rank,4)
assert.equal(mine.previous_forecast_rank,2)
assert.equal(mine.forecast_change,-2) // The user's regression: #2 -> #4 is DOWN.
assert.equal(view.games[0].favorite.team_id,3)
assert.ok(Math.abs(view.games[0].margin + .99)<1e-9)
assert.equal(mine.weekly_projection,131.73)
assert.equal(mine.ppg,120)
assert.equal(mine.ppg_change,10)
assert.equal(mine.season_pace,1680)
assert.equal(mine.season_pace_change,140)
assert.equal(mine.completed_points,360)
assert.equal(mine.record,'1–1–1')
// Do not present old snapshots, mismatched releases or a different season as weekly movement.
for (const h of [{...history,version:'other'},{...history,season:2025},{...history,horizon:'next4'},{...history,snapshots:[{...history.snapshots[0],through_week:1}]},{...history,snapshots:[{...history.snapshots[0],through_week:3}]}]) {
 assert.equal(leagueOutlook(data,rankings,h).teams[1].previous_forecast_rank,null)
}
const incomplete=structuredClone(history);incomplete.snapshots[0].rankings.pop()
assert.ok(leagueOutlook(data,rankings,incomplete).teams.every(t=>t.previous_forecast_rank==null))
const missing=structuredClone(data);missing.teams[1].schedule.splice(1,1)
assert.equal(leagueOutlook(missing,rankings).teams[1].season_pace,null)
const zero=structuredClone(data);zero.teams[1].schedule.forEach(g=>g.score=0)
assert.equal(leagueOutlook(zero,rankings).teams[1].season_pace,0)
assert.equal(leagueOutlook({...data,week:1},rankings).teams[1].ppg,null)
assert.equal(leagueOutlook({...data,week:2},rankings).teams[1].ppg_change,null)
const tied=structuredClone(matchups);tied.matchups[0].away.model_projection=131.73
assert.equal(leagueOutlook(data,rankings,history,tied).games[0].favorite,null)
assert.equal(leagueOutlook(data,rankings,history,tied).games[0].margin,0)
tied.matchups[0].away.model_projection=null
assert.equal(leagueOutlook(data,rankings,history,tied).games[0].margin,null)
assert.equal(leagueOutlook(data,rankings,history,{...matchups,season:2025}).games.length,0)
assert.equal(leagueOutlook(data,rankings,history,{...matchups,requested_week:3}).games.length,0)
for (const patch of [{source:'espn_observations'}, {version:'other'}, {current_week:3}, {through_week:2}, {captured_at:'old-lineup'}]) {
 assert.equal(leagueOutlook(data,rankings,history,{...matchups,...patch}).games.length,0)
}
// Weekly forecasts do not depend on the separate remaining-season ranking request.
assert.equal(leagueOutlook(data,undefined,undefined,matchups).teams[1].weekly_projection,131.73)
const uncovered=structuredClone(matchups);uncovered.matchups[0].home.model_projection=null
assert.equal(leagueOutlook(data,rankings,history,uncovered).teams[1].weekly_projection,null)
assert.equal(leagueOutlook(data,rankings,history,uncovered).games[0].favorite,null)
assert.ok(leagueOutlook(data,{...rankings,report:{season:2025}},history).teams.every(t=>t.nextgen_team_rank==null))
// In-season constraints with a real zero remain covered; K/DST never inflate skill-roster rank.
const constrained=structuredClone(data);constrained.players.push({player_id:'out',position:'RB',owner_team_id:2,lineup_slot:'IR'},{player_id:'k',position:'K',owner_team_id:2,lineup_slot:'K'})
const rr={...rankings,rankings:[...rankings.rankings,{player_id:'out',prediction:0,overall_rank:null}]}
const hh={...history,snapshots:[{...history.snapshots[0],rankings:[...history.snapshots[0].rankings,{player_id:'out',prediction:0}]}]}
assert.equal(leagueOutlook(constrained,rr,hh).teams[1].forecast_change,-2)
console.log('League outlook passed: forecast movement, compatible history, missing data, scoring pace, partial weeks, ties and matchup sources.')
