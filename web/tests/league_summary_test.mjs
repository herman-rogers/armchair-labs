import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { transform } from 'esbuild'
const source = await readFile(new URL('../src/leagueSummary.ts', import.meta.url), 'utf8')
const { code } = await transform(source, {loader:'ts',format:'esm'})
const {leagueSummary, recentTransactions, forecastFor} = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
const data = {season:2026, week:3, teams:[
  {team_id:1, schedule:[{week:1,opponent_team_id:2,score:0,outcome:'L'},{week:4,opponent_team_id:2,score:999,outcome:'U'}]},
  {team_id:2, schedule:[{week:1,opponent_team_id:1,score:10,outcome:'W'}]},
  {team_id:3, schedule:[]},
], players:[
  {player_id:'a',owner_team_id:1,position:'QB',lineup_slot:'QB'},
  {player_id:'b',owner_team_id:1,position:'RB',lineup_slot:'BE'},
  {player_id:'k',owner_team_id:1,position:'K',lineup_slot:'K'},
  {player_id:'c',owner_team_id:2,position:'WR',lineup_slot:'WR'},
  {player_id:'d',owner_team_id:3,position:'TE',lineup_slot:'TE'},
  {player_id:'unknown',owner_team_id:3,position:'RB',lineup_slot:'IR'},
],transactions:[{date:null},{date:'2026-09-22T00:00:00Z'},{date:'2026-09-24T00:00:00Z'}]}
const rankings={report:{season:2026},rankings:[
  {player_id:'a',prediction:200,evidence_status:'validated_forecast',overall_rank:2},
  {player_id:'b',prediction:0,evidence_status:'reference',constraint:'Reported absence',overall_rank:null},
  {player_id:'c',prediction:200,evidence_status:'validated_forecast',overall_rank:2},
  {player_id:'d',prediction:80,evidence_status:'reference',overall_rank:90},
]}
const teams=leagueSummary(data,rankings)
assert.deepEqual(teams.map(t=>t.nextgen_team_rank),[1,1,null])
assert.equal(teams[0].forecast_points,200)
assert.equal(teams[0].forecast_count,2) // A genuine constrained zero is covered.
assert.deepEqual(teams.map(t=>t.average_overall_rank),[2,2,90]) // Null ranks and missing forecasts are excluded.
assert.deepEqual(teams.map(t=>t.ranked_count),[1,1,1])
const withBenchRank = structuredClone(rankings)
withBenchRank.rankings[1].overall_rank = 101
assert.equal(leagueSummary(data,withBenchRank)[0].average_overall_rank,51.5) // Includes bench; divides by ranked players, not roster size.
const withReserve = structuredClone(data)
withReserve.players[1].lineup_slot = 'IR'
assert.equal(leagueSummary(withReserve,withBenchRank)[0].average_overall_rank,51.5)
assert.deepEqual(leagueSummary({...data,players:[]},rankings).map(t=>t.average_overall_rank),[null,null,null])
assert.equal(teams[0].unmodeled_count,1)
assert.equal(teams[0].reference_count,1)
assert.equal(teams[0].constrained_count,1)
assert.equal(teams[0].points_for,0) // Actual zero, never future schedule placeholder.
assert.equal(teams[0].points_against,10)
assert.equal(teams[2].forecast_points,null)
assert.equal(teams[2].known_points,80)
assert.equal(teams[2].points_for,null)
assert.equal(forecastFor('d',rankings).overall_rank,90) // Ownership never renumbers players.
assert.deepEqual(leagueSummary(data).map(t=>t.nextgen_team_rank),[null,null,null])
assert.deepEqual(leagueSummary(data).map(t=>t.average_overall_rank),[null,null,null])
assert.deepEqual(leagueSummary(data,{...rankings,report:{season:2025}}).map(t=>t.average_overall_rank),[null,null,null])
assert.deepEqual(leagueSummary(data,{...rankings,report:{season:2025}}).map(t=>t.forecast_points),[null,null,null])
assert.equal(recentTransactions(data)[0].date,'2026-09-24T00:00:00Z')
const partial = structuredClone(data); partial.teams[1].schedule=[]
assert.equal(leagueSummary(partial,rankings)[0].points_against,null)
console.log('League summary checks passed: coverage, ties, constrained zeros, global ranks, missing scores, season mismatch, chronology.')
