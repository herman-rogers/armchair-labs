import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { QueryClient } from '@tanstack/react-query'

async function moduleAt(path) {
  const result = await build({ entryPoints: [new URL(path, import.meta.url).pathname], bundle: true, write: false, format: 'esm', platform: 'node' })
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`)
}
const { allPages } = await moduleAt('../src/api/pagination.ts')
const { rankingsQuery, leagueQuery, directoryQuery, profileQuery, filterDirectory, teamAnalysisQuery } = await moduleAt('../src/api/queries/index.ts')

assert.equal(teamAnalysisQuery(new URLSearchParams('b=2&a=1')).queryKey[2], 'a=1&b=2')
const directory = { report: { season: 2026 }, total: 3, players: [
  { player_id: 'a', player_display_name: 'A [literal]', position: 'WR', rookie_season: 2026, current_candidate: true, college_linked: true },
  { player_id: 'b', player_display_name: 'Beta', position: 'WR', rookie_season: null, current_candidate: true },
  { player_id: 'c', player_display_name: 'Gamma', position: 'QB', rookie_season: 2020, current_candidate: false },
] }
assert.deepEqual(filterDirectory(directory, { scope: 'current', position: 'ALL', population: 'rookie', search: '[' }).players.map(p => p.player_id), ['a'])
assert.equal(filterDirectory(directory, { scope: 'current', position: 'ALL', population: 'returner', search: '' }).total, 0)
assert.equal(directory.total, 3)

const ids = Array.from({length: 1501}, (_, i) => i)
const offsets = []
const complete = await allPages(new URLSearchParams('search=a'), async p => {
  offsets.push(Number(p.get('offset')))
  return {total: ids.length, rows: ids.slice(Number(p.get('offset')), Number(p.get('offset')) + Number(p.get('limit')))}
}, 'rows')
assert.deepEqual(complete.rows, ids)
assert.deepEqual(offsets, [0, 500, 1000, 1500])
await assert.rejects(allPages(new URLSearchParams(), async p => ({total: p.get('offset') === '0' ? 501 : 502, rows: Array(500).fill(0)}), 'rows'))

const requests = []
let aborted = false
const originalFetch = globalThis.fetch
globalThis.fetch = async (url, options) => {
  requests.push({url, options})
  if (url.includes('/profiles/player')) return new Promise((_, reject) => {
    options.signal.addEventListener('abort', () => { aborted = true; reject(new DOMException('Aborted', 'AbortError')) }, {once:true})
  })
  return new Response(JSON.stringify(url.includes('/rankings') ? {total:1,rankings:[{player_id:'a'}]} : {players:[]}), {status:200})
}
try {
  const client = new QueryClient()
  await Promise.all([client.fetchQuery(rankingsQuery('rest_of_season', 'release-a')), client.fetchQuery(rankingsQuery('rest_of_season', 'release-a'))])
  assert.equal(requests.length, 1)
  await client.fetchQuery(rankingsQuery('rest_of_season', 'release-a'))
  assert.equal(requests.length, 1)
  await client.fetchQuery(rankingsQuery('rest_of_season', 'release-b'))
  assert.equal(requests.length, 2)
  assert.equal(requests[1].options.headers['X-Data-Catalog'], 'release-b')
  assert.ok(!requests[0].url.includes('search='))
  const pending = client.fetchQuery(profileQuery(new URLSearchParams('player_id=a'), 'release-a')).catch(() => {})
  await client.cancelQueries({queryKey: ['player-profile']})
  await pending
  assert.equal(aborted, true)
  client.setQueryData(leagueQuery('release-a').queryKey, {players:[]})
  client.setQueryData(directoryQuery('release-a').queryKey, directory)
  await client.invalidateQueries({queryKey: ['league-observations']})
  assert.equal(client.getQueryState(leagueQuery('release-a').queryKey).isInvalidated, true)
  assert.equal(client.getQueryState(directoryQuery('release-a').queryKey).isInvalidated, false)
  const analysis = teamAnalysisQuery(new URLSearchParams('team=LA'), 'release-a')
  const before = requests.length
  await client.fetchQuery(analysis)
  assert.equal(requests.length, before + 1)
  assert.equal(requests.at(-1).url, '/api/nextgen/team-analysis?team=LA')
  await client.fetchQuery(analysis)
  assert.equal(requests.length, before + 1)
  await client.invalidateQueries({queryKey: analysis.queryKey})
  await client.fetchQuery(analysis)
  assert.equal(requests.length, before + 2)
  const previous = {report: {table_version: 'one'}}
  assert.equal(analysis.placeholderData(previous, {queryKey: ['team-analysis', 'release-b', 'team=LA']}), undefined)
  assert.equal(analysis.placeholderData(previous, {queryKey: ['team-analysis', 'release-a', 'team=BUF']}), undefined)
  assert.equal(analysis.placeholderData(previous, {queryKey: ['team-analysis', 'release-a', 'players=a&team=LA']}), previous)
  client.clear()
} finally { globalThis.fetch = originalFetch }
console.log('Query sharing, release isolation, local search, pagination, cancellation, and league invalidation passed.')
