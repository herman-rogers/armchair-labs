import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { transform } from 'esbuild'
const source = await readFile(new URL('../src/standingsHistory.ts', import.meta.url), 'utf8')
const { code } = await transform(source, { loader: 'ts', format: 'esm' })
const { standingsHistory } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
const game = (week, outcome, score) => ({ week, outcome, score })
const data = { week: 4, teams: [
  { team_id: 1, schedule: [game(1,'W',100),game(2,'L',80),game(3,'T',90),game(4,'W',999)] },
  { team_id: 2, schedule: [game(1,'L',80),game(2,'W',110),game(3,'T',90),game(4,'L',0)] },
] }
const history = standingsHistory(data)
assert.deepEqual(history.map(s => s.teams.map(t => t.rank)), [[1,2],[2,1],[2,1]])
assert.equal(history.length, 3) // Excludes the current partial week.
const tied = structuredClone(data); tied.teams[1].schedule[1].score = 100
assert.deepEqual(standingsHistory(tied)[1].teams.map(t => t.rank), [1,1])
const missing = structuredClone(data); missing.teams[1].schedule.splice(1,1)
assert.deepEqual(standingsHistory(missing).map(s => s.teams.map(t => t.rank)), [[1,2],[null,null],[null,null]])
const zero = { week: 2, teams: [{team_id:1,schedule:[game(1,'T',0)]},{team_id:2,schedule:[game(1,'T',0)]}] }
assert.deepEqual(standingsHistory(zero)[0].teams.map(t=>t.rank), [1,1])
assert.deepEqual(standingsHistory({...data,week:1}), [])
console.log('Standings history passed: movement, ties, zeros, incomplete weeks, current-week exclusion.')
