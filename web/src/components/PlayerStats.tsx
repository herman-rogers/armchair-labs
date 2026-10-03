import type { Catalog } from '../api/nextgen'
import { useCatalog } from '../api/queries/hooks'
import { measurementsQuery } from '../api/queries'
import { useUrlState } from '../navigation'
import { useQuery } from '@tanstack/react-query'
import { useDataRelease } from '../dataRelease'
import { fixed, percent } from '../format'

const groups: Record<string, string[]> = {
  Production: ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'league_points'],
  Opportunity: ['attempts', 'carries', 'targets', 'receptions', 'snap_share'],
  Efficiency: ['passing_yards_per_attempt', 'rushing_yards_per_carry', 'receiving_yards_per_target'],
  Consistency: ['mean', 'median', 'std', 'cv', 'top2_positive_share', 'mean_without_top2', 'between_season_std'],
}
const periods = [
  ['prior', 'Previous completed season'],
  ['recent3', 'Previous three seasons'],
  ['career', 'Completed NFL career'],
  ['current', 'Current season · partial'],
]
const format = (value: unknown, key: string) =>
  ['snap_share', 'top2_positive_share'].includes(key) ? percent(value, 1) : fixed(value)

/** Measurement catalog entries keyed by stat name (`measurement:targets` → `targets`). */
const measurementDefinitions = (catalog?: Catalog) => new Map(catalog?.entries.filter(entry => entry.kind === 'measurement')
  .map(entry => [entry.id.replace('measurement:', ''), entry]))

export function PlayerStats({ playerId, name }: { playerId: string; name: string }) {
  const { token } = useDataRelease()
  const [period, setPeriod] = useUrlState('period', 'prior')
  const catalog = useCatalog()
  const query = useQuery(measurementsQuery(period, token))
  const player = query.data?.players.find(row => row.player_id === playerId)
  const definitions = measurementDefinitions(catalog.data)

  return <section className="player-stats" aria-label={`${name} player stats`}>
    <div className="profile-heading">
      <div><span className="eyebrow">Recorded performance</span><h3>Player stats</h3></div>
      <label>Observation period<select value={period} onChange={event => setPeriod(event.target.value)}>
        {periods.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select></label>
    </div>
    {(query.isLoading || catalog.isLoading) && <p role="status">Loading player stats…</p>}
    {(query.isError || catalog.isError) && <p role="alert">Player stats unavailable: {query.error?.message ?? catalog.error?.message}</p>}
    {query.isSuccess && !player && <p>No recorded stats for this player in the selected period.</p>}
    {player && catalog.data && <>
      <p className="legend">Latest published measurements · {player.first_season ?? 'Unknown'}–{player.last_season ?? 'unknown'}
        {period === 'current' && player.through_week != null && ` · through Week ${player.through_week}`}
        {' · '}{player.observed_weeks ?? 'Unknown'} observed weeks
        {player.coverage && ` · ${String(player.coverage).replaceAll('_', ' ')}`}
        {' · '}— means unavailable, not zero.</p>
      <div className="player-stat-groups">{Object.entries(groups).map(([group, keys]) => <section key={group} aria-label={group}>
        <h4>{group}</h4>
        <dl>{keys.filter(key => definitions.has(key)).map(key => {
          const definition = definitions.get(key)!
          return <div key={key}><dt title={definition.definition ?? undefined}>{definition.label}</dt><dd>{format(player[key], key)}</dd></div>
        })}</dl>
      </section>)}</div>
      <p className="legend">Consistency describes observed scoring variation, not medical risk. Compare periods and sample sizes together.</p>

    </>}
  </section>
}

export function PlayerStatDefinitions() {
  const catalog = useCatalog()
  const definitions = measurementDefinitions(catalog.data)
  if (catalog.isError) return <p role="alert">Stat definitions unavailable: {catalog.error.message}</p>
  if (!catalog.data) return <p role="status">Loading stat definitions…</p>
  return <section className="player-stat-definitions"><h4>Player stat definitions</h4>
    <dl className="profile-definitions">{[...new Set(Object.values(groups).flat())].filter(key => definitions.has(key)).map(key => {
      const definition = definitions.get(key)!
      return <div key={key}><dt>{definition.label}</dt><dd>{definition.definition ?? definition.reason}</dd></div>
    })}</dl>
  </section>
}
