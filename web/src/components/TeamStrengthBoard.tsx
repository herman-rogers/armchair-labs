import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import type { LeagueObservations } from '../api/league'
import type { PowerRankingsResponse, PowerTeam } from '../api/powerRankings'
import { powerRankingsQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { espnStandings, weekRange } from '../espnStandings'
import { fixed, rank } from '../format'
import { teamPath, useUrlState } from '../navigation'
import { QueryError, SelectField } from './Controls'
import { SourceMetrics, SourceTag } from './StatCards'

const pct = (v: number | null | undefined) => v == null ? '—' : `${fixed(v * 100)}%`
const record = (w: number, l: number, t: number) => `${w}–${l}${t ? `–${t}` : ''}`

const SORTS = { model: 'Model strength', playoffs: 'Playoff odds', espn: 'ESPN standings' } as const
type Sort = keyof typeof SORTS

export function TeamStrengthBoard({ data, campaignId }: { data: LeagueObservations; campaignId: string }) {
  const { token } = useDataRelease()
  const query = useQuery(powerRankingsQuery(campaignId, token))
  const [sortParam, setSort] = useUrlState('rank', 'model', { replace: true })
  const power = query.data?.status === 'ok' && query.data.season === data.season && query.data.week === data.week ? query.data : undefined
  const sort: Sort = !power ? 'espn' : sortParam in SORTS ? sortParam as Sort : 'model'
  const espn = espnStandings(data)
  const model = new Map(power?.teams.map(t => [t.team_id, t]))
  const odds = (id: number) => model.get(id)?.playoff_odds ?? model.get(id)?.projected_wins ?? 0
  const position = (id: number) => sort === 'espn' ? espn.get(id)?.standing
    : sort === 'playoffs' ? 1 + data.teams.filter(o => odds(o.team_id) > odds(id)).length
    : model.get(id)?.rating_rank
  const teams = [...data.teams].sort((a, b) => (position(a.team_id) ?? Infinity) - (position(b.team_id) ?? Infinity))
  return <section className="strength-board" aria-label="League power rankings">
    <QueryError query={query} label="Model power rankings unavailable">. ESPN results are shown instead.</QueryError>
    {query.isPending && <p role="status">Loading model power rankings…</p>}
    {query.data?.status === 'unavailable' && <p className="notice">{query.data.issue} ESPN results are shown instead.</p>}
    <div className="strength-board-head">
      <PowerLegend power={power} sort={sort} />
      {power && <div className="analysis-controls"><SelectField label="Rank by" value={sort} onChange={e => setSort(e.target.value)}>
        {Object.entries(SORTS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </SelectField></div>}
    </div>
    <ol className="strength-team-grid">{teams.map(team => {
      const m = model.get(team.team_id), e = espn.get(team.team_id)!
      return <li key={team.team_id}>
        <Link className={`strength-team-card${team.team_id === data.my_team_id ? ' mine' : ''}`} to={teamPath(team)}>
          <div className="strength-team-head">
            <span className="strength-team-rank" title={SORTS[sort]}>{position(team.team_id) ?? '—'}</span>
            <span className="strength-team-name"><strong title={team.team_name}>{team.team_name}</strong>{team.team_id === data.my_team_id && <span className="badge mine">You</span>}</span>
          </div>
          <SourceMetrics rows={[
            ...(power ? [{ source: 'model' as const, label: `${power.campaign_name} forecast`, items: modelItems(m, power) }] : []),
            { source: 'espn', label: `ESPN results through Week ${data.week - 1}`, items: [
              { label: 'Record', value: record(e.wins, e.losses, e.ties), title: e.standing ? `${rank(e.standing)} in ESPN standings` : undefined },
              { label: 'Pts / wk', value: fixed(e.ppg), title: `${rank(e.ppg_rank)} in points per week` },
              { label: 'All-play', value: pct(e.all_play), title: `${rank(e.all_play_rank)} against every team` },
            ] },
          ]} />
        </Link>
      </li>
    })}</ol>
    {power && <PowerMethod power={power} />}
  </section>
}

function modelItems(m: PowerTeam | undefined, power: PowerRankingsResponse) {
  return [
    { label: 'Strength', value: fixed(m?.rating), title: `${rank(m?.rating_rank)} · expected starting-lineup points per week` },
    { label: 'Playoffs', value: power.playoff_teams ? pct(m?.playoff_odds) : '—', title: m?.average_seed != null ? `Average seed ${fixed(m.average_seed)}` : undefined },
    { label: 'Proj W–L', value: m?.projected_wins != null ? `${fixed(m.projected_wins)}–${fixed(m.projected_losses)}` : '—', title: 'Current ESPN record plus simulated remaining games' },
  ]
}

function PowerLegend({ power, sort }: { power?: PowerRankingsResponse; sort: Sort }) {
  if (!power) return <p className="legend">Ranked by ESPN standings: record, then points scored. Select a team for its full breakdown.</p>
  const basis = sort === 'espn' ? 'ESPN standings (record, then points scored)'
    : sort === 'playoffs' ? `${power.campaign_name} simulated playoff odds`
    : `${power.campaign_name} rest-of-season starting-lineup strength (${weekRange(power.weeks)})`
  return <p className="legend">Ranked by {basis}. <SourceTag source="model" /> figures are forecasts from the model selected above; <SourceTag source="espn" /> figures are actual results. Hover a figure for its rank; select a team for the full breakdown.</p>
}

/** Shared by the board and team pages so both explain the same numbers. */
export function PowerMethod({ power }: { power: PowerRankingsResponse }) {
  const name = power.campaign_name ?? 'The model'
  return <details className="outlook-method"><summary>How the model rankings work</summary>
    <p><strong>Strength</strong> is {name}'s expected points per week from each team's best legal starting lineup ({power.slots?.join(', ')}) over {weekRange(power.weeks).toLowerCase()}, re-picked every week from each player's week-by-week forecast. Players on bye count as zero.</p>
    <p><strong>Playoffs</strong>, average seed and <strong>Proj W–L</strong> come from {power.simulations?.toLocaleString()} simulated seasons. Every remaining matchup is played using those weekly means plus random swings the size of this league's observed week-to-week spread ({fixed(power.score_sd)} points, from {power.score_sd_samples} ESPN team scores), then added to the current ESPN record.{power.playoff_teams ? ` The top ${power.playoff_teams} teams make the playoffs.` : ''}</p>
    <ul>{power.limitations?.map(l => <li key={l}>{l}</li>)}</ul>
    <p>ESPN figures are actual results. All-play is the share of other teams' scores beaten each week, with ties counting half.</p>
  </details>
}
