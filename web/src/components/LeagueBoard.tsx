import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchOpponents, fetchTransactions } from '../api/client'
import type { LeagueTeam, MetricVersion } from '../api/types'
import { boardMetricLabel } from '../metricPresentation'
import { ComparePanel } from './ComparePanel'
import { RosterPanel } from './RosterPanel'

type SortKey =
  | 'team_rank'
  | 'ranking_total'
  | 'risk_adjusted_total'
  | 'wins'
  | 'expected_weekly_points'
  | 'weekly_floor'
  | 'weekly_risk'
  | 'lineup_coverage'
  | 'bench_rescue_points'
  | 'faab_remaining'

type WorkspaceView = 'compare' | 'roster'

const SORT_LABELS: Record<SortKey, string> = {
  team_rank: 'Power',
  ranking_total: 'Overall VOR',
  risk_adjusted_total: 'RA-VOR',
  wins: 'Record',
  expected_weekly_points: 'Expected',
  weekly_floor: 'Floor',
  weekly_risk: 'Risk',
  lineup_coverage: 'Coverage',
  bench_rescue_points: 'Bench',
  faab_remaining: 'FAAB',
}

function formatWhen(value: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function MetricHeader({
  metric,
  active,
  direction,
  onSort,
  title,
}: {
  metric: SortKey
  active: boolean
  direction: 'asc' | 'desc'
  onSort: (key: SortKey) => void
  title: string
}) {
  return (
    <th
      title={title}
      aria-sort={active ? (direction === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <button type="button" className="board-sort" onClick={() => onSort(metric)}>
        {SORT_LABELS[metric]}
        <span className="sort-mark" aria-hidden="true">
          {active ? (direction === 'asc' ? '↑' : '↓') : '↕'}
        </span>
      </button>
    </th>
  )
}

/**
 * One home for league context: scan every team's metrics, pick a rival, then compare
 * the legal lineups or inspect either roster without changing sections.
 */
export function LeagueBoard({
  teams,
  version,
}: {
  teams: LeagueTeam[]
  version: MetricVersion
}) {
  const myTeam = teams.find((team) => team.is_mine) ?? teams[0]
  const firstRival = teams.slice().sort((a, b) => a.team_rank - b.team_rank).find(
    (team) => team.team_id !== myTeam.team_id,
  ) ?? myTeam

  const [pair, setPair] = useState({ left: myTeam.team_id, right: firstRival.team_id })
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>('compare')
  const [rosterTeamId, setRosterTeamId] = useState(myTeam.team_id)
  const [sort, setSort] = useState<{ key: SortKey; direction: 'asc' | 'desc' }>({
    key: 'team_rank',
    direction: 'asc',
  })

  const opponents = useQuery({
    queryKey: ['opponents', version],
    queryFn: () => fetchOpponents(version),
  })
  const transactions = useQuery({ queryKey: ['transactions'], queryFn: fetchTransactions })

  const thinByTeam = useMemo(() => {
    const result = new Map<string, { position: string; best_vor: number }[]>()
    for (const row of opponents.data?.teams ?? []) {
      const list = result.get(row.owner_team_name) ?? []
      list.push({ position: row.position, best_vor: row.best_vor })
      result.set(row.owner_team_name, list)
    }
    for (const rows of result.values()) rows.sort((a, b) => a.best_vor - b.best_vor)
    return result
  }, [opponents.data])

  const rankedTeams = useMemo(() => {
    const value = (team: LeagueTeam): number => {
      if (sort.key === 'faab_remaining') return team.faab_remaining ?? -1
      return team[sort.key]
    }
    return teams.slice().sort((a, b) => {
      const difference = value(a) - value(b)
      return (sort.direction === 'asc' ? difference : -difference) || a.team_rank - b.team_rank
    })
  }, [sort, teams])

  const leftTeam = teams.find((team) => team.team_id === pair.left) ?? myTeam
  const rightTeam = teams.find((team) => team.team_id === pair.right) ?? firstRival
  const rosterTeam = teams.find((team) => team.team_id === rosterTeamId) ?? myTeam
  const hasDivisions = teams.some((team) => team.division_name || team.division_id !== null)
  const divisionGroups = useMemo(() => {
    if (!hasDivisions) return [{ label: null as string | null, teams: rankedTeams }]
    const groups = new Map<string, LeagueTeam[]>()
    for (const team of rankedTeams) {
      const label = team.division_name ?? (team.division_id !== null ? `Division ${team.division_id}` : 'No division')
      groups.set(label, [...(groups.get(label) ?? []), team])
    }
    return [...groups.entries()]
      .sort((a, b) => Math.min(...a[1].map((t) => t.team_rank)) - Math.min(...b[1].map((t) => t.team_rank)))
      .map(([label, members]) => ({ label, teams: members }))
  }, [hasDivisions, rankedTeams])

  const changeSort = (key: SortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { key, direction: key === 'team_rank' || key === 'weekly_risk' ? 'asc' : 'desc' },
    )
  }

  const selectTeam = (team: LeagueTeam) => {
    setRosterTeamId(team.team_id)
    if (team.team_id !== pair.left) setPair((current) => ({ ...current, right: team.team_id }))
  }

  return (
    <section className="league-board">
      <div className="league-intro">
        <div>
          <span className="eyebrow">League command center</span>
          <h2>See the room. Find the edge.</h2>
          <p>
            Power ranks, lineup outcomes, roster depth, and head-to-head comparisons in
            one view. Select any team below to bring it into the workspace.
          </p>
        </div>
        <span className="model-badge">{version.toUpperCase()} model</span>
      </div>

      <div className="board-kpis" aria-label={`${myTeam.team_name} summary`}>
        <div className="board-kpi primary">
          <span>Your power rank</span>
          <strong>#{myTeam.team_rank}</strong>
          <small>
            {myTeam.ranking_total.toFixed(1)} {boardMetricLabel(myTeam.ranking_metric)}
          </small>
        </div>
        <div className="board-kpi">
          <span>Expected lineup</span>
          <strong>{myTeam.expected_weekly_points.toFixed(1)}</strong>
          <small>skill points / week</small>
        </div>
        <div className="board-kpi">
          <span>Weekly floor</span>
          <strong>{myTeam.weekly_floor.toFixed(1)}</strong>
          <small>with ±{myTeam.weekly_risk.toFixed(1)} risk</small>
        </div>
        <div className="board-kpi">
          <span>Lineup coverage</span>
          <strong>{Math.round(myTeam.lineup_coverage * 100)}%</strong>
          <small>+{myTeam.bench_rescue_points.toFixed(1)} bench rescue</small>
        </div>
      </div>

      <div className="board-section-head">
        <div>
          <h3>League power board</h3>
          <p>
            Power rank uses each roster’s best legal lineup on the same model value as
            the player board. Expected points, availability, floor, and risk remain
            separate; lower risk is steadier and higher values are better elsewhere.
          </p>
        </div>
        <span className="count">{teams.length} teams · K/DST excluded</span>
      </div>

      {divisionGroups.map((group) => (
        <div className="table-wrap power-table-wrap" key={group.label ?? 'league'}>
          {group.label && <h4 className="division-head">{group.label}</h4>}
          <table className="power-table">
            <thead>
              <tr>
                <th className="left sticky-team">Team</th>
                <MetricHeader metric="team_rank" active={sort.key === 'team_rank'} direction={sort.direction} onSort={changeSort} title={`League-wide power rank: best legal full-strength lineup on ${boardMetricLabel(myTeam.ranking_metric)}`} />
                <MetricHeader metric="ranking_total" active={sort.key === 'ranking_total'} direction={sort.direction} onSort={changeSort} title={`Best legal full-strength lineup summed on ${boardMetricLabel(myTeam.ranking_metric)}: per-game points above a replacement lineup`} />
                <MetricHeader metric="risk_adjusted_total" active={sort.key === 'risk_adjusted_total'} direction={sort.direction} onSort={changeSort} title="Risk-adjusted VOR: expected best-active-lineup VOR across availability scenarios, less 0.674 x its spread (an approximate 25th percentile). Penalises stars-and-scrubs rosters whose absent stars expose a weak bench." />
                <MetricHeader metric="wins" active={sort.key === 'wins'} direction={sort.direction} onSort={changeSort} title="Current ESPN record" />
                <MetricHeader metric="expected_weekly_points" active={sort.key === 'expected_weekly_points'} direction={sort.direction} onSort={changeSort} title="Availability-aware points from the best legal active lineup" />
                <MetricHeader metric="weekly_floor" active={sort.key === 'weekly_floor'} direction={sort.direction} onSort={changeSort} title="Approximate 25th-percentile weekly lineup score" />
                <MetricHeader metric="weekly_risk" active={sort.key === 'weekly_risk'} direction={sort.direction} onSort={changeSort} title="Standard deviation of weekly lineup points; lower is steadier" />
                <MetricHeader metric="lineup_coverage" active={sort.key === 'lineup_coverage'} direction={sort.direction} onSort={changeSort} title="Chance the roster can fill every skill-position starting slot" />
                <MetricHeader metric="bench_rescue_points" active={sort.key === 'bench_rescue_points'} direction={sort.direction} onSort={changeSort} title="Expected weekly points outside the full-strength lineup" />
                <th className="left">Thin at</th>
                <MetricHeader metric="faab_remaining" active={sort.key === 'faab_remaining'} direction={sort.direction} onSort={changeSort} title="Remaining free-agent budget" />
              </tr>
            </thead>
            <tbody>
              {group.teams.map((team) => {
                const selected = team.team_id === pair.left || team.team_id === pair.right
                const thin = thinByTeam.get(team.team_name)?.slice(0, 2) ?? []
                return (
                  <tr key={team.team_id} className={`${team.is_mine ? 'mine-row' : ''} ${selected ? 'selected-row' : ''}`}>
                    <td className="left sticky-team">
                      <button type="button" className="team-link" onClick={() => selectTeam(team)}>
                        <span>{team.team_name}</span>
                        <small>{team.owner ?? 'Owner unavailable'}</small>
                        {team.fallback_players > 0 && (
                          <small title="No board or market value was available, so the league power simulation used ESPN projected season points for these players.">
                            {team.fallback_players} ESPN fallback
                          </small>
                        )}
                      </button>
                      {team.is_mine && <span className="badge mine">You</span>}
                    </td>
                    <td><span className="power-rank">#{team.team_rank}</span></td>
                    <td className="metric-emphasis" title={`${boardMetricLabel(team.ranking_metric)}`}>{team.ranking_total.toFixed(1)}</td>
                    <td title={`expected ${team.expected_lineup_vor.toFixed(1)} ± ${team.lineup_vor_risk.toFixed(1)} lineup VOR across availability scenarios`}>{team.risk_adjusted_total.toFixed(1)}</td>
                    <td><span className="record">{team.wins}–{team.losses}</span></td>
                    <td>{team.expected_weekly_points.toFixed(1)}</td>
                    <td>{team.weekly_floor.toFixed(1)}</td>
                    <td>±{team.weekly_risk.toFixed(1)}</td>
                    <td>{Math.round(team.lineup_coverage * 100)}%</td>
                    <td>+{team.bench_rescue_points.toFixed(1)}</td>
                    <td className="left">
                      <span className="thin-list">
                        {thin.length > 0 ? thin.map((entry) => (
                          <span key={entry.position} className={`pos ${entry.position}`} title={`Best ${entry.position}: ${entry.best_vor.toFixed(1)} ${version !== 'v1' ? 'overall projection VOR' : 'historical VOR'}`}>
                            {entry.position}
                          </span>
                        )) : <span className="faint">—</span>}
                      </span>
                    </td>
                    <td>{team.faab_remaining === null ? <span className="faint">—</span> : `$${team.faab_remaining}`}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ))}

      <div className="comparison-workspace">
        <div className="board-section-head workspace-head">
          <div>
            <span className="eyebrow">Team workspace</span>
            <h3>{workspaceView === 'compare' ? 'Lineup comparison' : 'Roster detail'}</h3>
          </div>
          <div className="view-switch" role="tablist" aria-label="Team workspace view">
            <button type="button" role="tab" aria-selected={workspaceView === 'compare'} onClick={() => setWorkspaceView('compare')}>Compare lineups</button>
            <button type="button" role="tab" aria-selected={workspaceView === 'roster'} onClick={() => setWorkspaceView('roster')}>View roster</button>
          </div>
        </div>

        {workspaceView === 'compare' ? (
          <ComparePanel
            teams={teams}
            left={pair.left}
            right={pair.right}
            version={version}
            onChange={(side, teamId) => {
              setPair((current) => ({ ...current, [side]: teamId }))
              setRosterTeamId(teamId)
            }}
          />
        ) : (
          <div>
            <div className="roster-choice" aria-label="Roster to inspect">
              {[leftTeam, rightTeam].filter((team, index, values) => values.findIndex((candidate) => candidate.team_id === team.team_id) === index).map((team) => (
                <button key={team.team_id} type="button" className="chip" aria-pressed={rosterTeam.team_id === team.team_id} onClick={() => setRosterTeamId(team.team_id)}>
                  {team.team_name}{team.is_mine ? ' · You' : ''}
                </button>
              ))}
            </div>
            <RosterPanel team={rosterTeam} version={version} />
          </div>
        )}
      </div>

      <details className="league-activity">
        <summary>
          <span>
            <b>League activity</b>
            <small>Recent claims, drops, and the prices your league has paid</small>
          </span>
          <span className="count">{transactions.data?.total ?? 0} moves</span>
        </summary>
        {transactions.isError && <div className="notice">League activity is unavailable.</div>}
        {transactions.data && transactions.data.transactions.length > 0 ? (
          <div className="table-wrap activity-table">
            <table>
              <thead><tr><th className="left">When</th><th className="left">Team</th><th className="left">Action</th><th className="left">Player</th><th>Bid</th></tr></thead>
              <tbody>
                {transactions.data.transactions.slice(0, 25).map((entry, index) => (
                  <tr key={`${entry.date}-${entry.player_name}-${index}`}>
                    <td className="left faint">{formatWhen(entry.date)}</td>
                    <td className="left">{entry.team_name ?? '—'}</td>
                    <td className="left dim">{entry.kind ?? '—'}</td>
                    <td className="left name">{entry.player_name ?? '—'}</td>
                    <td>{entry.bid_amount ? `$${entry.bid_amount}` : <span className="faint">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : transactions.data ? (
          <div className="notice">No transactions yet this season.</div>
        ) : null}
      </details>
    </section>
  )
}
