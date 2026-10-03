import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { LeaguePlayer, MetricVersion, Position } from '../api/types'
import { forecastColumns } from './ForecastColumns'
import { RankingExplanation } from './PlayerDetails'
import { BoardTable } from './BoardTable'
import { ForecastContext } from './ForecastContext'
import { Freshness } from './Freshness'
import { PlayerTable } from './PlayerTable'
import { IDENTITY, OWNERSHIP } from './playerColumns'
import { SKILL_POSITIONS } from '../positions'
import { statusQuery } from '../api/queries'
import { boardQuery, leaguePlayersQuery } from '../api/queries/archive'
import { useUrlState } from '../navigation'

const FLAGS = ['age', 'ESPN-only'] as const

type Ownership = 'all' | 'free_agent' | 'rostered' | 'mine'

const OWNERSHIP_FILTERS: { id: Ownership; label: string; hint: string }[] = [
  { id: 'all', label: 'Everyone', hint: 'Every ranked player.' },
  { id: 'free_agent', label: 'Available', hint: 'Confirmed claimable right now.' },
  { id: 'rostered', label: 'Rostered', hint: 'Owned by a team in your league.' },
  { id: 'mine', label: 'Mine', hint: 'On your roster.' },
]

/**
 * Every ranked player, cross-referenced with league status.
 *
 * Falls back to the pure metric board when ESPN is not connected, because the ranking
 * is useful on its own — it just cannot tell you who is available.
 */
export function PlayersView({ version }: { version: MetricVersion }) {
  const status = useQuery(statusQuery())
  const [positionList, setPositionList] = useUrlState('positions', '')
  const positions = useMemo(() => new Set(positionList.split(',').filter(Boolean) as Position[]), [positionList])
  const [flagParam, setFlagParam] = useUrlState('flag', '')
  const flag = flagParam || null
  const [ownershipParam, setOwnership] = useUrlState('ownership', 'all')
  const ownership = ownershipParam as Ownership
  const [search, setSearch] = useUrlState('q', '', { replace: true })

  const league = useQuery(leaguePlayersQuery(version))
  // Only fetched when the league join is unavailable, so an unauthenticated user still
  // gets the board rather than an error screen.
  const board = useQuery({ ...boardQuery(version), enabled: league.isError })

  const connected = !league.isError && !!league.data
  const rows: LeaguePlayer[] = connected
    ? league.data.players
    : ((board.data?.players ?? []) as LeaguePlayer[])

  const needle = search.trim().toLowerCase()
  const filtered = rows.filter((player) => {
    if (positions.size && !positions.has(player.position)) return false
    if (flag && !player.flags.includes(flag)) return false
    if (needle && !player.player_display_name.toLowerCase().includes(needle)) return false
    if (connected && ownership !== 'all') {
      if (ownership === 'mine' && !player.is_mine) return false
      if (ownership === 'free_agent' && player.availability !== 'free_agent') return false
      if (ownership === 'rostered' && player.availability !== 'rostered') return false
    }
    return true
  })

  const togglePosition = (position: Position) => {
    const next = new Set(positions)
    if (next.has(position)) next.delete(position)
    else next.add(position)
    setPositionList([...next].join(','))
  }

  if (league.isLoading) return <div className="notice">Loading players…</div>
  if (!connected && board.isLoading) return <div className="notice">Loading the board…</div>
  if (!connected && board.isError) {
    return <div className="notice">{(board.error as Error).message}</div>
  }

  return (
    <>
      <div className="controls">
        {SKILL_POSITIONS.map((position) => (
          <button
            key={position}
            type="button"
            className="chip"
            aria-pressed={positions.has(position)}
            onClick={() => togglePosition(position)}
          >
            {position}
          </button>
        ))}
        <span style={{ width: 8 }} />
        {FLAGS.map((name) => (
          <button
            key={name}
            type="button"
            className="chip"
            aria-pressed={flag === name}
            onClick={() => setFlagParam(flag === name ? '' : name)}
          >
            {name}
          </button>
        ))}
        {connected && (
          <>
            <span style={{ width: 8 }} />
            {OWNERSHIP_FILTERS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                className="chip"
                aria-pressed={ownership === entry.id}
                title={entry.hint}
                onClick={() => setOwnership(entry.id)}
              >
                {entry.label}
              </button>
            ))}
          </>
        )}
        <input
          className="search"
          aria-label="Search players"
          placeholder="Search players…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <span className="spacer" />
        <span className="count">
          {filtered.length} of {rows.length}
        </span>
      </div>

      {!connected && (
        <p className="legend tight faint">
          Showing the metric board only — ESPN is not connected, so ownership is unknown.
          Sign in with <code>uv run engine auth login</code> to cross-reference against your
          league.
        </p>
      )}

      {status.data && <ForecastContext status={status.data} version={version} />}
      <RankingExplanation version={version} />
      {connected ? (
        <PlayerTable
          players={filtered}
          version={version}
          columns={[...IDENTITY, OWNERSHIP, ...forecastColumns<LeaguePlayer>(version)]}
          defaultSort="rank"
        />
      ) : (
        <BoardTable players={filtered} version={version} />
      )}

      {connected && <Freshness data={league.data} />}
      {connected && version !== 'v1' && (
        <p className="legend tight faint">
          Projection football metrics come from nflverse. ESPN contributes ownership, fantasy
          lineup, transactions, and the live injury badge. ESPN projected points do not
          affect model-backed ranks. A player without prior NFL production is visibly
          flagged ESPN-only and placed by ESPN’s current PPR draft-room rank; model
          metrics remain blank. The separate league power simulation may use ESPN points
          as a labeled fallback for players the board cannot project.
        </p>
      )}
    </>
  )
}
