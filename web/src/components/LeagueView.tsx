import { NavLink, Outlet, useParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchLeague, refreshLeague } from '../api/client'
import type { MetricVersion } from '../api/types'
import { Freshness } from './Freshness'
import { LeagueBoard } from './LeagueBoard'
import { MatchupsPanel } from './MatchupsPanel'
import { WirePanel } from './WirePanel'
import { DraftPanel } from './DraftPanel'
import { PlayersView } from './PlayersView'
import { DataTable } from './DataTable'
import { ARCHIVED_VIEWS, archivePath } from '../navigation'

/** Archived league tabs; the router accepts only the ids in `ARCHIVED_VIEWS.league`. */
const LEAGUE_VIEW_TABS: { id: typeof ARCHIVED_VIEWS.league[number]; label: string; hint: string }[] = [
  { id: 'overview', label: 'Overview', hint: 'Power ranks, roster detail, and team comparisons' },
  { id: 'matchups', label: 'Matchups', hint: 'The schedule and each weekly lineup edge' },
  { id: 'players', label: 'Players', hint: 'Search the league player pool and inspect individual stats' },
  { id: 'wire', label: 'Waiver Wire', hint: 'Free agents ranked against what is actually available' },
  { id: 'draft', label: 'Draft recap', hint: 'The draft recap scored against the board, pick by pick' },
]

/** The archived league workspace reads the v2 metric snapshot. */
const version: MetricVersion = 'v2'

/** Layout for `/research/archive/league/*`. Everything that needs live ESPN state. */
export function LeagueView() {
  const queryClient = useQueryClient()

  const league = useQuery({
    queryKey: ['league', version],
    queryFn: () => fetchLeague(version),
  })
  const refresh = useMutation({
    mutationFn: () => refreshLeague(version),
    // Everything on these screens derives from one snapshot, so a refresh invalidates
    // all of it rather than leaving panels disagreeing about what week it is.
    onSuccess: () => queryClient.invalidateQueries(),
  })

  if (league.isError) {
    const message = (league.error as Error).message
    const needsAuth = message.toLowerCase().includes('auth')
    return (
      <div className="notice">
        <h2>{needsAuth ? 'Not signed in to ESPN' : 'League data unavailable'}</h2>
        {needsAuth ? (
          <>
            Run <code>uv run patron auth login</code> — it opens a browser, you sign in
            normally, and it captures the session.
          </>
        ) : (
          message
        )}
      </div>
    )
  }

  if (!league.data) return <div className="notice">Connecting to your league…</div>

  if (league.data.teams.length === 0) {
    return <div className="notice">No teams were returned for this league.</div>
  }

  return (
    <>
      <p className="archive-banner">ARCHIVED: old power rankings, matchup simulations and waiver/draft valuations are unapproved research scenarios.</p>
      <div className="section-heading"><div><span className="eyebrow">Current league</span>
        <h2>{league.data.league_name}</h2><p>Every team, every week. Scores and ownership from your league snapshot.</p></div>
        <span className="instrument-tag">Week {league.data.week} · {league.data.season}</span>
      </div>
      <div className="subnav" aria-label="Archived league views">
        {LEAGUE_VIEW_TABS.map((entry) => (
          <NavLink key={entry.id} className="subtab" to={archivePath('league', entry.id)} title={entry.hint} preventScrollReset>
            {entry.label}
          </NavLink>
        ))}
        <span className="spacer" />
        <Freshness
          data={league.data}
          onRefresh={() => refresh.mutate()}
          refreshing={refresh.isPending}
        />
      </div>
      <Outlet />
    </>
  )
}

/** `/research/archive/league/:view`. */
export function LeagueViewTab() {
  const { view } = useParams()
  const league = useQuery({ queryKey: ['league', version], queryFn: () => fetchLeague(version) })
  if (!league.data) return null
  const hasMyTeam = league.data.teams.some((team) => team.is_mine)
  return (
    <>
      {view === 'overview' && <section className="standings-section"><h3>Standings</h3>
        <DataTable rows={league.data.teams} defaultSort="wins" rowKey={t => t.team_id} columns={[
          { key: 'team_name', label: 'Team', title: 'League team.', align: 'left' },
          { key: 'wins', label: 'W', title: 'Actual wins.' },
          { key: 'losses', label: 'L', title: 'Actual losses.', initial: 'asc' },
          { key: 'division_name', label: 'Division', title: 'League division.', align: 'left', render: t => t.division_name ?? '—' },
          { key: 'faab_remaining', label: 'FAAB', title: 'Remaining waiver budget.', render: t => t.faab_remaining ?? '—' },
        ]} />
      </section>}
      {view === 'overview' && !hasMyTeam && (
        <div className="notice">
          <h2>Choose your team</h2>
          Re-run <code>uv run patron auth login</code> to identify your team, or set{' '}
          <code>ESPN_TEAM_ID</code> in <code>.env</code>.
        </div>
      )}
      {view === 'overview' && <LeagueBoard teams={league.data.teams} version={version} />}
      {view === 'players' && <PlayersView version={version} />}
      {view === 'matchups' && <MatchupsPanel version={version} />}
      {view === 'wire' && <WirePanel version={version} />}
      {view === 'draft' && <DraftPanel version={version} />}
    </>
  )
}
