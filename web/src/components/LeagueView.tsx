import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchLeague, refreshLeague } from '../api/client'
import type { MetricVersion } from '../api/types'
import { Freshness } from './Freshness'
import { LeagueBoard } from './LeagueBoard'
import { MatchupsPanel } from './MatchupsPanel'
import { WirePanel } from './WirePanel'
import { DraftPanel } from './DraftPanel'

type Tab = 'board' | 'matchups' | 'wire' | 'draft'

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: 'board', label: 'League Board', hint: 'Power ranks, roster detail, and team comparisons' },
  { id: 'matchups', label: 'Matchups', hint: 'The schedule and each weekly lineup edge' },
  { id: 'wire', label: 'Waiver Wire', hint: 'Free agents ranked against what is actually available' },
  { id: 'draft', label: 'Draft', hint: 'The draft recap scored against the board, pick by pick' },
]

/** Everything that needs live ESPN state. */
export function LeagueView({ version }: { version: MetricVersion }) {
  const [tab, setTab] = useState<Tab>('board')
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

  const hasMyTeam = league.data.teams.some((team) => team.is_mine)

  return (
    <>
      <div className="subnav">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            className="subtab"
            aria-selected={tab === entry.id}
            onClick={() => setTab(entry.id)}
            title={entry.hint}
          >
            {entry.label}
          </button>
        ))}
        <span className="spacer" />
        <Freshness
          data={league.data}
          onRefresh={() => refresh.mutate()}
          refreshing={refresh.isPending}
        />
      </div>

      {tab === 'board' && !hasMyTeam && (
        <div className="notice">
          <h2>Choose your team</h2>
          Re-run <code>uv run patron auth login</code> to identify your team, or set{' '}
          <code>ESPN_TEAM_ID</code> in <code>.env</code>.
        </div>
      )}
      {tab === 'board' && hasMyTeam && <LeagueBoard teams={league.data.teams} version={version} />}
      {tab === 'matchups' && <MatchupsPanel version={version} />}
      {tab === 'wire' && <WirePanel version={version} />}
      {tab === 'draft' && <DraftPanel version={version} />}
    </>
  )
}
