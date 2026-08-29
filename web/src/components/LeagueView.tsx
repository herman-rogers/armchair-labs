import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchLeague, refreshLeague } from '../api/client'
import type { MetricVersion } from '../api/types'
import { Freshness } from './Freshness'
import { ComparePanel } from './ComparePanel'
import { MatchupsPanel } from './MatchupsPanel'
import { RosterPanel } from './RosterPanel'
import { TeamsPanel } from './TeamsPanel'
import { WirePanel } from './WirePanel'

type Tab = 'roster' | 'matchups' | 'compare' | 'wire' | 'teams'

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: 'roster', label: 'My Roster', hint: 'Health, value, and who to start' },
  { id: 'matchups', label: 'Matchups', hint: 'Any week of the schedule, side by side' },
  { id: 'compare', label: 'Compare', hint: 'Any two rosters, slot by slot' },
  { id: 'wire', label: 'Wire', hint: 'Free agents ranked against what else is free' },
  { id: 'teams', label: 'League', hint: 'Rival depth and what the room is paying' },
]

/** Everything that needs live ESPN state. */
export function LeagueView({ version }: { version: MetricVersion }) {
  const [tab, setTab] = useState<Tab>('roster')
  // Compare holds its own pair rather than reusing the roster tab. Sending a team
  // click to "My Roster" meant the tab lied about what it was showing and you lost
  // your place getting back.
  const [comparePair, setComparePair] = useState<{ left: number; right: number } | null>(null)
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

  const myTeam = league.data.teams.find((team) => team.is_mine) ?? null

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

      {tab === 'roster' && myTeam && <RosterPanel team={myTeam} version={version} />}
      {tab === 'roster' && !myTeam && (
        <div className="notice">
          No team is marked as yours. Re-run <code>uv run patron auth login</code> to pick
          your team, or set <code>ESPN_TEAM_ID</code> in <code>.env</code>.
        </div>
      )}

      {tab === 'matchups' && <MatchupsPanel version={version} />}

      {tab === 'compare' && (
        <ComparePanel
          teams={league.data.teams}
          left={comparePair?.left ?? myTeam?.team_id ?? league.data.teams[0].team_id}
          right={
            comparePair?.right ??
            (league.data.teams.find((team) => !team.is_mine)?.team_id ??
              league.data.teams[0].team_id)
          }
          version={version}
          onChange={(side, teamId) =>
            setComparePair((current) => {
              const base = {
                left: current?.left ?? myTeam?.team_id ?? league.data.teams[0].team_id,
                right:
                  current?.right ??
                  (league.data.teams.find((team) => !team.is_mine)?.team_id ??
                    league.data.teams[0].team_id),
              }
              return { ...base, [side]: teamId }
            })
          }
        />
      )}

      {tab === 'wire' && <WirePanel version={version} />}

      {tab === 'teams' && (
        <TeamsPanel
          teams={league.data.teams}
          version={version}
          onSelectTeam={(team) => {
            // Open the comparison against your own team rather than replacing the
            // roster tab's contents.
            setComparePair({
              left: myTeam?.team_id ?? team.team_id,
              right: team.team_id,
            })
            setTab('compare')
          }}
        />
      )}

    </>
  )
}
