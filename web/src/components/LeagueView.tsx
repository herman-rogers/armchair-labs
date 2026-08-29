import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchLeague, refreshLeague } from '../api/client'
import type { LeagueTeam, MetricVersion } from '../api/types'
import { Freshness } from './Freshness'
import { RosterPanel } from './RosterPanel'
import { TeamsPanel } from './TeamsPanel'
import { WirePanel } from './WirePanel'

type Tab = 'roster' | 'wire' | 'teams'

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: 'roster', label: 'My Roster', hint: 'Health, value, and who to start' },
  { id: 'wire', label: 'Wire', hint: 'Free agents ranked against what else is free' },
  { id: 'teams', label: 'League', hint: 'Rival depth and what the room is paying' },
]

/** Everything that needs live ESPN state. */
export function LeagueView({ version }: { version: MetricVersion }) {
  const [tab, setTab] = useState<Tab>('roster')
  const [selectedTeam, setSelectedTeam] = useState<LeagueTeam | null>(null)
  const queryClient = useQueryClient()

  const league = useQuery({ queryKey: ['league'], queryFn: fetchLeague })
  const refresh = useMutation({
    mutationFn: refreshLeague,
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
  const rosterTeam = selectedTeam ?? myTeam

  return (
    <>
      <div className="subnav">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            className="subtab"
            aria-selected={tab === entry.id}
            onClick={() => {
              setTab(entry.id)
              if (entry.id === 'roster') setSelectedTeam(null)
            }}
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

      {tab === 'roster' && rosterTeam && (
        <>
          <h2 className="section-head">
            {rosterTeam.team_name}
            {!rosterTeam.is_mine && (
              <button type="button" className="chip" onClick={() => setSelectedTeam(null)}>
                back to mine
              </button>
            )}
          </h2>
          <RosterPanel team={rosterTeam} version={version} />
        </>
      )}
      {tab === 'roster' && !rosterTeam && (
        <div className="notice">
          No team is marked as yours. Re-run <code>uv run patron auth login</code> to pick
          your team, or set <code>ESPN_TEAM_ID</code> in <code>.env</code>.
        </div>
      )}

      {tab === 'wire' && <WirePanel version={version} />}

      {tab === 'teams' && (
        <TeamsPanel
          teams={league.data.teams}
          version={version}
          onSelectTeam={(team) => {
            setSelectedTeam(team)
            setTab('roster')
          }}
        />
      )}
    </>
  )
}
