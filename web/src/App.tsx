import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchLeagueStatus, fetchStatus } from './api/client'
import type { MetricVersion } from './api/types'
import { LeagueView } from './components/LeagueView'
import { PlayersView } from './components/PlayersView'

type View = 'league' | 'players'

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'league', label: 'League', hint: 'Your roster, the wire, and what rivals hold' },
  { id: 'players', label: 'Players', hint: 'Every ranked player, cross-referenced' },
]

const VERSION_HINTS: Record<MetricVersion, string> = {
  v1: 'Historical production',
  v2: 'Forward projection',
}

export default function App() {
  const [view, setView] = useState<View>('league')
  const [version, setVersion] = useState<MetricVersion>('v2')

  const status = useQuery({ queryKey: ['status'], queryFn: fetchStatus })
  const leagueStatus = useQuery({ queryKey: ['league-status'], queryFn: fetchLeagueStatus })

  // Fall back if the selected metric generation has not been built.
  useEffect(() => {
    if (status.data && !status.data.metric_versions[version]?.available) {
      setVersion(status.data.metric_versions.v2?.available ? 'v2' : 'v1')
    }
  }, [status.data, version])

  const league = status.data?.league
  const join = leagueStatus.data?.join

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-row">
          <div>
            <h1>{league?.name ?? 'Patron Saints'}</h1>
            <p>
              {league
                ? `${league.team_count}-team, full PPR with big-play bonuses · ${league.seasons.join('/')} tape for the ${league.draft_season} season`
                : 'Loading league configuration…'}
            </p>
          </div>
          <div className="version-switch" role="tablist" aria-label="Metric version">
            {(['v1', 'v2'] as MetricVersion[]).map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={version === name}
                disabled={!status.data?.metric_versions[name]?.available}
                onClick={() => setVersion(name)}
                title={VERSION_HINTS[name]}
              >
                <span>{name.toUpperCase()}</span>
                <small>{VERSION_HINTS[name]}</small>
              </button>
            ))}
          </div>
        </div>

        <nav className="mainnav" role="tablist" aria-label="Section">
          {VIEWS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={view === entry.id}
              onClick={() => setView(entry.id)}
              title={entry.hint}
            >
              {entry.label}
            </button>
          ))}
        </nav>
      </header>

      {status.isError && (
        <div className="notice">
          <h2>Cannot reach the API</h2>
          Start it with <code>just api</code> (or <code>uv run patron serve</code>).
        </div>
      )}

      {status.data && !status.data.board_available && (
        <div className="notice">
          <h2>No board built yet</h2>
          Run <code>just board</code> to pull nflverse data and generate one, then reload.
        </div>
      )}

      {status.data?.board_available && (
        <>
          {view === 'league' && <LeagueView version={version} />}
          {view === 'players' && <PlayersView version={version} />}
        </>
      )}

      {join && join.unmatched > 0 && (
        <p className="legend faint">
          Join health: {join.matched}/{join.total} ESPN players matched to the board (
          {Math.round(join.match_rate * 100)}%). {join.unmatched} have no prior-season tape —
          2026 rookies and returns from injury — and {join.not_ranked} are kickers or defenses,
          which this league tiers rather than ranks.
        </p>
      )}
    </div>
  )
}
