import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchLeagueStatus, fetchStatus } from './api/client'
import type { MetricVersion } from './api/types'
import { LeagueView } from './components/LeagueView'
import { PlayersView } from './components/PlayersView'
import { MetricReportView } from './components/MetricReportView'

type View = 'league' | 'players' | 'metrics'

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'league', label: 'League', hint: 'Your roster, the wire, and what rivals hold' },
  { id: 'players', label: 'Players', hint: 'Every ranked player, cross-referenced' },
  { id: 'metrics', label: 'Metric Report', hint: 'Definitions and rolling backtest evidence' },
]

/**
 * The metric systems, named by what they answer rather than only by version.
 *
 * V1 is the actual historical board. V2 is the forward board: its position-specific
 * rankers were selected from the rolling backtest and its overall order uses the
 * fitted PPG ranker on one cross-position scale.
 */
function versionOptions(boardSeason?: number, draftSeason?: number) {
  return [
    {
      id: 'v1' as MetricVersion,
      label: boardSeason ? `V1 · ${boardSeason} Actual` : 'V1 · Actual',
      hint: 'What each player actually scored last season, under this league\u2019s scoring.',
    },
    {
      id: 'v2' as MetricVersion,
      label: draftSeason ? `V2 · ${draftSeason} Projection` : 'V2 · Projection',
      hint: 'Forward ranks selected by position from the rolling backtest.',
    },
    {
      id: 'adaptive' as MetricVersion,
      label: draftSeason ? `Adaptive · ${draftSeason} Shadow` : 'Adaptive · Shadow',
      hint: 'Frozen Adaptive PPG-selector forecast; experimental and prospectively graded.',
    },
  ]
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
          <div className="metric-switch" role="tablist" aria-label="Metric basis">
            {versionOptions(league?.board_season, league?.draft_season).map((option) => (
              <button
                key={option.id}
                type="button"
                role="tab"
                aria-selected={version === option.id}
                disabled={!status.data?.metric_versions[option.id]?.available}
                onClick={() => setVersion(option.id)}
                title={option.hint}
              >
                {option.label}
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
          {view === 'metrics' && <MetricReportView available={status.data.metric_report.available} />}
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
