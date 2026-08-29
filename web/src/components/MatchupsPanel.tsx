import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchMatchups } from '../api/client'
import type { MetricVersion } from '../api/types'
import { TeamCompare } from './TeamCompare'

/**
 * The season, week by week.
 *
 * Opens on the current week and on your own game, because that is the question being
 * asked nine times out of ten. Every other matchup is one click away, and any week is
 * one click away, which is what makes a schedule scan possible at all.
 */
export function MatchupsPanel({ version }: { version: MetricVersion }) {
  const [week, setWeek] = useState<number | null>(null)
  const [expanded, setExpanded] = useState<number | null>(0)

  const matchups = useQuery({
    queryKey: ['matchups', version, week],
    queryFn: () => fetchMatchups(version, week ?? undefined),
  })

  if (matchups.isError) return <div className="notice">{(matchups.error as Error).message}</div>
  if (!matchups.data) return <div className="notice">Loading matchups…</div>

  const data = matchups.data
  const weeks = Array.from({ length: data.regular_season_weeks }, (_, i) => i + 1)

  return (
    <section>
      <div className="week-strip" role="tablist" aria-label="Week">
        {weeks.map((number) => (
          <button
            key={number}
            type="button"
            role="tab"
            aria-selected={number === data.requested_week}
            className={number === data.current_week ? 'current' : undefined}
            onClick={() => {
              setWeek(number)
              setExpanded(0)
            }}
            title={number === data.current_week ? 'Current week' : `Week ${number}`}
          >
            {number}
          </button>
        ))}
      </div>

      <div className="matchup-list">
        {data.matchups.map((matchup, index) => {
          const open = expanded === index
          const homeAhead = matchup.margin > 0
          return (
            <div
              key={`${matchup.home.team_id}-${matchup.away.team_id}`}
              className={`matchup ${matchup.involves_me ? 'mine' : ''} ${open ? 'open' : ''}`}
            >
              <button
                type="button"
                className="matchup-row"
                onClick={() => setExpanded(open ? null : index)}
              >
                <span className={`matchup-team left ${homeAhead ? 'ahead' : ''}`}>
                  {matchup.home.team_name}
                </span>
                <span className="matchup-scores">
                  <span className={homeAhead ? 'ahead' : ''}>
                    {matchup.home.total.toFixed(1)}
                  </span>
                  <span className="faint">vs</span>
                  <span className={!homeAhead && matchup.margin !== 0 ? 'ahead' : ''}>
                    {matchup.away.total.toFixed(1)}
                  </span>
                </span>
                <span className={`matchup-team right ${!homeAhead && matchup.margin !== 0 ? 'ahead' : ''}`}>
                  {matchup.away.team_name}
                </span>
                {matchup.involves_me && <span className="badge mine">You</span>}
                <span className="faint chevron">{open ? '−' : '+'}</span>
              </button>

              {open && (
                <TeamCompare
                  left={matchup.home}
                  right={matchup.away}
                  margin={matchup.margin}
                  showHeads={false}
                />
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
