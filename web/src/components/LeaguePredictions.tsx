import { lazy, Suspense, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { LeagueObservations } from '../api/league'
import { leagueScoringQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { buildPredictionLadder } from '../predictions'
import { fixed, percent } from '../format'
import { QueryError } from './Controls'
import { ChartFrame } from './Charts'
import { PredictionLadder } from './PredictionLadder'
import { WeeklyRankings } from './WeeklyRankings'
import { useUrlNumber } from '../navigation'
import './ScoringTrends.css'

const LeagueScoringTrends = lazy(() => import('./LeagueScoringTrends').then(m => ({ default: m.LeagueScoringTrends })))

/** `/league/predictions`: how the chosen campaign's forecasts have fared, game by game. */
export function LeaguePredictions({ data, campaignId }: { data: LeagueObservations; campaignId: string }) {
  const { token } = useDataRelease()
  const [showTrends, setShowTrends] = useState(false)
  const scoring = useQuery(leagueScoringQuery(campaignId, token))
  const sameSeason = scoring.data?.season === data.season
  const model = scoring.data && sameSeason ? buildPredictionLadder(data, scoring.data) : undefined
  const defaultWeek = model?.weeks.find(w => w.week === model.through_week) ?? model?.weeks.at(-1)
  const [selectedWeek, setSelectedWeek] = useUrlNumber('week', defaultWeek?.week ?? data.week)
  const week = model?.weeks.find(w => w.week === selectedWeek) ?? defaultWeek
  return <section className="league-predictions" aria-label="Predictions">
    <section className="scoring-card" aria-label="Predicted versus actual by matchup">
      <QueryError query={scoring} label="Scoring forecasts unavailable" />
      {scoring.isPending && <p role="status">Loading scoring forecasts…</p>}
      {scoring.data && !sameSeason && <p className="notice">The campaign's saved forecasts are for another season; matchup predictions are unavailable.</p>}
      {scoring.data?.stale && <p className="notice">This chart uses a saved league snapshot. Refresh data for the latest lineups and results.</p>}
      {model && <>
        {week && <WeeklyRankings week={week} previous={model.weeks.find(w => w.week === week.week - 1)} teamCount={model.teams.length} />}
        <ChartFrame title="Predictions by matchup" description="Choose a week to see who played, who won, and how the model scored each team." plain>
          <dl className="prediction-season-summary">
            <div><dt>Season picks right</dt><dd>{model.picks.correct} / {model.picks.evaluated} <small>{model.picks.evaluated ? `· ${percent(model.picks.correct / model.picks.evaluated)}` : '· none graded'}</small></dd></div>
            <div><dt>Average score error</dt><dd>{fixed(model.mae)} <small>pts per team</small></dd></div>
            <div><dt>Model scoring bias</dt><dd>{fixed(model.bias == null ? null : Math.abs(model.bias))} <small>{model.bias == null ? 'no results' : model.bias === 0 ? 'pts · on target' : `pts ${model.bias > 0 ? 'too low' : 'too high'}`}</small></dd></div>
          </dl>
          <PredictionLadder model={model} week={week} onWeekChange={setSelectedWeek} />
        </ChartFrame>
        <div className="prediction-foot">
          <span>Season summary through Week {model.through_week}{model.picks.ties ? ` · ${model.picks.ties} ties ungraded` : ''}</span>
          <span>Totals include K/DST</span>
        </div>
      </>}
    </section>
    <details className="prediction-trends" onToggle={e => setShowTrends(e.currentTarget.open)}>
      <summary>Explore scoring trends by team</summary>
      {showTrends && <Suspense fallback={<p role="status">Loading scoring charts…</p>}><LeagueScoringTrends campaignId={campaignId} /></Suspense>}
    </details>
  </section>
}
