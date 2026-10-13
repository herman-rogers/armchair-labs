import { useId } from 'react'
import type { LadderPair, LadderWeek, PredictionLadderModel } from '../predictions'
import { fixed, percent, signed } from '../format'
import './PredictionLadder.css'

function pickLabel(pair: LadderPair) {
  if (pair.status === 'correct') return '✓ Pick right'
  if (pair.status === 'wrong') return '✕ Pick wrong'
  if (pair.status === 'tie') return 'Tie · ungraded'
  if (pair.status === 'pending') return 'Pick pending'
  return pair.home.predicted != null && pair.away.predicted != null ? 'Model predicts a tie' : 'Forecast missing'
}

function weekSummary(week: LadderWeek) {
  if (week.evaluated) return `${week.correct}/${week.evaluated} picks right`
  if (week.graded) return 'No graded picks'
  return 'Awaiting results'
}

function MatchupCard({ pair, index }: { pair: LadderPair; index: number }) {
  const titleId = useId()
  const final = pair.home.outcome != null && pair.away.outcome != null
  return <article className="prediction-matchup" aria-labelledby={titleId}>
    <header>
      <h4 id={titleId}>Matchup {index + 1} <span>· {final ? 'Final' : 'Not final'}</span></h4>
      <span className={`prediction-verdict ${pair.status}`}>{pickLabel(pair)}</span>
    </header>
    <table aria-label={`${pair.home.name} versus ${pair.away.name}`}>
      <thead><tr><th scope="col">Team</th><th scope="col">Predicted</th><th scope="col">Actual</th><th scope="col">Error</th></tr></thead>
      <tbody>{[pair.home, pair.away].map(team => <tr key={team.team_id}>
        <th scope="row">
          <span className="prediction-team-name">{team.name}</span>
          <span className="prediction-team-meta">
            <span className={`prediction-outcome ${team.outcome ?? ''}`}>{team.outcome === 'won' ? 'Won' : team.outcome === 'lost' ? 'Lost' : team.outcome === 'tie' ? 'Tied' : 'Not final'}</span>
            {pair.pick_id === team.team_id && <span className="prediction-model-pick">Model pick</span>}
            {team.mine && <span>Your team</span>}
          </span>
        </th>
        <td>{fixed(team.predicted)}</td>
        <td className={`prediction-actual ${team.outcome === 'won' ? 'won' : ''}`}>{fixed(final ? team.actual : null)}</td>
        <td className="prediction-error">{signed(final && team.actual != null && team.predicted != null ? team.actual - team.predicted : null, 1)}</td>
      </tr>)}</tbody>
    </table>
  </article>
}

/** A week of paired scorecards, with the season's pick record always visible. */
export function PredictionLadder({ model, week, onWeekChange }: { model: PredictionLadderModel; week?: LadderWeek; onWeekChange: (week: number) => void }) {
  const panelId = useId()
  if (!week) return <p className="notice">No matchups are available for these forecasts yet.</p>
  const errors = week.pairs.flatMap(p => [p.home, p.away])
    .filter(t => t.outcome != null && t.actual != null && t.predicted != null)
    .map(t => Math.abs(t.actual! - t.predicted!))
  const mae = errors.length ? errors.reduce((sum, n) => sum + n, 0) / errors.length : null
  const ungraded = week.pairs.filter(p => p.status === 'none').length
  const pending = week.pairs.filter(p => p.home.outcome == null || p.away.outcome == null).length

  return <div className="prediction-matchups">
    <nav className="prediction-weeks" aria-label="Prediction weeks">
      {model.weeks.map(w => <button key={w.week} type="button" aria-pressed={w.week === week.week} aria-controls={panelId} onClick={() => onWeekChange(w.week)}>
        <strong>Week {w.week}</strong>
        <span>{weekSummary(w)}</span>
        <span className="prediction-week-track" aria-hidden="true">{w.pairs.map(p => <i key={p.home.team_id} className={p.status} />)}</span>
        <small>{w.ties ? `${w.ties} tied · ` : ''}{w.graded ? 'Final' : 'Pending'}</small>
      </button>)}
    </nav>
    <section id={panelId} aria-label={`Week ${week.week} matchups`}>
      <div className="prediction-week-heading">
        <div><h3>Week {week.week} <span>· {week.graded ? 'Results' : 'Forecasts & results'}</span></h3>
          <p>{week.evaluated ? `${weekSummary(week)} (${percent(week.correct / week.evaluated)})` : pending ? `${pending} awaiting results` : weekSummary(week)}
            {week.ties > 0 && ` · ${week.ties} ${week.ties === 1 ? 'tie' : 'ties'} ungraded`}
            {ungraded > 0 && ` · ${ungraded} without a winner pick`}
            {pending > 0 && week.evaluated > 0 && ` · ${pending} awaiting results`}
          </p>
        </div>
        {mae != null && <p className="prediction-week-error">Average score error <strong>{fixed(mae)} pts</strong></p>}
      </div>
      <div className="prediction-matchup-grid">{week.pairs.map((pair, index) => <MatchupCard key={pair.home.team_id} pair={pair} index={index} />)}</div>
      <p className="prediction-help">Error = actual − predicted. A positive number means the team scored more than predicted. Ties and missing picks are excluded from pick accuracy.</p>
    </section>
  </div>
}
