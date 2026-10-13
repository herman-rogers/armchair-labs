import { Link } from 'react-router'
import { weeklyRankings, type LadderWeek } from '../predictions'
import { fixed, rank } from '../format'
import { teamPath } from '../navigation'

export function WeeklyRankings({ week, previous, teamCount }: { week: LadderWeek; previous?: LadderWeek; teamCount: number }) {
  const rows = weeklyRankings(week, previous, teamCount)
    .sort((a, b) => (a.projectedRank ?? Infinity) - (b.projectedRank ?? Infinity) || a.name.localeCompare(b.name))
  return <section className="prediction-rankings" aria-label={`Week ${week.week} rankings`}>
    <div className="rank-snapshot-heading"><h3>Week {week.week} · Rank snapshot</h3><span>Projected order · {previous ? `change vs W${previous.week}` : 'includes K/DST'}</span></div>
    <ol className="rank-snapshot-list">
      {rows.map(row => <li key={row.team_id} className={row.mine ? 'is-mine' : undefined}>
        <Link to={teamPath({ team_id: row.team_id, team_name: row.name })} title={`${row.name}${row.mine ? ' · Your team' : ''}. Projected rank ${rank(row.projectedRank)}${row.actualRank != null ? `; actual rank ${rank(row.actualRank)}` : ''}`}>
          <strong className="rank-snapshot-number">{rank(row.projectedRank)}</strong>
          <span className="rank-snapshot-team"><span>{row.name}{row.mine ? ' · You' : ''}</span><small>{fixed(row.predicted)} proj{row.actual != null ? ` · ${fixed(row.actual)} actual` : ''}</small></span>
          {row.rankChange != null && row.rankChange !== 0 && <small className="rank-snapshot-change" aria-label={`${Math.abs(row.rankChange)} places ${row.rankChange > 0 ? 'up' : 'down'} from Week ${week.week - 1}`}>{row.rankChange > 0 ? '↑' : '↓'}{Math.abs(row.rankChange)}</small>}
        </Link>
      </li>)}
    </ol>
  </section>
}
