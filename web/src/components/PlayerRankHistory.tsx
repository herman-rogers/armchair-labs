import type { Ranking, RankingHistory } from '../api/nextgen'
import { RankMovement } from './RankMovement'

export function PlayerRankHistory({ ranking, history }: { ranking: Ranking; history?: RankingHistory }) {
  const rows = [
    ...(history?.snapshots.map(snapshot => ({ week: snapshot.through_week, ...snapshot.rankings.find(r => r.player_id === ranking.player_id) })) ?? []),
    { week: ranking.through_week, overall_rank: ranking.overall_rank, position_rank: ranking.position_rank },
  ]
  return <section className="rank-history"><h4>Weekly rank history</h4>
    <p>Published ranks through each completed week · {ranking.horizon === 'next4' ? 'next four weeks' : 'rest of season'}. Ranks use the full player pool.</p>
    <div className="table-wrap"><table><thead><tr><th>Through week</th><th>Overall rank</th><th>Position rank</th><th>Overall change</th></tr></thead>
      <tbody>{rows.map((row, index) => <tr key={row.week}><td>{row.week}</td><td>{row.overall_rank ?? '—'}</td><td>{row.position_rank == null ? '—' : `${ranking.position}${row.position_rank}`}</td><td><RankMovement current={row.overall_rank} previous={rows[index - 1]?.overall_rank} /></td></tr>)}</tbody>
    </table></div>
    {!history?.snapshots.length && <p>No earlier published weekly ranks are available.</p>}
  </section>
}
