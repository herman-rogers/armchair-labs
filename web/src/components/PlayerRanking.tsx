import { useUrlState } from '../navigation'
import { ForecastBasis } from './ForecastBasis'
import { useQuery } from '@tanstack/react-query'
import { rankingsQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'

export function PlayerRanking({ playerId, season, week }: { playerId: string; name: string; season: number; week: number }) {
  const { token } = useDataRelease()
  const [horizon, setHorizon] = useUrlState('horizon', 'rest_of_season')
  const query = useQuery(rankingsQuery(horizon, token))
  if (query.isError) return <p className="notice">Current-season ranking unavailable: {query.error.message}</p>
  if (!query.data) return <p>Loading current-season ranking…</p>
  if (query.data.report.season !== season || query.data.report.through_week !== week) return null
  const r = query.data.rankings.find(r => r.player_id === playerId)
  if (!r) return <p>No published current-season ranking for this player.</p>
  return <section><h4>NextGen · {horizon === 'next4' ? 'next four weeks' : 'rest of season'}</h4>
    <label>Forecast horizon <select value={horizon} onChange={event => setHorizon(event.target.value)}><option value="rest_of_season">Rest of regular season</option><option value="next4">Next four weeks</option></select></label>
    <p><strong>{r.position_rank == null ? 'Unranked' : `${r.position}${r.position_rank}`} · {r.prediction.toFixed(1)} {horizon === 'next4' ? 'points over the next four weeks' : 'remaining points'}</strong> · {r.evidence_status === 'validated_forecast' ? 'Historically validated forecast' : 'Reference forecast'}.</p>
    <p>Production through Week {r.through_week}; weeks {r.through_week + 1}–{r.end_week}. Published {new Date(query.data.report.published_at ?? query.data.report.generated_at).toLocaleString()}.</p>
    <ForecastBasis ranking={r} />
  </section>
}
