import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router'
import { get } from '../api/client'

type Freshness = {
  stale: boolean
  warnings: string[]
  observations: { through_week?: number }
  forecasts: { campaign: string; required_origin_week?: number; stale: boolean }
  injuries: { observed_at?: string; stale: boolean }
}

export function DataFreshness() {
  const [params] = useSearchParams()
  const campaign = params.get('campaign') ?? ''
  const query = useQuery({
    queryKey: ['freshness', campaign],
    queryFn: ({ signal }) => get<Freshness>(`/api/freshness${campaign ? `?campaign=${encodeURIComponent(campaign)}` : ''}`, undefined, signal),
    staleTime: 30_000, refetchInterval: 60_000, retry: false,
  })
  if (query.isError) return <p className="notice" role="alert">Data freshness could not be checked. Saved results may be out of date.</p>
  if (!query.data?.stale) return null
  return <div className="notice" role="status" aria-label="Data freshness warning">
    <strong>Some inputs need attention.</strong>
    <ul>{query.data.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul>
    <p>Use Refresh data to update observations and forecasts. Injury collection and game-result updates are checked separately.</p>
  </div>
}
