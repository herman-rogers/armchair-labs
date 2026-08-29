import type { Freshness as FreshnessData } from '../api/types'

function describeAge(seconds: number): string {
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

/**
 * How old the league data is, and whether it is a fallback.
 *
 * Always visible rather than tucked into a tooltip: every number on these screens is
 * a snapshot of a league that moves during the day, and acting on a stale wire is the
 * specific mistake this is here to prevent.
 */
export function Freshness({
  data,
  onRefresh,
  refreshing,
}: {
  data: FreshnessData | undefined
  onRefresh?: () => void
  refreshing?: boolean
}) {
  if (!data) return null

  return (
    <div className="freshness">
      {data.stale && (
        <span className="badge warn" title="The last refresh failed. This is the previous snapshot.">
          ESPN unreachable
        </span>
      )}
      <span className="faint">
        Week {data.week} · synced {describeAge(data.age_seconds)}
      </span>
      {onRefresh && (
        <button type="button" className="chip" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? 'Refreshing…' : 'Refresh'}
        </button>
      )}
    </div>
  )
}
