export function RankMovement({ current, previous }: { current: number | null | undefined; previous: number | null | undefined }) {
  if (current == null) return <span className="faint">Unavailable</span>
  if (previous == null) return <span className="faint">No prior rank</span>
  const change = previous - current
  return <span className={change > 0 ? 'rank-up' : change < 0 ? 'rank-down' : 'faint'} aria-label={change ? `${change > 0 ? 'Up' : 'Down'} ${Math.abs(change)} places` : 'Unchanged'}>
    {change > 0 ? `↑ ${change}` : change < 0 ? `↓ ${-change}` : '— Unchanged'}
  </span>
}
