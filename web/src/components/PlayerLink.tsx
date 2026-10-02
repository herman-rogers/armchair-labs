import { useQueryClient } from '@tanstack/react-query'
import { useDataRelease } from '../dataRelease'
import { profileQuery } from '../api/queries'
import type { ReactNode } from 'react'
import { Link } from 'react-router'
import { collegePath, playerPath } from '../navigation'

/** Ordinary route link with intent prefetching; browser new-tab behavior stays native. */
export function ProfileRouteLink({ to, children, className = 'player-profile-link' }: { to: string; children: ReactNode; className?: string }) {
  const client = useQueryClient()
  const { token } = useDataRelease()
  const prefetch = () => {
    const url = new URL(to, window.location.origin)
    const [, kind, id] = url.pathname.split('/')
    if (!id || !['players', 'college'].includes(kind)) return
    const params = new URLSearchParams({ [kind === 'players' ? 'player_id' : 'college_id']: decodeURIComponent(id) })
    if (url.searchParams.has('season')) {
      params.set('season', url.searchParams.get('season')!)
      params.set('week', url.searchParams.get('week') ?? '18')
    }
    void client.prefetchQuery(profileQuery(params, token))
  }
  return <Link onMouseEnter={prefetch} onFocus={prefetch} className={className} to={to}>{children}</Link>
}

export function PlayerLink({ playerId, collegeId, children = 'View profile', className = 'player-profile-link' }: {
  playerId?: string | null; collegeId?: string; children?: ReactNode; className?: string
}) {
  if (!playerId && !collegeId) return <>{children}</>
  return <ProfileRouteLink className={className} to={playerId ? playerPath(playerId) : collegePath(collegeId!)}>{children}</ProfileRouteLink>
}
