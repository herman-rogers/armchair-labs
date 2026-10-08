import { Link, useLocation } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { breadcrumbsFor, type Breadcrumb } from '../breadcrumbs'
import { leagueQuery, profileQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'

function TeamLabel({ id, fallback }: { id: string; fallback: string }) {
  const { token } = useDataRelease()
  const query = useQuery(leagueQuery(token))
  return query.data?.teams.find(team => String(team.team_id) === id)?.team_name ?? fallback
}

function ProfileLabel({ entity, fallback }: { entity: NonNullable<Breadcrumb['entity']>; fallback: string }) {
  const { token } = useDataRelease()
  const { search } = useLocation()
  const season = new URLSearchParams(search).get('season')
  // Share the profile's query key, including its historical cutoff.
  const params = new URLSearchParams({ [entity.kind === 'player' ? 'player_id' : 'college_id']: entity.id, ...(season ? { season, week: '18' } : {}) })
  const query = useQuery(profileQuery(params, token))
  return query.data?.identity.player_display_name ?? fallback
}

/** Always rendered by the app shell, including loading, error and missing pages. */
export function Breadcrumbs({ resolveNames = true }: { resolveNames?: boolean }) {
  const { pathname, search } = useLocation()
  const crumbs = breadcrumbsFor(pathname, search)
  return <nav className="breadcrumbs" aria-label="Breadcrumb">
    <ol>{crumbs.map((crumb, index) => {
      const current = index === crumbs.length - 1
      const label = resolveNames && crumb.entity
        ? crumb.entity.kind === 'team' ? <TeamLabel id={crumb.entity.id} fallback={crumb.label} /> : <ProfileLabel entity={crumb.entity} fallback={crumb.label} />
        : crumb.label
      return <li key={`${pathname}-${index}`}>
        {index > 0 && <span className="breadcrumb-separator" aria-hidden="true">/</span>}
        {current ? <span aria-current="page">{label}</span> : crumb.href ? <Link to={crumb.href}>{label}</Link> : <span>{label}</span>}
      </li>
    })}</ol>
  </nav>
}
