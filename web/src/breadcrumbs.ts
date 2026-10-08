import { matchPath } from 'react-router'
import { navigationPage } from './pageNavigation'
import { PROFILE_SECTIONS, leagueWeekPath } from './navigation'

export type Breadcrumb = {
  label: string
  href?: string
  entity?: { kind: 'player' | 'college' | 'team'; id: string }
}

/** Location hierarchy, independent of browsing history so deep links work too. */
export function breadcrumbsFor(pathname: string, search = ''): Breadcrumb[] {
  const navigation = navigationPage(pathname)
  // Section destinations follow sidebar order; labels and parent pages share its metadata.
  const parents: Breadcrumb[] = navigation ? [
    { label: navigation.group.label, href: navigation.group.pages[0].path },
    { label: navigation.page.title, href: navigation.page.path },
  ] : []
  for (const kind of ['players', 'college'] as const) {
    const profile = matchPath(`/${kind}/:id/:section?`, pathname)
    if (profile) {
      const { id, section } = profile.params
      const base = `/${kind}/${encodeURIComponent(id!)}`
      const sectionLabel = PROFILE_SECTIONS.find(([key]) => key === section)?.[1]
      return [...parents,
        { label: kind === 'players' ? 'Player profile' : 'College profile', entity: { kind: kind === 'players' ? 'player' : 'college', id: id! }, ...(sectionLabel ? { href: base + search } : {}) },
        ...(sectionLabel ? [{ label: sectionLabel }] : []),
      ]
    }
  }
  const matchup = matchPath('/league/matchups/:week/:homeId/:awayId', pathname)
  if (matchup) {
    const week = Number(matchup.params.week)
    const validWeek = /^[1-9]\d*$/.test(matchup.params.week ?? '') && week <= 25
    return [...parents,
      ...(validWeek ? [{ label: `Week ${week} matchups`, href: leagueWeekPath(week) }] : []), { label: 'Matchup details' }]
  }
  const team = matchPath('/league/teams/:id/:slug?', pathname)
  if (team) return [...parents,
    { label: 'Team details', entity: { kind: 'team', id: team.params.id! } }]
  const archive = matchPath('/research/archive/:workspace/:view?', pathname)
  if (archive) {
    const labels: Record<string, Record<string, string>> = {
      intelligence: { players: 'Players', 'league-impact': 'League impact', research: 'Research' },
      league: { overview: 'Overview', matchups: 'Matchups', players: 'Players', wire: 'Waiver Wire', draft: 'Draft recap' },
    }
    const { workspace, view } = archive.params
    if (workspace && labels[workspace] && (!view || labels[workspace][view])) {
      return [...parents,
        { label: workspace === 'league' ? 'Archived league' : 'Archived intelligence', ...(view ? { href: `/research/archive/${workspace}` } : {}) },
        ...(view ? [{ label: labels[workspace][view] }] : [])]
    }
  }
  if (navigation?.page.path === pathname.replace(/\/$/, '')) {
    return [parents[0], { label: navigation.page.title }]
  }
  return pathname === '/' ? [] : [{ label: 'Page not found' }]
}
