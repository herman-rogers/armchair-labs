/**
 * Navigation conventions — the one way this app tracks "where am I?".
 *
 * The URL is the single source of truth for navigation, and React Router owns the URL.
 * That gives refresh, shared links, new tabs and the Back/Forward buttons the same view
 * the reader was looking at.
 *
 * 1. Places are paths. A page or profile section is a route in `routes.tsx`, reached
 *    with `<Link>`/`<NavLink>` (never a button that flips component state). Page navigation renders `<NavLink>`s, which mark the active one with `aria-current="page"`.
 * 2. View configuration is search params. Filters, sort, page, selected week or team:
 *    read and write them with `useUrlState`, `useUrlPage` or `useUrlParams` below.
 *    These push a history entry, so Back undoes them one step at a time. Free-text
 *    inputs pass `{ replace: true }` so each keystroke does not become a Back step.
 *    One interaction makes one navigation: when a change also resets other params
 *    (filters reset the page), declare them with `resets` or use `useUrlParams` — two
 *    separate setter calls in the same event would overwrite each other. A panel that
 *    shares its page with others prefixes its params (`exp_`, `report_`) so they don't
 *    collide.
 * 3. Component state is only for ephemeral UI that nobody would link to: an expanded
 *    table row, an open `<details>`, a "show all rows" truncation toggle, an export
 *    error. (The theme is a stored preference, not navigation.)
 * 4. Scroll position belongs to `<ScrollRestoration>` in the root layout. Search-param
 *    updates keep the reader where they are (`preventScrollReset`); a new path starts at
 *    the top; Back/Forward restore the saved position. The one addition: a profile
 *    section path (`/players/:id/stats`) scrolls to that section on the profile page.
 *
 * Do not pass `location.state` to remember a previous page, and do not call
 * `window.history` or `window.location` directly; the browser history already does it.
 */
import { createSearchParams, useSearchParams, type To } from 'react-router'

type ParamValue = string | number | boolean | null | undefined

/** Path plus search params; `null`, `undefined`, `''` and `false` values are left out. */
export function to(pathname: string, params: Record<string, ParamValue> = {}): To {
  const search = createSearchParams(Object.entries(params)
    .filter(([, value]) => value != null && value !== '' && value !== false)
    .map(([key, value]): [string, string] => [key, String(value)])).toString()
  return search ? { pathname, search: `?${search}` } : { pathname }
}

/** Player profile sections, each addressable as `/players/:playerId/:section`. */
export const PROFILE_SECTIONS = [['overview', 'Overview'], ['stats', 'Player stats'], ['history', 'Season history'], ['role', 'Role & participation'],
  ['similar', 'Similar careers'], ['tracking', 'Next Gen Stats'], ['forecasts', 'Forecasts'], ['evidence', 'Sources & gaps']] as const
export type ProfileSection = typeof PROFILE_SECTIONS[number][0]
export const isProfileSection = (value: string | undefined): value is ProfileSection => PROFILE_SECTIONS.some(([id]) => id === value)

export const playerPath = (playerId: string, section?: ProfileSection) =>
  `/players/${encodeURIComponent(playerId)}${section ? `/${section}` : ''}`
export const collegePath = (collegeId: string, section?: ProfileSection) =>
  `/college/${encodeURIComponent(collegeId)}${section ? `/${section}` : ''}`
export const intelligencePath = (view: 'rankings' | 'teams' | 'qb-passing' | 'rookies') => `/intelligence/${view}`
export const leaguePath = (view: 'overview' | 'matchups' | 'teams' | 'free-agents' | 'transactions' | 'draft') => `/league/${view}`
/** Stable team identity plus a readable, rename-safe URL. */
export const teamPath = (team: { team_id: number; team_name: string }) => {
  const slug = team.team_name.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'team'
  return `${leaguePath('teams')}/${team.team_id}/${slug}`
}
export const matchupPath = (week: number, homeId: number, awayId: number) => `/league/matchups/${week}/${homeId}/${awayId}`
export const leagueWeekPath = (week: number) => `/league/overview?week=${week}#matchups`
export const researchPath = (view: 'evidence' | 'qb-experiments' | 'forecasts' | 'archive') => `/research/${view}`
/** Views of the archived workspaces, each a route under `/research/archive/<workspace>/:view`. */
export const ARCHIVED_VIEWS = {
  intelligence: ['players', 'league-impact', 'research'],
  league: ['overview', 'matchups', 'players', 'wire', 'draft'],
} as const
export const isArchivedView = <W extends keyof typeof ARCHIVED_VIEWS>(workspace: W, view: string | undefined): view is typeof ARCHIVED_VIEWS[W][number] =>
  (ARCHIVED_VIEWS[workspace] as readonly string[]).includes(view ?? '')
export const archivePath = <W extends keyof typeof ARCHIVED_VIEWS>(workspace: W, view?: typeof ARCHIVED_VIEWS[W][number]) =>
  `/research/archive/${workspace}${view ? `/${view}` : ''}`

export type NavigateOptions = { replace?: boolean }

/**
 * Update several search params in one navigation. Values equal to `null`, `undefined`
 * or `''` are removed, so defaults stay out of the URL.
 *
 * Built on React Router's `setSearchParams`. It commits synchronously (`flushSync`):
 * inputs bound to the URL must show each keystroke at once, or fast typing drops
 * characters. That is also why routes avoid loaders (see `routes.tsx`).
 */
export function useUrlParams() {
  const [params, setParams] = useSearchParams()
  const update = (changes: Record<string, ParamValue>, { replace = false }: NavigateOptions = {}) => {
    setParams(previous => {
      const next = new URLSearchParams(previous)
      for (const [key, value] of Object.entries(changes)) {
        if (value == null || value === '' || value === false) next.delete(key)
        else next.set(key, String(value))
      }
      return next
    }, { replace, preventScrollReset: true, flushSync: true })
  }
  return [params, update] as const
}

/**
 * One search param as state. The fallback is not written to the URL.
 * `resets` names params cleared in the same navigation (usually `['page']`).
 */
export function useUrlState(key: string, fallback: string, { replace = false, resets = [] }: NavigateOptions & { resets?: string[] } = {}) {
  const [params, update] = useUrlParams()
  const set = (value: string) =>
    update({ ...Object.fromEntries(resets.map(name => [name, null])), [key]: value === fallback ? null : value }, { replace })
  return [params.get(key) ?? fallback, set] as const
}

/** A numeric search param; invalid or missing values read as the fallback. */
export function useUrlNumber(key: string, fallback: number, options: NavigateOptions & { resets?: string[] } = {}) {
  const [value, set] = useUrlState(key, String(fallback), options)
  const parsed = Number(value)
  return [Number.isFinite(parsed) ? parsed : fallback, (next: number) => set(String(next))] as const
}

/** A boolean search param stored as `key=1`. */
export function useUrlFlag(key: string, options: NavigateOptions & { resets?: string[] } = {}) {
  const [value, set] = useUrlState(key, '', options)
  return [value === '1', (next: boolean) => set(next ? '1' : '')] as const
}

export type UrlPage = { param: string; offset: number; limit: number; onPage: (offset: number) => void }

/**
 * Pagination stored as a 1-based `page` search param. Pass the result straight to
 * `<DataTable pagination>`; the table clears it when its URL sort changes.
 */
export function useUrlPage(limit: number, param = 'page'): UrlPage {
  const [params, update] = useUrlParams()
  const page = Math.max(1, Number.parseInt(params.get(param) ?? '1', 10) || 1)
  return { param, limit, offset: (page - 1) * limit,
    onPage: offset => update({ [param]: offset > 0 ? Math.floor(offset / limit) + 1 : null }) }
}
