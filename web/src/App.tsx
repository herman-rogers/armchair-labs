import { isRouteErrorResponse, Link, Outlet, ScrollRestoration, useRouteError, useLocation } from 'react-router'
import { DataReleaseProvider } from './components/DataRelease'
import { useQuery } from '@tanstack/react-query'
import { statusQuery } from './api/queries'
import { useEffect } from 'react'
import { PageNavigation } from './components/PageNavigation'
import { useCurrentPage } from './pageNavigation'
import { ThemeToggle } from './components/ThemeToggle'

/** Root layout for every route in `routes.tsx`. */
export default function App() {
  const status = useQuery(statusQuery())
  const { pathname } = useLocation()
  const page = useCurrentPage()
  // Team routes supply the selected team's name for meaningful browser history.
  useEffect(() => { if (page?.path !== '/league/teams') document.title = `${page?.title ?? 'Player analytics'} · Sweaty Plays` }, [page?.title, page?.path])

  return <div className="app instrument-dashboard">
    <a className="skip-link" href="#page-content">Skip to content</a>
    <header className="masthead">
      <div className="masthead-row">
        <div className="dashboard-identity"><span className="eyebrow">League analytics</span>
          <h1><Link to="/">Sweaty Plays</Link></h1></div>
        <div className="masthead-actions">
          <div className="dashboard-meta">
            <span>{status.data?.league.draft_season ?? '—'} season</span>
            <span>{status.data?.league.team_count ?? '—'} teams</span>
            <span>Full PPR · league scoring</span>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
    {status.isError && <div className="notice"><h2>Release status unavailable</h2><p>{status.error.message}</p></div>}
    <DataReleaseProvider><div className="page-shell">
      <PageNavigation key={pathname} />
      <div id="page-content" className="page-content" tabIndex={-1}><Outlet /></div>
    </div></DataReleaseProvider>
    <ScrollRestoration />
  </div>
}

export function NotFound() {
  return <main className="notice"><h2>Page not found</h2><Link to="/">Back to dashboard</Link></main>
}

/** Route error boundary: a failing page shows an error inside the app shell, not a blank screen. */
export function RouteError() {
  const error = useRouteError()
  const message = isRouteErrorResponse(error) ? `${error.status} ${error.statusText}` : error instanceof Error ? error.message : 'Unknown error'
  return <main className="notice" role="alert"><h2>This page could not be displayed</h2><p>{message}</p><Link to="/">Back to dashboard</Link></main>
}
