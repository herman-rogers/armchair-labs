import { useState } from 'react'
import { Link, useLocation } from 'react-router'
import { PAGE_GROUPS } from '../pageNavigation'

export function PageNavigation() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  const active = (path: string) => pathname === path || pathname.startsWith(`${path}/`) || (path === '/league/overview' && pathname.startsWith('/league/matchups/'))
  return <aside className="page-sidebar">
    <button className="button navigation-toggle" type="button" aria-expanded={open} aria-controls="page-navigation" onClick={() => setOpen(!open)}>Browse pages <span aria-hidden="true">{open ? '−' : '+'}</span></button>
    <nav id="page-navigation" className={open ? 'page-navigation is-open' : 'page-navigation'} aria-label="Main navigation">
      {PAGE_GROUPS.map(group => <section key={group.label}>
        <h2>{group.label}</h2>
        <ul>{group.pages.map(page => <li key={page.path}><Link to={page.path} className={active(page.path) ? 'active' : undefined} aria-current={active(page.path) ? 'page' : undefined} onClick={() => setOpen(false)}>{page.title}</Link></li>)}</ul>
      </section>)}
    </nav>
  </aside>
}
