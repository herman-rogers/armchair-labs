import { Link, Navigate, useLocation, useParams } from 'react-router'
import { isProfileSection, collegePath, playerPath } from '../navigation'
import { PlayerProfile } from './PlayerProfile'

/** `/players/:playerId/:section?` and `/college/:collegeId/:section?`; the router validates `section`. */
export function PlayerPage() {
  const { playerId, collegeId, section } = useParams()
  const { hash, search } = useLocation()
  // Bookmarks from before sections were paths (`#profile-stats`) move to the section path;
  // this is a component redirect because loaders never see the URL hash.
  const legacySection = hash.replace(/^#profile-/, '')
  if (!section && hash.startsWith('#profile-') && isProfileSection(legacySection)) {
    return <Navigate replace to={{ pathname: playerId ? playerPath(playerId, legacySection) : collegePath(collegeId!, legacySection), search }} />
  }
  return <main className="player-page">
    <nav aria-label="Breadcrumb"><Link to="/">← Dashboard</Link></nav>
    <PlayerProfile key={playerId ?? collegeId} playerId={playerId} collegeId={collegeId} section={isProfileSection(section) ? section : undefined} />
  </main>
}
