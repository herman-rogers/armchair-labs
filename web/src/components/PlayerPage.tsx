import { Link, Navigate, useLocation, useParams } from 'react-router'
import { isProfileSection, collegePath, playerPath, type ProfileSection } from '../navigation'
import { PlayerProfile } from './PlayerProfile'

/** `/players/:playerId/:section?` and `/college/:collegeId/:section?`. */
export function PlayerPage() {
  const { playerId, collegeId, section } = useParams()
  const { hash, search } = useLocation()
  // Bookmarks from before sections were paths (`#profile-stats`) move to the section path.
  const legacySection = hash.replace(/^#profile-/, '')
  const profilePath = (to?: ProfileSection) => playerId ? playerPath(playerId, to) : collegePath(collegeId!, to)
  if (!section && hash.startsWith('#profile-') && isProfileSection(legacySection)) {
    return <Navigate replace to={{ pathname: profilePath(legacySection), search }} />
  }
  // Unknown sections fall back to the top of the profile, keeping view params.
  if (section && !isProfileSection(section)) return <Navigate replace to={{ pathname: profilePath(), search }} />
  return <main className="player-page">
    <nav aria-label="Breadcrumb"><Link to="/">← Dashboard</Link></nav>
    <PlayerProfile key={playerId ?? collegeId} playerId={playerId} collegeId={collegeId} section={isProfileSection(section) ? section : undefined} />
  </main>
}
