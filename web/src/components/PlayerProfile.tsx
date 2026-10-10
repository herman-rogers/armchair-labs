import { NavLink, useLocation, useNavigationType } from 'react-router'
import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { collegeQuery, leagueQuery, playerForecastsQuery, playerQuery } from '../api/queries'
import { describeHorizon, type PlayerForecast } from '../api/chairstats'
import type { CollegeSeasonRow, SeasonRow, WeekRow } from '../api/tables'
import { InjuryNotes } from './LeagueAttention'
import { PROFILE_SECTIONS, collegePath, playerPath, useUrlState, type ProfileSection } from '../navigation'
import { PlayerLink } from './PlayerLink'
import { DataTable, type Column } from './DataTable'
import { PlayerPassing } from './QBPassing'
import { useDataRelease } from '../dataRelease'
import { QueryError } from './Controls'
import { fixed, grouped } from '../format'

/**
 * The profile is one long page; a section path scrolls to that section. Following a
 * section link (or arriving on a section URL) scrolls to it. Back/Forward onto content
 * that is already rendered leaves the position <ScrollRestoration> restored.
 */
function useScrollToSection(section: ProfileSection | undefined, ready: boolean) {
  const navigationType = useNavigationType()
  const wasReady = useRef(ready)
  useEffect(() => {
    const arrived = ready && !wasReady.current
    wasReady.current = ready
    if (!ready || !section) return
    if (navigationType === 'POP' && !arrived) return
    document.getElementById(`profile-${section}`)?.scrollIntoView()
  }, [section, ready, navigationType])
}

const seasonColumns: Column<SeasonRow>[] = [
  { key: 'season', label: 'Season', title: 'NFL regular season.', initial: 'asc' },
  { key: 'team', label: 'Team', title: 'Recorded team.', align: 'left', render: r => String(r.team ?? '—') },
  { key: 'position', label: 'Pos', title: 'Recorded position.', align: 'left', render: r => String(r.position ?? '—') },
  { key: 'games', label: 'Games', title: 'Games with recorded statistics.' },
  { key: 'attempts', label: 'Att', title: 'Pass attempts.' },
  { key: 'passing_yards', label: 'Pass yds', title: 'Passing yards.', render: r => grouped(r.passing_yards) },
  { key: 'passing_tds', label: 'Pass TD', title: 'Passing touchdowns.' },
  { key: 'carries', label: 'Car', title: 'Rushing carries.' },
  { key: 'rushing_yards', label: 'Rush yds', title: 'Rushing yards.', render: r => grouped(r.rushing_yards) },
  { key: 'rushing_tds', label: 'Rush TD', title: 'Rushing touchdowns.' },
  { key: 'targets', label: 'Tgt', title: 'Receiving targets.' },
  { key: 'receptions', label: 'Rec', title: 'Receptions.' },
  { key: 'receiving_yards', label: 'Rec yds', title: 'Receiving yards.', render: r => grouped(r.receiving_yards) },
  { key: 'receiving_tds', label: 'Rec TD', title: 'Receiving touchdowns.' },
  { key: 'total_pts', label: 'League pts', title: 'League-scored fantasy points, including configured bonuses.', render: r => fixed(r.total_pts) },
  { key: 'ppg', label: 'Pts / game', title: 'League points per recorded game.', render: r => fixed(r.ppg) },
]
const weekColumns: Column<WeekRow>[] = [
  { key: 'season', label: 'Season', title: 'NFL season.', initial: 'asc' },
  { key: 'week', label: 'Week', title: 'Regular-season week.', initial: 'asc' },
  { key: 'team', label: 'Team', title: 'Recorded team.', align: 'left', render: r => r.team ?? '—' },
  { key: 'opponent_team', label: 'Opp', title: 'Opponent.', align: 'left', render: r => r.opponent_team ?? '—' },
  { key: 'attempts', label: 'Att', title: 'Pass attempts.' },
  { key: 'passing_yards', label: 'Pass yds', title: 'Passing yards.' },
  { key: 'passing_tds', label: 'Pass TD', title: 'Passing touchdowns.' },
  { key: 'carries', label: 'Car', title: 'Carries.' },
  { key: 'rushing_yards', label: 'Rush yds', title: 'Rushing yards.' },
  { key: 'targets', label: 'Tgt', title: 'Targets.' },
  { key: 'receptions', label: 'Rec', title: 'Receptions.' },
  { key: 'receiving_yards', label: 'Rec yds', title: 'Receiving yards.' },
  { key: 'league_points', label: 'League pts', title: 'League-scored fantasy points for the game.', render: r => fixed(r.league_points) },
]
const forecastColumns: Column<PlayerForecast>[] = [
  { key: 'campaign_name', label: 'Campaign', title: 'Campaign bundle that produced the forecast.', align: 'left', initial: 'asc' },
  { key: 'model_id', label: 'Model', title: 'Model inside the campaign.', align: 'left' },
  { key: 'horizon', label: 'Horizon', title: 'Forecast horizon.', align: 'left', render: r => describeHorizon(r.horizon) },
  { key: 'vintage', label: 'As of', title: 'Season and observation cutoff week of the saved forecast.', align: 'left', value: r => r.season * 100 + r.origin_week, render: r => `${r.season} · week ${r.origin_week}` },
  { key: 'overall_rank', label: 'Overall rank', title: 'Rank across every position in that campaign and horizon.', initial: 'asc' },
  { key: 'position_rank', label: 'Position rank', title: 'Rank within the model.', initial: 'asc', render: r => r.position_rank == null ? '—' : `${r.position}${r.position_rank}` },
  { key: 'prediction', label: 'Points', title: 'Predicted fantasy points over the horizon.', render: r => fixed(r.prediction) },
  { key: 'points_per_game', label: 'Per game', title: 'Horizon points divided by scheduled games.', render: r => fixed(r.points_per_game) },
  { key: 'candidates', label: 'Candidates', title: 'Players the model forecast in that vintage.' },
]

function CollegeHistory({ collegeId }: { collegeId: string }) {
  const { token } = useDataRelease()
  const query = useQuery(collegeQuery(collegeId, token))
  const columns: Column<CollegeSeasonRow>[] = [
    { key: 'season', label: 'Season', title: 'College season.', initial: 'asc' },
    { key: 'teams', label: 'Teams', title: 'Teams that season; transfers list more than one.', align: 'left', render: r => Array.isArray(r.teams) ? r.teams.join(', ') : String(r.teams ?? '—') },
    { key: 'college_position', label: 'Pos', title: 'Recorded college position.', align: 'left', render: r => String(r.college_position ?? '—') },
    { key: 'observed_stat_games', label: 'Games', title: 'Games with captured statistics.' },
    { key: 'passing_yards', label: 'Pass yds', title: 'Passing yards.', render: r => grouped(r.passing_yards) },
    { key: 'rushing_yards', label: 'Rush yds', title: 'Rushing yards.', render: r => grouped(r.rushing_yards) },
    { key: 'receptions', label: 'Rec', title: 'Receptions.', render: r => fixed(r.receptions, 0) },
    { key: 'receiving_yards', label: 'Rec yds', title: 'Receiving yards.', render: r => grouped(r.receiving_yards) },
    { key: 'receiving_tds', label: 'Rec TD', title: 'Receiving touchdowns.', render: r => fixed(r.receiving_tds, 0) },
    { key: 'coverage', label: 'Coverage', title: 'Captured team games over scheduled games; not player availability.', render: r => typeof r.coverage === 'number' ? `${fixed(r.coverage * 100, 0)}%` : '—' },
  ]
  return <>
    <QueryError query={query} label="College history unavailable" />
    {query.data && <DataTable rows={query.data.seasons} columns={columns} defaultSort="season" rowKey={r => `${r.college_id}-${r.season}`} emptyMessage="No captured college seasons." />}
  </>
}

export function PlayerProfile({ playerId, collegeId, section }: { playerId?: string; collegeId?: string; section?: ProfileSection }) {
  const { token } = useDataRelease()
  const location = useLocation()
  const [season, setSeason] = useUrlState('season', '')
  const player = useQuery({ ...playerQuery(playerId ?? '', season, token), enabled: !!playerId })
  const forecasts = useQuery({ ...playerForecastsQuery(playerId ?? ''), enabled: !!playerId })
  const league = useQuery({ ...leagueQuery(token), enabled: !!playerId })
  const currentPlayer = league.data?.players.find(p => p.player_id === playerId)
  useScrollToSection(section, player.isSuccess || !!collegeId)
  const name = player.data?.player.display_name
  useEffect(() => {
    const previous = document.title
    if (name) document.title = `${name} · Sweaty Plays`
    return () => { document.title = previous }
  }, [name])
  if (collegeId && !playerId) {
    return <section className="player-profile" aria-label="College profile">
      <div className="profile-heading"><div><span className="eyebrow">College profile</span><h3>{collegeId}</h3></div></div>
      <CollegeHistory collegeId={collegeId} />
    </section>
  }
  if (player.isError) return <p className="notice">{player.error.message}</p>
  if (!player.data) return <p role="status">Loading player…</p>
  const data = player.data, p = data.player
  const seasons = [...new Set(data.weeks.map(w => w.season))].sort((a, b) => b - a)
  const linked = data.college.filter(c => c.status === 'linked')
  const sectionLink = (id: ProfileSection) => ({ pathname: playerId ? playerPath(playerId, id) : collegePath(collegeId ?? '', id), search: location.search })
  return <section className="player-profile" aria-label={`${p.display_name} player profile`}>
    <div className="profile-heading"><div><span className="eyebrow">Player profile</span><h3>{p.display_name}</h3>
      <p>{p.draft_year ? `Drafted ${p.draft_year}${p.draft_round ? ` · round ${p.draft_round}` : ''}${p.draft_pick ? ` · pick ${p.draft_pick}` : ''}${p.draft_team ? ` · ${p.draft_team}` : ''}` : p.rookie_season ? `NFL entry ${p.rookie_season}` : 'Draft history unknown'}{p.birth_date ? ` · born ${p.birth_date}` : ''}</p></div>
      <label>Game log season<select aria-label="Game log season" value={season} onChange={e => setSeason(e.target.value)}><option value="">All seasons</option>{seasons.map(y => <option key={y} value={y}>{y}</option>)}</select></label>
    </div>
    <p className="legend">NFL observations through {data.observation_cutoff.season ?? '—'}, Week {data.observation_cutoff.through_week ?? '—'}. Forecasts come from every published campaign; none is selected over another.</p>
    <nav className="profile-section-links" aria-label="Player profile sections">{PROFILE_SECTIONS.map(([id, label]) =>
      <NavLink key={id} to={sectionLink(id)} preventScrollReset>{label}</NavLink>)}</nav>
    <section id="profile-overview" className="profile-section" aria-labelledby="profile-overview-heading"><h3 id="profile-overview-heading">Overview</h3>
      {currentPlayer && <p>{currentPlayer.owner_team_name ? `Rostered by ${currentPlayer.owner_team_name}` : 'Free agent'} · {currentPlayer.lineup_slot ?? 'no slot'} · ESPN status {currentPlayer.injury_status ?? 'not reported'}</p>}
      {currentPlayer && <InjuryNotes player={currentPlayer} />}
      <dl className="profile-facts">
        <div><dt>Seasons observed</dt><dd>{data.seasons.length}</dd></div>
        <div><dt>Games observed</dt><dd>{data.weeks.length}</dd></div>
        <div><dt>Latest team</dt><dd>{data.weeks.at(-1)?.team ?? '—'}</dd></div>
        <div><dt>College links</dt><dd>{linked.length}</dd></div>
      </dl>
    </section>
    <section id="profile-forecasts" className="profile-section" aria-labelledby="profile-forecasts-heading"><h3 id="profile-forecasts-heading">Forecasts</h3>
      <QueryError query={forecasts} label="Campaign forecasts unavailable" />
      {forecasts.data && <DataTable rows={forecasts.data.forecasts} columns={forecastColumns} defaultSort="campaign_name" rowKey={r => `${r.campaign_id}-${r.model_id}`} emptyMessage="No campaign has a saved forecast for this player."
        renderDetails={r => <div className="player-details"><dl className="profile-facts">{Object.entries(r.predictions).map(([target, value]) => <div key={target}><dt>{target.replaceAll('_', ' ')}</dt><dd>{fixed(value)}</dd></div>)}</dl>
          <p className="legend">{r.candidate_basis ?? 'Candidate'} · status {r.availability_status ?? 'unknown'}{r.availability_as_of ? ` as of ${new Date(r.availability_as_of).toLocaleDateString()}` : ''}.</p></div>} />}
      {playerId && <PlayerPassing playerId={playerId} />}
    </section>
    <section id="profile-history" className="profile-section" aria-labelledby="profile-history-heading"><h3 id="profile-history-heading">Season history</h3>
      <DataTable rows={data.seasons} columns={seasonColumns} defaultSort="season" rowKey={r => r.season} emptyMessage="No completed NFL seasons observed." />
    </section>
    <section id="profile-games" className="profile-section" aria-labelledby="profile-games-heading"><h3 id="profile-games-heading">Game log</h3>
      <DataTable rows={data.weeks} columns={weekColumns} defaultSort="season" rowKey={r => `${r.season}-${r.week}`} emptyMessage="No recorded games." />
    </section>
    <section id="profile-college" className="profile-section" aria-labelledby="profile-college-heading"><h3 id="profile-college-heading">College</h3>
      {linked.length ? linked.map(c => <div key={c.college_id}><p><PlayerLink collegeId={c.college_id}>{c.college_name ?? c.college_id}</PlayerLink> · {c.first_college_season ?? '?'}–{c.last_college_season ?? '?'} · {c.method?.replaceAll('_', ' ') ?? 'linked'}{c.evidence ? ` · ${c.evidence}` : ''}</p><CollegeHistory collegeId={c.college_id} /></div>)
        : <p>No accepted college identity link. Missing is not zero.</p>}
    </section>
  </section>
}
