import { NavLink, useLocation, useNavigationType } from 'react-router'
import { profileQuery, leagueQuery, directoryQuery, filterDirectory, prefetchProfile } from '../api/queries'
import { InjuryNotes } from './LeagueAttention'
import { PROFILE_SECTIONS, collegePath, playerPath, useUrlPage, useUrlState, type ProfileSection } from '../navigation'
import { PlayerLink } from './PlayerLink'
import { PlayerStats, PlayerStatDefinitions } from './PlayerStats'
import { useEffect, useId, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { CollegeSeason, NFLSeason, PlayerProfileData, Production, ProfileIdentity, TrackingMetric, TrackingSeason } from '../api/profiles'
import type { PlayerFilters } from '../playerFilters'
import { DataTable, type Column } from './DataTable'
import { SimilarCareers } from './SimilarCareers'
import { PlayerRanking } from './PlayerRanking'
import { PlayerPassing } from './QBPassing'
import { useDataRelease } from '../dataRelease'
import { Pager, QueryError } from './Controls'
import { fixed, percent } from '../format'

const roles: Record<string, string> = { high: 'High participation', rotation: 'Rotational participation', limited: 'Limited participation', unknown: 'Participation unknown' }

const trajectoryLabels = { points_per_observed_week: 'Points / observed week', snap_share: 'Offensive snap share', targets_per_observed_week: 'Targets / observed week', carries_per_observed_week: 'Carries / observed week' }
type TrajectoryMetric = keyof typeof trajectoryLabels

function Trajectory({ seasons }: { seasons: NFLSeason[] }) {
  const [requested, setMetric] = useUrlState('trajectory', 'points_per_observed_week')
  const metric: TrajectoryMetric = requested in trajectoryLabels ? requested as TrajectoryMetric : 'points_per_observed_week'
  const labelId = useId()
  const labels = trajectoryLabels
  const valid = seasons.map(s => s[metric]).filter((v): v is number => v != null)
  const maximum = Math.max(...valid, 1)
  const minimum = Math.min(...valid, 0)
  const x = (i: number) => 48 + i * 630 / Math.max(seasons.length - 1, 1)
  const y = (v: number) => 140 - (v - minimum) / (maximum - minimum) * 110
  return <div className="profile-trajectory">
    <label htmlFor={labelId}>Career trajectory</label>
    <select id={labelId} value={metric} onChange={e => setMetric(e.target.value)}>
      {Object.entries(labels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select>
    {valid.length ? <svg viewBox="0 0 720 178" role="img" aria-label={`${labels[metric]} by NFL season; values also appear in the season history`}>
      <line x1="48" x2="685" y1={y(0)} y2={y(0)} className="profile-axis" />
      <text x="4" y="32">{metric === 'snap_share' ? percent(maximum, 0, 'Unknown') : fixed(maximum)}</text>
      {seasons.map((s, i) => {
        const value = s[metric], prior = i > 0 ? seasons[i - 1][metric] : null
        return <g key={s.season}>
          {value != null && prior != null && s.season === seasons[i - 1].season + 1 && <line className="profile-trend-line" x1={x(i - 1)} y1={y(prior)} x2={x(i)} y2={y(value)} />}
          {value != null && <circle className={s.partial ? 'profile-dot partial' : 'profile-dot'} cx={x(i)} cy={y(value)} r="5"><title>{s.season}: {metric === 'snap_share' ? percent(value, 0, 'Unknown') : fixed(value)} · {s.teams.join(', ')} · {s.observed_weeks} observed weeks{s.partial ? ' · partial season' : ''}</title></circle>}
          <text x={x(i)} y="166" textAnchor="middle">{s.season}{s.partial ? '*' : ''}</text>
        </g>
      })}
    </svg> : <p>No observations for this measure.</p>}
    <p className="legend">* Partial season. Gaps stay unknown. Participation and production describe the recorded role; they do not isolate talent.</p>
  </div>
}

function productionColumns<T extends Production>(data: PlayerProfileData): Column<T>[] {
  return [
    { key: 'observed_weeks', label: 'Weeks', title: data.metrics.observed_weeks.definition },
    { key: 'points_per_observed_week', label: 'Pts / week', title: data.metrics.points_per_observed_week.definition, render: r => fixed(r.points_per_observed_week) },
    { key: 'targets_per_observed_week', label: 'Targets / week', title: data.metrics.targets_per_observed_week.definition, render: r => fixed(r.targets_per_observed_week) },
    { key: 'carries_per_observed_week', label: 'Carries / week', title: data.metrics.carries_per_observed_week.definition, render: r => fixed(r.carries_per_observed_week) },
    { key: 'snap_share', label: 'Snap share', title: data.metrics.snap_share.definition, render: r => percent(r.snap_share, 0, 'Unknown') },
    { key: 'snap_observations', label: 'Snap samples', title: 'Weeks with recorded offensive snap share; missing weeks are not zero shares.' },
  ]
}

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
    const target = document.getElementById(`profile-${section}`)
    if (!target?.parentElement) return
    target.scrollIntoView()
    // Sections above (player stats, similar careers) load separately and push this one
    // down. Keep it in place while they settle, until the reader scrolls themselves.
    const keep = new ResizeObserver(() => target.scrollIntoView())
    keep.observe(target.parentElement)
    const stop = () => keep.disconnect()
    const timer = window.setTimeout(stop, 3000)
    const events = ['wheel', 'touchstart', 'keydown'] as const
    events.forEach(name => window.addEventListener(name, stop, { once: true, passive: true }))
    return () => { stop(); window.clearTimeout(timer); events.forEach(name => window.removeEventListener(name, stop)) }
  }, [section, ready, navigationType])
}

export function PlayerProfile({ playerId, collegeId, section }: { playerId?: string; collegeId?: string; section?: ProfileSection }) {
  const { token } = useDataRelease()
  const location = useLocation()
  const [season, setSeason] = useUrlState('season', '')
  const params = new URLSearchParams({ ...(playerId ? { player_id: playerId } : { college_id: collegeId ?? '' }), ...(season ? { season, week: '18' } : {}) })
  const query = useQuery(profileQuery(params, token))
  const league = useQuery({ ...leagueQuery(token), enabled: !season && !!playerId })
  const client = useQueryClient()
  const requestKey = params.toString()
  const period = new URLSearchParams(location.search).get('period') ?? 'prior'
  const horizon = new URLSearchParams(location.search).get('horizon') ?? 'rest_of_season'
  useEffect(() => { prefetchProfile(client, new URLSearchParams(requestKey), token, period, horizon) }, [client, requestKey, token, period, horizon])
  const currentPlayer = !season ? league.data?.players.find(player => player.player_id === playerId) : undefined
  useScrollToSection(section, query.isSuccess)
  useEffect(() => {
    const previous = document.title
    if (query.data) document.title = `${query.data.identity.player_display_name} · Sweaty Plays`
    return () => { document.title = previous }
  }, [query.data])
  if (query.isError) return <p className="notice">{query.error.message}</p>
  if (!query.data) return <p role="status">Loading player career and evidence…</p>
  const data = query.data, p = data.identity
  const current = data.nfl_seasons.find(row => row.season === data.cutoff.season && row.partial)
  const trackingMetrics = data.tracking ? (Object.keys(data.tracking.metrics) as TrackingMetric[]).filter(key =>
    data.tracking!.metrics[key].positions.includes(p.position ?? '') || data.tracking_seasons.some(row => row[key] != null)) : []
  const trackingColumns: Column<TrackingSeason>[] = [
    { key: 'season', label: 'Season', title: 'Completed regular season; no partial-season tracking is imputed.', initial: 'asc' },
    ...trackingMetrics.map(key => ({ key, label: data.tracking!.metrics[key].label,
      title: `${data.tracking!.metrics[key].definition} Unit: ${data.tracking!.metrics[key].unit}.`,
      render: (row: TrackingSeason) => row[key] == null ? '—' : `${fixed(row[key], 2)} ${data.tracking!.metrics[key].unit}` })),
    { key: 'coverage', label: 'Coverage', title: 'Available relevant metrics; this is not the tracking play count.',
      render: row => `${trackingMetrics.filter(key => row[key] != null).length} / ${trackingMetrics.length} measures` },
  ]
  const nflColumns: Column<NFLSeason>[] = [
    { key: 'season', label: 'Season', title: 'NFL regular season; current partial seasons are labeled.', initial: 'asc', render: r => `${r.season}${r.partial ? ' · partial' : ''}` },
    { key: 'teams', label: 'Teams', title: 'Teams observed in this season, in week order.', align: 'left', render: r => r.teams.join(' → ') || 'Unknown' },
    ...productionColumns<NFLSeason>(data),
    { key: 'league_points', label: 'Total points', title: data.metrics.league_points.definition, render: r => fixed(r.league_points) },
  ]
  const collegeColumns: Column<CollegeSeason>[] = [
    { key: 'season', label: 'Season', title: 'College season including bowl games.', initial: 'asc' },
    { key: 'college_team', label: 'School / stint', title: 'Transfers remain separate observations.', align: 'left' },
    { key: 'passing_yards', label: 'Pass yards', title: 'Recorded passing yards; check coverage.' },
    { key: 'rushing_yards', label: 'Rush yards', title: 'Recorded rushing yards; check coverage.' },
    { key: 'receiving_yards', label: 'Rec yards', title: 'Recorded receiving yards; check coverage.' },
    { key: 'share_receiving_yards', label: 'Rec yard share', title: 'Share of captured team receiving yards; incomplete team data may distort it.', render: r => percent(r.share_receiving_yards, 0, 'Unknown') },
    { key: 'coverage', label: 'Coverage', title: 'Captured team games / scheduled completed team games; not player availability.', render: r => percent(r.coverage, 0, 'Unknown') },
    { key: 'complete_team_season', label: 'Quality', title: 'Complete schedule and reconciled team totals.', render: r => r.complete_team_season ? 'Passed' : 'Incomplete / review' },
  ]
  const years = Array.from({ length: Math.max(0, data.available_through.season - 2001) }, (_, i) => 2001 + i).reverse()
  return <section className="player-profile" aria-label={`${p.player_display_name} player profile`}>
    <div className="profile-heading"><div><span className="eyebrow">Player profile</span><h3>{p.player_display_name}</h3>
      <p>{p.position ?? 'Position unknown'}{p.rookie_season ? ` · NFL entry ${p.rookie_season}` : ''}{p.draft_pick ? ` · Draft pick ${p.draft_pick}` : ''}</p></div>
      <label>History through<select aria-label="Profile history through" value={season} onChange={e => setSeason(e.target.value)}><option value="">Latest captured observations</option>{years.map(y => <option key={y} value={y}>{y} season</option>)}</select></label>
    </div>
    <p className="legend">NFL observations through {data.cutoff.season}, Week {data.cutoff.week}. College history through the preceding season. Identity details reflect the captured registry.</p>
    <nav className="profile-section-links" aria-label="Player profile sections">{PROFILE_SECTIONS.filter(([id]) => id !== 'stats' || p.player_id).map(([id, label]) =>
      <NavLink key={id} to={{ pathname: playerId ? playerPath(playerId, id) : collegePath(collegeId ?? '', id), search: location.search }} preventScrollReset>{label}</NavLink>)}</nav>
    <section id="profile-overview" className="profile-section" aria-labelledby="profile-overview-heading"><h3 id="profile-overview-heading">Overview</h3>
      {currentPlayer && <InjuryNotes player={currentPlayer} />}
      <dl className="profile-facts">
        <div><dt>Completed NFL seasons observed</dt><dd>{fixed(data.features.completed_seasons_observed, 0)}</dd></div>
        <div><dt>Career points / observed week</dt><dd>{fixed(data.features.career_points_per_observed_week)}</dd></div>
        <div><dt>Recent three-season scoring average</dt><dd>{fixed(data.features.recent3_points_per_observed_week)}</dd></div>
        <div><dt>College stints captured</dt><dd>{data.college_seasons.length}</dd></div>
      </dl>
      <Trajectory seasons={data.nfl_seasons} />
      <div className="detail-grid"><section><h4>Career changes</h4>{data.transitions.length ? <ol className="profile-events">{data.transitions.map((e, i) => <li key={i}><strong>{e.season} · {e.level}</strong> {e.description}</li>)}</ol> : <p>No observed transitions at this cutoff.</p>}</section>
        <section><h4>Current opportunity</h4>{current ? <>
          <p>{current.season} through Week {current.through_week} · {current.observed_weeks} observed weeks · {percent(current.snap_share, 0, 'Unknown')} offensive snap share.</p>
          <p>{fixed(current.targets_per_observed_week)} targets and {fixed(current.carries_per_observed_week)} carries per observed week.</p>
        </> : <p>No partial-season observations at this cutoff.</p>}
        {data.missing_evidence.slice(0, 3).map(text => <p className="legend" key={text}>{text}</p>)}</section></div>
    </section>
    {p.player_id && <section id="profile-stats" className="profile-section" aria-label="Player stats">
      <PlayerStats key={p.player_id} playerId={p.player_id} name={p.player_display_name} />
    </section>}
    <section id="profile-history" className="profile-section" aria-labelledby="profile-history-heading"><h3 id="profile-history-heading">Season history</h3><h4>NFL seasons</h4><DataTable rows={data.nfl_seasons} columns={nflColumns} defaultSort="season" rowKey={r => r.season} rowLabel={r => `${r.season} season`} emptyMessage="No NFL observations at this cutoff."
        renderDetails={r => <div className="player-details"><h5>{r.season} · production and efficiency</h5>
          <p>{r.positions.join(' / ')} · {r.offensive_weeks ?? 'Unknown'} observed offensive weeks across {r.snap_observations} snap-share records.</p>
          <p>{fixed(r.passing_yards_per_attempt)} passing yards / attempt · {fixed(r.receiving_yards_per_target)} receiving yards / target · {fixed(r.rushing_yards_per_carry)} rushing yards / carry.</p>
          <p>{fixed(r.targets, 0)} recorded targets · {fixed(r.carries, 0)} carries · {fixed(r.attempts, 0)} pass attempts. These describe recorded production and opportunity; incomplete observations remain unknown.</p>
        </div>} />
      <h4>College seasons and transfers</h4><DataTable rows={data.college_seasons} columns={collegeColumns} defaultSort="season" rowKey={r => `${r.college_id}-${r.season}-${r.team_id}`} emptyMessage="No accepted college history at this cutoff. Missing is not zero." /></section>
    <section id="profile-role" className="profile-section" aria-labelledby="profile-role-heading"><h3 id="profile-role-heading">Role & participation</h3><h4>Production in different participation levels</h4><p>High: at least 70% of offensive snaps. Rotational: 35–69%. Limited: below 35%. These describe participation, not starts or routes. Different teammates, ages and matchups also affect production.</p>
      <DataTable rows={data.role_history} columns={[{ key: 'label', label: 'Participation', title: 'Descriptive offensive snap-share band.', align: 'left' }, ...productionColumns<typeof data.role_history[number]>(data)]} defaultSort="observed_weeks" rowKey={r => r.role} />
      <section><h4>Week-by-week role periods</h4><DataTable rows={data.role_periods} columns={[
        { key: 'season', label: 'Season', title: 'NFL season.', initial: 'asc' }, { key: 'start_week', label: 'From', title: 'First observed week in the period.' }, { key: 'end_week', label: 'Through', title: 'Last consecutive observed week. Gaps and team changes split periods.' },
        { key: 'team', label: 'Team', title: 'Observed team.', align: 'left' }, { key: 'role', label: 'Role evidence', title: 'Observed snap-share band.', align: 'left', render: r => roles[r.role] }, ...productionColumns<typeof data.role_periods[number]>(data),
      ]} defaultSort="season" rowKey={r => `${r.season}-${r.start_week}`} emptyMessage="No observed NFL role periods." /></section>
      <section><h4>Recorded injury and practice reports ({data.injury_reports.length})</h4><p>Historical reports describe reported status. Missing reports do not establish health, and no causal injury adjustment is inferred.</p>
        <DataTable rows={data.injury_reports} columns={[{ key: 'season', label: 'Season', title: 'Report season.' }, { key: 'week', label: 'Week', title: 'Report week.' }, { key: 'report_primary_injury', label: 'Reported injury', title: 'Source injury description.', align: 'left' }, { key: 'report_status', label: 'Game status', title: 'Recorded game designation.', align: 'left' }, { key: 'practice_status', label: 'Practice', title: 'Recorded practice status.', align: 'left' }]} defaultSort="season" rowKey={r => `${r.season}-${r.week}-${r.team}-${r.report_status}-${r.practice_status}`} emptyMessage="No captured injury reports." /></section>
    </section>
    <section id="profile-similar" className="profile-section" aria-labelledby="profile-similar-heading"><h3 id="profile-similar-heading">Similar careers</h3>
      {p.player_id ? <SimilarCareers playerId={p.player_id} season={season ? data.cutoff.season : undefined} week={season ? data.cutoff.week : undefined} /> : <p>No linked NFL identity is available for career comparisons.</p>}
    </section>
    <section id="profile-tracking" className="profile-section" aria-labelledby="profile-tracking-heading"><h3 id="profile-tracking-heading">Next Gen Stats</h3><h4>NFL Next Gen Stats</h4>
      <p>{data.tracking?.note ?? 'No tracking summaries were captured for this profile release.'}</p>
      {trackingMetrics.length > 0 && <DataTable rows={data.tracking_seasons} columns={trackingColumns} defaultSort="season" rowKey={row => row.season} emptyMessage="No completed-season tracking summaries at this cutoff." />}
      <p className="legend">— means unknown or unavailable. Values retain the provider's season aggregation; they are not averaged across a career.</p>
    </section>
    <section id="profile-forecasts" className="profile-section" aria-labelledby="profile-forecasts-heading"><h3 id="profile-forecasts-heading">Forecasts</h3>
      {!season && p.player_id && <PlayerRanking playerId={p.player_id} name={p.player_display_name} season={data.cutoff.season} week={data.cutoff.week} />}
      {!season && p.player_id && p.position === 'QB' && <PlayerPassing playerId={p.player_id} />}
      <h4>Preseason reference forecasts</h4>
      <p>Saved estimates for the selected season, using information available at the listed cutoff. These do not update for current-week news or provide rest-of-season, waiver or trade values.</p>
      <DataTable rows={data.approved_forecasts} columns={[
        { key: 'label', label: 'Outcome and model', title: 'Forecast permitted by the current analysis policy for this outcome and position.', align: 'left' },
        { key: 'season', label: 'Season', title: 'Forecast regular season.' },
        { key: 'cutoff', label: 'Information through', title: 'Saved forecast information cutoff.', align: 'left' },
        { key: 'prediction', label: 'Estimate', title: 'Full-season forecast in the listed units.', render: row => fixed(row.prediction) },
        { key: 'unit', label: 'Units', title: 'The outcome units; different outcomes are not combined.', align: 'left' },
        { key: 'actual', label: 'Completed outcome', title: 'Shown only after the entire selected season has completed.', render: row => fixed(row.actual) },
      ]} defaultSort="label" rowKey={row => `${row.season}-${row.target}-${row.model}`} emptyMessage="No approved preseason forecasts at this cutoff. Archived models are available in Research archive." />
      {data.sources.analysis && <p className="legend">Forecast release: {data.sources.analysis}. Historical reference baselines do not establish a market or decision advantage.</p>}
    </section>
    <section id="profile-evidence" className="profile-section" aria-labelledby="profile-evidence-heading"><h3 id="profile-evidence-heading">Sources & gaps</h3>
      <h4>Coverage and identity</h4>{data.missing_evidence.map(text => <p key={text}>{text}</p>)}
      {data.links.map(link => <p key={link.college_id}>College ESPN {link.college_id} · {link.status} · {link.method?.replaceAll('_', ' ') ?? 'No accepted NFL match'}{link.evidence ? ` · ${link.evidence}` : ''}</p>)}

      <PlayerStatDefinitions />
      <section><h4>Measure definitions</h4><dl className="profile-definitions">{Object.entries(data.metrics).map(([key, m]) => <div key={key}><dt>{m.label}</dt><dd>{m.definition}</dd></div>)}</dl></section>
      {data.limits.map(text => <p className="legend" key={text}>{text}</p>)}
      <p className="legend">Captured {new Date(data.sources.observations_saved_at).toLocaleString()}.</p>
      <p className="legend faint">{data.sources.gold && <>Data: {data.sources.gold} · </>}Profile: {data.version} · Analysis: {data.sources.analysis ?? 'Unavailable'} · College: {data.sources.college}</p>
    </section>
  </section>
}

export function PlayerProfileLink({ playerId, collegeId }: { playerId?: string; collegeId?: string }) {
  return <p className="profile-expander"><PlayerLink playerId={playerId} collegeId={collegeId}>Open complete player profile →</PlayerLink></p>
}

export function PlayerProfiles({ filters }: { filters: PlayerFilters }) {
  const { token } = useDataRelease()
  const [scope, setScope] = useUrlState('scope', 'current', { resets: ['page'] })
  const page = useUrlPage(50)
  const query = useQuery({ ...directoryQuery(token), select: data => filterDirectory(data, { ...filters, scope }) })
  const columns: Column<ProfileIdentity>[] = [
    { key: 'player_display_name', label: 'Player', title: 'Open the complete player profile.', align: 'left', initial: 'asc' }, { key: 'position', label: 'Position', title: 'Latest captured fantasy position.', align: 'left' },
    { key: 'rookie_season', label: 'NFL entry', title: 'Recorded NFL entry season.' }, { key: 'nfl_seasons', label: 'NFL seasons observed', title: 'Captured seasons, including a current partial season.' },
    { key: 'college_linked', label: 'College history', title: 'Only accepted identity links join college and NFL records.', align: 'left', render: r => r.college_linked ? 'Linked' : 'No accepted link' },
  ]
  return <section aria-label="Player profiles"><h3>Career history</h3><p>Follow a player from college through NFL seasons, changes in role, and current opportunity.</p>
    <div className="analysis-controls">
      <label>Profile scope<select value={scope} onChange={e => setScope(e.target.value)}><option value="current">Current candidates</option><option value="all">All captured NFL careers</option><option value="college_linked">Linked college / NFL careers</option></select></label>
    </div>
    <QueryError query={query} />{query.isFetching && <p role="status">Loading profiles…</p>}
    {query.data && <><p className="legend">{query.data.total.toLocaleString()} matching players · observations through {query.data.report.season}, Week {query.data.report.through_week}. Select the College → NFL population to browse college careers.</p>
      <DataTable rows={query.data.players} pagination={page} sortParam="sort" columns={columns} defaultSort="player_display_name" rowKey={r => r.player_id ?? r.player_display_name} rowLabel={r => r.player_display_name} emptyMessage="No matching profiles. Try All captured NFL careers." profileHref={r => r.player_id ? playerPath(r.player_id) : undefined} />
      <Pager page={page} total={query.data.total} />
    </>}
  </section>
}
