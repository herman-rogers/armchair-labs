import { PlayerProfileLink } from './PlayerProfile'
import type { PlayerFilters } from '../playerFilters'
import { useQuery } from '@tanstack/react-query'
import { fetchPlayerOutlook } from '../api/client'
import type { OutlookPlayer } from '../api/types'
import { DataTable, type Column } from './DataTable'
import { useUrlState } from '../navigation'
import { useDataRelease } from '../dataRelease'

const number = (v: number | null | undefined) => v == null ? '—' : v.toFixed(1)
const percent = (v: number | null | undefined) => v == null ? 'Unknown' : `${(v * 100).toFixed(0)}%`
const owner = (r: OutlookPlayer) => r.is_mine ? 'My team' : r.owner_team_name
  ?? (r.availability === 'free_agent' ? 'Available' : 'Unknown')
const cohortLabel = (value: string) => value === 'all' ? 'All eligible' : value === 'rookie' ? 'Rookies'
  : value === 'small_prior_sample' ? 'Prior season <10 offensive weeks' : 'Prior season ≥10 offensive weeks'

export function PlayerOutlook({ filters }: { filters: PlayerFilters }) {
  const { token: dataRelease } = useDataRelease()
  const { search, position, population } = filters
  const [ownership, setOwnership] = useUrlState('outlook_roster', 'all')
  const [priorSample, setPriorSample] = useUrlState('outlook_prior', 'all')
  const [coverage, setCoverage] = useUrlState('outlook_coverage', 'all')
  const report = useQuery({ queryKey: ['player-outlook', dataRelease], queryFn: () => fetchPlayerOutlook(dataRelease), retry: false })
  const data = report.data
  const evidence = data?.validation.filter(v => v.cutoff_week === data.through_week
    && (position === 'ALL' || v.position === position)) ?? []
  const rows = (data?.players ?? []).filter(r => (position === 'ALL' || r.position === position)
    && (ownership === 'all' || (ownership === 'mine' ? r.is_mine : r.availability === ownership))
    && (population === 'all' || r.population === population)
    && (priorSample === 'all' || (r.prior_offensive_weeks != null && r.prior_offensive_weeks < 10))
    && (coverage === 'all' || (r.forecast_next4 != null) === (coverage === 'scored'))
    && r.player_display_name.toLowerCase().includes(search.trim().toLowerCase()))
  const columns: Column<OutlookPlayer>[] = [
    { key: 'position_rank', label: 'Pos rank', title: 'Experimental four-week point order within the position, not trade value.', initial: 'asc', render: r => r.position_rank == null ? '—' : `#${r.position_rank}` },
    { key: 'player_display_name', label: 'Player', title: 'Open for usage, volatility, samples, and limitations.', align: 'left' },
    { key: 'position', label: 'Pos', title: 'Fantasy position at the observation cutoff.', align: 'left' },
    { key: 'ownership', label: 'Roster', title: 'Latest saved league ownership; unknown does not mean available.', align: 'left', value: owner, render: owner },
    { key: 'injury_status', label: 'Saved health', title: 'Current saved ESPN context only; not used to adjust this forecast.', align: 'left' },
    { key: 'forecast_next4', label: '4-week pts / baseline', title: 'Exploratory outlook / simple recent-usage model. Both include absences and byes. Higher is not proof of value.', render: r => `${number(r.forecast_next4)} / ${number(r.usage_baseline_next4)}` },
    { key: 'usage_change', label: 'Opportunity trend', title: 'Recent three calendar weeks vs the preceding two, per scheduled game. Carries + targets (QB: attempts + carries). Descriptive, not a probability.', align: 'left', render: r => r.opportunity_trend },
    { key: 'recent_snap_share', label: 'Role evidence', title: 'Recent offensive snap share including verified zero-offense games; role change needs at least two recent and two earlier observations. Not routes.', align: 'left', render: r => `${percent(r.recent_snap_share)} · ${r.role_evidence}` },
    { key: 'participation_pace_next4', label: 'Off. weeks: pace / model', title: 'Simple participation pace / exploratory model: weeks with any offensive snaps across the four-week horizon. Includes role loss and missed time—not injury probability. Do not multiply points by this again.', render: r => `${number(r.participation_pace_next4)} / ${number(r.expected_offensive_weeks)}` },
    { key: 'outcome_range', label: '80% outcome range', title: 'Published only after historical calibration and incremental-value gates pass for this position and applicable cohorts. Not a guaranteed floor/ceiling.', render: r => r.outcome_range ? `${number(r.outcome_range.low)}–${number(r.outcome_range.high)}` : 'Withheld' },
  ]
  return <section aria-label="Player outlook">
    <div className="section-heading"><div><h3>Four-week outlook</h3>
      <p>Expected points, opportunity, role stability, and offensive participation in one research view.</p></div>
      <span className="instrument-tag">Experimental · not promoted</span></div>
    {report.isLoading && <p role="status">Verifying the outlook and its repaired historical inputs…</p>}
    {report.isError && <p className="notice">{report.error.message}</p>}
    {data && <>
      <p className="legend">{data.season} · observations through Week {data.through_week} · outlook covers Weeks {data.horizon[0]}–{data.horizon[1]}.
        {' '}Observations saved {new Date(data.observations_saved_at).toLocaleString()}.
        {' '}{data.players.filter(r => r.forecast_next4 != null).length} / {data.players.length} players scored.</p>
      <p className="legend faint">Outlook: {data.version} · repaired history: {data.history.version}. V1 and published boards are unchanged.</p>
      {(data.newer_week_possible || data.observation_age_hours > 48) && <p className="notice">This saved outlook may be stale. Games after Week {data.through_week} are not included.</p>}
      {(!data.ownership_available || data.ownership_stale) && <p className="notice">League ownership is unavailable or stale. Verify availability before acting.</p>}
      <p className="notice">No confidence scores are published. Outcome ranges are withheld unless the historical gates pass. Offensive participation is not medical availability; current injuries and transactions do not automatically update these estimates.</p>
      <div className="analysis-controls">
        <label>Outlook ownership<select value={ownership} onChange={e => setOwnership(e.target.value)}>
          <option value="all">Everyone</option><option value="free_agent">Available</option><option value="mine">My players</option><option value="rostered">Rostered</option><option value="unknown">Unknown</option>
        </select></label>
        <label>Prior sample<select value={priorSample} onChange={e => setPriorSample(e.target.value)}>
          <option value="all">Any sample size</option><option value="small">Fewer than 10 prior offensive weeks</option>
        </select></label>
        <label>Outlook coverage<select value={coverage} onChange={e => setCoverage(e.target.value)}>
          <option value="all">All candidates</option><option value="scored">With outlook</option><option value="missing">Unscored / unknown</option>
        </select></label>
      </div>
      {evidence.filter(v => v.cohort === (population === 'rookie' ? 'rookie' : priorSample === 'small' ? 'small_prior_sample' : 'all')).map(v =>
        <p className="legend" key={v.position}>{v.position}: outlook error {number(v.mae.outlook)} points vs recent-usage {number(v.mae.recent_usage)} and recent-points pace {number(v.mae.recent_points)}.
          {' '}{v.n.toLocaleString()} held-out forecasts across {v.folds.length} seasons; lower is better.
          {!v.publication_checks.point_mae_vs_usage && ' A reliable improvement over recent usage has not been established.'}
          {v.availability.outlook_mae != null && v.availability.recent_participation_mae != null
            && v.availability.outlook_mae >= v.availability.recent_participation_mae
            && ' For offensive participation, the simple pace baseline has been more accurate than the model.'}</p>)}
      <DataTable rows={rows} columns={columns} defaultSort="forecast_next4" rowKey={r => r.player_id}
        rowLabel={r => r.player_display_name} rowClass={r => r.is_mine ? 'mine-row' : undefined}
        emptyMessage="No players match these outlook filters."
        renderDetails={r => <><PlayerProfileLink playerId={r.player_id} /><details className="player-details"><summary>Four-week outlook diagnostics</summary>
          <p>{r.forecast_status}. Team at cutoff: {r.team ?? 'Unknown'} · {r.population}.
            {' '}{r.observed_offensive_weeks} observed offensive weeks this season; {r.prior_offensive_weeks ?? 'unknown'} last season.
            {' '}{r.snap_observations} snap observations; {r.snap_feed_complete ? 'weekly feed and identity checks passed' : 'participation coverage incomplete'}.</p>
          <p>Recent scheduled-game averages: {number(r.recent_targets_pg)} targets · {number(r.recent_carries_pg)} carries
            {r.position === 'QB' && <> · {number(r.recent_attempts_pg)} pass attempts</>} · {number(r.recent_points_pg)} points.
            {' '}Continuing recent points pace over {r.future_team_games} scheduled games gives {number(r.points_pace_next4)} points.</p>
          <p>Opportunity change: {number(r.usage_change)} per game. Snap-share change: {r.snap_change == null ? 'insufficient history' : `${number(r.snap_change * 100)} percentage points`}.
            {' '}Role stability diagnostic (snap-share standard deviation): {r.snap_std == null ? 'insufficient history (needs ≥3 observations)' : `${number(r.snap_std * 100)} percentage points`}. Lower means less observed variation, not job security.</p>
          <p>Participation model: {number(r.expected_offensive_weeks)} offensive weeks vs {number(r.participation_pace_next4)} from recent participation pace.
            {' '}Conditional snap-share estimate when on offense: {percent(r.expected_active_snap_share)}. Neither measure is an injury forecast.</p>
          <p>Recent scoring variation: {r.points_std == null ? 'insufficient history' : `${number(r.points_std)} points standard deviation`}.
            {' '}Largest game's share of recent positive points: {percent(r.spike_share)}; long-TD bonus share: {percent(r.bonus_share)}.
            {' '}These flag concentrated scoring to investigate—not a validated “flashy player” or regression label.</p>
          <p>{r.range_status}. Current saved health: {r.injury_status ?? 'Unknown'} (display only).
            {' '}Verify current role, news, schedule, roster fit, and acquisition cost; this is not a rest-of-season trade valuation.</p>
        </details></>} />
      <details className="reference-section"><summary>Validation: does the added complexity help?</summary>
        <p>Tests use only earlier seasons, with two separate earlier seasons reserved for range calibration. The simple usage model uses recent points, targets, carries, pass attempts, snap share, and scheduled games. Recent-points pace is a second baseline. All errors compare identical players and horizons; these are not medical or market-price tests.</p>
        <div className="table-wrap"><table><thead><tr><th>Pos / cohort</th><th>Forecasts</th><th>Points error: outlook / usage / pace</th><th>80% coverage</th><th>Range gates</th></tr></thead>
          <tbody>{evidence.map(v => <tr key={`${v.position}-${v.cohort}`}><td>{v.position} · {cohortLabel(v.cohort)}</td><td>{v.n}</td>
            <td>{number(v.mae.outlook)} / {number(v.mae.recent_usage)} / {number(v.mae.recent_points)}</td><td>{percent(v.observed_coverage)}</td>
            <td>{v.range_gate_passed ? 'Passed historically' : 'Not passed'}</td></tr>)}</tbody></table></div>
        {evidence.map(v => <p key={`${v.position}-${v.cohort}`}>{v.position} · {cohortLabel(v.cohort)}:
          {' '}offensive-week error {number(v.availability.outlook_mae)} vs participation-pace {number(v.availability.recent_participation_mae)} ({v.availability.n} forecasts);
          {' '}conditional-role error {percent(v.role.outlook_mae)} vs holding recent share {percent(v.role.hold_recent_share_mae)} ({v.role.n} forecasts).
          {' '}Season-block 95% interval for point-error improvement over usage: {v.season_bootstrap_95_improvement_vs_usage.map(number).join(' to ')} points.
          {' '}Unpassed range checks: {Object.entries(v.publication_checks).filter(([, passed]) => !passed).map(([key]) => key.replaceAll('_', ' ')).join(', ') || 'none'}.</p>)}
        <p>{data.design.method}</p><p>{data.design.publication_policy}</p>
        <ul>{data.limitations.map(text => <li key={text}>{text}</li>)}</ul>
      </details>
    </>}
  </section>
}
