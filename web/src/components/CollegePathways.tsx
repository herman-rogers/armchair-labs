import { useDebounced } from '../api/queries/useDebounced'
import type { PlayerFilters } from '../playerFilters'
import { playerPath, collegePath } from '../navigation'
import { useQuery } from '@tanstack/react-query'
import { DataTable, type Column } from './DataTable'
import { collegeQuery } from '../api/queries'
import { useUrlState, useUrlPage } from '../navigation'
import { useDataRelease } from '../dataRelease'

type Scores = Record<string, number>
type Evidence = { position: string; target: string; era: string; n: number; mae: Scores; folds: { year: number }[] }
type Study = {
  version: string; college_start: number; college_end: number; current_forecast_year: number
  nfl_complete_through: number; limitations: string[]; nfl_backtest: Evidence[]; college_backtest: Evidence[]
  identity: { college_players: number; linked_nfl_players: number; statuses: { status: string; len: number }[]; identifier_conflicts: unknown[] }
  cohorts: { forecast_year: number; position: string; candidates: number; linked: number; eligible: number }[]
}
type Player = {
  player_id: string; player_display_name: string; position: string; forecast_year: number
  college_ids: string[]; college_team: string | null; eligible: boolean; ineligible_reason: string | null
  link_methods: string[]; draft_pick: number | null; latest_receiving_yards: number | null
  points_college_only: number | null; points_draft_only: number | null; points_college_plus_draft: number | null
  games_college_plus_draft: number | null; nfl_year1_points: number | null
}
type Identity = { college_id: string; college_name: string; player_id: string | null; nfl_name: string | null
  first_college_season: number; last_college_season: number; status: string; method: string | null; evidence: string | null }

const number = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1)
const labels: Record<string, string> = { nfl_year1_points: 'NFL rookie-year points', nfl_year1_games: 'NFL rookie-year games', nfl_first3_points: 'NFL first-three-year points', next_college_yards: 'Next college season yards' }
const populationLabels: Record<string, string> = { passing_yards: 'College passing', rushing_yards: 'College rushing', receiving_yards: 'College receiving' }
const methodLabels: Record<string, string> = { provider_id_name_chronology: 'Shared ID + corroboration', exact_name_birth_date: 'Name + birth date', exact_name_school_entry_window: 'Name + school + timing', reviewed_override: 'Reviewed exception' }
const methodLabel = (method: string | null) => method ? methodLabels[method] ?? method : 'No accepted match'


export function CollegePathways({ view, filters, audit = false }: { view: 'players' | 'identities' | 'evidence'; filters?: PlayerFilters; audit?: boolean }) {
  const { token: dataRelease } = useDataRelease()
  const [year, setYear] = useUrlState('class', 'current', { resets: ['page'] })
  const [auditSearch, setAuditSearch] = useUrlState('q', '', { replace: true, resets: ['page'] })
  const search = useDebounced(filters?.search ?? auditSearch)
  const position = filters?.position ?? 'ALL'
  const [identityStatus, setIdentityStatus] = useUrlState('status', 'all', { resets: ['page'] })
  const page = useUrlPage(100)
  const defaultSort = view === 'players' ? '-points_college_plus_draft' : 'college_name'
  const [requestedSort] = useUrlState('sort', defaultSort)
  const fields = view === 'players' ? ['player_display_name', 'forecast_year', 'position', 'college_team', 'link_methods', 'points_college_only', 'points_draft_only', 'points_college_plus_draft', 'nfl_year1_points', 'eligible'] : ['college_name', 'college_id', 'first_college_season', 'last_college_season', 'status', 'nfl_name', 'method']
  const sort = fields.includes(requestedSort.replace(/^-/, '')) ? requestedSort : defaultSort
  const paging = { limit: String(page.limit), offset: String(page.offset), sort }
  const study = useQuery(collegeQuery<Study>('', new URLSearchParams(), dataRelease))
  const actualYear = year === 'current' ? study.data?.current_forecast_year : year === 'all' ? undefined : Number(year)
  const params = new URLSearchParams({ ...paging, position, search, ...(actualYear ? { season: String(actualYear) } : {}) })
  const players = useQuery({ ...collegeQuery<{ total: number; players: Player[] }>('players', params, dataRelease), enabled: !!study.data && view === 'players', retry: false })
  const identityParams = new URLSearchParams({ ...paging, search, status: identityStatus, position })
  const identities = useQuery({ ...collegeQuery<{ total: number; players: Identity[] }>('identities', identityParams, dataRelease), enabled: !!study.data && view === 'identities', retry: false })
  if (study.isError) return <p className="notice">{study.error.message}</p>
  if (!study.data) return <p role="status">Verifying college and NFL research…</p>
  const data = study.data
  const columns: Column<Player>[] = [
    { key: 'player_display_name', label: 'Player', title: 'NFL entry candidate, including unlinked and unscored players.', align: 'left', initial: 'asc' },
    { key: 'forecast_year', label: 'NFL entry', title: 'NFL rookie cohort year. These are preseason rookie forecasts, not current rest-of-season forecasts.' },
    { key: 'position', label: 'Pos', title: 'Audited NFL forecast position.', align: 'left' },
    { key: 'college_team', label: 'Final college', title: 'School in the last observed pre-NFL season.', align: 'left' },
    { key: 'link_methods', label: 'Identity evidence', title: 'Exact provider ID or corroborated name match; not a confidence score.', align: 'left', render: r => r.link_methods.map(methodLabel).join(', ') || 'Unlinked' },
    { key: 'points_college_only', label: 'College-only points', title: 'Experimental rookie-year league points, conditional on NFL entry; excludes actual draft capital.', render: r => number(r.points_college_only) },
    { key: 'points_draft_only', label: 'Draft baseline', title: 'Same-player rookie-year forecast using recorded NFL draft capital only.', render: r => number(r.points_draft_only) },
    { key: 'points_college_plus_draft', label: 'College + draft', title: 'Experimental post-draft rookie-year points. Not promoted or automatically blended into current outlook.', render: r => number(r.points_college_plus_draft) },
    { key: 'nfl_year1_points', label: 'Actual rookie points', title: 'Completed regular-season league-exact points. Pending seasons remain unknown.', render: r => number(r.nfl_year1_points) },
    { key: 'eligible', label: 'Coverage', title: 'Missing links or incomplete final seasons withhold research scores.', align: 'left', render: r => r.eligible ? 'Eligible' : r.ineligible_reason?.replaceAll('_', ' ') ?? 'Unscored' },
  ]
  const identityColumns: Column<Identity>[] = [
    { key: 'college_name', label: 'College player', title: 'Includes players without NFL matches.', align: 'left', initial: 'asc' },
    { key: 'college_id', label: 'College ESPN ID', title: 'Source identity retained across transfers.', align: 'left' },
    { key: 'first_college_season', label: 'First observed', title: 'First captured season, not necessarily career start.' },
    { key: 'last_college_season', label: 'Last observed', title: 'Last captured college season.' },
    { key: 'status', label: 'Link status', title: 'Unmatched is not a negative NFL outcome. Review means conflicting evidence.', align: 'left' },
    { key: 'nfl_name', label: 'NFL candidate', title: 'Accepted NFL name; review candidates are not accepted joins.', align: 'left' },
    { key: 'method', label: 'Evidence', title: 'Identity resolution rule.', align: 'left', render: r => methodLabel(r.method) },
  ]
  const evidence = [...data.nfl_backtest, ...data.college_backtest].filter(r => r.era === 'modern_2018_plus')
  const evidenceColumns: Column<Evidence>[] = [
    { key: 'position', label: 'Population', title: 'NFL position or college production category.', align: 'left', initial: 'asc', render: r => populationLabels[r.position] ?? r.position },
    { key: 'target', label: 'Outcome', title: 'Totals over the stated horizon, not PPG.', align: 'left', render: r => labels[r.target] ?? r.target },
    { key: 'n', label: 'Held-out forecasts', title: 'Identical comparison rows for all methods within this line. College players may appear in multiple seasons.' },
    { key: 'baseline', label: 'Baseline error', title: 'Mean absolute error: NFL draft-only; college last-season yardage. Lower is better.', value: r => r.mae.draft_only ?? r.mae.last_season, render: r => number(r.mae.draft_only ?? r.mae.last_season) },
    { key: 'model', label: 'Added-college error', title: 'Mean absolute error: NFL college+draft; college ridge. No model promotion implied.', value: r => r.mae.college_plus_draft ?? r.mae.college_model, render: r => number(r.mae.college_plus_draft ?? r.mae.college_model) },
  ]
  return <section>
    <h3>{view === 'evidence' ? 'College backtests' : view === 'identities' ? audit ? 'College identity audit' : 'College careers' : 'NFL translation forecasts'}</h3>
    <p>{data.college_start}–{data.college_end} college history · {data.identity.college_players.toLocaleString()} college identities · {data.identity.linked_nfl_players.toLocaleString()} linked NFL players · NFL outcomes complete through {data.nfl_complete_through}.</p>
    <p className="notice">Research only. College-only and post-draft projections are separate experiments—not upgrades automatically applied to Next-gen or Player outlook. Unmatched players are unknown, not NFL failures.</p>
    <div className="analysis-controls">
      {audit && view === 'identities' && <label>Find player<input type="search" value={auditSearch} onChange={e => setAuditSearch(e.target.value)} placeholder="Player name…" /></label>}
      {view === 'players' && <>
        <label>NFL entry class<select value={year} onChange={e => setYear(e.target.value)}><option value="current">Latest ({data.current_forecast_year})</option><option value="all">All classes</option>{[...new Set(data.cohorts.map(r => r.forecast_year))].reverse().map(y => <option key={y}>{y}</option>)}</select></label>
      </>}
      {audit && view === 'identities' && <label>Identity status<select value={identityStatus} onChange={e => setIdentityStatus(e.target.value)}>{['all', 'linked', 'review', 'unmatched'].map(s => <option key={s}>{s}</option>)}</select></label>}
    </div>
    {view === 'players' && <>
      {players.isError && <p className="notice">{players.error.message}</p>}
      {players.isLoading && <p role="status">Loading linked forecasts…</p>}
      {players.data && <><p className="legend">Showing {players.data.players.length} / {players.data.total} candidates. Missing forecasts are withheld, not zero. Open a row for college → NFL → current outlook.</p>
        <DataTable serverSide pagination={page} sortParam="sort" rows={players.data.players} columns={columns} defaultSort="points_college_plus_draft" rowKey={r => `${r.player_id}-${r.forecast_year}`} rowLabel={r => r.player_display_name}
          emptyMessage="No matching NFL entrants. Try All classes or the college identity audit." profileHref={r => playerPath(r.player_id)} /></>}
    </>}
    {view === 'identities' && <>
      {identities.isError && <p className="notice">{identities.error.message}</p>}
      {identities.isLoading && <p role="status">Loading identity ledger…</p>}
      {identities.data && <><p className="legend">Showing {identities.data.players.length} / {identities.data.total}; narrow by name. {data.identity.identifier_conflicts.length} conflicting supplemental provider aliases are withheld.</p>
        <DataTable serverSide pagination={page} sortParam="sort" rows={identities.data.players} columns={identityColumns} defaultSort="college_name" rowKey={r => r.college_id} rowLabel={r => r.college_name} emptyMessage="No matching identities." profileHref={r => collegePath(r.college_id)} /></>}
    </>}
    {view !== 'evidence' && <CollegePagination page={page} total={(view === 'players' ? players.data : identities.data)?.total ?? 0} />}
    {view === 'evidence' && <>
      <p className="legend">2018+ held-out forecasts, trained only on earlier completed outcomes. Lower error is better. College forecasts are conditional on returning to an observed college roster; NFL forecasts are conditional on the NFL-entry cohort. These are not NFL-entry probabilities.</p>
      <DataTable rows={evidence} columns={evidenceColumns} defaultSort="position" rowKey={r => `${r.position}-${r.target}`} />
      <details><summary>Coverage by NFL class</summary><table><thead><tr><th>Class</th><th>Position</th><th>Candidates</th><th>Linked</th><th>Eligible</th></tr></thead><tbody>{data.cohorts.map(r => <tr key={`${r.forecast_year}-${r.position}`}><td>{r.forecast_year}</td><td>{r.position}</td><td>{r.candidates}</td><td>{r.linked}</td><td>{r.eligible}</td></tr>)}</tbody></table></details>
    </>}
    <details><summary>Data boundaries and method</summary><ul>{data.limitations.map(line => <li key={line}>{line}</li>)}</ul><p>Version: {data.version}</p></details>
  </section>
}

function CollegePagination({ page, total }: { page: { offset: number; limit: number; onPage: (offset: number) => void }; total: number }) {
  return <div className="profile-pagination">
    <button className="button" disabled={!page.offset} onClick={() => page.onPage(Math.max(0, page.offset - page.limit))}>Previous</button>
    <span>{Math.min(page.offset + (total ? 1 : 0), total)}–{Math.min(page.offset + page.limit, total)} of {total}</span>
    <button className="button" disabled={page.offset + page.limit >= total} onClick={() => page.onPage(page.offset + page.limit)}>Next</button>
  </div>
}
