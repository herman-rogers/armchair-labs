import { QueryError } from './Controls'
import { useQuery } from '@tanstack/react-query'
import { similarityQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { DataTable } from './DataTable'

type Season = { season: number; career_year?: number; observed_weeks: number; comparison_weeks: number; missing_weeks: number[]; verified_zero_weeks: number; [key: string]: string | number | number[] | null | undefined }
type ZeroReview = { season?: number; week?: number; source_url: string; evidence: string }
type Career = { player_id: string; player_display_name: string; distance: number; from_season: number; through_season: number; observed_weeks: number; comparison_weeks: number; verified_zero_weeks: number; zero_stat_reviews: ZeroReview[]; seasons: Season[] }
type Similarity = { matches: Career[]; target?: Career; coverage: Season[]; metrics: string[]; seasons_compared: number; eligible_peers: number; complete_through: number; reason: string | null; method: string; limitations: string }
const labels: Record<string, string> = { attempts: 'Pass attempts', passing_yards: 'Pass yards', carries: 'Carries', rushing_yards: 'Rush yards', targets: 'Targets', receptions: 'Receptions', receiving_yards: 'Rec yards' }

function ZeroSources({ career }: { career: Career }) {
  if (!career.zero_stat_reviews.length) return null
  return <p>{career.player_display_name}: verified zero-stat appearances included. {career.zero_stat_reviews.map(r => <span key={`${r.season}-${r.week}`}><a href={r.source_url} target="_blank" rel="noreferrer" title={r.evidence}>{r.season}, Week {r.week} source</a>{' '}</span>)}</p>
}

export function SimilarCareers({ playerId, season, week }: { playerId: string; season?: number; week?: number }) {
  const { token } = useDataRelease()
  const params = new URLSearchParams({ player_id: playerId, ...(season != null ? { season: String(season), week: String(week ?? 18) } : {}) })
  const query = useQuery(similarityQuery<Similarity>(params, token))
  const data = query.data
  return <section aria-label="Similar careers"><h4>Statistically similar careers</h4>
    <p>Closest available same-position career patterns, including historical players. These are descriptive comparisons, not Next Gen rankings or predictions of future success.</p>
    <QueryError query={query} />
    {!data && !query.isError && <p role="status">Comparing captured career statistics…</p>}
    {data && <>
      {data.reason ? <p className="notice">{data.reason}</p> : <><p>Comparing the first {data.seasons_compared} NFL season(s) against {data.eligible_peers} eligible careers. Only completed seasons through {data.complete_through} are used. Rates use weeks with all comparison statistics available. Open a match for season-by-season statistics.</p>
        {[data.target!, ...data.matches].some(c => c.comparison_weeks < c.observed_weeks) && <p className="notice">Some seasons have partial statistical coverage. Missing weeks are excluded from rates and shown below; low-activity omissions can make those rates look higher.</p>}
        <DataTable rows={data.matches} columns={[
          { key: 'player_display_name', label: 'Similar player', title: 'Same captured primary position, matched at the same career stage.', align: 'left' },
          { key: 'distance', label: 'Stat distance', title: 'Root mean squared standardized difference across the displayed measures and career years. Lower is closer; this is not a probability.', initial: 'asc', render: r => r.distance.toFixed(2) },
          { key: 'from_season', label: 'Compared seasons', title: 'Only these career seasons enter the comparison.', render: r => `${r.from_season}–${r.through_season}` },
          { key: 'comparison_weeks', label: 'Weeks covered', title: 'Weeks with all comparison statistics / observed participation weeks. Includes source-verified zero-stat appearances.', render: r => `${r.comparison_weeks} / ${r.observed_weeks}` },
        ]} defaultSort="distance" rowKey={r => r.player_id} rowLabel={r => `${r.player_display_name} comparison`} renderDetails={r => <>
          <p>All measures below are per covered week, using the same weeks for every measure. Each career year is compared separately.</p>
          <ZeroSources career={r} />
          <DataTable rows={[...(data.target?.seasons ?? []).map(y => ({...y, name: data.target!.player_display_name})), ...r.seasons.map(y => ({...y, name: r.player_display_name}))]} columns={[
            {key: 'career_year', label: 'Career year', title: 'Year since recorded NFL entry.', initial: 'asc'},
            {key: 'name', label: 'Player', title: 'The selected player and comparison player.', align: 'left'},
            {key: 'season', label: 'Season', title: 'Calendar season; eras are not adjusted.'},
            {key: 'comparison_weeks', label: 'Weeks covered', title: 'Weeks used in every rate / observed participation weeks.', render: y => `${y.comparison_weeks} / ${y.observed_weeks}`},
            {key: 'missing_weeks', label: 'Omitted weeks', title: 'Weeks with missing or invalid comparison statistics; values are not assumed to be zero.', render: y => y.missing_weeks.join(', ') || 'None'},
            {key: 'verified_zero_weeks', label: 'Verified zeros', title: 'Source-reviewed zero-stat appearances included in covered weeks.'},
            ...data.metrics.map(m => ({key: m, label: `${labels[m] ?? m} / covered week`, title: 'Rate across weeks with all comparison measures available, including verified zero-stat appearances.', render: (y: Season) => typeof y[m] === 'number' ? Number(y[m]).toFixed(1) : '—'})),
          ]} defaultSort="career_year" rowKey={y => `${y.name}-${y.season}`} />
        </>} />
      </>}
      {data.target && <ZeroSources career={data.target} />}
      {data.coverage.length > 0 && <details><summary>Selected player’s statistical coverage</summary>
        <DataTable rows={data.coverage} columns={[
          {key: 'season', label: 'Season', title: 'Completed career season.', initial: 'asc'},
          {key: 'comparison_weeks', label: 'Weeks covered', title: 'At least eight covered weeks and 80% of observed weeks are required in each season.', render: r => `${r.comparison_weeks} / ${r.observed_weeks}`},
          {key: 'missing_weeks', label: 'Omitted weeks', title: 'Observed weeks without all comparison statistics.', render: r => r.missing_weeks.join(', ') || 'None'},
          {key: 'verified_zero_weeks', label: 'Verified zeros', title: 'Source-reviewed zero-stat appearances included in the rates.'},
        ]} defaultSort="season" rowKey={r => String(r.season)} />
      </details>}
      <p className="legend">{data.limitations}</p><details><summary>How comparisons are calculated</summary><p>{data.method}</p></details>
    </>}
  </section>
}
