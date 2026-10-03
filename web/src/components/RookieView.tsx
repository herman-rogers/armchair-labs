import { useQuery } from '@tanstack/react-query'
import { rookiesQuery } from '../api/queries'
import { useDataRelease } from '../dataRelease'
import { DataTable } from './DataTable'
import { Link } from 'react-router'
import { playerPath, researchPath, to, type UrlPage } from '../navigation'
import { Pager, QueryError } from './Controls'
import { fixed } from '../format'

export function RookieView({ search, position, page }: { search: string; position: string; page: UrlPage }) {
  const { token } = useDataRelease()
  const params = new URLSearchParams({ search, position })
  const query = useQuery({ ...rookiesQuery(params, token) })
  return <section aria-label="Rookie analysis"><h3>{query.data?.season ?? ''} rookie class</h3>
    <p>Draft capital, captured NFL opportunity, and linked college histories. Open a player to follow their college-to-NFL progression.</p>
    <p className="legend">Experimental rookie comparisons remain in Research. Search, filtering and sorting cover the full rookie class before pagination.</p>
    <Link className="button" to={to(researchPath('archive'), { population: 'rookie' })}>Open rookie research archive</Link>
    <QueryError query={query} />
    {query.isFetching && <p role="status">Loading rookies…</p>}
    {query.data && <><p className="legend">{query.data.total} rookies · NFL observations through {query.data.season}, Week {query.data.through_week}.</p>
      <DataTable rows={query.data.players} pagination={page} sortParam="sort" columns={[
        {key:'player_display_name',label:'Player',title:'Current rookie cohort, including missing NFL production.',align:'left',initial:'asc'},
        {key:'position',label:'Position',title:'Audited fantasy position.',align:'left'},
        {key:'draft_round',label:'NFL draft round',title:'Recorded NFL draft round; unknown remains blank.',initial:'asc'},
        {key:'draft_pick',label:'NFL draft pick',title:'Actual draft pick, not a model rank.',initial:'asc'},
        {key:'college_linked',label:'College history',title:'Only accepted identity links join college and NFL histories.',render:r => r.college_linked ? 'Linked' : 'Unknown'},
        {key:'observed_weeks',label:'Observed weeks',title:'Captured scoring or positive-snap observations; not a medical availability estimate.'},
        {key:'league_points',label:'Points so far',title:'Recorded league points through the capture cutoff.',render:r => fixed(r.league_points)},
        {key:'targets',label:'Targets',title:'Captured receiving targets; incomplete totals remain unknown.'},
        {key:'carries',label:'Carries',title:'Captured rushing carries; incomplete totals remain unknown.'},
      ]} defaultSort="draft_pick" rowKey={r => r.player_id} rowLabel={r => r.player_display_name} profileHref={r => playerPath(r.player_id)} emptyMessage="No rookies match these filters. Clear the player search or position filter." />
      <Pager page={page} total={query.data.total} noun="rookies" />
    </>}
  </section>
}
