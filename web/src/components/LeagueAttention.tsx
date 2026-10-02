import { Link } from 'react-router'
import type { AttentionFlag, LeagueObservations } from '../api/nextgen'
import { leaguePath, to } from '../navigation'

type Player = LeagueObservations['players'][number]
export function AttentionBadges({ flags }: { flags?: AttentionFlag[] }) {
  return <span className="league-flags">{flags?.filter(flag => flag.kind !== 'news').map((flag, i) => <span key={`${flag.kind}-${i}`} className={`badge ${flag.severity === 'urgent' ? 'out' : 'questionable'}`}>{flag.label}</span>)}</span>
}
export function PlayerStatus({ player }: { player: Player }) {
  const flag = player.attention?.find(flag => flag.kind === 'injury' || flag.kind === 'unknown_status')
  const tone = flag ? flag.severity === 'urgent' ? 'out' : 'questionable' : 'rostered'
  return <span className={`badge ${tone}`} title="Captured ESPN status">{player.injury_status?.replaceAll('_', ' ') ?? 'Not reported'}</span>
}
export function PlayerNotes({ player }: { player: Player }) {
  return <div className="player-notes"><AttentionBadges flags={player.attention?.filter(flag => flag.kind !== 'injury' && flag.kind !== 'unknown_status')} /><InjuryNotes player={player} compact /></div>
}
export function InjuryNotes({ player, compact = false }: { player: Player; compact?: boolean }) {
  return <>{!!player.injury_news?.length && <details className={`injury-notes${compact ? ' compact' : ''}`}><summary aria-label={`Injury reports for ${player.player_display_name}`}>{compact ? 'Reports' : 'Injury reports'} ({player.injury_news.length})</summary><div className="injury-notes-content"><p>Review the date and source. Reports may differ from ESPN tags and do not establish today's game availability.</p>
    <ul>{player.injury_news.map((note, i) => <li key={i}><strong>{note.known_on}</strong> · {note.evidence_status.replaceAll('_', ' ')}: {note.summary} <a href={note.source_url} target="_blank" rel="noreferrer">Source</a></li>)}</ul></div></details>}</>
}
export function LeagueAttention({ data }: { data: LeagueObservations }) {
  const players = data.players.filter(p => p.is_mine && p.attention?.length)
    .sort((a, b) => Number(b.attention?.some(f => f.severity === 'urgent')) - Number(a.attention?.some(f => f.severity === 'urgent')))
  const urgent = players.filter(p => p.attention?.some(flag => flag.severity === 'urgent')).length
  return <section className="league-attention" aria-label="Needs attention">
    <details className="attention-details">
      <summary><strong>Needs attention</strong><span>{players.length ? `${players.length} flagged player${players.length === 1 ? '' : 's'}` : 'No player alerts'}</span>
        {urgent > 0 && <span className="badge out">{urgent} urgent</span>}
        {data.stale && <span className="badge questionable">Saved snapshot</span>}
        {!!data.attention_notices?.length && <span className="badge questionable">{data.attention_notices.length} roster alerts</span>}
        {data.injury_news_warning && <span className="badge questionable">Report coverage limited</span>}
      </summary>
    <p className="legend">ESPN snapshot: {new Date(data.captured_at).toLocaleString()}. Refresh league updates ESPN; injury reports are separately captured and have their own dates. ACTIVE/NORMAL is a provider tag, not confirmation of health or a starting role.</p>
    {data.stale && <p className="notice">Saved league data is stale. Use Refresh league before making lineup or acquisition decisions.</p>}
    {data.injury_news_warning && <p className="notice">{data.injury_news_warning}</p>}
    {!!data.attention_notices?.length && <ul>{data.attention_notices.map((n, i) => <li key={i}><AttentionBadges flags={[n]} /></li>)}</ul>}
    {!!players.length && <ul className="attention-players">{players.map(p => <li key={p.espn_id}><strong>{p.player_display_name}</strong> <span className="faint">{p.lineup_slot ?? 'Slot unknown'}</span> <AttentionBadges flags={p.attention} /><InjuryNotes player={p} /></li>)}</ul>}
    {!players.length && <p>No player alerts found in the captured roster. This is not an all-clear; verify statuses and lineup locks in ESPN.</p>}
    </details>
    {!!players.length && <Link className="button" to={to(leaguePath('rosters'), { attention: 1 })}>Review players</Link>}
  </section>
}
