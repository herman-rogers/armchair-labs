import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { trackingGamesQuery, trackingPlayQuery, trackingPlaysQuery, trackingSourcesQuery } from '../api/queries'
import type { TrackingEntity, TrackingPlay, TrackingPlaySummary } from '../api/tracking'
import { deriveMetrics, downLabel, eventLabel, fieldPosition, frameSeconds, mph } from '../replay'
import { useUrlParams } from '../navigation'
import { fixed } from '../format'
import { QueryError, SelectField } from './Controls'
import { FieldCanvas, type FieldCanvasHandle } from './FieldCanvas'
import './GameEngine.css'

const SPEEDS = [0.25, 0.5, 1, 2] as const
const SIDE_ORDER: Record<string, number> = { offense: 0, defense: 1, ball: 2 }
const POSITION_ORDER = ['QB', 'RB', 'FB', 'WR', 'TE', 'T', 'G', 'C', 'DE', 'DT', 'NT', 'OLB', 'ILB', 'MLB', 'LB', 'CB', 'SS', 'FS', 'S', 'DB']

function PlayRow({ play, current, onPick }: { play: TrackingPlaySummary; current: boolean; onPick: () => void }) {
  return <li>
    <button type="button" aria-current={current} onClick={onPick} title={play.description ?? undefined}>
      <span className="play-meta">
        <b>Q{play.quarter ?? '—'} {play.clock ?? ''}</b>
        <span>{downLabel(play.down, play.distance) ?? 'No down'}</span>
        <span>{play.possession ?? ''}{play.yardline_number != null ? ` · ${fieldPosition(play.yardline_side, play.yardline_number)}` : ''}</span>
        <span className="play-type">{play.play_type}</span>
      </span>
      <span className="play-desc">{play.description?.replace(/^\([^)]*\)\s*/, '') ?? `Play ${play.play_id}`}</span>
    </button>
  </li>
}

function rowOrder(a: TrackingEntity, b: TrackingEntity) {
  const side = (SIDE_ORDER[a.side ?? 'ball'] ?? 9) - (SIDE_ORDER[b.side ?? 'ball'] ?? 9)
  if (side) return side
  const pa = POSITION_ORDER.indexOf(a.position ?? ''), pb = POSITION_ORDER.indexOf(b.position ?? '')
  if (pa !== pb) return (pa < 0 ? 99 : pa) - (pb < 0 ? 99 : pb)
  return a.name.localeCompare(b.name)
}

/** The live numbers for every tracked entity at one frame. */
function LiveTable({ play, frame, selectedId, onSelect }: { play: TrackingPlay; frame: number; selectedId: string | null; onSelect: (id: string | null) => void }) {
  const metrics = useMemo(() => deriveMetrics(play), [play])
  const rows = useMemo(() => [...play.entities].sort(rowOrder), [play])
  const hasBall = play.entities.some(e => e.kind === 'ball')
  const throwFrame = play.events.find(e => e.name === 'pass_forward')?.frame
  const num = (v: number | null | undefined, digits = 1) => <td className="num">{fixed(v, digits)}</td>
  return <div className="table-wrap replay-table"><table>
    <thead><tr>
      <th>Player</th><th>Pos</th><th>Team</th>
      <th className="num" title="Speed at this frame, yards per second.">yd/s</th>
      <th className="num" title="Speed at this frame, miles per hour.">mph</th>
      <th className="num" title="Acceleration at this frame, yards per second squared.">accel</th>
      <th className="num" title="Yards covered since the first tracked frame of the play.">covered</th>
      <th className="num" title="Highest speed so far on this play, miles per hour.">peak mph</th>
      {hasBall && <th className="num" title="Yards from the ball.">to ball</th>}
      <th className="num" title="Yards to the closest player on the other side of the ball.">nearest opp</th>
      <th className="num" title="Orientation in degrees; 0 faces the far sideline, 90 faces the right end zone.">facing</th>
    </tr></thead>
    <tbody>{rows.map(entity => {
      const m = metrics.get(entity.id)!
      const untracked = entity.x[frame] == null
      const afterThrow = throwFrame != null && frame > throwFrame && entity.tracked_in_flight === false
      return <tr key={entity.id} aria-selected={entity.id === selectedId} onClick={() => onSelect(entity.id === selectedId ? null : entity.id)}>
        <td>{entity.name}{untracked && <span className="untracked">{afterThrow ? 'not tracked after throw' : 'not in frame'}</span>}</td>
        <td>{entity.position ?? (entity.kind === 'ball' ? 'ball' : '—')}</td>
        <td className={entity.side ? `side-${entity.side}` : undefined}>{entity.team ?? '—'}</td>
        {num(m.speed[frame], 2)}{num(mph(m.speed[frame]))}{num(m.accel[frame])}{num(m.distance[frame])}{num(mph(m.maxSpeed[frame]))}
        {hasBall && num(m.toBall[frame])}
        {num(entity.kind === 'ball' ? null : m.nearestOpponent[frame])}
        {num(entity.o[frame], 0)}
      </tr>
    })}</tbody>
  </table></div>
}

/** One play's playback. Mounted fresh per play (keyed by the parent) so frame and play state reset. */
function Replay({ play, onEnded, selectedId, onSelect }: { play: TrackingPlay; onEnded: () => void; selectedId: string | null; onSelect: (id: string | null) => void }) {
  const field = useRef<FieldCanvasHandle>(null)
  const position = useRef(0)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState<number>(1)
  const last = play.frames - 1
  const ended = useRef(onEnded)
  useEffect(() => { ended.current = onEnded }, [onEnded])

  const seek = useCallback((t: number) => {
    position.current = Math.max(0, Math.min(last, t))
    field.current?.draw(position.current)
    setFrame(Math.round(position.current))
  }, [last])

  useEffect(() => {
    if (!playing) return
    let handle = 0, previous: number | null = null
    const tick = (now: number) => {
      if (previous != null) {
        position.current = Math.min(last, position.current + (now - previous) / 1000 * play.frame_rate * speed)
        field.current?.draw(position.current)
        const whole = Math.round(position.current)
        setFrame(current => current === whole ? current : whole)
        if (position.current >= last) { setPlaying(false); ended.current(); return }
      }
      previous = now
      handle = requestAnimationFrame(tick)
    }
    handle = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(handle)
  }, [playing, speed, play, last])

  const toggle = () => { if (!playing && position.current >= last) seek(0); setPlaying(p => !p) }
  const onKey = (event: React.KeyboardEvent) => {
    if ((event.target as HTMLElement).tagName === 'SELECT') return
    if (event.key === ' ') { event.preventDefault(); toggle() }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); setPlaying(false); seek(position.current - 1) }
    else if (event.key === 'ArrowRight') { event.preventDefault(); setPlaying(false); seek(position.current + 1) }
    else if (event.key === 'Home') { event.preventDefault(); seek(0) }
    else if (event.key === 'End') { event.preventDefault(); setPlaying(false); seek(last) }
  }
  const current = play.events.find(e => e.frame === frame)
  const ctx = play.context
  return <div className="replay" tabIndex={0} onKeyDown={onKey} aria-label="Play replay; space plays or pauses, arrow keys step frames">
    <div className="replay-context">
      <span className="score">{play.teams.away} {ctx.away_score ?? '—'} @ {play.teams.home} {ctx.home_score ?? '—'}</span>
      <span className="situation"><b>Q{ctx.quarter ?? '—'} {ctx.clock ?? ''}</b> · {downLabel(ctx.down, ctx.distance) ?? 'no down'} · <b>{play.teams.offense}</b> ball{ctx.yardline_number != null ? ` on ${fieldPosition(ctx.yardline_side, ctx.yardline_number)}` : ''} · moving {play.play_direction ?? '—'}</span>
      {ctx.formation && <span className="tag">{ctx.formation.replaceAll('_', ' ').toLowerCase()}</span>}
      {ctx.coverage && <span className="tag">{ctx.coverage.replaceAll('_', ' ').toLowerCase()}</span>}
      {ctx.route && <span className="tag">target route: {ctx.route.toLowerCase()}</span>}
      {ctx.result_yards != null && <span className="tag">{ctx.result_yards} yd{ctx.pass_result ? ` · ${ctx.pass_result}` : ''}</span>}
      {ctx.epa != null && <span className="tag" title="Expected points added, as recorded by the archive.">EPA {fixed(ctx.epa, 2)}</span>}
      {ctx.description && <p className="description">{ctx.description}</p>}
    </div>
    <div className="field-wrap"><FieldCanvas ref={field} play={play} selectedId={selectedId} onSelect={onSelect} /></div>
    <div className="playback">
      <div className="transport">
        <button type="button" className="button" onClick={() => { setPlaying(false); seek(0) }} title="First frame (Home)">⏮</button>
        <button type="button" className="button" onClick={() => { setPlaying(false); seek(position.current - 1) }} title="Previous frame (←)">◂</button>
        <button type="button" className="button" onClick={toggle} aria-pressed={playing} title="Play or pause (space)">{playing ? 'Pause' : 'Play'}</button>
        <button type="button" className="button" onClick={() => { setPlaying(false); seek(position.current + 1) }} title="Next frame (→)">▸</button>
      </div>
      <div className="scrubber">
        <input type="range" min={0} max={last} step={1} value={frame} aria-label="Frame" onChange={e => { setPlaying(false); seek(Number(e.target.value)) }} />
        <div className="marks" aria-hidden="true">{play.events.map(e => <span key={`${e.frame}-${e.name}`} className="mark" style={{ left: `${last ? e.frame / last * 100 : 0}%` }}>{eventLabel(e.name)}</span>)}</div>
      </div>
      <div className="readout">
        <span>frame {frame + 1}/{play.frames} · {frameSeconds(frame, play.frame_rate)}</span>
        {current && <span className="event">{eventLabel(current.name)}</span>}
        <select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}>{SPEEDS.map(s => <option key={s} value={s}>{s}×</option>)}</select>
      </div>
    </div>
    <LiveTable play={play} frame={frame} selectedId={selectedId} onSelect={onSelect} />
  </div>
}

/** `/tracking/games`: replay tracked games play by play with live movement numbers. */
export function GameEngine() {
  const [params, update] = useUrlParams()
  const sources = useQuery(trackingSourcesQuery())
  const available = sources.data?.sources.filter(s => s.available) ?? []
  const source = available.find(s => s.id === params.get('source')) ?? available[0]
  const games = useQuery(trackingGamesQuery(source?.id ?? ''))
  const game = games.data?.games.find(g => g.game_id === params.get('game')) ?? games.data?.games[0]
  const plays = useQuery(trackingPlaysQuery(source?.id ?? '', game?.game_id ?? ''))
  const play = plays.data?.plays.find(p => p.play_id === params.get('play')) ?? plays.data?.plays[0]
  const detail = useQuery(trackingPlayQuery(source?.id ?? '', game?.game_id ?? '', play?.play_id ?? ''))
  const [autoAdvance, setAutoAdvance] = useState(true)
  const selectedId = params.get('player')

  const pickPlay = (playId: string, replace = false) => update({ play: playId, player: null }, { replace })
  const index = plays.data?.plays.findIndex(p => p.play_id === play?.play_id) ?? -1
  const neighbour = (offset: number) => plays.data?.plays[index + offset]
  const onEnded = () => {
    if (!autoAdvance) return
    const next = neighbour(1)
    if (next) window.setTimeout(() => pickPlay(next.play_id, true), 700)
  }

  return <>
    <div className="analysis-controls">
      <SelectField label="Tracking archive" value={source?.id ?? ''} onChange={e => update({ source: e.target.value, game: null, play: null, player: null })} disabled={!available.length}>
        {available.map(s => <option key={s.id} value={s.id}>{s.label} · {s.games} games</option>)}
      </SelectField>
      <SelectField label="Game" value={game?.game_id ?? ''} onChange={e => update({ game: e.target.value, play: null, player: null })} disabled={!games.data}>
        {games.data?.games.map(g => <option key={g.game_id} value={g.game_id}>Week {g.week} · {g.away} @ {g.home}{g.home_score != null ? ` (${g.away_score}–${g.home_score})` : ''} · {g.plays} plays</option>)}
      </SelectField>
      <button type="button" className="button" disabled={!neighbour(-1)} onClick={() => pickPlay(neighbour(-1)!.play_id)}>Previous play</button>
      <button type="button" className="button" disabled={!neighbour(1)} onClick={() => pickPlay(neighbour(1)!.play_id)}>Next play</button>
      <label className="replay-options"><input type="checkbox" checked={autoAdvance} onChange={e => setAutoAdvance(e.target.checked)} /> Continue to the next play</label>
    </div>
    <QueryError query={sources} label="Tracking archives unavailable" />
    <QueryError query={games} label="Games unavailable" />
    <QueryError query={plays} label="Plays unavailable" />
    <QueryError query={detail} label="Play unavailable" />
    {sources.data && !available.length && <p className="notice">No tracking archive is ingested locally. See the tracking data runbook.</p>}
    {source && <p className="legend"><b>{source.label}.</b> {source.description}</p>}
    <div className="game-engine">
      <nav className="play-list" aria-label="Plays in this game">
        <h3>{game ? `Week ${game.week} · ${game.away} @ ${game.home}` : 'Plays'}{plays.data ? ` · ${plays.data.plays.length}` : ''}</h3>
        {plays.isPending && source && game && <p role="status" style={{ padding: '10px 14px' }}>Loading plays…</p>}
        <ol>{plays.data?.plays.map(p => <PlayRow key={p.play_id} play={p} current={p.play_id === play?.play_id} onPick={() => pickPlay(p.play_id)} />)}</ol>
      </nav>
      <div>
        {detail.isPending && play && <p role="status">Loading play…</p>}
        {detail.data && <Replay key={`${detail.data.source}/${detail.data.game_id}/${detail.data.play_id}`} play={detail.data} onEnded={onEnded} selectedId={selectedId} onSelect={id => update({ player: id }, { replace: true })} />}
        {detail.data && <div className="replay-options">
          <span><span className="swatch" style={{ background: 'var(--wr)' }} /> offense ({detail.data.teams.offense})</span>
          <span><span className="swatch" style={{ background: 'var(--sell)' }} /> defense ({detail.data.teams.defense})</span>
          <span><span className="swatch" style={{ background: '#4cc3ff' }} /> line of scrimmage</span>
          <span><span className="swatch" style={{ background: '#ffd23f' }} /> first down{detail.data.ball_land ? ' · × ball landing spot' : ''}</span>
          <span>Click a marker or row to follow a player; the tick on each marker shows which way the player faces. Faded markers are last known positions.</span>
        </div>}
      </div>
    </div>
  </>
}
