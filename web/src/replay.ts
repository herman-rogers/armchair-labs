/**
 * Pure replay arithmetic for the game engine: per-frame numbers derived from observed
 * trajectories, interpolation between frames, and labels. No React, no canvas.
 */
import type { Series, TrackingEntity, TrackingPlay } from './api/tracking'

/** 1 yard per second is 3600/1760 miles per hour. */
export const MPH_PER_YARD_PER_SECOND = 3600 / 1760

export const FIELD_LENGTH = 120
export const FIELD_WIDTH = 160 / 3

export interface EntityMetrics {
  /** Observed speed, or speed from successive positions where the archive has none. */
  speed: Series
  accel: Series
  /** Yards covered since the first observed frame. */
  distance: Series
  /** Highest speed observed up to and including each frame. */
  maxSpeed: Series
  /** Yards to the ball, when the archive tracks the ball. */
  toBall: Series
  /** Yards to the closest player on the other side of the ball. */
  nearestOpponent: Series
}

const present = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)
const hypot = (x0: number, y0: number, x1: number, y1: number) => Math.hypot(x1 - x0, y1 - y0)

/** Speed with gaps filled from positions; the archive's own value wins where it exists. */
export function fillSpeed(entity: TrackingEntity, frameRate: number): Series {
  const n = entity.x.length
  const out: Series = new Array(n).fill(null)
  for (let i = 0; i < n; i++) {
    if (present(entity.s[i])) { out[i] = entity.s[i]; continue }
    const x = entity.x[i], y = entity.y[i]
    if (!present(x) || !present(y)) continue
    const j = i > 0 && present(entity.x[i - 1]) && present(entity.y[i - 1]) ? i - 1
      : i + 1 < n && present(entity.x[i + 1]) && present(entity.y[i + 1]) ? i + 1 : -1
    if (j >= 0) out[i] = Math.round(hypot(x, y, entity.x[j]!, entity.y[j]!) * frameRate * 100) / 100
  }
  return out
}

export function deriveMetrics(play: TrackingPlay): Map<string, EntityMetrics> {
  const ball = play.entities.find(e => e.kind === 'ball')
  const n = play.frames
  const result = new Map<string, EntityMetrics>()
  for (const entity of play.entities) {
    const speed = fillSpeed(entity, play.frame_rate)
    const distance: Series = new Array(n).fill(null)
    const maxSpeed: Series = new Array(n).fill(null)
    const toBall: Series = new Array(n).fill(null)
    const nearestOpponent: Series = new Array(n).fill(null)
    let covered = 0, peak: number | null = null, last = -1
    const opponents = entity.kind === 'player' && entity.side && entity.side !== 'ball'
      ? play.entities.filter(o => o.kind === 'player' && o.side && o.side !== 'ball' && o.side !== entity.side) : []
    for (let i = 0; i < n; i++) {
      const x = entity.x[i], y = entity.y[i]
      if (!present(x) || !present(y)) continue
      if (last >= 0) covered += hypot(entity.x[last]!, entity.y[last]!, x, y)
      last = i
      distance[i] = Math.round(covered * 100) / 100
      const s = speed[i]
      if (present(s)) peak = peak == null ? s : Math.max(peak, s)
      maxSpeed[i] = peak
      if (ball && ball !== entity && present(ball.x[i]) && present(ball.y[i])) toBall[i] = Math.round(hypot(x, y, ball.x[i]!, ball.y[i]!) * 100) / 100
      let nearest: number | null = null
      for (const o of opponents) {
        if (!present(o.x[i]) || !present(o.y[i])) continue
        const d = hypot(x, y, o.x[i]!, o.y[i]!)
        if (nearest == null || d < nearest) nearest = d
      }
      nearestOpponent[i] = nearest == null ? null : Math.round(nearest * 100) / 100
    }
    result.set(entity.id, { speed, accel: entity.a, distance, maxSpeed, toBall, nearestOpponent })
  }
  return result
}

/** Linear interpolation at fractional frame `t`; falls back to whichever neighbour exists. */
export function sample(series: Series, t: number): number | null {
  const i = Math.floor(t), j = Math.min(i + 1, series.length - 1)
  const a = series[i], b = series[j]
  if (present(a) && present(b)) return a + (b - a) * (t - i)
  if (present(a)) return a
  if (present(b)) return b
  return null
}

/** Last observed frame index at or before `t`, or -1. */
export function lastObserved(series: Series, t: number): number {
  for (let i = Math.min(Math.floor(t), series.length - 1); i >= 0; i--) if (present(series[i])) return i
  return -1
}

export const ordinal = (n: number) => `${n}${['th', 'st', 'nd', 'rd'][(n % 100 > 10 && n % 100 < 14) ? 0 : Math.min(n % 10, 4) === 4 ? 0 : n % 10]}`

export function downLabel(down: number | null, distance: number | null) {
  if (!down) return null
  return `${ordinal(down)} & ${distance == null ? '—' : distance === 0 ? 'goal' : distance}`
}

export function fieldPosition(side: string | null, number: number | null) {
  if (number == null) return null
  return number === 50 ? 'the 50' : `${side ?? ''} ${number}`.trim()
}

/** `autoevent_passforward` → "pass forward (auto)"; `first_contact` → "first contact". */
export function eventLabel(name: string) {
  const auto = name.startsWith('autoevent_')
  const stem = (auto ? name.slice('autoevent_'.length) : name)
    .replace(/^passforward$/, 'pass_forward').replace(/^ballsnap$/, 'ball_snap').replace(/^passinterrupted$/, 'pass_interrupted')
    .replaceAll('_', ' ')
  return auto ? `${stem} (auto)` : stem
}

export const mph = (yardsPerSecond: number | null) => present(yardsPerSecond) ? yardsPerSecond * MPH_PER_YARD_PER_SECOND : null

/** Playback clock for a frame index: `12` frames at 10 Hz → "1.2 s". */
export const frameSeconds = (frame: number, frameRate: number) => `${(frame / frameRate).toFixed(1)} s`
