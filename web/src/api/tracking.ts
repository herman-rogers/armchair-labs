import { get } from './client'

/** `/api/tracking/sources`: each replayable tracking archive. */
export interface TrackingSource {
  id: string
  label: string
  season: number
  /** `full` tracks all 22 players and the ball; `coverage` tracks the passing game only. */
  tracked: 'full' | 'coverage'
  description: string
  available: boolean
  games: number
  issue?: string
}

export interface TrackingGame {
  game_id: string
  season: number
  week: number
  date: string | null
  home: string
  away: string
  home_score: number | null
  away_score: number | null
  plays: number
}

export interface TrackingPlaySummary {
  play_id: string
  order: number
  quarter: number | null
  clock: string | null
  down: number | null
  distance: number | null
  possession: string | null
  defense: string | null
  yardline_side: string | null
  yardline_number: number | null
  description: string | null
  result_yards: number | null
  pass_result: string | null
  play_type: 'pass' | 'run'
  home_score: number | null
  away_score: number | null
  frames: number
  events: string[]
}

/** One value per frame; `null` where the archive has no observation for that frame. */
export type Series = (number | null)[]

export interface TrackingEntity {
  id: string
  kind: 'player' | 'ball'
  name: string
  position: string | null
  team: string | null
  side: 'offense' | 'defense' | 'ball' | null
  role?: string | null
  /** Coverage archives only: whether this player is tracked after the throw. */
  tracked_in_flight?: boolean
  x: Series
  y: Series
  /** Speed in yards per second. */
  s: Series
  /** Acceleration in yards per second squared. */
  a: Series
  /** Orientation in degrees, clockwise from the +y sideline direction. */
  o: Series
  /** Direction of motion in degrees, same convention as `o`. */
  dir: Series
}

export interface TrackingPlay {
  source: string
  game_id: string
  play_id: string
  frame_rate: number
  frames: number
  play_direction: 'left' | 'right' | null
  line_of_scrimmage: number | null
  first_down_line: number | null
  ball_land: { x: number; y: number } | null
  teams: { home: string; away: string; offense: string; defense: string }
  context: {
    quarter: number | null
    clock: string | null
    down: number | null
    distance: number | null
    yardline_side: string | null
    yardline_number: number | null
    description: string | null
    result_yards: number | null
    pass_result: string | null
    formation: string | null
    defenders_in_box: number | null
    home_score: number | null
    away_score: number | null
    epa: number | null
    ball_carrier: string | null
    coverage: string | null
    route: string | null
  }
  events: { frame: number; name: string }[]
  entities: TrackingEntity[]
}

const base = '/api/tracking'
const id = (value: string) => encodeURIComponent(value)

export const fetchTrackingSources = (signal?: AbortSignal) =>
  get<{ sources: TrackingSource[] }>(`${base}/sources`, undefined, signal)
export const fetchTrackingGames = (source: string, signal?: AbortSignal) =>
  get<{ games: TrackingGame[] }>(`${base}/${id(source)}/games`, undefined, signal)
export const fetchTrackingPlays = (source: string, game: string, signal?: AbortSignal) =>
  get<{ plays: TrackingPlaySummary[] }>(`${base}/${id(source)}/games/${id(game)}/plays`, undefined, signal)
export const fetchTrackingPlay = (source: string, game: string, play: string, signal?: AbortSignal) =>
  get<TrackingPlay>(`${base}/${id(source)}/games/${id(game)}/plays/${id(play)}`, undefined, signal)
