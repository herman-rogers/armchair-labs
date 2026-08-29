/** Shapes returned by the Patron read API. Mirrors `patron.pipeline.BOARD_EXPORT_COLUMNS`. */

export type Position = 'QB' | 'RB' | 'WR' | 'TE'

export interface Player {
  rank: number
  player_id: string
  player_display_name: string
  position: Position
  team: string
  /** VOR after manual overrides — what the board actually sorts by. */
  adj_vor: number
  /** VOR as computed from the tape, before any manual judgement. */
  vor: number
  ppg: number
  flags: string
  games: number
  season_pts: number
  /** Big-play bonus points. This league's private edge, kept visible. */
  bonus_pts: number
  floor: number | null
  volatility: number | null
  wtd_opp: number
  target_share: number | null
  air_yards_share: number | null
  wopr: number | null
  td_over_exp: number
  age_at_season: number | null
  repl_ppg: number
  override_delta: number
  override_reason: string | null
}

export interface BoardResponse {
  total: number
  offset: number
  limit: number
  players: Player[]
}

export interface Status {
  board_available: boolean
  built_at: string | null
  player_count: number
  league: {
    name: string
    team_count: number
    board_season: number
    draft_season: number
    seasons: number[]
    vor_baseline_rank: Record<string, number>
  }
  espn_connected: boolean
  phase: number
}
