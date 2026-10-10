import { get } from './client'

export type Cutoff = { season: number | null; through_week: number | null }
export interface DirectoryPlayer { player_id: string; player_display_name: string; position: string; team: string | null }
export interface PlayerDirectory { observation_cutoff: Cutoff; total: number; players: DirectoryPlayer[] }
export interface PlayerIdentity {
  player_id: string; display_name: string; birth_date?: string | null; espn_id?: string | null; pfr_id?: string | null
  rookie_season?: number | null; draft_year?: number | null; draft_round?: number | null; draft_pick?: number | null; draft_team?: string | null
}
export type SeasonRow = Record<string, number | string | null> & { player_id: string; season: number; position: string | null; team: string | null }
export interface WeekRow {
  player_id: string; season: number; week: number; team: string | null; opponent_team: string | null; position: string | null
  completions: number | null; attempts: number | null; passing_yards: number | null; passing_tds: number | null; passing_interceptions: number | null
  carries: number | null; rushing_yards: number | null; rushing_tds: number | null; targets: number | null; receptions: number | null
  receiving_yards: number | null; receiving_tds: number | null; league_points: number | null
}
export interface CollegeLink {
  college_id: string; college_name: string | null; first_college_season: number | null; last_college_season: number | null
  status: string; method: string | null; evidence: string | null
}
export interface PlayerDetail { player: PlayerIdentity; observation_cutoff: Cutoff; seasons: SeasonRow[]; weeks: WeekRow[]; college: CollegeLink[] }
export interface Rookie {
  player_id: string; player_display_name: string; position: string; team: string | null; draft_pick: number | null
  draft_round: number | null; draft_team: string | null; birth_date: string | null
  college_id: string | null; college_name: string | null; first_college_season: number | null; last_college_season: number | null; college_link_status: string | null
}
export interface Rookies { season: number; total: number; rookies: Rookie[] }
export type CollegeSeasonRow = Record<string, number | string | boolean | string[] | null> & { college_id: string; season: number }
export interface CollegeSeasons { college_id: string; seasons: CollegeSeasonRow[] }

export const fetchPlayers = (params: URLSearchParams, token?: string, signal?: AbortSignal) =>
  get<PlayerDirectory>(`/api/tables/players?${params}`, token, signal)
export const fetchPlayer = (playerId: string, season: string, token?: string, signal?: AbortSignal) =>
  get<PlayerDetail>(`/api/tables/players/${encodeURIComponent(playerId)}${season ? `?season=${season}` : ''}`, token, signal)
export const fetchRookies = (token?: string, signal?: AbortSignal) => get<Rookies>('/api/tables/rookies', token, signal)
export const fetchCollege = (collegeId: string, token?: string, signal?: AbortSignal) =>
  get<CollegeSeasons>(`/api/tables/college/${encodeURIComponent(collegeId)}`, token, signal)
