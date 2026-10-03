import type { Position } from './api/types'

/** The fantasy skill positions every model and table covers, in display order. */
export const SKILL_POSITIONS: readonly Position[] = ['QB', 'RB', 'WR', 'TE']
/** Position filter choices: all skill positions, or one. */
export const POSITION_FILTERS: readonly string[] = ['ALL', ...SKILL_POSITIONS]
/** League rosters also carry kickers and defenses. */
export const LEAGUE_POSITION_FILTERS: readonly string[] = [...POSITION_FILTERS, 'K', 'D/ST']

export const isSkillPosition = (position: string | null | undefined): position is Position =>
  (SKILL_POSITIONS as readonly string[]).includes(position ?? '')

/** Sort key for display order (QB, RB, WR, TE); other values sort first. */
export const positionOrder = (position: string) => (SKILL_POSITIONS as readonly string[]).indexOf(position)
