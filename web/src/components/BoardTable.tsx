import type { MetricVersion, Player } from '../api/types'
import {
  OVERALL_VOR_TITLE,
  POSITION_RANK_TITLE,
  SEASON_EQUIVALENT_TITLE,
  rankBasis,
  rankBasisTitle,
} from '../metricPresentation'
import { DataTable, type Column as DataColumn } from './DataTable'
import { Flags } from './Flags'

type Column = DataColumn<Player>

const number = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? <span className="faint">—</span> : value.toFixed(digits)

const percent = (value: number | null | undefined) =>
  value === null || value === undefined ? (
    <span className="faint">—</span>
  ) : (
    `${(value * 100).toFixed(0)}%`
  )

const IDENTITY_COLUMNS: Column[] = [
  {
    key: 'rank',
    label: '#',
    title: 'Overall board rank, after manual overrides where present.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="rank">{p.rank}</span>,
  },
  {
    key: 'player_display_name',
    label: 'Player',
    title: 'Player name.',
    align: 'left',
    initial: 'asc',
    render: (p) => (
      <span className="name">
        {p.player_display_name}
        {p.override_reason ? (
          <>
            {' '}
            <span className="override" title={p.override_reason}>
              *
            </span>
          </>
        ) : null}
        {p.projection_reason ? (
          <>
            {' '}
            <span className="override" title={p.projection_reason}>
              ◇
            </span>
          </>
        ) : null}
      </span>
    ),
  },
  {
    key: 'position',
    label: 'Pos',
    title: 'Position.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className={`pos ${p.position}`}>{p.position}</span>,
  },
  {
    key: 'team',
    label: 'Tm',
    title: 'Team as of the last game of the season.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="team">{p.team}</span>,
  },
]

const V1_COLUMNS: Column[] = [
  {
    key: 'adj_vor',
    label: 'VOR',
    title:
      'Value over replacement: points per game above the freely-available player at this position. What the board sorts by.',
    render: (p) => <span className="strong">{p.adj_vor.toFixed(1)}</span>,
  },
  {
    key: 'ppg',
    label: 'PPG',
    title: 'League-scored points per game, including big-play bonuses.',
    render: (p) => number(p.ppg),
  },
  {
    key: 'flags',
    label: 'Flags',
    title: 'Historical heuristics and sample warnings. Hover any chip for its limits.',
    align: 'left',
    initial: 'asc',
    render: (p) => <Flags value={p.flags} />,
  },
  {
    key: 'games',
    label: 'G',
    title: 'Games played. Small samples carry an Ngms flag.',
    render: (p) => <span className="dim">{p.games}</span>,
  },
  {
    key: 'floor',
    label: 'Floor',
    title: 'Historical 25th-percentile weekly score; a noisy lineup descriptor, especially in small samples.',
    render: (p) => <span className="dim">{number(p.floor)}</span>,
  },
  {
    key: 'volatility',
    label: 'Vol',
    title:
      'Standard deviation of weekly scores. Buy it deliberately when you need a ceiling, avoid it when you are favoured.',
    render: (p) => <span className="dim">{number(p.volatility)}</span>,
  },
  {
    key: 'bonus_pts',
    label: 'Bonus',
    title:
      "Big-play bonus points from 40+ and 50+ yard touchdowns. This league's private edge, kept visible.",
    render: (p) => (
      <span className={p.bonus_pts > 0 ? 'dim' : 'faint'}>{p.bonus_pts.toFixed(0)}</span>
    ),
  },
  {
    key: 'target_share',
    label: 'Tgt%',
    title: 'Share of the team’s targets, weighted by weekly usage.',
    render: (p) => <span className="dim">{percent(p.target_share)}</span>,
  },
  {
    key: 'air_yards_share',
    label: 'AY%',
    title:
      'Share of the team’s air yards. Matters extra here: deep targets feed the bonus brackets.',
    render: (p) => <span className="dim">{percent(p.air_yards_share)}</span>,
  },
  {
    key: 'age_at_season',
    label: 'Age',
    title: 'Age as of Sept 1 of the season being played.',
    render: (p) => <span className="dim">{number(p.age_at_season)}</span>,
  },
]

const V2_COLUMNS: Column[] = [
  {
    key: 'projected_team',
    label: 'Proj Tm',
    title: 'Team whose forward offensive, quarterback, and teammate context is applied.',
    align: 'left',
    initial: 'asc',
    render: (p) => <span className="team">{p.projected_team ?? p.team}</span>,
  },
  {
    key: 'v2_overall_vor',
    label: 'Overall VOR',
    title: OVERALL_VOR_TITLE,
    render: (p) => <span className="strong">{number(p.v2_overall_vor)}</span>,
  },
  {
    key: 'v2_position_rank',
    label: 'Pos Rk',
    title: POSITION_RANK_TITLE,
    render: (p) => <span className="dim">{p.v2_position_rank ?? '—'}</span>,
  },
  {
    key: 'v2_rank_value',
    label: 'Rank basis',
    title: 'The configured projection and raw value used for this player’s position rank.',
    render: (p) => <span className="dim" title={rankBasisTitle(p)}>{rankBasis(p)}</span>,
  },
  {
    key: 'market_ecr',
    label: 'ECR',
    title: 'FantasyPros consensus positional rank at the draft snapshot. Kept as information; it is the fallback basis only for players the model cannot rate.',
    render: (p) => (
      <span className={p.rank_source === 'market' ? 'strong' : 'dim'}>
        {p.market_ecr == null ? '—' : `${p.market_position ?? ''}${Math.round(p.market_ecr)}`}
        {p.rank_source === 'market' ? ' ★' : ''}
      </span>
    ),
  },
  {
    key: 'proj_ppg',
    label: 'Proj PPG',
    title: 'V2 projection per active game (history and stat-line branches); a fitted-ranker input, not the sort.',
    render: (p) => number(p.proj_ppg),
  },
  {
    key: 'season_equivalent_ppg',
    label: 'Avail PPG',
    title: SEASON_EQUIVALENT_TITLE,
    render: (p) => <span className="dim">{number(p.season_equivalent_ppg)}</span>,
  },
  {
    key: 'expected_games',
    label: 'Exp G',
    title: 'Expected active games, estimated from participation and injury-report history and shrunk toward the league prior.',
    render: (p) => <span className="dim">{number(p.expected_games)}</span>,
  },
  {
    key: 'fitted_season_points',
    label: 'Season Pts',
    title: 'Fitted season points: fitted PPG multiplied by the fitted games projection.',
    render: (p) => <span className="dim">{number(p.fitted_season_points, 0)}</span>,
  },
  {
    key: 'ppg',
    label: 'Actual PPG',
    title: 'Most recent season’s league-scored PPG. A strong naive baseline and supporting evidence, not the overall V2 sort.',
    render: (p) => <span className="dim">{number(p.ppg)}</span>,
  },
  {
    key: 'projection_confidence',
    label: 'Conf',
    title: 'Sample support for the projection. Team moves and manual role assumptions reduce it.',
    render: (p) => <span className="dim">{percent(p.projection_confidence)}</span>,
  },
  {
    key: 'depth_chart_rank',
    label: 'Depth',
    title: 'Latest published nflverse depth-chart rank. This automatically affects projected role; it is not ESPN fantasy lineup status.',
    render: (p) => <span className="dim">{p.depth_chart_rank ?? '—'}</span>,
  },
  {
    key: 'projected_targets_pg',
    label: 'Tgt/G',
    title: 'Projected targets per game from route opportunities, shrunk target share, and team pass volume.',
    render: (p) => <span className="dim">{number(p.projected_targets_pg)}</span>,
  },
  {
    key: 'projected_carries_pg',
    label: 'Car/G',
    title: 'Projected carries per game from shrunk carry share and team rushing volume.',
    render: (p) => <span className="dim">{number(p.projected_carries_pg)}</span>,
  },
  {
    key: 'projected_route_participation',
    label: 'Route%',
    title: 'Projected share of team dropbacks with an eligible-player route opportunity. Participation cannot distinguish a released route from pass protection.',
    render: (p) => <span className="dim">{percent(p.projected_route_participation)}</span>,
  },
  {
    key: 'projected_end_zone_targets_pg',
    label: 'EZ Tgt/G',
    title: 'Projected end-zone targets per game from nflverse play-by-play usage.',
    render: (p) => <span className="dim">{number(p.projected_end_zone_targets_pg, 2)}</span>,
  },
  {
    key: 'projected_goal_line_carries_pg',
    label: 'GL Car/G',
    title: 'Projected carries from the opponent five-yard line or closer per game.',
    render: (p) => <span className="dim">{number(p.projected_goal_line_carries_pg, 2)}</span>,
  },
  {
    key: 'flags',
    label: 'Flags',
    title: 'Historical model signals retained as supporting evidence.',
    align: 'left',
    initial: 'asc',
    render: (p) => <Flags value={p.flags} />,
  },
  {
    key: 'age_at_season',
    label: 'Age',
    title: 'Age as of Sept 1 of the projected season; v2 applies a gradual position-specific curve.',
    render: (p) => <span className="dim">{number(p.age_at_season)}</span>,
  },
]

const ADAPTIVE_COLUMNS: Column[] = V2_COLUMNS.map((column) =>
  column.key === 'fitted_season_points'
    ? {
        key: 'adaptive_season_points',
        label: 'Adaptive Pts',
        title: 'Frozen 2026 Adaptive PPG-selector forecast on a season-points scale.',
        render: (p) => <span className="strong">{number(p.adaptive_season_points, 0)}</span>,
      }
    : column,
)

export function BoardTable({ players, version }: { players: Player[]; version: MetricVersion }) {
  const projected = version === 'adaptive' ? ADAPTIVE_COLUMNS : V2_COLUMNS
  const columns = version !== 'v1' ? [...IDENTITY_COLUMNS.slice(0, 3), ...projected] : [...IDENTITY_COLUMNS, ...V1_COLUMNS]

  // Remount per version: the column sets differ, so a carried-over sort key could
  // reference a column the new set does not show.
  return (
    <DataTable
      key={version}
      rows={players}
      columns={columns}
      defaultSort="rank"
      rowKey={(player) => player.player_id}
    />
  )
}
