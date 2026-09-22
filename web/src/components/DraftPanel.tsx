import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchDraft } from '../api/client'
import type { DraftPickAnalysis, DraftTeamGrade, MetricVersion } from '../api/types'
import { boardMetricLabel } from '../metricPresentation'
import { DataTable, type Column } from './DataTable'

function signed(value: number | null) {
  if (value == null) return '—'
  return `${value > 0 ? '+' : ''}${value}`
}

/** Sorting verdicts alphabetically is meaningless; order them best-to-worst. */
const VERDICT_ORDER: Record<DraftPickAnalysis['verdict'], number> = {
  steal: 0,
  fair: 1,
  reach: 2,
  unrated: 3,
}

function gradeColumns(metric: string, onTeam: (teamId: number) => void): Column<DraftTeamGrade>[] {
  return [
    {
      key: 'grade_rank',
      label: '#',
      title: 'Grade rank: teams ordered by captured value.',
      initial: 'asc',
      render: (team) => <span className="strong">#{team.grade_rank}</span>,
    },
    {
      key: 'team_name',
      label: 'Team',
      title: 'Click a team to filter the pick list below to its picks.',
      align: 'left',
      initial: 'asc',
      render: (team) => (
        <>
          <button type="button" className="team-link" onClick={() => onTeam(team.team_id)}>
            <span>{team.team_name}</span>
          </button>
          {team.is_mine && <span className="badge mine">You</span>}
        </>
      ),
    },
    {
      key: 'captured_value',
      label: 'Captured value',
      title: `Sum of the team’s rated picks on ${metric}.`,
      render: (team) => <span className="metric-emphasis">{team.captured_value.toFixed(1)}</span>,
    },
    {
      key: 'value_vs_board',
      label: 'vs board',
      title: 'Sum of pick number minus board rank; positive means value fell to the team.',
      render: (team) => signed(team.value_vs_board),
    },
    {
      key: 'value_vs_market',
      label: 'vs market',
      title: 'The same, against the FantasyPros consensus instead of the board.',
      render: (team) => signed(team.value_vs_market),
    },
    { key: 'steals', label: 'Steals', title: 'Picks that went 15+ spots later than the board’s rank.' },
    { key: 'reaches', label: 'Reaches', title: 'Picks that went 15+ spots earlier than the board’s rank.' },
    {
      key: 'value_left_on_board',
      label: 'Left on board',
      title: 'Sum, over the team’s picks, of how much better the best still-available player was. Lower is better.',
      initial: 'asc',
      render: (team) => team.value_left_on_board.toFixed(1),
    },
    {
      key: 'rated_picks',
      label: 'Rated / picks',
      title: 'How many of the team’s picks the board could rate; kickers and defenses are tiered, not ranked.',
      render: (team) => (
        <span className="dim">
          {team.rated_picks}/{team.picks}
        </span>
      ),
    },
  ]
}

function pickColumns(metric: string): Column<DraftPickAnalysis>[] {
  return [
    {
      key: 'overall',
      label: 'Pick',
      title: 'Round.pick, with the overall pick number.',
      initial: 'asc',
      render: (pick) => (
        <span className="dim">
          {pick.round}.{pick.round_pick} <small>({pick.overall})</small>
        </span>
      ),
    },
    {
      key: 'team_name',
      label: 'Team',
      title: 'The drafting team.',
      align: 'left',
      initial: 'asc',
      render: (pick) => <span className="dim">{pick.team_name}</span>,
    },
    {
      key: 'player_display_name',
      label: 'Player',
      title: 'The player taken. ★ marks placement by consensus market rank, with no model tape.',
      align: 'left',
      initial: 'asc',
      render: (pick) => (
        <span className="name">
          {pick.player_display_name}
          {pick.rank_source === 'market' && (
            <span title="Placed by consensus market rank; no model tape"> ★</span>
          )}
        </span>
      ),
    },
    {
      key: 'position',
      label: 'Pos',
      title: 'Position.',
      initial: 'asc',
      render: (pick) => pick.position ?? '—',
    },
    {
      key: 'board_rank',
      label: 'Board #',
      title: 'The board’s overall rank for the player.',
      initial: 'asc',
      render: (pick) => pick.board_rank ?? '—',
    },
    {
      key: 'position_rank',
      label: 'Pos rk',
      title: 'The board’s positional rank.',
      initial: 'asc',
      render: (pick) => (
        <span className="dim">
          {pick.position_rank != null && pick.position ? `${pick.position}${pick.position_rank}` : '—'}
        </span>
      ),
    },
    {
      key: 'board_value',
      label: 'Value',
      title: `The player’s ${metric}.`,
      render: (pick) => (
        <span className="strong">{pick.board_value == null ? '—' : pick.board_value.toFixed(2)}</span>
      ),
    },
    {
      key: 'espn_draft_rank',
      label: 'ESPN #',
      title: 'ESPN PPR draft-room rank at the snapshot.',
      initial: 'asc',
      render: (pick) => <span className="dim">{pick.espn_draft_rank ?? '—'}</span>,
    },
    {
      key: 'value_vs_board',
      label: 'vs board',
      title: 'Pick number minus board rank; positive = the player went later than the board values him.',
      render: (pick) => signed(pick.value_vs_board),
    },
    {
      key: 'value_vs_market',
      label: 'vs market',
      title: 'Pick number minus consensus market rank.',
      render: (pick) => <span className="dim">{signed(pick.value_vs_market)}</span>,
    },
    {
      key: 'verdict',
      label: 'Verdict',
      title: 'Steal or reach at ±15 spots against the board. Sorts best to worst.',
      initial: 'asc',
      value: (pick) => VERDICT_ORDER[pick.verdict],
      render: (pick) => <span className={`draft-verdict ${pick.verdict}`}>{pick.verdict}</span>,
    },
    {
      key: 'best_available_gap',
      label: 'Best available',
      title: 'Top board players still on the board when the pick was made; sorts by how much value was left.',
      align: 'left',
      render: (pick) => (
        <span className="dim">
          {pick.best_available.map((entry) => `${entry.player_display_name} (#${entry.board_rank})`).join(', ') || '—'}
          {pick.best_available_gap != null && pick.best_available_gap > 0 && (
            <small className="cell-sub">
              {' '}
              +{pick.best_available_gap.toFixed(1)} {metric} left
            </small>
          )}
        </span>
      ),
    },
  ]
}

/** The draft recap scored against the board: team grades, then every pick with exact best-available. */
export function DraftPanel({ version }: { version: MetricVersion }) {
  const draft = useQuery({ queryKey: ['draft', version], queryFn: () => fetchDraft(version) })
  const [teamId, setTeamId] = useState<number | 'all'>('all')

  const picks = useMemo(() => {
    const rows = draft.data?.picks ?? []
    return teamId === 'all' ? rows : rows.filter((pick) => pick.team_id === teamId)
  }, [draft.data, teamId])

  if (draft.isLoading) return <div className="notice">Loading draft recap…</div>
  if (draft.isError || !draft.data) {
    return <div className="notice">Could not load the draft: {(draft.error as Error)?.message}</div>
  }
  if (draft.data.pick_count === 0) {
    return (
      <div className="notice">
        <h2>No draft recap in the snapshot</h2>
        Refresh the league state after the draft; ESPN publishes the recap once the draft completes.
      </div>
    )
  }

  const metric = boardMetricLabel(draft.data.ranking_metric)
  return (
    <section className="draft-panel">
      <div className="board-section-head">
        <div>
          <h3>Draft grades</h3>
          <p>
            Captured value sums each team’s picks on {metric}. Value vs board is pick number minus
            the board’s rank (positive = the player went later than the board values him); steals
            and reaches are ±15 spots. Value left on board sums, over a team’s picks, how much
            better the best still-available player was.
          </p>
        </div>
        <span className="count">{draft.data.pick_count} picks · {draft.data.teams.length} teams</span>
      </div>
      <DataTable
        rows={draft.data.teams}
        columns={gradeColumns(metric, setTeamId)}
        defaultSort="grade_rank"
        rowKey={(team) => team.team_id}
        rowClass={(team) => (team.is_mine ? 'mine-row' : undefined)}
      />

      <div className="board-section-head">
        <div>
          <h3>Picks</h3>
          <p>Best available is exact: the top board players not yet taken when the pick was made.</p>
        </div>
        <label className="field">
          Team
          <select
            value={teamId}
            onChange={(event) => setTeamId(event.target.value === 'all' ? 'all' : Number(event.target.value))}
          >
            <option value="all">All teams</option>
            {draft.data.teams.map((team) => (
              <option key={team.team_id} value={team.team_id}>{team.team_name}</option>
            ))}
          </select>
        </label>
      </div>
      <DataTable
        rows={picks}
        columns={pickColumns(metric)}
        defaultSort="overall"
        rowKey={(pick) => pick.overall}
        rowClass={(pick) => (pick.is_mine ? 'mine-row' : undefined)}
        emptyMessage="No picks for this team in the snapshot."
      />
    </section>
  )
}
