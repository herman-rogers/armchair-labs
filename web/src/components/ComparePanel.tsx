import { useQuery } from '@tanstack/react-query'
import type { LeagueTeam, MetricVersion } from '../api/types'
import { compareQuery } from '../api/queries/archive'
import { TeamCompare } from './TeamCompare'

/**
 * Any two teams, side by side.
 *
 * Both sides are selectable, so following a thought — "who else is thin at tight
 * end?" — is a dropdown change rather than a trip back through a team list. That
 * fluidity is the whole point: the old flow hijacked the roster tab and lost your
 * place.
 */
export function ComparePanel({
  teams,
  left,
  right,
  version,
  onChange,
}: {
  teams: LeagueTeam[]
  left: number
  right: number
  version: MetricVersion
  onChange: (side: 'left' | 'right', teamId: number) => void
}) {
  const comparison = useQuery({ ...compareQuery(left, right, version), enabled: left !== right })

  const picker = (side: 'left' | 'right', value: number) => (
    <select
      className="team-select"
      value={value}
      onChange={(event) => onChange(side, Number(event.target.value))}
      aria-label={`${side} team`}
    >
      {teams.map((team) => (
        <option key={team.team_id} value={team.team_id}>
          {team.team_name}
          {team.is_mine ? ' (you)' : ''}
        </option>
      ))}
    </select>
  )

  return (
    <section>
      <div className="compare-controls">
        {picker('left', left)}
        <button
          type="button"
          className="chip"
          onClick={() => {
            onChange('left', right)
            onChange('right', left)
          }}
          title="Swap sides"
        >
          ⇄
        </button>
        {picker('right', right)}
      </div>

      {left === right && (
        <div className="notice">Pick two different teams to compare.</div>
      )}
      {comparison.isError && (
        <div className="notice">{(comparison.error as Error).message}</div>
      )}
      {comparison.isLoading && left !== right && <div className="notice">Comparing…</div>}
      {comparison.data && (
        <TeamCompare
          left={comparison.data.left}
          right={comparison.data.right}
          margin={comparison.data.margin}
          version={version}
          context="rosters"
        />
      )}
    </section>
  )
}
