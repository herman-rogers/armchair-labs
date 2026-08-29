import { useQuery } from '@tanstack/react-query'
import { fetchOpponents, fetchTransactions } from '../api/client'
import type { LeagueTeam, MetricVersion } from '../api/types'

/** ESPN times arrive as ISO strings; render them the way a person reads a date. */
function formatWhen(value: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

/**
 * Where every roster is thin, and what the league has been paying.
 *
 * Both are opponent intelligence. Positional depth tells you which trade offer lands
 * and which claim blocks someone; the transaction log is the league's revealed price
 * signal — what a rival actually spent tells you more than what they say they value.
 */
export function TeamsPanel({
  teams,
  version,
  onSelectTeam,
}: {
  teams: LeagueTeam[]
  version: MetricVersion
  onSelectTeam: (team: LeagueTeam) => void
}) {
  const opponents = useQuery({
    queryKey: ['opponents', version],
    queryFn: () => fetchOpponents(version),
  })
  const transactions = useQuery({ queryKey: ['transactions'], queryFn: fetchTransactions })

  const byTeam = new Map<string, { position: string; best_vor: number; best_player: string }[]>()
  for (const row of opponents.data?.teams ?? []) {
    const list = byTeam.get(row.owner_team_name) ?? []
    list.push(row)
    byTeam.set(row.owner_team_name, list)
  }

  return (
    <section>
      <div className="team-grid">
        {teams.map((team) => {
          const rows = (byTeam.get(team.team_name) ?? [])
            .slice()
            .sort((a, b) => a.best_vor - b.best_vor)
          return (
            <button
              type="button"
              key={team.team_id}
              className={`team-card ${team.is_mine ? 'mine' : ''}`}
              onClick={() => onSelectTeam(team)}
            >
              <div className="team-head">
                <span className="team-name">{team.team_name}</span>
                {team.is_mine && <span className="badge mine">You</span>}
              </div>
              <div className="team-meta faint">
                {team.owner ?? 'unknown'} · {team.wins}-{team.losses}
                {team.faab_remaining !== null && <> · ${team.faab_remaining} FAAB</>}
              </div>
              {rows.length > 0 && (
                <div className="team-thin">
                  <span className="faint">thinnest:</span>{' '}
                  {rows.slice(0, 2).map((row) => (
                    <span key={row.position} className="thin-item">
                      <span className={`pos ${row.position}`}>{row.position}</span>
                      <span className="dim">{row.best_vor.toFixed(1)}</span>
                    </span>
                  ))}
                </div>
              )}
            </button>
          )
        })}
      </div>

      <h3 className="section-head">Recent transactions</h3>
      <p className="faint tight">
        What the league actually pays. A bid clearing real money is the market repricing a
        player in public.
      </p>
      {transactions.data && transactions.data.transactions.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="left">When</th>
                <th className="left">Team</th>
                <th className="left">Action</th>
                <th className="left">Player</th>
                <th>Bid</th>
              </tr>
            </thead>
            <tbody>
              {transactions.data.transactions.slice(0, 25).map((entry, index) => (
                <tr key={`${entry.date}-${entry.player_name}-${index}`}>
                  <td className="left faint">{formatWhen(entry.date)}</td>
                  <td className="left">{entry.team_name ?? '—'}</td>
                  <td className="left dim">{entry.kind ?? '—'}</td>
                  <td className="left name">{entry.player_name ?? '—'}</td>
                  <td>
                    {entry.bid_amount ? (
                      `$${entry.bid_amount}`
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="notice">
          No transactions yet this season. The league has not made a move since the draft.
        </div>
      )}
    </section>
  )
}
