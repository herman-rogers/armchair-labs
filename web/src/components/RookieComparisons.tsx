import { useQuery } from '@tanstack/react-query'
import { fetchRookieWatch } from '../api/client'
import { DataTable } from './DataTable'

const number = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1)

/** Historical rookie analogs at the shared profile's captured cutoff. */
export function RookieComparisons({ playerId, season, week }: { playerId: string; season: number; week: number }) {
  const report = useQuery({ queryKey: ['rookie-watch'], queryFn: fetchRookieWatch, retry: false })
  if (report.isError) return <p className="legend">Rookie comparisons unavailable: {report.error.message}</p>
  if (!report.data) return <p role="status">Loading historical rookie comparisons…</p>
  const data = report.data
  if (data.season !== season || data.through_week !== week) return <p className="legend">No rookie comparison snapshot at this profile cutoff.</p>
  const player = data.players.find(r => r.player_id === playerId)
  if (!player) return <p className="legend">No captured rookie comparisons for this player.</p>
  const evidence = data.backtest.positions.find(r => r.position === player.position)
  return <section aria-label="Historical rookie comparisons">
    <h4>Historical rookie comparisons</h4>
    <p>{data.season} through Week {data.through_week} · comparison horizon: Weeks {data.horizon[0]}–{data.horizon[1]}.</p>
    <p>{player.forecast_status}. Analog estimate: {number(player.forecast_next4)} points; current-pace baseline: {number(player.pace_next4)}.
      {' '}Historical spread: {number(player.analog_p10)}–{number(player.analog_p90)} points. This spread is not a calibrated prediction interval.</p>
    <p>{player.observed_stat_weeks} box-score weeks and {player.snap_observations} offensive snap observations.
      {' '}Latest completed week: {player.latest_targets ?? 'Unknown'} targets · {player.latest_carries ?? 'Unknown'} carries.</p>
    <p>{player.neighbor_count} comparable rookies used from {player.history_count} earlier players at this position and cutoff. Five closest shown:</p>
    <DataTable rows={player.analogs} columns={[
      { key: 'player_display_name', label: 'Historical player', title: 'Comparable rookie at the same NFL cutoff.', align: 'left' },
      { key: 'season', label: 'Season', title: 'Historical rookie season.', initial: 'asc' },
      { key: 'points_per_week', label: 'Points / week at cutoff', title: 'Recorded scoring pace before the comparison horizon.', render: r => number(r.points_per_week) },
      { key: 'targets', label: 'Targets', title: 'Targets through the historical cutoff.' },
      { key: 'carries', label: 'Carries', title: 'Carries through the historical cutoff.' },
      { key: 'next4_actual', label: 'Following four weeks', title: 'Observed points after that historical cutoff.', render: r => number(r.next4_actual) },
    ]} defaultSort="season" rowKey={r => `${r.season}-${r.player_display_name}`} emptyMessage="No historical comparisons for this sample." />
    <details className="reference-section"><summary>Rookie comparison evidence</summary>
      {evidence && <p>{player.position}: analog error {number(evidence.forecast_next4_mae)} points vs current-pace error {number(evidence.pace_next4_mae)}
        {' '}({evidence.scored} / {evidence.eligible} eligible rookies forecast; lower is better).</p>}
      <p>{data.backtest.basis} {data.method}</p>
      <ul>{data.limitations.map(text => <li key={text}>{text}</li>)}</ul>
      <p className="legend">Historical comparison dataset: {data.history?.version ?? 'See source metadata'}. Saved {new Date(data.saved_at).toLocaleString()}.</p>
    </details>
  </section>
}
