import type { TeamStrength } from '../api/teamStrength'
import { fixed } from '../format'
import { AnalysisPanel } from './AnalysisPanel'
import { StatList } from './StatCards'
import './TeamCorrelationRisk.css'

const signed = (n: number | null | undefined, digits = 1) => n == null ? '—' : `${n > 0 ? '+' : ''}${n.toFixed(digits)}`
const percent = (n: number | null | undefined) => n == null ? '—' : `${signed(n)}%`
const interval = (v: number[] | null) => v ? `${signed(v[0], 2)} to ${signed(v[1], 2)}` : 'Unavailable'

export function TeamCorrelationRisk({ team }: { team: TeamStrength }) {
  const risk = team.correlation_risk
  if (!risk || risk.status === 'unavailable') return <AnalysisPanel compact id="strength-correlation" heading="Historical correlation risk">
    <p className="notice">Historical connections unavailable.{risk?.issue && ` ${risk.issue}`}</p>
  </AnalysisPanel>
  const connections = risk.connections ?? []
  return <AnalysisPanel compact id="strength-correlation" heading="Historical correlation risk"
    meta={`${risk.history_start}–${risk.history_end} · completed seasons`} aria-label="Historical correlation risk">
    <p className="legend">Recorded starters · same NFL team · QB / WR / TE · descriptive, not a forecast</p>
    {risk.unknown_team_starters ? <p className="notice">{risk.unknown_team_starters} starter(s) have no NFL team identity; connection coverage is incomplete.</p> : null}
    {risk.unmapped_starters ? <p className="notice">{risk.unmapped_starters} starter(s) have no unique historical player identity. Unmatched history stays unavailable.</p> : null}
    {!connections.length ? <p className="legend">No same-NFL-team QB/WR/TE connections among the recorded starters. Other roster relationships are not measured here.</p> : <>
      <div className="correlation-observed">
        <h4>This season · recorded team results</h4>
        <StatList label="Observed correlation contribution" items={[
          { label: 'Team scoring σ', value: `${fixed(team.results.volatility)} pts` },
          { label: 'Connections · 2Σcov', value: `${signed(risk.observed?.covariance_effect)} pts²` },
          { label: 'Share of observed team variance', value: percent(risk.observed?.team_variance_share_pct) },
          { label: 'Completed weeks', value: risk.observed?.n ?? '—', note: `${risk.observed?.covered_connections ?? 0}/${connections.length} connections reconciled` },
        ]} />
        <p className="legend">{(risk.observed?.n ?? 0) < 10 ? 'Short sample · ' : ''}Retrospective variance accounting. Other players can amplify or offset these swings.</p>
      </div>
      <div className="correlation-connections">
        {connections.map(connection => {
          const h = connection.historical
          return <section className="correlation-connection" key={connection.id} aria-label={`${connection.a.name} and ${connection.b.name} historical connection`}>
            <div className="correlation-title"><h4>{connection.a.name} <span>+ {connection.b.name}</span></h4><span>{connection.nfl_team} · {connection.a.position}/{connection.b.position}</span></div>
            <p className="legend">{h.n} shared starts · {h.seasons.join(', ') || 'No prior-season overlap'}</p>
            {h.status !== 'available' ? <p className="legend">{h.status === 'unmapped_identity' ? 'Historical player identity unavailable.' : 'Insufficient shared starts for a variance estimate.'}</p> : <>
              <StatList items={[
                {label: 'Historical correlation r', value: signed(h.r, 2), note: `95% interval: ${interval(h.interval)}`},
                {label: 'Combined scoring σ', value: `${fixed(h.sd)} pts`, note: `Zero-covariance baseline: ${fixed(h.independent_sd)} pts`},
                {label: 'Shared movement · 2cov', value: `${signed(h.covariance_effect)} pts²`, note: 'Common games · season-adjusted'},
                {label: 'Share of pair variance', value: percent(h.variance_share_pct), note: 'Denominator: this pair’s variance'},
              ]} />
              <div className="correlation-variance" aria-label="Historical pair variance decomposition">
                <span>Individual variances <b>{fixed(h.independent_variance)}</b></span>
                <span>Shared movement <b>{signed(h.covariance_effect)}</b></span>
                <span>Combined variance <b>{fixed(h.variance)} pts²</b></span>
              </div>
              <details><summary>Season breakdown & uncertainty</summary>
                <p className="legend">Variance change vs zero covariance: {percent(h.variance_change_pct)}. Bootstrap 95% interval: {h.variance_change_interval ? `${percent(h.variance_change_interval[0])} to ${percent(h.variance_change_interval[1])}` : 'unavailable'}.</p>
                <div className="correlation-table-scroll"><table className="correlation-season-table"><thead><tr><th>Season</th><th>Games</th><th>r</th><th>2cov · pts²</th><th>Pair variance share</th></tr></thead><tbody>
                  {h.by_season.map(year => <tr key={year.season}><th>{year.season}</th><td>{year.n}</td><td>{signed(year.r, 2)}</td><td>{signed(year.covariance_effect)}</td><td>{percent(year.variance_share_pct)}</td></tr>)}
                </tbody></table></div>
              </details>
            </>}
            <p className="correlation-current legend">This season: {connection.observed.status === 'available' ? `${signed(connection.observed.covariance_effect)} pts² · ${percent(connection.observed.team_variance_share_pct)} of observed team variance · n = ${connection.observed.n}` : `Attribution unavailable · ${connection.observed.n} shared recorded starts; requires the full completed-week sample.`}</p>
          </section>
        })}
      </div>
    </>}
    <details className="outlook-method"><summary>Sources & interpretation</summary>
      <p>Historical samples cover the previous five completed seasons, excluding this season. Stable ESPN-to-player IDs connect teammates across every fantasy roster. Each pair uses shared starter games for the NFL team shown; games with other clubs are excluded. Means are removed separately within each season.</p>
      <p>Combined variance = individual variances + 2 × covariance. Positive shared movement increases observed variance; negative movement offsets it. Shares can be negative or exceed 100% when other terms offset them. Pair samples differ, so historical effects are not added into a roster risk estimate. These are exploratory historical associations, not causal effects or predicted outcomes.</p>
      <p>This-season attribution uses recorded fantasy scores, including K/DST in the team total. Both players must have started in every completed, reconciled week (at least three). Prior-season history uses configured league scoring. Neither calculation feeds team forecasts or the recorded scoring standard deviation.</p>
      <p>Historical source: {risk.source?.table} · {risk.source?.source_version} · {risk.source?.version}</p>
    </details>
  </AnalysisPanel>
}
