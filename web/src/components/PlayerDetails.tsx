import { PlayerProfileLink } from './PlayerProfile'
import type { MetricVersion, Player } from '../api/types'
import { forecastSource, forecastValues, rankBasis, rankerLabel } from '../metricPresentation'
import { fixed, percent } from '../format'

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="stat-facts">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function PlayerDetails({ player: p, version }: { player: Player; version: MetricVersion }) {
  const forecast = forecastValues(p, version)
  return (
    <div className="player-details">
      <h4>{p.player_display_name} · forecast and evidence</h4>
      <PlayerProfileLink playerId={p.player_id} />
      <div className="detail-grid">
        <section>
          <h5>{version === 'v1' ? 'Preserved draft reference' : 'Selected forecast'}</h5>
          <Facts
            rows={
              version === 'v1'
                ? [
                    ['Historical PPG', fixed(p.ppg)],
                    ['Games counted', fixed(p.games, 0)],
                    ['Season points', fixed(p.season_pts, 0)],
                  ]
                : [
                    ['Active-game PPG', fixed(forecast.activePPG)],
                    ['Expected games', fixed(forecast.expectedGames)],
                    ['Season points', fixed(forecast.seasonPoints, 0)],
                    ['Season-equivalent PPG', fixed(p.season_equivalent_ppg)],
                    ['Source', forecastSource(p, version)],
                    ['Forecast cutoff', p.forecast_as_of ?? 'Not recorded'],
                    ['Sample support', percent(p.forecast_sample_support ?? p.projection_confidence)],
                  ]
            }
          />
          {version === 'adaptive' && (
            <p>
              The frozen season total has no active-game PPG or expected-games estimate. League
              points and risk use a separate V2 simulation.
            </p>
          )}
          {version !== 'v1' && (
            <p>
              Sample support describes the inputs; it is not a probability that the forecast is
              correct. Season totals use unrounded inputs.
            </p>
          )}
        </section>
        <section>
          <h5>Why this rank?</h5>
          <Facts
            rows={
              version === 'v1'
                ? [
                    ['Draft VOR', fixed(p.adj_vor)],
                    ['Replacement PPG', fixed(p.repl_ppg)],
                    ['Manual adjustment', fixed(p.override_delta)],
                  ]
                : [
                    ['Overall VOR', fixed(p.v2_overall_vor)],
                    ['Position rank', fixed(p.v2_position_rank, 0)],
                    ['Position basis', p.v2_rank_key ? rankerLabel(p.v2_rank_key) : 'Not recorded'],
                    ['Basis value', rankBasis(p)],
                    ['Position VOR', fixed(p.v2_rank_vor)],
                    ['Consensus position rank', fixed(p.market_ecr, 0)],
                    ['Market snapshot', p.market_snapshot ?? 'Not recorded'],
                  ]
            }
          />
          <p>
            {version === 'v1'
              ? 'The original draft ordering and manual adjustments are preserved.'
              : 'Overall order compares season value above positional replacement. Position rank follows its own configured basis and can differ.'}
          </p>
          {p.override_reason && <p>Manual adjustment: {p.override_reason}</p>}
        </section>
        <section>
          <h5>Historical production</h5>
          <Facts
            rows={[
              ['PPG', fixed(p.ppg)],
              ['Games', fixed(p.games, 0)],
              ['Season points', fixed(p.season_pts, 0)],
              ['Weekly 25th percentile', fixed(p.floor)],
              ['Weekly standard deviation', fixed(p.volatility)],
              ['Big-play bonus points', fixed(p.bonus_pts, 0)],
              ['Target share', percent(p.target_share)],
              ['Air-yards share', percent(p.air_yards_share)],
              ['Age in reference season', fixed(p.age_at_season)],
            ]}
          />
          <p>
            {version === 'v1'
              ? 'PPG uses the draft reference’s production-row denominator.'
              : 'Historical PPG uses participation-observed active games.'}{' '}
            Weekly percentiles and variability describe past performance.
          </p>
        </section>
      </div>
      {version !== 'v1' && (
        <details className="research-inputs">
          <summary>Research inputs and projection assumptions</summary>
          <p>
            Supporting estimates used to construct the forecast. These are not additional
            independent predictions.
          </p>
          <Facts
            rows={[
              ['Component PPG', fixed(p.component_proj_ppg)],
              ['Historical prior PPG', fixed(p.historical_ppg_prior)],
              ['Blended input PPG', fixed(p.proj_ppg)],
              ['Input expected games', fixed(p.expected_games)],
              ['Targets / game', fixed(p.projected_targets_pg)],
              ['Carries / game', fixed(p.projected_carries_pg)],
              ['Dropback on-field share (proxy)', percent(p.projected_route_participation)],
              ['End-zone targets / game', fixed(p.projected_end_zone_targets_pg, 2)],
              ['Depth rank', fixed(p.depth_chart_rank, 0)],
              ['Depth snapshot', p.depth_chart_date ?? 'Not recorded'],
              ['Projected NFL team', p.projected_team ?? 'Not recorded'],
            ]}
          />
          <p>On-field dropback participation is not verified routes run: a back or tight end may be blocking. It is a historical projection input, not a live charted-route feed.</p>
          {p.projection_reason && <p>{p.projection_reason}</p>}
        </details>
      )}
    </div>
  )
}

export function RankingExplanation({ version }: { version: MetricVersion }) {
  return (
    <p className="ranking-explanation">
      {version === 'v1' ? (
        <>
          <b>Draft reference.</b> Preserved ordering and historical production, including manual
          adjustments.
        </>
      ) : (
        <>
          <b>Overall rank</b> uses season value above positional replacement. <b>Position rank</b>{' '}
          uses its configured projection; open Details to see the basis.
          {version === 'adaptive' &&
            ' Adaptive supplies a frozen season total only; PPG and games are not estimated.'}
        </>
      )}
    </p>
  )
}
