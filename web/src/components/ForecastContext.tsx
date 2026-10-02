import type { MetricVersion, Status } from '../api/types'
import { SYSTEM_LABELS } from '../metricPresentation'

const STATES: Record<string, { label: string; description: string }> = {
  current: {
    label: 'Verified artifact',
    description: 'The saved forecast matches its recorded data and model dependencies.',
  },
  frozen_reference: {
    label: 'Verified draft archive',
    description: 'The preserved draft reference matches its archived snapshot.',
  },
  stale: {
    label: 'Stale artifact',
    description:
      'The saved forecast no longer matches its recorded inputs. Its numbers may differ from the latest model evidence.',
  },
  degraded: {
    label: 'Incomplete coverage',
    description:
      'The saved board has players without a usable forecast. Check each player’s source.',
  },
  unverified: {
    label: 'Unverified artifact',
    description:
      'This saved board has no verified provenance. Its forecast cutoff and model version may be incomplete.',
  },
}

export function ForecastContext({ status, version }: { status: Status; version: MetricVersion }) {
  const metadata = status.metric_versions[version]
  const state = metadata.provenance?.state ?? 'unverified'
  const info = STATES[state] ?? STATES.unverified
  const dates = metadata.forecast_as_of ?? []
  return (
    <section className={`forecast-context ${state}`} aria-label="Selected forecast context">
      <div className="forecast-context-head">
        <strong>
          {SYSTEM_LABELS[version]} · {status.league.draft_season}
        </strong>
        <span className="artifact-state">{metadata.available ? info.label : 'Not available'}</span>
        <span className="forecast-cutoff">
          {version === 'v1'
            ? `${status.league.board_season} historical production`
            : dates.length
              ? `Forecast cutoff ${dates.join(' / ')}`
              : 'Forecast cutoff not recorded'}
        </span>
      </div>
      <p>
        {version === 'v1'
          ? 'The original draft ordering and manual adjustments, preserved for reference.'
          : version === 'adaptive'
            ? 'Experimental season totals frozen before the season for a prospective evaluation.'
            : 'The production preseason forecast: active-game PPG, expected games, and season points.'}{' '}
        Ownership and injury badges can update independently of these numbers.
      </p>
      <details>
        <summary>Forecast source and freshness</summary>
        <p>{info.description}</p>
        <dl className="stat-facts">
          <div>
            <dt>Saved artifact</dt>
            <dd>
              {metadata.built_at ? new Date(metadata.built_at).toLocaleString() : 'Not recorded'}
            </dd>
          </div>
          <div>
            <dt>Players</dt>
            <dd>{metadata.player_count.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Artifact ID</dt>
            <dd className="artifact-id">{metadata.provenance?.artifact_id ?? 'Not recorded'}</dd>
          </div>
        </dl>
        <p>
          The saved-artifact date records publication, not a new forecast cutoff.
          {version === 'adaptive' &&
            ' League points, floor, and risk use a separate V2 simulation; the frozen forecast supplies season totals and ranking value only.'}
        </p>
      </details>
    </section>
  )
}
