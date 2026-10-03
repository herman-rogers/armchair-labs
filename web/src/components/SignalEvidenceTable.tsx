import { useState } from 'react'
import type { MetricBacktestResult, MetricCatalogEntry } from '../api/types'
import { DataTable, type Column } from './DataTable'
import { percent, signed } from '../format'

export function SignalEvidenceTable({
  rows,
  catalog,
}: {
  rows: MetricBacktestResult[]
  catalog: MetricCatalogEntry[]
}) {
  const [showAll, setShowAll] = useState(false)
  const strength = (row: MetricBacktestResult) => {
    const value = row.partial_spearman ?? row.spearman
    return value == null ? null : Math.abs(value)
  }
  const columns: Column<MetricBacktestResult>[] = [
    {
      key: 'label',
      label: 'Signal',
      title: 'Open Details for the definition and seasonal evidence.',
      align: 'left',
    },
    { key: 'group', label: 'Family', title: 'Input family.', align: 'left' },
    {
      key: 'assessment',
      label: 'Assessment',
      title: 'Report classification under its configured thresholds.',
      render: (row) => (
        <span className={`metric-assessment ${row.assessment}`}>{row.assessment}</span>
      ),
    },
    {
      key: 'strength',
      label: 'Adjusted |ρ|',
      title:
        'Absolute partial Spearman correlation, falling back to raw correlation if unavailable.',
      value: strength,
      render: (row) => strength(row)?.toFixed(3) ?? '—',
    },
    {
      key: 'spearman',
      label: 'Spearman',
      title: 'Rank association with the next-season outcome.',
      render: (row) => signed(row.spearman, 3),
    },
    {
      key: 'partial_spearman',
      label: 'Partial vs prior',
      title: 'Rank association after controlling for historical PPG prior.',
      render: (row) => signed(row.partial_spearman, 3),
    },
    {
      key: 'direction_consistency',
      label: 'Consistency',
      title: 'Share of years agreeing with the pooled direction.',
      render: (row) => percent(row.direction_consistency, 1),
    },
    {
      key: 'coverage',
      label: 'Coverage',
      title: 'Share of eligible observations with this signal.',
      render: (row) => percent(row.coverage, 1),
    },
    { key: 'n', label: 'N', title: 'Observed player-seasons.' },
    { key: 'folds', label: 'Seasons', title: 'Completed forecast seasons.' },
  ]
  return (
    <>
      <div className="section-heading">
        <span className="count">{rows.length} matching signals</span>
        {rows.length > 25 && (
          <button
            type="button"
            className="chip"
            aria-pressed={showAll}
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? 'Show strongest 25' : `Show all ${rows.length} signals`}
          </button>
        )}
      </div>
      <DataTable
        key={rows[0] ? `${rows[0].window}-${rows[0].position}-${rows[0].target}` : 'empty'}
        rows={showAll ? rows : rows.slice(0, 25)}
        columns={columns}
        defaultSort="strength"
        rowKey={(row) => row.metric}
        rowLabel={(row) => row.label}
        emptyMessage="No signals match these filters."
        renderDetails={(row) => (
          <div className="player-details">
            <h4>{row.label}</h4>
            <p>
              {catalog.find((entry) => entry.key === row.metric)?.description ??
                'No definition recorded.'}
            </p>
            <div className="research-kpis">
              <div>
                <span>Fold 2.5–97.5% interval</span>
                <strong className="compact-number">
                  {signed(row.spearman_ci_low, 3)} to {signed(row.spearman_ci_high, 3)}
                </strong>
                <small>Reported Spearman interval</small>
              </div>
              <div>
                <span>Observed / eligible</span>
                <strong>
                  {row.n.toLocaleString()} / {row.eligible.toLocaleString()}
                </strong>
                <small>{percent(row.coverage, 1)} signal coverage</small>
              </div>
              <div>
                <span>Partial versus prior</span>
                <strong>{signed(row.partial_spearman, 3)}</strong>
                <small>Association beyond historical PPG</small>
              </div>
            </div>
            <div className="signal-seasons">
              {row.fold_results.map((fold) => (
                <div key={fold.forecast_season}>
                  <b>{fold.forecast_season}</b>
                  <div className="correlation-track">
                    <i
                      style={{
                        left: `${50 + Math.min(fold.spearman ?? 0, 0) * 50}%`,
                        width: `${Math.abs(fold.spearman ?? 0) * 50}%`,
                      }}
                    />
                  </div>
                  <span>{signed(fold.spearman, 3)}</span>
                  <small>N = {fold.n}</small>
                </div>
              ))}
            </div>
            <p className="legend">
              Bars show each season’s Spearman correlation on a −1 to +1 scale, centered at zero.
              Missing observations are labeled —.
            </p>
          </div>
        )}
      />
    </>
  )
}
