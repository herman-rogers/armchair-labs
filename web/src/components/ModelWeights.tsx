import { useState } from 'react'
import type { FittedModelSummary } from '../api/types'
import { rankerLabel } from '../metricPresentation'
import { PLAYER_MODELS, researchModelLabel, researchModelDescription } from '../researchModels'
import { useUrlFlag, useUrlState } from '../navigation'
import { DataTable, type Column } from './DataTable'

export function ModelWeights({ entries }: { entries: FittedModelSummary[] }) {
  const [model, setModel] = useUrlState('weights_model', 'fitted_ppg')
  const [search, setSearch] = useUrlState('weights_q', '', { replace: true })
  const [all, setAll] = useState(false)
  const [showAdvancedModels, setShowAdvancedModels] = useUrlFlag('weights_all_models')
  const savedModels = [...new Set(entries.map((entry) => entry.model))]
  const models = savedModels.filter(name => showAdvancedModels || name in PLAYER_MODELS)
  const active = models.includes(model) ? model : models[0]
  const selected = entries.filter((entry) => entry.model === active)
  const sources = [
    ...new Set(selected.flatMap((entry) => Object.keys(entry.selected_source_counts ?? {}))),
  ].map((source) => ({ source }))
  const features = [...new Set(selected.flatMap((entry) => Object.keys(entry.coefficients ?? {})))]
    .map((feature) => ({
      feature,
      magnitude: Math.max(...selected.map((entry) => Math.abs(entry.coefficients?.[feature] ?? 0))),
    }))
    .filter((row) => rankerLabel(row.feature).toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => b.magnitude - a.magnitude)
  type Row = (typeof features)[number]
  const columns: Column<Row>[] = [
    {
      key: 'feature',
      label: 'Input feature',
      title: 'Model input, standardized to its training distribution.',
      align: 'left',
      render: (row) => rankerLabel(row.feature),
    },
    {
      key: 'magnitude',
      label: 'Largest |weight|',
      title: 'Largest absolute coefficient across positions, used for initial sorting.',
      render: (row) => row.magnitude.toFixed(3),
    },
    ...selected.map(
      (entry): Column<Row> => ({
        key: entry.position,
        label: entry.position,
        title: 'Latest standardized coefficient ± its standard deviation across refits.',
        value: (row) => entry.coefficients?.[row.feature],
        render: (row) => {
          const value = entry.coefficients?.[row.feature]
          const sd = entry.coefficient_sd_across_refits?.[row.feature]
          return value == null ? (
            '—'
          ) : (
            <span className={value >= 0 ? 'positive-value' : 'negative-value'}>
              {value >= 0 ? '+' : ''}
              {value.toFixed(3)}
              {sd != null && <span className="dim"> ±{sd.toFixed(3)}</span>}
            </span>
          )
        },
      }),
    ),
  ]
  return (
    <section id="research-weights" className="analysis-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Model anatomy</span>
          <h3>What the fitted models use</h3>
        </div>
      </div>
      <p>
        Latest fitted weights, with variability across historical refits. A coefficient is the
        change in model output per training-standard-deviation of an input, holding other inputs
        fixed; correlated inputs make causal interpretations unreliable.
      </p>
      <div className="controls report-controls">
        <label className="research-wide-control">
          Weight model
          <select
            aria-label="Weight model"
            value={active ?? ''}
            disabled={!models.length}
            onChange={(event) => setModel(event.target.value)}
          >
            {!models.length && <option value="">No player-model weights · choose advanced</option>}
            {models.map((name) => (
              <option key={name} value={name}>
                {researchModelLabel(name)}
              </option>
            ))}
          </select>
        </label>
        <label>Model list
          <select aria-label="Weight model list" value={showAdvancedModels ? 'all' : 'player'}
            onChange={event => setShowAdvancedModels(event.target.value === 'all')}>
            <option value="player">Player forecasts</option>
            <option value="all">All research outputs (advanced)</option>
          </select>
        </label>
        <input
          className="search"
          aria-label="Search model inputs"
          placeholder="Search model inputs…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <button className="chip" type="button" aria-pressed={all} onClick={() => setAll(!all)}>
          {all ? 'Show strongest 20' : `Show all ${features.length} inputs`}
        </button>
      </div>
      {active ? <p className="legend">{researchModelDescription(active)}</p>
        : <p className="notice">No player-model weights are saved. Choose All research outputs (advanced) to inspect components and selectors.</p>}
      <p className="legend">Only models with saved weight summaries are listed. Combined forecasts may use component models rather than having their own coefficients.</p>
      <div className="research-kpis">
        {selected.map((entry) => (
          <div key={entry.position}>
            <span>
              {entry.position} · forecast {entry.latest_forecast_season}
            </span>
            <strong>
              {entry.selected_source
                ? (entry.selection_folds ?? '—')
                : (entry.n_train?.toLocaleString() ?? '—')}
            </strong>
            <small>
              {entry.selected_source
                ? `prior selection folds · score ${entry.selection_score?.toFixed(3) ?? '—'}`
                : `training rows · ${entry.refits} refits · λ ${entry.ridge_lambda ?? '—'}`}
            </small>
            {entry.selected_source && (
              <small>
                Selected {researchModelLabel(entry.selected_source)} · {entry.selection_folds ?? 0} prior
                folds
              </small>
            )}
          </div>
        ))}
      </div>
      {features.length > 0 ? (
        <DataTable
          rows={all ? features : features.slice(0, 20)}
          columns={columns}
          defaultSort="magnitude"
          rowKey={(row) => row.feature}
        />
      ) : (
        <p className="notice">
          {selected.some((entry) => entry.selected_source)
            ? 'This selector chooses a source model rather than fitting feature weights.'
            : 'No coefficients match this model and search.'}
        </p>
      )}
      {sources.length > 0 && (
        <>
          <h4>Historical source selections</h4>
          <p>
            How often each source was chosen across completed and pending refits in this artifact.
          </p>
          <DataTable
            rows={sources}
            defaultSort="source"
            rowKey={(row) => row.source}
            columns={[
              {
                key: 'source',
                label: 'Selected source',
                title: 'Component forecast chosen by the adaptive selector.',
                align: 'left',
                initial: 'asc',
                render: (row) => researchModelLabel(row.source),
              },
              ...selected.map((entry) => ({
                key: entry.position,
                label: entry.position,
                title: 'Number of refits selecting this source.',
                value: (row: { source: string }) => entry.selected_source_counts?.[row.source] ?? 0,
              })),
            ]}
          />
        </>
      )}
      <p className="legend">
        These latest weights describe the pending forecast. Historical rankings above use each
        season’s own fit. They are not coefficients fitted on the selected evidence window.
      </p>
    </section>
  )
}
