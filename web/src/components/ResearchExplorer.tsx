import { useId, useState } from 'react'
import type { MetricRankingFoldResult, MetricRankingResult, MetricReport } from '../api/types'
import { rankerLabel } from '../metricPresentation'
import { PLAYER_MODELS, researchModelLabel, researchModelDescription } from '../researchModels'
import { useUrlFlag, useUrlState } from '../navigation'
import { DataTable, type Column } from './DataTable'
import { POSITION_FILTERS } from '../positions'
import { saveFile, toCsv } from '../download'
import { fixed, percent, pointLift } from '../format'

type Score = 'hit_rate' | 'ndcg'
type Fold = { season: number; model?: MetricRankingFoldResult; baseline?: MetricRankingFoldResult }

function downloadFolds(folds: Fold[], model: string, baseline: string) {
  const rows = [
    [
      'Season',
      'Model',
      'Benchmark',
      'Model pool N',
      'Benchmark pool N',
      'Model hit rate',
      'Benchmark hit rate',
      'Model NDCG',
      'Benchmark NDCG',
    ],
    ...folds.map((f) => [
      f.season,
      model,
      baseline,
      f.model?.n,
      f.baseline?.n,
      f.model?.hit_rate,
      f.baseline?.hit_rate,
      f.model?.ndcg,
      f.baseline?.ndcg,
    ]),
  ]
  saveFile(new Blob([toCsv(rows)], { type: 'text/csv;charset=utf-8' }), 'forecast-fold-comparison.csv')
}

function FoldChart({
  folds,
  score,
  selected,
  onSelect,
  model,
  baseline,
}: {
  folds: Fold[]
  score: Score
  selected: number | undefined
  onSelect: (season: number) => void
  model: string
  baseline: string
}) {
  const title = useId()
  const x = (index: number) => 52 + (index / Math.max(folds.length - 1, 1)) * 820
  const y = (value: number) => 195 - value * 160
  function path(side: 'model' | 'baseline') {
    let joined = false
    return folds
      .map((fold, index) => {
        const value = fold[side]?.[score]
        if (value == null) {
          joined = false
          return ''
        }
        const command = `${joined ? 'L' : 'M'}${x(index)},${y(value)}`
        joined = true
        return command
      })
      .join(' ')
  }
  return (
    <figure className="fold-chart">
      <figcaption>
        <span className="chart-key model">{model}</span>
        <span className="chart-key benchmark">{baseline}</span>
      </figcaption>
      <div className="chart-scroll">
        <svg viewBox="0 0 920 230" role="img" aria-labelledby={title}>
          <title id={title}>
            {score === 'hit_rate' ? 'Top-K hit rate' : 'NDCG'} by forecast season. Exact values
            appear in the fold table.
          </title>
          {[0, 0.25, 0.5, 0.75, 1].map((value) => (
            <g key={value}>
              <line x1="52" x2="890" y1={y(value)} y2={y(value)} className="chart-grid" />
              <text x="42" y={y(value) + 4} textAnchor="end">
                {score === 'hit_rate' ? `${value * 100}%` : value}
              </text>
            </g>
          ))}
          <path d={path('baseline')} className="chart-line benchmark" />
          <path d={path('model')} className="chart-line model" />
          {folds.map((fold, index) => (
            <g key={fold.season}>
              {fold.season === selected && (
                <line x1={x(index)} x2={x(index)} y1="25" y2="200" className="chart-selection" />
              )}
              {(['baseline', 'model'] as const).map((side) => {
                const value = fold[side]?.[score]
                return value == null ? null : (
                  <circle
                    key={side}
                    cx={x(index)}
                    cy={y(value)}
                    r={fold.season === selected ? 5 : 3}
                    className={`chart-point ${side}`}
                  >
                    <title>
                      {fold.season} · {side === 'model' ? model : baseline}:{' '}
                      {score === 'hit_rate' ? percent(value, 1) : fixed(value, 3)}
                    </title>
                  </circle>
                )
              })}
              {(folds.length < 14 || index % 2 === 0 || index === folds.length - 1) && (
                <text x={x(index)} y="219" textAnchor="middle">
                  {fold.season}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
      <div className="season-picker" aria-label="Inspect a forecast season">
        {folds.map((f) => (
          <button
            type="button"
            key={f.season}
            aria-pressed={selected === f.season}
            onClick={() => onSelect(f.season)}
          >
            {f.season}
          </button>
        ))}
      </div>
    </figure>
  )
}

export function ResearchExplorer({ report, window }: { report: MetricReport; window: string }) {
  // Params are prefixed `explorer_`: the explorer shares its page with other research panels.
  const [population, setPopulation] = useUrlState('explorer_population', 'all')
  const [position, setPosition] = useUrlState('explorer_position', 'ALL')
  const [target, setTarget] = useUrlState('explorer_outcome', 'actual_availability_value')
  const [model, setModel] = useUrlState('explorer_model', 'fitted_season_points')
  const [baseline, setBaseline] = useUrlState('explorer_baseline', 'market_ecr_score')
  const [scoreParam, setScore] = useUrlState('explorer_score', 'hit_rate')
  const score = scoreParam as Score
  const [seasonParam, setSeasonParam] = useUrlState('explorer_season', '')
  const selectedSeason = seasonParam ? Number(seasonParam) : undefined
  const setSelectedSeason = (season: number) => setSeasonParam(String(season))
  const [showAll, setShowAll] = useState(false)
  const [showAdvancedModels, setShowAdvancedModels] = useUrlFlag('explorer_all_models')
  const populations = [
    { id: 'all', label: 'All evaluated players', rows: report.ranking_results },
    ...[...new Set(report.population_ranking_results?.map((row) => row.population) ?? [])].map(
      (id) => ({
        id,
        label: id === 'rookie' ? 'Rookies' : rankerLabel(id),
        rows: report.population_ranking_results!.filter((row) => row.population === id),
      }),
    ),
    ...[...new Set(report.ranking_sensitivity_results?.map((row) => row.population) ?? [])].map(
      (id) => ({
        id,
        label: id === 'cutoff_rostered' ? 'Rostered at cutoff' : 'Week 1 roster proxy',
        rows: report.ranking_sensitivity_results!.filter((row) => row.population === id),
      }),
    ),
  ]
  const pool = (populations.find((entry) => entry.id === population) ?? populations[0]).rows.filter(
    (row) => row.window === window,
  )
  const positions = POSITION_FILTERS.filter((pos) =>
    pool.some((row) => row.position === pos),
  )
  const activePosition = positions.includes(position) ? position : positions[0]
  const targets = [
    ...new Set(pool.filter((row) => row.position === activePosition).map((row) => row.target)),
  ]
  const activeTarget = targets.includes(target) ? target : targets[0]
  const rows = pool.filter((row) => row.position === activePosition && row.target === activeTarget)
  const candidates = rows.filter((row) => row.role === 'candidate')
  const visibleCandidates = candidates.filter(row => showAdvancedModels || row.ranker in PLAYER_MODELS)
  const benchmarks = rows.filter((row) => row.role === 'baseline')
  const selected = visibleCandidates.find((row) => row.ranker === model) ?? visibleCandidates[0]
  const benchmark = benchmarks.find((row) => row.ranker === baseline) ?? benchmarks[0]
  const seasons = [
    ...new Set(
      [...(selected?.fold_results ?? []), ...(benchmark?.fold_results ?? [])].map(
        (f) => f.forecast_season,
      ),
    ),
  ].sort((a, b) => a - b)
  const folds = seasons.map((season) => ({
    season,
    model: selected?.fold_results.find((f) => f.forecast_season === season),
    baseline: benchmark?.fold_results.find((f) => f.forecast_season === season),
  }))
  const paired = folds.filter((f) => f.model?.[score] != null && f.baseline?.[score] != null)
  const meanLift = paired.length
    ? paired.reduce((sum, f) => sum + f.model![score]! - f.baseline![score]!, 0) / paired.length
    : null
  const wins = paired.filter((f) => f.model![score]! > f.baseline![score]! + 0.00005).length
  const losses = paired.filter((f) => f.model![score]! < f.baseline![score]! - 0.00005).length
  const focus = folds.find((f) => f.season === selectedSeason) ?? folds.at(-1)
  const ranked = rows.filter(row => row.role === 'baseline' || showAdvancedModels || row.ranker in PLAYER_MODELS)
    .sort((a, b) => (b[score] ?? -Infinity) - (a[score] ?? -Infinity))
  const market =
    population === 'all' && activePosition === 'ALL'
      ? report.market_disagreement_results?.find(
          (row) =>
            row.window === window &&
            row.candidate === selected?.ranker &&
            row.target === activeTarget,
        )
      : undefined
  const columns: Column<MetricRankingResult>[] = [
    {
      key: 'ranker',
      label: 'Ranker',
      title: 'Select a candidate to inspect its seasonal performance.',
      align: 'left',
      render: (row) => (
        <button
          type="button"
          className="text-button"
          onClick={() => (row.role === 'baseline' ? setBaseline(row.ranker) : setModel(row.ranker))}
        >
          {researchModelLabel(row.ranker)}
        </button>
      ),
    },
    {
      key: 'role',
      label: 'Role',
      title: 'Historical or market benchmark versus candidate forecast.',
      align: 'left',
      render: (row) => <span className={`ranking-role ${row.role}`}>{row.role}</span>,
    },
    {
      key: 'hit_rate',
      label: 'Hit @K',
      title: 'Share of the selected top K that are also in the actual top K.',
      render: (row) => percent(row.hit_rate, 1),
    },
    {
      key: 'ndcg',
      label: 'NDCG',
      title: 'Rank-discounted actual value relative to ideal ordering; higher is better.',
      render: (row) => fixed(row.ndcg, 3),
    },
    {
      key: 'pool_spearman',
      label: 'Rank correlation',
      title: 'Spearman rank correlation in the evaluated pool.',
      render: (row) => fixed(row.pool_spearman, 3),
    },
    {
      key: 'folds',
      label: 'Seasons',
      title: 'Completed forecast seasons contributing to this result.',
      render: (row) => row.folds,
    },
    {
      key: 'hit_rate_lift',
      label: 'Reported hit lift',
      title:
        'Reported lift versus this row’s best baseline; may differ from the benchmark selected above.',
      render: (row) => (
        <span title={row.best_baseline ? `Versus ${researchModelLabel(row.best_baseline)}` : undefined}>
          {pointLift(row.hit_rate_lift)}
        </span>
      ),
    },
  ]
  return (
    <section className="research-explorer" aria-label="Forecast research explorer">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Forecast research</span>
          <h3>Compare performance across seasons</h3>
        </div>
        <span className="count">
          {report.data_summary.completed_rows.toLocaleString()} completed player-seasons in report
        </span>
      </div>
      <p className={`evaluation-note ${report.evaluation_contract ? '' : 'legacy'}`}>
        {report.evaluation_contract ??
          'Legacy comparison: models were evaluated on different player pools. These results are diagnostic and cannot establish a fair model-selection verdict.'}
      </p>
      <div className="controls report-controls">
        <label>
          Population
          <select
            aria-label="Population"
            value={populations.some((p) => p.id === population) ? population : 'all'}
            onChange={(e) => setPopulation(e.target.value)}
          >
            {populations.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Research position
          <select
            aria-label="Research position"
            value={activePosition ?? ''}
            onChange={(e) => setPosition(e.target.value)}
          >
            {positions.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>
        <label>
          Research outcome
          <select
            aria-label="Research outcome"
            value={activeTarget ?? ''}
            onChange={(e) => setTarget(e.target.value)}
          >
            {targets.map((t) => (
              <option key={t} value={t}>
                {rankerLabel(t)}
              </option>
            ))}
          </select>
        </label>
        <label className="research-wide-control">
          Model to inspect
          <select
            aria-label="Model to inspect"
            value={selected?.ranker ?? ''}
            disabled={!visibleCandidates.length}
            onChange={(e) => setModel(e.target.value)}
          >
            {!visibleCandidates.length && <option value="">No player forecasts in this slice</option>}
            {visibleCandidates.map((row) => (
              <option key={row.ranker} value={row.ranker}>
                {researchModelLabel(row.ranker)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Model list
          <select aria-label="Evidence model list" value={showAdvancedModels ? 'all' : 'player'}
            onChange={e => setShowAdvancedModels(e.target.value === 'all')}>
            <option value="player">Player forecasts</option>
            <option value="all">All research outputs (advanced)</option>
          </select>
        </label>
        <label>
          Benchmark
          <select
            aria-label="Benchmark"
            value={benchmark?.ranker ?? ''}
            onChange={(e) => setBaseline(e.target.value)}
          >
            {benchmarks.map((row) => (
              <option key={row.ranker} value={row.ranker}>
                {rankerLabel(row.ranker)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Compare by
          <select
            aria-label="Compare by"
            value={score}
            onChange={(e) => setScore(e.target.value as Score)}
          >
            <option value="hit_rate">Top-K hit rate</option>
            <option value="ndcg">NDCG</option>
          </select>
        </label>
      </div>
      {selected && <p className="legend">{researchModelDescription(selected.ranker)}</p>}
      <p className="legend">Only models with saved results for this outcome, position, and window are listed. Advanced exposes components and diagnostic experiments; hidden outputs are retained, not deleted.</p>
      {!selected ? (
        <div className="notice">{candidates.length && !showAdvancedModels
          ? 'No player forecasts in this slice. Choose All research outputs (advanced) to inspect saved components.'
          : 'No ranking results for this window and population.'}</div>
      ) : (
        <>
          <div className="research-kpis">
            <div>
              <span>Hit rate · top {selected.k}</span>
              <strong>{percent(selected.hit_rate, 1)}</strong>
              <small>{selected.folds} reported seasons</small>
            </div>
            <div>
              <span>Ranked value · NDCG</span>
              <strong>{fixed(selected.ndcg, 3)}</strong>
              <small>1.000 is the ideal ordering</small>
            </div>
            <div>
              <span>Δ vs selected benchmark</span>
              <strong>{score === 'hit_rate' ? pointLift(meanLift) : fixed(meanLift, 3)}</strong>
              <small>
                {paired.length} shared season{paired.length === 1 ? '' : 's'} · equal weight
              </small>
            </div>
            <div>
              <span>Season wins / ties / losses</span>
              <strong>
                {paired.length ? `${wins} / ${paired.length - wins - losses} / ${losses}` : '—'}
              </strong>
              <small>By {score === 'hit_rate' ? 'hit rate' : 'NDCG'} on shared seasons</small>
            </div>
          </div>
          <FoldChart
            folds={folds}
            score={score}
            selected={focus?.season}
            onSelect={setSelectedSeason}
            model={researchModelLabel(selected.ranker)}
            baseline={benchmark ? rankerLabel(benchmark.ranker) : 'No benchmark'}
          />
          {focus && (
            <div className="fold-focus" aria-live="polite">
              <b>{focus.season}</b>
              <span>
                Model{' '}
                {score === 'hit_rate' ? percent(focus.model?.hit_rate, 1) : fixed(focus.model?.ndcg, 3)}
              </span>
              <span>
                Benchmark{' '}
                {score === 'hit_rate'
                  ? percent(focus.baseline?.hit_rate, 1)
                  : fixed(focus.baseline?.ndcg, 3)}
              </span>
              <span>
                Pool N: {focus.model?.n ?? '—'} / {focus.baseline?.n ?? '—'}
              </span>
              <span>
                Top-K actual mean: {fixed(focus.model?.top_k_actual_mean, 2)} · ideal{' '}
                {fixed(focus.model?.ideal_top_k_actual_mean, 2)}
              </span>
            </div>
          )}
          <div className="research-split">
            <div>
              <h4>Reported uncertainty</h4>
              <p>
                {selected.hit_rate_lift_ci_low != null && selected.hit_rate_lift_ci_high != null ? (
                  <>
                    Hit-rate lift interval:{' '}
                    <b>
                      {pointLift(selected.hit_rate_lift_ci_low)} to{' '}
                      {pointLift(selected.hit_rate_lift_ci_high)}
                    </b>
                    , versus {researchModelLabel(selected.best_baseline ?? 'unreported baseline')}. This is
                    the report’s fold-based interval, not a prediction interval for a player.
                  </>
                ) : (
                  'No hit-rate lift interval was saved for this model and slice.'
                )}
              </p>
            </div>
            <div>
              <h4>Coverage and comparability</h4>
              <p>
                Pool N describes the players evaluated in each fold. Missing seasons are excluded
                from paired differences; missing values are never plotted as zero. Different player
                pools can change what counts as a hit.
              </p>
            </div>
          </div>
          <div className="section-heading">
            <h4>Ranker leaderboard</h4>
            <button
              type="button"
              className="chip"
              aria-pressed={showAll}
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? 'Show top 8' : `Show all ${ranked.length} rankers`}
            </button>
          </div>
          <DataTable
            key={`${window}-${population}-${activePosition}-${activeTarget}-${score}-${showAll}`}
            rows={showAll ? ranked : ranked.slice(0, 8)}
            columns={columns}
            defaultSort={score}
            rowKey={(row) => row.ranker}
          />
          <details className="analysis-details">
            <summary>Exact fold values and coverage ({folds.length} seasons)</summary>
            <button
              type="button"
              className="chip export-button"
              onClick={() => downloadFolds(folds, selected.ranker, benchmark?.ranker ?? '')}
            >
              Download fold CSV
            </button>
            <DataTable
              rows={folds}
              defaultSort="season"
              rowKey={(row) => row.season}
              columns={[
                {
                  key: 'season',
                  label: 'Season',
                  title: 'Held-out forecast season.',
                  initial: 'asc',
                },
                {
                  key: 'model_n',
                  label: 'Model pool N',
                  title: 'Players evaluated for the model.',
                  value: (f) => f.model?.n,
                },
                {
                  key: 'baseline_n',
                  label: 'Benchmark pool N',
                  title: 'Players evaluated for the benchmark.',
                  value: (f) => f.baseline?.n,
                },
                {
                  key: 'outcome_n',
                  label: 'Outcome pool N',
                  title: 'Full actual-outcome universe for the model, when recorded.',
                  value: (f) => f.model?.outcome_n,
                },
                {
                  key: 'missing_scores',
                  label: 'Missing forecasts',
                  title: 'Outcome rows without a forecast, when recorded.',
                  value: (f) => f.model?.missing_scores,
                },
                {
                  key: 'coverage',
                  label: 'Model coverage',
                  title:
                    'Fraction of the outcome universe with a forecast; unavailable in older artifacts.',
                  value: (f) => f.model?.coverage,
                  render: (f) => percent(f.model?.coverage, 1),
                },
                {
                  key: 'model_hit',
                  label: 'Model hit rate',
                  title: 'Model top-K hit rate.',
                  value: (f) => f.model?.hit_rate,
                  render: (f) => percent(f.model?.hit_rate, 1),
                },
                {
                  key: 'baseline_hit',
                  label: 'Benchmark hit rate',
                  title: 'Benchmark top-K hit rate.',
                  value: (f) => f.baseline?.hit_rate,
                  render: (f) => percent(f.baseline?.hit_rate, 1),
                },
                {
                  key: 'model_ndcg',
                  label: 'Model NDCG',
                  title: 'Model ranked value relative to ideal.',
                  value: (f) => f.model?.ndcg,
                  render: (f) => fixed(f.model?.ndcg, 3),
                },
                {
                  key: 'baseline_ndcg',
                  label: 'Benchmark NDCG',
                  title: 'Benchmark ranked value relative to ideal.',
                  value: (f) => f.baseline?.ndcg,
                  render: (f) => fixed(f.baseline?.ndcg, 3),
                },
              ]}
            />
          </details>
          {market && (
            <details className="analysis-details">
              <summary>When this model disagrees with the market</summary>
              <p>
                Common-player comparison against {rankerLabel(market.market)} · top {market.k} ·{' '}
                {market.folds} seasons. Totals below count selections across folds.
              </p>
              <div className="research-kpis">
                <div>
                  <span>Model-only hits</span>
                  <strong>{market.model_only_hits}</strong>
                  <small>Successful picks the market left out</small>
                </div>
                <div>
                  <span>Market-only hits</span>
                  <strong>{market.market_only_hits}</strong>
                  <small>Successful picks the model left out</small>
                </div>
                <div>
                  <span>Contrarian precision</span>
                  <strong>{percent(market.contrarian_precision, 1)}</strong>
                  <small>{market.contrarian_false_positives} false positives</small>
                </div>
                <div>
                  <span>Market misses recovered</span>
                  <strong>{percent(market.missed_value_capture_rate, 1)}</strong>
                  <small>Share of actual top-K market misses</small>
                </div>
              </div>
              <p>
                Mean common pool: {fixed(market.common_players_mean, 0)} players · mean top-K overlap:{' '}
                {fixed(market.top_k_overlap_mean, 1)} · net swap value per fold:{' '}
                {fixed(market.net_swap_value_per_fold, 2)} in the selected outcome’s units.
              </p>
              {!!market.value_capture_bands?.length && (
                <DataTable
                  rows={market.value_capture_bands}
                  defaultSort="rank_gap"
                  rowKey={(row) => row.rank_gap}
                  columns={[
                    {
                      key: 'rank_gap',
                      label: 'Minimum rank gap',
                      title: 'Minimum rank disagreement for a contrarian pick.',
                      initial: 'asc',
                    },
                    {
                      key: 'calls',
                      label: 'Calls',
                      title: 'Number of model-only selections across folds.',
                    },
                    {
                      key: 'hits',
                      label: 'Hits',
                      title: 'Model-only selections that reached the actual top K.',
                    },
                    {
                      key: 'precision',
                      label: 'Precision',
                      title: 'Hits divided by calls.',
                      render: (row) => percent(row.precision, 1),
                    },
                    {
                      key: 'false_positive_cost',
                      label: 'False-positive cost',
                      title: 'Reported missed value from unsuccessful calls, in outcome units.',
                      render: (row) => fixed(row.false_positive_cost, 2),
                    },
                  ]}
                />
              )}
            </details>
          )}
        </>
      )}
    </section>
  )
}
