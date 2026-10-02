import type { Ranking } from '../api/nextgen'

const recipes: Record<string, string> = {
  prior: 'Prior production reference', current: 'Current production reference', blend: 'Prior + current reference',
  calibrated_current: 'Calibrated current production', calibrated_blend: 'Calibrated prior + current production',
  profile_ridge: 'Profile regression', profile_boost: 'Profile boosting', enriched_boost: 'Enriched profile boosting',
}

export function ForecastBasis({ ranking: r }: { ranking: Ranking }) {
  return <section className="forecast-explanation" aria-label="Forecast basis">
    <div className="forecast-explanation-grid">
      <div><span className="eyebrow">Forecast basis</span><h3>{recipes[r.recipe] ?? r.recipe}</h3>
        <span className="badge rostered">{r.evidence_status === 'validated_forecast' ? 'Historically validated' : 'Reference forecast'}</span>
        <h4>Why this forecast is used</h4>
        <ul>{r.evidence_reason.split(';').map(reason => reason.trim()).filter(Boolean).map((reason, index) => <li key={index}>{reason}</li>)}</ul>
      </div>
      <div className="forecast-history"><h4>Observed history</h4><dl>
        <div><dt>Current-season weeks</dt><dd>{r.current_weeks}</dd></div>
        <div><dt>Prior-season weeks</dt><dd>{r.prior_weeks}</dd></div>
      </dl><p>Missing history is not zero talent.</p></div>
    </div>
    {r.constraint && <p className="notice">{r.constraint} <a href={r.constraint_source ?? undefined} target="_blank" rel="noreferrer">Source · {r.constraint_known_on}</a>. Withheld from ranking; unconstrained statistical estimate was {r.unconstrained_prediction.toFixed(1)} points.</p>}
  </section>
}
