# QB passing: production uncertainty and production evidence

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

Implemented September 24, 2026. Open **Intelligence → QB passing**, or the
forecast section of a QB profile. This document describes the original delivery
`nextgen_qb_passing_20260924_r2`, using research `qb_passing_20260924_r3`.
The [variation experiment](../research/qb_passing_variations_2026-09-24.md) supersedes its
current forecasts in five approved target/horizon scopes and adds four-week QB
fantasy integration. Other ranking scopes retain their references.
`data/current.json` remains the authority. The existing ranking release and the
September 24 registry revalidation decisions are preserved.

## What is now available

- Next scheduled team game, next four scheduled team games, and remaining-season
  passing-yard reference forecasts for 116 current candidates.
- Observed current production displayed separately from future production.
- Recency- and attempt-weighted career YPA evidence, its sample size, and the
  corresponding observed career rate. Zero attempts add no execution evidence.
- Reference residual ranges targeting 80% coverage, with historical coverage,
  sample groups and absolute/squared-error results in the dashboard.
- Dated roster events and reported season-ending absences, preserving the
  unadjusted estimate and source. A reported absence is a scenario restriction,
  not a calibrated zero-width interval or a diagnosis.
- Explicit research inspection for conditional and direct boosting. Their scores
  and probabilities are excluded from current production forecasts.

These are outcome-specific passing forecasts. They do not alter fantasy-point
rankings, make a medical risk prediction, or claim a confirmed starting role.

## Historical experiment and results

Passing observations begin in 2001; candidate training begins in 2004. Every
evaluated year uses all earlier completed candidate seasons. Evaluation covers
2007–2025, with 2019–2025 reported separately. Weekly origins include every
completed NFL week. Current 2026 outcomes are never training labels.

The six fixed candidates are prior-season reference, current pace, career
reference, adaptive ridge, direct boosting, and conditional boosting. Reference
and challenger selection use only earlier out-of-fold results. The first three
evaluation years use the declared career reference until selection has enough
earlier folds. Hyperparameters and thresholds were frozen before fitting.

Modern **weekly** results, equal-season errors in future passing yards:

| Horizon | Prior-year MAE | Current-pace MAE | Published reference MAE | Current-pace RMSE | Reference RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Next team game | 62.72 | 37.27 | 37.98 | 72.19 | 63.73 |
| Next four team games | 201.18 | 121.41 | 120.45 | 232.49 | 201.73 |
| Remaining season | 423.09 | 301.18 | 274.08 | 599.08 | 505.76 |

RMSE expresses the primary mean-forecast squared loss in yards. The adaptive
reference improves RMSE against current pace, while next-game MAE is slightly
worse. Full-history reference MAEs are 41.32, 130.47 and 285.34 respectively.
Ranges cover 84.3%, 81.7% and 81.7% of matched modern outcomes respectively;
these are empirical reference coverage rates, not medical confidence intervals.
Repeated player/origin/horizon records are not independent observations.

The challenger selection policy passes this run's retrospective checks, but
prospective validation is pending. Its modern next-game MAE is 35.27 and RMSE
62.14; remaining-season MAE is 276.45 and RMSE 488.21. The latter improves the
primary squared loss while slightly worsening MAE against the reference.
All intervals and overlapping comparisons remain exploratory. No challenger
received serving approval.

**Career averaging does not universally improve preseason accuracy.** The
full-history career reference has preseason MAE 827.81 versus 781.41 for the
repaired prior-season reference. Modern values are 795.45 versus 768.40. Career
information helps recovery cases but cannot establish a new starting job.

| Preseason case | Original saved boost | New career reference | Actual yards |
| --- | ---: | ---: | ---: |
| Tom Brady, 2009 | 1,510 | 2,907 | 4,398 |
| Patrick Mahomes, 2018 | 888 | 305 | 5,097 |
| Lamar Jackson, 2019 | 1,206 | 1,400 | 3,127 |

These examples are diagnostics already known before the experiment, not an
independent holdout. The generic preseason role-transition problem remains.
The new unconditional production model should not be judged as an intrinsic
ability estimator, nor should injury outcomes be deleted from total-yard scoring.

## Data, timing and model mechanics

Execution uses all available career attempts, season half-life three years,
and 100 prior attempts at the earlier league YPA. An eleven-attempt injury
fragment has little weight against hundreds of earlier attempts. Workload history
uses scheduled exposure, not only games with recorded stats. Age, draft capital,
prior production, career workload and current/last-game usage enter the models.
College production and complete historical starting-depth-chart assignments are
not inputs to this fixed experiment.

Conditional boosting estimates three workload states: zero attempts, 1–14
attempts, and 15+ attempts. Conditional attempts and a yardage residual around
the career execution estimate account for remaining opportunity/execution
dependence. It is not a multiplication of independent marginal predictions.
Known partial absence is an input to the unconditional model, never an additional
discount afterward. Fully unavailable horizons force zero consistently.

The run includes 4,259 resolved dated QB events from the trusted transaction
sources plus reviewed supplemental retirement announcements. Known-bad ESPN
historical transaction dates are excluded. Supplements cover Brady, Brees,
Rivers, Roethlisberger and Luck; this is not a complete retirement census.
The general parser supports subsequent activation/signing events, conflicts and
later observed participation. Brady's 2023 retirement now constrains all new
forecasts to zero without modifying his career passing record.

Source occurrence/publication dates remain distinct from capture and forecast
times. Historical data are reconstructed provider snapshots, not original
historical vintages. Historical injury reports without trustworthy publication
dates are not used to infer health. Current reviewed news has preserved source
bytes and timestamps. Team schedules use the current/last recorded team, with an
explicit league-week fallback where a team is absent; last-recorded teams can be
stale following a release. The UI exposes dated off-roster evidence separately.

The saved legacy cohort matches **all 1,960 preseason outcomes exactly**, including
passing yards recorded after a QB changes position to TE/WR. The delivery audit
retains the original booster and baseline comparisons unchanged. Additional dated
constraints in the new run mean comparisons with that original booster combine
input and model changes.

## Serving, archives and operations

The reference endpoints are `/api/nextgen/qb-passing` and
`/api/nextgen/qb-passing/evidence`. The latter requires `scope=research` to expose
challengers. Current responses use an explicit field allowlist and the shared
eligibility/incident policy. Missing, altered or suspended releases fail closed.
The catalog cannot silently drop QB passing on a later refresh.

Research revisions 1 and 2 are retained and quarantined in the registry. Revision
1 had a mismatched no-history fallback and omitted some changed-position outcomes;
revision 2 still applied partial availability twice. Revision 3 corrects both
paths. They are not a new active revalidation queue. Earlier deliveries and
catalog pointers remain available for reproduction and rollback.

Run a new immutable forecast issue after refreshing the verified input products:

```sh
.venv/bin/python research/build_qb_passing.py --version qb_passing_NEW
.venv/bin/python research/publish_qb_passing.py --source qb_passing_NEW --version nextgen_qb_passing_NEW --publish
```

Both commands accept `--analysis-version STAGED_ANALYSIS` when preparing a new
gold/profile release before its atomic catalog publication. They never edit a
previous experiment. A refresh is explicit; the dashboard does not silently
refit models on every page load. Historical preseason estimates remain frozen.

Prospective forecasts were frozen on September 24 before their future target
games. Score the same issue against later gold observations, without refitting:

```sh
.venv/bin/python research/score_qb_passing.py --source qb_passing_20260924_r3 --version qb_passing_score_NEW
```

The first scoring pass correctly reports **348 pending horizons and zero scored**.
Partial target horizons are not treated as completed outcomes. Because schedules
have dates but no verified kickoff times, same-day issues are conservatively
excluded from prospective timing certification when scored. The scorer cannot
promote a model. Future prospective evidence and richer dated starting-role
information remain required for the corresponding stronger claims.

## Verification

47 focused Python tests passed across temporal safety, short-season evidence,
availability accounting, API scope, existing ranking publication and profiles.
Production frontend build and focused Python/TypeScript lint passed. Dedicated
QB browser checks cover desktop/mobile, horizon switches, research isolation,
reported absence and profile integration. Existing Evidence browser checks passed.
The build still reports the existing unsupported Node 21 warning; it completed.
