# Deep feature and ensemble search

See the [completed results](../deep_readout.md) and
[combined-library dashboard](../runs/deep_report_001/published_library/index.html).

This extends Future Player Lab with a materially larger tree library and learned
combinations. It is a finite, explicitly recorded research campaign. “All possible
models” and the absolute maximum future accuracy cannot be certified by a search.

## Executed protocol

The prepared matrix has 69,584 examples, 4,770 players, 17,396 completed candidate
player-seasons and 2,929 candidate inputs. Raw NFL observations begin in 2001;
eligible training seasons begin in 2004. Every expanding annual fit retains all
earlier completed seasons, with no random train/test row split. The main campaign
uses base forecast folds 2013–2025, ensemble evaluations from 2016, and primary
policy evaluation 2019–2025. Earlier held-out years train the ensemble; another
three earlier ensemble-evaluation years choose its method. A later year's labels
never choose its model, feature transformation, weights, combination method or
interval. Reused historical years are retrospective development evidence.

Positions: QB, RB, WR and TE. Horizons: preseason full season, next calendar week,
next four calendar weeks, and remaining regular season. In-season origin is week
2, matching the published historical ranking forecasts. The broader previous lab
also tested week 8; this campaign does not certify performance at every origin.

The primary target is league fantasy points, with MSE, MAE, ranking and calibration
evaluated separately. Auxiliary multi-output and hurdle models use workload, yards,
touchdowns and observed player-week counts to support point prediction. Unknown
auxiliary labels remain masked, not converted into zero.

## Inputs and learned features

- Raw recent sequences expanded from 24 to 48 observations; four recent/career
  windows, variability, slopes, counts and history gaps.
- Verified preseason statistical inventory: 1,373 numeric/boolean fields, including
  constants and entirely missing fields when they have a valid input definition.
  Training-only availability determines whether a learner can consume them.
- College, draft, age, career, team-context, usage, historical tracking summaries,
  dated availability facts, and descriptive scoring distributions already present
  in the verified inventory. Fourteen market inputs have a separately named view.
- Raw, summary, inventory+raw, all own-data, market-augmented, and basic+raw views.
  In this matrix, inventory+raw and all own-data resolve to the same 2,915 columns;
  they are aliases, not two independent sources of information. The six path labels
  therefore represent five distinct input sets.
- Training-only tree-selected features, learned pair products and stabilized ratios,
  and tree-leaf response coordinates. Unsupervised PCA/kernel representations and
  supervised neural/PLS alternatives remain in the library.

No predictor is selected by the earlier individual-statistical significance screen,
full-data correlations or alias list. Held-out outcomes and unsafe fitted/proxy
outputs stay excluded. Missing measurements stay unknown until model fitting.
An in-season row updates NFL observation history; its inventory context remains
the dated preseason record. There is no complete historical live-news/starting-job
ledger or tracking-coordinate feed. More computation does not manufacture them.
Historical provider and college-identity vintages remain retrospective.

## Model library

54 fit paths yield 146 candidate forecasts per position/horizon/year:

| Engine | Paths | Capacity checkpoints |
|---|---:|---|
| LightGBM | 24 | 40, 120, 360 trees |
| XGBoost | 8 | 40, 120, 360 trees |
| CatBoost | 4 | 40, 120, 360 trees |
| sklearn histogram boosting | 4 | 40, 120, 360 trees |
| Extra trees / random forests | 6 | 40, 120, 360 trees |
| Ridge, SVR, kernel ridge, MLP, PLS, hurdle, neural-tree, multi-output extra trees | 8 | One defined recipe each |

Trees vary depth/leaf count, minimum leaf support, regularization, learning rate,
feature sampling, representation and recency weighting. The first six LightGBM
paths hold tree settings fixed across input views. Other representation/settings
comparisons can change several factors and must not be attributed to one feature.
Tree prefixes are reused exactly, not refitted and counted as independent models.
Complete coverage is 11,232 path/year fits and 30,368 capacity/year forecasts across
16 tasks. These counts are not independent trials or independent observations.
Inspect the completed manifest before representing any counts as actually finished.

A separate adaptive-capacity arm adds six pipelines: LightGBM, XGBoost and CatBoost,
each with own-data and market-augmented inputs. It permits up to 4,096 trees,
stopping after 80 rounds without improvement on the last earlier completed season.
The chosen count is then refitted on **all** earlier seasons. Neither feature
availability nor tree count is selected on the forecast year. These 1,248 pipeline/
year evaluations add six candidates to every ensemble library, for 152 new
candidates plus persistence. Preseason ensembles also include 31 verified saved
model candidates; the published-comparison arm can include two serving references.
Selecting the ceiling is explicitly recorded. Also inspect selections within the
80-round patience window of the ceiling, where the search budget may bind before
patience is exhausted. Capacity remains bounded by this protocol.

## Combining models

The library includes persistence. A second matched analysis adds the actual
published policy and reference, nine saved current-system baseline/challenger
outputs, and 36 saved QB point-forecast variations as eligible members. Every
additional value is an earlier-trained forecast for that same player/date/horizon;
in-sample fitted values and different-horizon forecasts cannot enter. The duplicate
QB four-week policy is included only once. The main combination menu:

- Top 1/2/3/5/10/20/all equal averages; inverse-error weights at three strengths.
- Forward ensemble selection with replacement, optimizing MSE, MAE, NDCG@24 or
  top-K point capture at the published position cutoffs (QB10/RB20/WR30/TE10).
  A model can enter because it cancels another model's errors even if it is weaker
  alone. Repeated selections create unequal weights.
- Nonnegative sum-to-one weights with four regularization strengths.
- Three ridge residual stacks and two shallow nonlinear residual stacks.
- Rookie/returner conditional weights with a global fallback when support is small.
- Rank-percentile combinations calibrated to points on earlier forecast outcomes.
- Increasing library-budget diagnostics at 8, 16, 32, 64 and all candidates.

Four primary policies choose the combination method using earlier MSE, MAE,
NDCG@24 or position-specific top-K capture. All fixed methods remain visible,
including unfavorable outcomes. Ranking
success does not establish point calibration. Signed stacks retain their raw
predictions; extreme/negative predictions and empirical interval coverage are
reported instead of silently clipping failures.

The historical-reuse study separately combines 32 saved model candidates from the
earlier representation and long-history labs, evaluating 2014–2025 after chronological
warmup. It performs no new base fits and is not independent evidence.

## Comparisons and search saturation

Published comparisons match identity, position, season, forecast origin, horizon
end and canonical actual league points. A source audit found nine published labels
that omit points recorded while a player's weekly position is outside QB/RB/WR/TE.
The original position-filtered totals reproduce exactly. A separate comparison
artifact retains those original labels and re-scores every saved prediction against
the full player-point total, including the nine affected rows. No forecasts or
published files change; no affected players are excluded. Any other disagreement
fails the comparison. Population
coverage is explicit because the lab uses the preseason candidate universe while
published in-season candidates can differ. The published QB four-week integration
is used where approved; the reference is used for other scopes. No season or
next-week production comparison is invented where a matching forecast is absent.

The saturation assessment compares later-season loss at increasing library sizes
and tree capacities, along with earlier-error correlations, effective ensemble
membership and selection stability. A plateau means no useful increment under
this protocol, not proof that a better model cannot exist. Do not choose a winner
after inspecting its outer-year score and describe it as a validated selection policy.
Season-bootstrap intervals are descriptive, unadjusted and based on reused years.
Multiple methods, overlapping player careers, vintage revisions and only seven
modern evaluation seasons limit claims. Nothing automatically promotes a model.

## Reproduce

Packages are isolated under the lab's ignored `.packages` directory. Current
versions: LightGBM 4.7.0, XGBoost 3.4.1 and CatBoost 1.2.10. On macOS their OpenMP
runtime is supplied by `libomp`. The application environment's installed modeling
packages are not replaced. Manifests record versions and implementation hashes.

```sh
uv pip install --target experiments/future_player_lab/.packages --no-deps \
  lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10 six==1.17.0 \
  pandas==3.0.6 graphviz==0.21 plotly==7.1.0 narwhals==2.26.0 python-dateutil==2.9.0.post0

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.deep.prepare --run-id NEW_DATA
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.deep.run --data NEW_DATA --run-id NEW_FITS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.deep.capacity_run --data NEW_DATA --run-id NEW_CAPACITY
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.deep.analyze --source NEW_FITS --capacity NEW_CAPACITY --run-id NEW_REPORT
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  experiments.future_player_lab.deep.reuse --run-id NEW_REUSE
```

New run IDs are create-only. Resume a stopped fit with the identical source,
configuration, package versions and verified inputs using `--resume`. Checkpoints
are hashed; a changed checkpoint is an error. Source files are copied before work.
The first fit run was interrupted for an auxiliary-output shape fix. Its 708
completed unaffected checkpoints were checksum verified and transferred to
`deep_search_002`; `migration.json` records every reused artifact. The numerical
source diff was restricted to the previously unsuccessful PLS/neural-tree branch.

For a stopped run, `deep.parallel_folds --source OLD_RUN --run-id NEW_RUN --workers N`
can redistribute unfinished work by individual year. It verifies and reuses
completed checkpoints; the numerical fit functions, earlier training rows, model
settings and seeds are unchanged. Separate-process equivalence tests verify both
fixed and adaptive execution. The executor records actual worker count separately
from the original prepared-data configuration, and snapshots its implementation.

Forward selection follows the method described in
[Caruana et al., Ensemble Selection from Libraries of Models](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf).
Stacking uses earlier out-of-year predictions rather than fitted training predictions,
consistent with the distinction in the
[scikit-learn ensemble documentation](https://sklearn.org/stable/modules/ensemble.html).
Our implementation replaces generic cross-validation with explicit chronological
folds and adds a separate earlier-year choice of ensemble method.
