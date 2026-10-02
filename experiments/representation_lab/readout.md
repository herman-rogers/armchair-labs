# First automated representation experiment — September 24, 2026

**Removing the old input gate helped. This first raw-history search did not outperform
the existing compact feature baseline.** The results support broad, unfiltered tree
inputs in further research; they do not yet support replacing the existing summaries
with the raw-history representations tested here.

Open the [searchable report](runs/discovery_complete_001/index.html),
[complete generated readout](runs/discovery_complete_001/report.md), or
[paired comparisons CSV](runs/discovery_complete_001/comparisons.csv).

The completed study covers preseason full-season league points for QB, RB, WR and TE:
2008–2025, 72 position/year folds, 14,475 distinct player-season forecasts per method,
15 fixed pipelines, three chronological selectors and 1,260 fitted models including
validation warmup. Every fit uses all earlier completed seasons beginning in 2004.
Each selector chooses its representation and model using the preceding three
chronological validation folds, each itself trained only on earlier seasons.

## Results

The compact **Basic tree** uses the existing 28-variable production/usage, draft,
age, experience and population specification. It is an experimental comparator,
not a claim that this is the complete currently served production system.

The **unfiltered inventory tree** uses all 1,325 own-data admissible candidates.
The **old-gate control** first keeps only its top 80 individual rank-correlated
inputs and excludes binary inputs. Both have identical tree settings, training
rows and preprocessing, isolating the gate within this new experiment. This is
not a replay of the old lab's complete ensemble.

The **raw search** chooses among raw weekly/annual history trees, a recent-only
tree, ridge, generic transformed ridge, PCA+SVR and tree-selected+SVR.

| Position | Unfiltered inventory: MSE reduction vs old gate | Unfiltered inventory: MSE reduction vs Basic | Raw search: MSE increase vs Basic |
| --- | ---: | ---: | ---: |
| QB | 17.0% | 7.1% | 11.7% |
| RB | 16.2% | 4.6% | 10.4% |
| WR | 8.4% | 5.3% | 6.8% |
| TE | 12.9% | 3.8% | 4.8% |

Unfiltered inventory improved on the old gate in 18/18 QB seasons, 18/18 RB,
16/18 WR and 17/18 TE. Paired season-bootstrap intervals for those comparisons
exclude zero. These are unadjusted, descriptive intervals on familiar historical
years; they are not independent prospective confirmation. The TE improvement
against Basic has an interval spanning zero.

Searching across inventory representations/models did **not** beat the fixed
unfiltered inventory tree in average MSE for any position. Its relative MSE was
0.7–2.2% higher, with most uncertainty intervals spanning zero. In joint raw/inventory
search, 71 of 72 selected pipelines used the existing inventory representations;
the remaining selection was a raw tree in one WR season. Trees accounted for
64 selections and SVR pipelines for eight. These are past-validation selections,
not choices made after observing each test season.

The raw-history QB search was particularly weak for rookies: mean seasonal subgroup
MSE was 5,883 versus 3,080 for Basic. The broad inventory tree was approximately
3,062. Retaining population breakdowns matters; an overall score can hide this.

## What the models repeatedly used

Held-out group permutation diagnostics for the raw tree most consistently identified:

| Position | Repeatedly useful measured input groups |
| --- | --- |
| QB | Prior-season league points, completions, passing yards, draft pick; some college passing and weight information |
| RB | Prior-season league points, rushing yards and carries, weight and age |
| WR | Prior-season league points, receiving yards and receptions, weight; some second-season history |
| TE | Prior-season receiving yards and league points, second-season receiving yards, age and weight |

Shuffling prior-season league points increased error in all 18 QB/RB/WR seasons.
Shuffling prior-season receiving yards increased error in all 18 TE/WR seasons.
See [permutation stability](runs/discovery_complete_001/permutation_stability.csv)
for every group and position. These diagnostics are model-specific; correlated
substitutes, missingness and unrealistic shuffled combinations affect interpretation.
Weight importance, for example, is not a causal claim about changing player weight.

[Feature-selection recurrence](runs/discovery_complete_001/feature_selection_stability.json)
and [adjacent tree-split co-occurrences](runs/discovery_complete_001/split_cooccurrences.json)
are saved as interaction hypotheses. They do **not** establish that an interaction
improves forecasts. This run also does not separate the value of retaining binary
inputs from the value of relaxing the broader correlation filter. No binary
indicator or interaction is promoted solely because it survived selection.

## Scope and limitations

The input contract reproduces the 1,339 admissible candidates and 40 two-valued
columns. Fourteen market columns are isolated in a separate fixed benchmark;
no market column enters the own-data selectors. Missingness flags accompany every
input, and fitting of imputation, scaling, PCA and tree selection stays within
training. Individual-statistic screen results and globally detected aliases never
select the inputs.

The raw arm has 1,021 columns: 13 measured weekly statistics over three previous
calendar seasons, older annual observations at lags 4–25, and 33 cutoff-known
player/status/college fields. Missing records remain unknown rather than zero.
This arm omits some tracking/context information present in the inventory, so
raw-versus-inventory results reflect both information and representation differences.
The conclusion is limited to this frozen search budget and these adapters. It is
not evidence that end-to-end representation learning cannot work. Neural networks,
symbolic interaction search and a full raw college/tracking sequence model were
not tested. There is also no direct all-input SVR arm, so the experiment cannot
isolate PCA/selection gains within SVR itself.

The fitted models and their outputs remain research-only. Production rankings,
publication pointers and the prospective snapshot retained their hashes.

## Verification and reproduction

Six targeted tests pass, including a binary XOR example rejected by the old gate,
future-week exclusion, missing-versus-zero preservation, training-only transforms,
PCA batch invariance and exclusion of current/future losses from selection.
The independent saved-artifact audit verified identical evaluation candidates,
all 216 selected forecast/model matches and 1,440 validation losses recomputed
from earlier scored predictions. All four runs and their combination have complete
manifests and unchanged protected inputs/publications. Ruff checks pass.

Run the full study sequentially into a new ID:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/representation_lab/run.py \
  --run-id reproduction_001
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/representation_lab/verify.py \
  experiments/representation_lab/runs/reproduction_001
```

The delivered run used four independent position processes with IDs
`discovery_qb_001`, `discovery_rb_001`, `discovery_wr_001`, `discovery_te_001`, each
created with `--positions QB` (or RB/WR/TE), then combined without fitting or
retuning. To recombine the same verified children into another unused ID:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/representation_lab/combine.py \
  --run-id combined_reproduction_001 \
  discovery_qb_001 discovery_rb_001 discovery_wr_001 discovery_te_001
```

The earlier `smoke_001` is plumbing-only. `discovery_001` was interrupted before
switching to concurrent position runs and is marked failed; it is not an efficacy
result. Source/config snapshots and dependency versions are preserved in the
completed child runs. Saved artifacts are locally ignored by Git and can be
regenerated. No production integration was added.
