# Opening Drive

The application serves `opening_drive_v3` and the preserved `opening_drive_v2`
comparison. V3 retains the eight original models and adds four weekly mean and
distribution models; see [v3 research, results, and reproduction](weekly_v3/README.md).
The separate [v3.1 interval challenger](weekly_v31/README.md) adds chronological
recalibration and immutable pregame comparisons. Once its run is sealed, the normal
refresh driver also captures and scores it; it does not replace the served v3 models.
V2 has `te_season` and `te_remaining` integrated into its bundle. `just campaigns consolidate` performs
the verified migration and archives v1, Scout Team and the separate TE bundle.
The commands below document historical training runs; their original campaign IDs
remain research provenance, not additional application choices. See the
[current application workflow](../../../docs/operations/model-campaigns.md).

`opening_drive_v1` is Armchair Labs' first shared production-forecast campaign for
RB, QB and WR. It extends the RB GPU tree research with a common reusable training
and inference contract. It is a **research preview**; publishing its independent
local API bundle does not promote it into the existing analysis release.

[Initial results and limitations](../../../docs/research/opening_drive_2026-10-09.md) ·
[Full evaluation tables](../runs/opening_drive_review_20261009_001/report.md)

For recurring comparisons against external projections and expert rankings, use
the [external campaign evaluation workflow](../../../docs/operations/model-campaigns.md#external-campaign-evaluation).
It evaluates frozen predictions after publication and never participates in model selection.

## Version 2 availability experiment

`opening_drive_v2` retains the four v1 recipes, targets, scoring, chronological
folds, affine/persistence controls, and preceding-three-year ensemble selection.
It reuses the checksummed v1 matrices and appends 23 availability features. The
candidate pool starts with v1; only the latest unambiguous formal retired status
excludes a player. This applies equally to training rows, validation, evaluation,
baselines, and current forecasts. Injury, reserve, suspended, released, and
unknown states remain eligible for aggregate remaining production. Retirement
announcements are features, not exclusions: the source can omit a subsequent
return announcement.

The state joins dated ESPN transaction clauses, completed-week nflverse rosters,
and timestamped injury reports. Ambiguous identity matches and conflicting
same-time states remain unknown. Transaction dates become usable the following
day at 12:00 UTC; weekly rosters become usable the day after that week's final
game at 12:00 UTC. Preseason is opening-game day at 12:00 UTC, before kickoff.
The after-Week-2 cutoff is the day after its final game at 12:00 UTC. Week 1
rosters never supply preseason status. Missing reports never mean healthy or
active, and future appearances never determine eligibility.

The archives were captured retrospectively; event dates are not archived
publication timestamps. Coverage varies by year, and source dates can be revised.
The source receipts retain those limitations. The complete paginated ESPN
capture is the transaction input. V1 candidate coverage and team-context features
remain fixed for this experiment, so expanding to all dated roster entrants or
rebuilding team features would be a separate change.

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.availability.espn \
  --output data/research/opening_drive_v2_sources/espn
uv run --no-sync python -m experiments.future_player_lab.campaigns.availability.data
uv run --no-sync python -m experiments.future_player_lab.campaigns.availability.run \
  --run-id opening_drive_v2_001
uv run --no-sync python -m experiments.future_player_lab.campaigns.availability.compare \
  --output experiments/future_player_lab/runs/opening_drive_v2_review_20261010_001
uv run --no-sync python -m experiments.future_player_lab.campaigns.availability.forecast
uv run --no-sync python -m experiments.future_player_lab.campaigns.replay \
  --campaign opening_drive_v2
```

Use a fresh source directory and run ID for semantic changes. The comparison
checks original matrix equality and pairs v1/v2 on the same eligible outcomes.
It also reports v1 selection with the eligibility gate alone, so the population
effect is visible separately from retraining plus new features. Saved models,
forecasts, availability evidence, and exclusion audits have independent checksums.
The v1 research and published bundle remain preserved.

Each position has a preseason model (`origin_week=0`, `horizon=season`) and a
remaining-season model after Week 2 (`origin_week=2`, `horizon=remaining`). Models
predict aggregate production through the regular-season end, including missed
games and reduced opportunity. They do not estimate an abstract overall ability
rating or individual future matchup/week outcomes.

## Train and evaluate

Prepare the [GPU research environment](../../../docs/operations/gpu_research_setup.md) first.
The commands preserve its installed packages with `--no-sync`. The `just campaign-train`,
`just campaign-score`, and `just campaign-replay` shortcuts also set thread limits.

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.run \
  --config experiments/future_player_lab/campaigns/configs/opening_drive_v1.json \
  --run-id opening_drive_v1_001
```

The default device is CUDA for XGBoost; use `--device cpu` for a portable research
run. LightGBM uses four CPU threads. `--positions QB WR RB` and
`--horizons remaining season` permit bounded subsets. Use a fresh run ID when
changing code, data, parameters or device; partial checkpoints are reusable only
with an identical hashed fit contract and intact checkpoint receipts.

The checked-in JSON names the campaign, positions, target/scoring contracts,
horizons, evaluation years and four candidate recipes. To create a later campaign,
copy it, change `campaign_id` and `name`, adjust supported recipe parameters, and
use a new run ID. This implementation intentionally rejects unsupported target or
evaluation-year changes; those require an explicit versioned adapter and tests.
The CLI defaults to QB, WR and RB with both horizons. Subset flags are useful for
evaluation; complete every configured cell before exporting a campaign.

Historical candidates use prior-season participants, participants observed by the
forecast origin, and exactly linked draft entrants. Future appearances never
control eligibility. Future absence in a completed season is a zero outcome;
unknown fields on a recorded observation remain unknown independently per target.

- Training starts with 2002 and expands chronologically through 2025.
- Candidate validation predicts 2018–2025 using only earlier seasons.
- Evaluation reports 2021–2025; each target's two-member ensemble is chosen using
  only the three previous validation seasons, with equal weight per year's MSE.
- Final 2026 models refit through 2025, selecting members using 2023–2025.
- Fixed candidates are XGBoost raw+draft, a shallower more regularized XGBoost,
  XGBoost with a selected provider-metric ablation, and LightGBM raw+draft.
- Targets are modeled as production per calendar week, then scaled to the forecast
  period. Baselines are calendar-week persistence and a historical affine
  calibration of persistence, fit only on earlier seasons.

These are historical **development evaluations**, not a previously untouched test
set. The RB guiding research already inspected some of these years. Each report
includes aggregate/year/target RMSE and MAE, raw candidate results, ensemble and
single selected tree, baseline comparisons and a cohort with previously known
appearances in at least half the elapsed calendar weeks.

## Targets and scoring

RB and WR independently predict carries, rushing yards/TDs, targets, receptions,
receiving yards/TDs, and direct PPR points. PPR here is
`0.1*(rushing_yards+receiving_yards)+receptions+6*(rushing_tds+receiving_tds)`;
passing production, lost fumbles and bonuses are excluded.

QB independently predicts attempts, completions, passing yards/TDs/interceptions,
carries, rushing yards/TDs, and direct points with four-point passing touchdowns:
`0.04*passing_yards+4*passing_tds-2*passing_interceptions+0.1*rushing_yards+6*rushing_tds`.
Receiving production, lost fumbles and bonuses are excluded. This explicit
contract should be checked before comparing any outside fantasy ranking.

Direct points and the arithmetic sum of independent stat predictions can differ.
Forecasts expose `component_points` separately, without silently replacing the
learned direct target. Per-scheduled-game points is simply aggregate points divided
by games; it is not a weekly model.

## Score saved models and expose the API

After all configured models finish:

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.forecast \
  --run-id opening_drive_v1_001
uv run engine campaigns check opening_drive_v1
uv run --no-sync python -m experiments.future_player_lab.campaigns.replay
```

The scorer defaults to the accepted Week 4 raw snapshot and the preserved richer
2026 supplements, with all observations restricted to each model's origin. Thus
these 2026 forecasts are **reconstructed cutoffs from later captures**, not claims
of forecasts issued before those games. No 2026 outcomes enter fitting. Snapshot,
supplement, model, feature and code hashes accompany the forecast.

This command scores the six saved models without fitting, saves feature matrices
and forecasts below the run, and creates a sealed copy under
`data/campaigns/opening_drive_v1/`. Replay verifies existing local forecasts before
reusing them; an existing published campaign is never overwritten. The API bundle
copies model weights and contracts as well as precomputed JSON so later cleanup of
a research run cannot break campaign access. The web request path never trains a
model or deserializes model weights.

```text
GET /api/campaigns
GET /api/campaigns/opening_drive_v1
GET /api/campaigns/opening_drive_v1/models/qb_remaining
GET /api/campaigns/opening_drive_v1/models/qb_remaining/metrics
GET /api/campaigns/opening_drive_v1/models/qb_remaining/forecasts?season=2026&origin_week=2
```

See the [API/store documentation](../../../docs/operations/model-campaigns.md)
for pagination and individual player filters.
No frontend changes, old release promotion, or legacy deletion are part of this
campaign setup.

## Limits and provenance

The broad numeric feature panel comes from verified raw observations and the RB
research's past-only aggregation semantics. Draft overall pick, round and years
since draft fix the omission found in the RB extension. Current-season delayed
participation data remains withheld; undated injury captures are not relabeled as
historical observations. Numeric source facts and aggregation windows remain
explicit rather than learned end to end.

QB passing box and NGS fields are available, with explicit passing history totals.
This initial build inherits rush/receiver play summaries; a new passer-specific
play-by-play adapter is a future candidate improvement, not claimed here.
The `xgb_no_provider_draft` ID is shorthand for a fixed exclusion mask, not a
claim that all provider-derived metrics are absent. In particular, nine QB
`passing_cpoe` fields survive the inherited exact-name rule; changing that mask
requires a separate controlled experiment.
Historical provider files are revised retrospective vintages. Last-observed team
can lag roster moves, and unlinked or undrafted newcomers can be absent before
playing. These limits affect every comparison and current forecast.

Original RB research remains preserved and referenced as lineage in this campaign;
new common-protocol RB fits are distinct models, not replacements silently
substituted into the earlier rankings.

## Independent evaluation review

Audit completed cells independently of training, then generate paired baseline
comparisons, high-prior-usage results, and ranking/selection diagnostics:

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.audit \
  experiments/future_player_lab/runs/opening_drive_v1_001/{rb,qb,wr}_{remaining,season} \
  --output /tmp/opening_drive_audit.json
uv run --no-sync python -m experiments.future_player_lab.campaigns.report \
  experiments/future_player_lab/runs/opening_drive_v1_001 \
  --output experiments/future_player_lab/runs/opening_drive_review_NEW
```

The audit checks saved hashes, matching evaluation rows, prior-year selection,
ensemble arithmetic, and recorded metrics. The report is descriptive and does not
change model selection or weights. High-prior-usage thresholds use only earlier
seasons; forecast-ranked selection compares 12 QB or 24 RB/WR slots on identical
candidate pools. Use a fresh report output directory for each review.

## Scout Team: preserved original research

`scout_team_v1` groups the pre-Opening-Drive RB GPU study and QB passing/variations
research. Import their original artifacts into the same campaign store:

```sh
just campaign-import-scout \
  --output experiments/future_player_lab/runs/scout_team_import_NEW
just campaigns check scout_team_v1
just campaigns show scout_team_v1
```

The import copies original RB weights and normalizes saved forecasts for the
common API. It preserves each study's provenance. QB forecasts retain the original
passing targets; they are not PPR predictions. The older QB workflow did not save
fitted estimators, so its imported members support saved-forecast queries only.
Training replacement QB estimators would create new artifacts in a separate task.
Use the original RB adapter for legacy-weight inference, not the Opening Drive
bundle replay runner. The import cannot overwrite an existing campaign ID.

## Full player rankings in the browser

### Same-ID v2 QB and rookie correction

The October 10 correction retains `opening_drive_v2`, while a new manifest hash
identifies its bytes. The candidate helper repairs exact-ID draft facts without
changing historical rows. The bounded QB experiment adds explicit opportunity
and efficiency features and evaluates attempts × predicted efficiency alongside
the original independent-target policies. Selection is strictly prior-year;
other QB targets and the RB/WR weights are preserved.

The correction's sealed run is
`runs/opening_drive_v2_update_20261010_001`. Training, staging, and readout commands
are separate; existing output directories fail closed rather than being reused:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 \
uv run --no-sync python -m experiments.future_player_lab.campaigns.qb_revision \
  --output experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001/qb_challenge \
  --cache experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001/inputs_candidate
uv run --no-sync python -m experiments.future_player_lab.campaigns.revise_v2 \
  --run experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001 \
  --registry experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001/staged_campaigns
uv run --no-sync python -m experiments.future_player_lab.campaigns.revision_readout \
  --run experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001 \
  --registry experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001/staged_campaigns
```

Staging checks preserved predictions and replays every saved target before
activation. The [replacement workflow](../../../docs/operations/model-campaigns.md)
archives the previous bundle and requires its exact hash. The experiment's
`export/readout.html` and `export/readout.json` report historical policy errors,
current selected policies, candidate coverage, and replay results.

### Building the player view

Build the standalone rankings page from verified saved forecasts:

```sh
uv run --no-sync python -m experiments.future_player_lab.campaigns.rankings_browser \
  --campaign opening_drive_v2 --season 2026 \
  --actual-snapshot data/raw/snapshots/weekly_20261008_w4_r3/manifest.json \
  --output data/outputs/opening_drive_rankings_2026
```

Open `data/outputs/opening_drive_rankings_2026/index.html`. The page defaults to
the latest saved cutoff and includes a preseason toggle, all modeled players,
overall and position ranks, search, position filters, component stats, and CSV
export. Rankings use the directly predicted points target; overall rank compares
raw fantasy points across positions and does not adjust for positional scarcity.
Filtering preserves ranks. The companion `rankings.json` retains forecast source
receipts and checksums. Building this view requires no model fitting or API server.

The optional actual snapshot adds season-to-date standard PPR points and ranks
through its completed-week cutoff. Actual ranks include all observed QB/RB/WR,
including players outside the model pool; ties share ranks and missing records
remain unknown. They cover a different period and a broader scoring contract
than the model projections, so the page does not call their difference accuracy.

## Tight end extension of v2

`opening_drive_v2_te` adds TE preseason and after-Week-2 models using the v2
recipes, regularization, chronological folds, prior-three-year selection, and
dated availability policy. It is a separately sealed extension because the
original v1/v2 bundles contain no TE cells. Existing bundles are not overwritten.

The fixed TE feature mask retains receiving volume, air yards, YAC, target share,
receiver tracking, offensive snaps, receiving/rushing play context (including
red-zone and inside-five usage), college receiving/rushing, age, physical measures,
draft capital, and team offensive context. Player passing, defense and kicking
fields are excluded. Availability features are appended after the mask. Missing
tracking or usage data stays missing; offensive snaps are not labeled as routes
run, and there is no blocking-quality target.

The eight outputs are targets, receptions, receiving yards/TDs, carries, rushing
yards/TDs, and direct PPR points. Occasional TE rushing remains in the scoring
contract. Selection diagnostics use top 12 TEs. The shared pipeline trains on
2002–2025 and evaluates 2021–2025 using only earlier seasons for each fit and
ensemble choice. There is no TE v1 comparison or untouched-holdout claim.

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 \
uv run --no-sync python -m experiments.future_player_lab.campaigns.tight_end \
  --run-id opening_drive_v2_te_003
uv run --no-sync python -m experiments.future_player_lab.campaigns.audit \
  experiments/future_player_lab/runs/opening_drive_v2_te_003/te_{remaining,season} \
  --output /tmp/opening_drive_v2_te_audit.json
uv run --no-sync python -m experiments.future_player_lab.campaigns.report \
  experiments/future_player_lab/runs/opening_drive_v2_te_003 \
  --output experiments/future_player_lab/runs/opening_drive_v2_te_review_NEW
uv run --no-sync python -m experiments.future_player_lab.campaigns.replay \
  --campaign opening_drive_v2_te
uv run --no-sync python -m experiments.future_player_lab.campaigns.rankings_browser \
  --campaign opening_drive_v2_te --season 2026 \
  --actual-snapshot data/raw/snapshots/weekly_20261008_w4_r3/manifest.json \
  --output data/outputs/opening_drive_v2_te_rankings_2026
```

The training command also scores and seals both models in the local campaign
store. `--device cpu` permits CPU fitting; `--export-only` scores an already
completed run. It verifies and copies the existing raw-history cache, constructs
new TE panels, and reuses the sealed v2 availability archive. The original v1
training CLI remains compatible with its three-position configuration.

[TE evaluation and artifacts](../../../docs/research/opening_drive_v2_te_2026-10-10.md).

For the accepted TE run, `opening_drive_v2_te_003/source_snapshot/` preserves the
training code independently of later workspace changes. Saved-bundle replay uses
the sealed matrix and weights; rebuilding raw features under changed code must
use a fresh run. Do not rewrite source receipts to bypass that check.

## Opening Drive v3.1 point-forecast extension

[V3.1 uncertainty feedback research](weekly_v31_points/README.md) tests whether
past forecast errors and uncertainty improve the expected points themselves.
It compares bias corrections, uncertainty feature ablations, model blends and
an appearance/conditional-points mixture using chronological fits and selection.
The interactive report is `data/outputs/opening_drive_v31_points_2026/index.html`.
This extends v3.1 research while preserving the earlier interval-only run and
its captured prospective forecasts. Research results do not change API serving.
