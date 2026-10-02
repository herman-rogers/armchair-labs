# Completed campaign record

## Final status — September 24, 2026 America/New_York

All authorized campaign work is complete. The execution notes below are retained
as history; there are no remaining fits, analysis jobs or pending permissions.

- `deep_search_003`: **fits_complete**, 11,232 verified checkpoints, exit 0.
- `deep_capacity_002`: **complete**, 1,248 verified checkpoints, exit 0.
- `deep_report_001`: **complete**, all 16 primary tasks and eight combined-library tasks.
- `deep_diagnostics_001`: **complete**; finish watcher exited 0.
- `deep_delivery_001`: final artifact verification, browser checks and readout hashes.
- Final narrative: `../deep_readout.md`; both library dashboards are linked there.
- Combined MSE policy improves published MSE in 8/8 tasks (2.70–25.53%), but MAE
  in only 3/8. The MAE policy improves MAE in 7/8. Earlier-selected top-three blends
  remain strong: primary MSE method selection beats that control in only 4/16 tasks.
- Adding saved models improves selected-MSE results in 8/8 matched tasks versus
  the new library alone. Broader inputs help; capacity returns diminish.
- No adaptive selected count approached the 4,096 ceiling. Neural convergence
  is not universal (33 modern fit warnings). Implausible extrapolations are
  retained and explicitly discussed rather than hidden.
- All 56 tests and Ruff pass; final browser checks cover 24 task views with four
  policy rows, filters, sorting and no JavaScript errors. Catalog hash unchanged.
- No promotion, serving changes or new 2026 forecast export. No global-optimum or
  prospective-superiority claim. No subagents used.

## Historical execution notes

User authorized all deeper ensemble recommendations: richer tree tuning, maximum
use of verified data, feature discovery, membership/weight optimization, stacking,
and matched comparisons to actual published predictions. Continue until completed;
do not mistake the completed supplementary study for the whole task.

## Active computation

- Main: `deep_search_003`, unified exec session 12923, logs
  `runs/deep_search_003.log`; 8 workers using independent annual folds,
  11,232 total path/fold checkpoints, 146 candidates.
- `_002` was stopped for parallelism after adaptive fitting freed two processors.
  All **9,498** completed checkpoints were checksum verified and reused in `_003`;
  **1,734** unfinished annual fits were queued. Numerical sources, configuration,
  package versions and prepared-data/source hashes matched exactly. `_002` remains
  preserved with `interrupted_for_parallelism` and a continuation pointer.
- Adaptive capacity: **complete, exit 0**, `deep_capacity_002`, session 79302, logs
  `runs/deep_capacity_002.log`; 4 workers, 1,248 checkpoints, six additional
  pipelines (3 engines × 2 feature views), max 4,096 trees, patience 80.
- All 1,248 adaptive fits finished. Selected tree-count min/median/max:
  CatBoost 52/320.5/1706; LightGBM 46/141/1335; XGBoost 36/148/787.
  No selected count was within 80 trees of the 4,096 ceiling. This is evidence
  against that ceiling binding in these recipes, not an unconstrained optimum.
- Original adaptive `_001` stopped for parallelism; all 151 completed checkpoints
  were verified against source, data, configuration and package hashes and reused
  in `_002`. The original manifest records interruption and continuation.
- Finish watcher: session 23041, logs `runs/deep_finish_004.log`; waits for both,
  then runs `deep.analyze --source deep_search_003 --capacity deep_capacity_002 --run-id deep_report_001`
  and `deep.diagnostics ... --run-id deep_diagnostics_001`.
- When main computation completes, consider moving adaptive capacity to more
  workers if still substantially unfinished. Stop its old executor and children
  safely, mark its manifest interrupted, then use the new **numerically identical**
  wrapper `deep.capacity_run --data deep_data_001 --run-id deep_capacity_003
  --reuse deep_capacity_002 --workers 8`. It verifies code/config/data/checkpoints.
  If doing this, stop/restart the finish watcher and point analysis at
  `--capacity deep_capacity_003`. Do not run two executors on the same run ID.
- Preferred redistribution executor is now **`deep.parallel_folds`**, which can
  continue either kind of run with independent per-year jobs. This avoids idle
  processors behind the last few long 13-year paths. Stop the owned parent and
  its workers, preserve/mark its old manifest interrupted, then use e.g.
  `--source deep_capacity_002 --run-id deep_capacity_003 --workers 8` (or 10 if
  the fixed search is complete). If adaptive fitting finishes first, it also
  supports `--source deep_search_002 --run-id deep_search_003 --workers 10`.
  Stop/restart the finish watcher with the resulting `--source`/`--capacity` IDs.
  It verifies numerical source, package versions, config, data and every reused
  checkpoint, and records source protocol/manifest hashes. Two tests verify
  **bit-identical** serial versus separate-process yearly forecasts for fixed and
  adaptive helpers. Live fit code remains unchanged. The new executor is prepared
  and **is now running for the fixed library**. Do not restart jobs just for its availability; use it
  when resources become free or remaining path-level scheduling leaves CPUs idle.

**Do not edit `deep/models.py`, `deep/data.py`, `deep/run.py`, or base lab Python
files while main fits run. Their source hashes are frozen. Do not edit
`deep/capacity.py` or `deep/capacity_run.py` while adaptive fits run.
**`deep/parallel_folds.py` is also frozen while `_003` runs.** Other report/document files can be
changed before the finish watcher begins analysis. Analyzer verifies its own
source hashes at completion.

## Completed

- `deep_data_001`: 69,584 rows, 2,929 inputs, 17,396 complete player-seasons,
  4,770 players; raw observations from 2001, candidates 2004–2025. Week-2 origins
  match current published backtests. All four positions/horizons. Primary points
  target; auxiliary models use five outputs. Inventory uses all valid numeric/
  boolean fields, without predictive-screen gates; 14 market inputs in named view.
- `deep_data_audit_001`: independently rebuilt with source snapshots;
  `deep_data_verification_001` verifies all seven core files byte-identical.
- `deep_reuse_001`: completed 32-model saved-library ensemble study for preseason
  2014–2025; four positions. No new base fits. `deep_reuse_sources_001` holds exact
  original implementation matching original hashes. Current reuse code has since
  added panel hash verification; originals preserved.
- `deep_reuse_comparison_001`: adaptive MSE policy vs equal top-three mean reduces
  MSE QB 4.186%, RB 1.151%, WR 1.117%, TE .234%; strongest fixed market tree still
  stronger on MSE than that policy. These are retrospective, unadjusted comparisons.
- `deep_benchmark_audit_002`: verified **nine** original published targets omit
  points when weekly position is outside QB/RB/WR/TE. Original filtered sums
  reproduce exactly. Canonical comparisons re-score the same saved predictions
  against full player points; no affected row excluded or forecast changed.
  `published_reconciled.parquet` retains original labels too. All 1,682 unknown
  2026 outcomes remain unknown. `deep_benchmark_audit_001` is marked failed for
  its initial failure to handle those null future labels.
- `deep_preview_001`: QB preseason integration test before adaptive candidates
  existed; complete preview only. 178 candidates = 146 new + persistence + 31
  archived fixed models. Top-three RMSE62.70, selected-MSE63.74, best fixed63.17.
  Not final campaign evidence; no production comparator for preseason.
- Earlier main `deep_search_001` stopped for PLS/neural-tree output-shape fix.
  Exactly 708 unaffected checkpoints verified/migrated to `_002`; migration ledger
  and old sources preserved. No numerical change to reused branches.
- All 56 focused tests (29 prior + 27 deep) pass, including separate-process
  annual-fold equivalence; final full output is `runs/deep_final_tests_002.log`.
  Ruff also passes.
  Includes future-label invariance, six tree-prefix equivalence tests,
  four five-output auxiliary adapters, three adaptive chronology checks, benchmark
  position-change reconciliation, complementarity and checksum protection.
- Ruff checks pass. Reuse dashboard browser test passed (Chrome): 3 policy rows,
  70 total models, position/view filters, no JS errors.
- Updated dashboard passed synthetic Chrome smoke for both library links,
  comparison/model view switching, sort resets, filters, friendly headings, no JS errors.
  Still check final real artifacts after rendering.
- Diagnostics now also report controlled feature-view increments, held-out tree
  capacity increments and the incremental value of adding saved existing models.
  Smoke-tested against the completed QB preview. Input views `inventory` and `all`
  are identical 2,915-column aliases; exclude that pair from feature-benefit claims.
- Input coverage diagnostics include position/population/source-group missingness.
  Modern preseason cells are 71–74% nonmissing by position (includes flags and
  derived inputs; not all direct observations). Between 10 and 47 columns are
  entirely missing in each position across modern years. The live source hashes
  were rechecked unchanged after diagnostics/UI edits.
- Added a fourth primary policy, `policy_capture`, selected only from earlier
  years using published position-specific K (QB10/RB20/WR30/TE10). Two greedy
  capture combination methods also join the library. Negative point totals are
  preserved; player-ID tie order is stable. Tests verify cutoff behavior,
  maximization direction and future-label invariance. Final dashboards should now
  show **four** policy rows and a Top-K point capture score column, so earlier
  three-row UI smoke expectations must be updated for the final real report.
- A new four-policy synthetic end-to-end smoke passed after the capture changes:
  chronology, matched comparisons, paired diagnostics, capture and uncertainty
  summaries, both Chrome dashboards, sorting and filters. Numerical fit sources
  were checked again against both live protocols and remained unchanged.
- Final analysis now accepts `--workers` (default 8), recorded in its manifest;
  it can use the freed performance cores once fitting completes. This changes
  execution parallelism only, not any model or ensemble method.
- Comparison summaries now include MAE/NDCG/capture bootstrap intervals and
  per-metric season-win counts. UI labels explicitly distinguish MSE wins from
  capture wins. `paired_methods` diagnostics now include rank/capture increments.
- Combined-library dashboard now opens on next-four forecasts and offers only
  its two supported horizons. Main dashboard explains absent published comparators
  for preseason/next-week instead of merely showing zero rows. Chrome default-
  navigation smoke passed. Still verify the final actual artifacts when ready.
- Search scope is now fixed. Complete both campaigns and analyze their results;
  do not keep adding model arms while waiting. Fix actual errors if discovered.

## Important pending checks and delivery

1. Wait for **both** long campaigns; monitor every ~60 seconds with concise updates.
2. Finish full ensemble analysis and diagnostics. If a job fails, preserve its run,
   fix and use a new ID. Actual reports must have complete manifests.
3. Read resulting summaries/paired comparisons, coverage, policy choices, capacity
   boundaries, optimizer warnings and negative predictions. Do not infer complete
   optimization or superiority from training loss/hindsight best candidates.
   Adaptive metadata records **selected** tree count, not all scanned rounds.
   Inspect counts within 80 rounds of 4,096 as possibly budget-bound even if
   `ceiling_selected` is false; do not infer an unconstrained optimum.
4. Browser-check final dashboards including published-comparison filters. Test the
   pipeline as necessary if further code changes are made.
5. Write a final readout and link from main README/initial_readout. Explain that
   152 new candidate policies plus archived models were tested; no absolute global
   optimum, no prospective confirmation, no production promotion.
6. Published benchmark is actual catalog release
   `nextgen_qb_variations_20260924_r2`: QB next4 uses approved `policy`; all other
   scopes use `reference`. Target audit explains nine position-filter omissions.
7. New report has primary library and `published_library/` where the published
   policy/reference themselves can join the ensemble. Saved actuals are reconciled
   before matching; forecast origin/end/player/outcome must agree.
8. No new 2026 forecast export was requested for this deeper stage; don't call
   2025 composition choices current-season forecasts.

No subagents used or authorized. No goal object exists. Shared worktree was already
very dirty; changes are confined to this lab except installation of Homebrew libomp.
New Python dependencies are isolated in ignored `.packages`, not replacing the
application environment. `deep/README.md` records versions, scope and commands.
