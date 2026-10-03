# Historical fold repairs — 2026-09-22

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

This change addresses route coverage, transaction identity, configurable weekly
history, and sparse-source adaptive selection. The original 15-item review was not
present in the repository; the latter two issues were taken as findings 9 and 10
from the supplied summary.

- Route availability now requires participation-derived observations. Play-by-play
  usage alone cannot establish route coverage. Red-zone usage has its own source
  indicator, so valid historical usage remains available.
- Transaction prose uses full-name token boundaries and excludes short-name aliases.
  Name ambiguity is checked against the complete player identity table, including
  players outside the returning-player fold. Ambiguous matches require a matching
  transaction team. Unique full names can still establish legitimate team changes.
- Weekly summaries support any positive history length. Additional years receive
  their own lag features; named three-year trends retain their three-year meaning.
  One-year histories have no year-over-year estimate. Feature cache keys include
  history length, and both report and discovery caches have new versions.
- Adaptive selectors compare identical prior folds and use the entire observed
  outcome universe to construct the ideal ranking. Missing predictions cannot
  remove an actual winner. A fold needs at least K predictions from every candidate;
  insufficient shared history uses the configured fallback. Current and future
  outcomes remain excluded from selection.

## Validation

- Offline suite on the original base: **420 passed, 35 network tests deselected**.
- After the concurrent stat-system merge: **431 passed, 35 network tests deselected**.
- Ruff and mypy pass; `git diff --check` passes.
- Rebuilt historical panel: route coverage is false for 2001–2015 and true for
  2016–2025; play-by-play coverage remains available across 2001–2025.
- Cached ESPN reproduction: Lamar Jackson's 2020 cutoff team changes from NE to
  BAL, with zero matched transactions.
- Across the retained returner folds, recalculating cutoff transactions changes
  467 team values, 234 status values, and 690 transaction counts. These are
  differences from saved inputs, not independently adjudicated transaction labels.

The matchups edits and frozen 2026 prospective snapshot are outside this change.

## Evaluation provenance

The historical rebuild started on `66f13c8` plus these four repairs. While it was
running, a separate stat-system implementation was merged as `c80b346` and the
working changes were restored. The running Python process retained the original
loaded implementation. Its comparison isolates this task from the later merge;
it is not a re-evaluation of that merged system. An isolated source copy and patch
were preserved during the merge.

## Rebuilt report

Artifacts are in `data/outputs/metrics_review_2026-09-22/`: the corrected JSON and
Markdown report, forecast parquet, saved baseline report, comparison, and patch.
Production report and board outputs were not replaced by this run.

The following are the report's overall top-60 results (modern = 2019–2025).

| Window | Ranker | Hit rate before → after | NDCG before → after |
|---|---|---:|---:|
| long_horizon | market_ecr_score | 0.6100 → 0.6100 | 0.6610 → 0.6610 |
| long_horizon | fitted_season_points | 0.5702 → 0.5702 | 0.6158 → 0.6158 |
| long_horizon | fitted_rich_weekly_combined | 0.5325 → 0.5307 | 0.5969 → 0.5936 |
| long_horizon | fitted_adaptive_ppg_hybrid | 0.5772 → 0.5789 | 0.6157 → 0.6180 |
| long_horizon | fitted_adaptive_season_hybrid | 0.5711 → 0.5781 | 0.6163 → 0.6188 |
| modern | market_ecr_score | 0.6333 → 0.6333 | 0.6596 → 0.6596 |
| modern | fitted_season_points | 0.5929 → 0.5929 | 0.6228 → 0.6228 |
| modern | fitted_rich_weekly_combined | 0.5738 → 0.5690 | 0.6042 → 0.5953 |
| modern | fitted_adaptive_ppg_hybrid | 0.6048 → 0.6095 | 0.6285 → 0.6330 |
| modern | fitted_adaptive_season_hybrid | 0.5952 → 0.6048 | 0.6208 → 0.6226 |

Neither Adaptive nor the rich-weekly candidate overtakes ECR in these reported
windows. Rich-weekly evidence weakens; this does not support promoting it.

This is a rebuild-versus-saved-report comparison, not a controlled ablation. The
saved forecast parquet had 17,103 rows; the rebuild has 17,021. One historical
rookie row (Ben Nikkel, 2024) and 82 pending 2026 rows disappear, and one pending
2026 row appears. On matched rows, actual PPG and season points are unchanged, and
hand-built projected PPG differs by at most 8.6e-14. Some fitted incumbent values
also change, despite unchanged aggregate incumbent ranking scores. These input and
fitting differences prevent attributing every score delta solely to the repairs.

The original report's differing candidate populations and the separately reported
transaction clause/trade issues further limit the interpretation. The concurrently
merged stat-system changes require their own evaluation; the numbers above are not
a new verdict on that merged system or a grade of the frozen 2026 forecast.
