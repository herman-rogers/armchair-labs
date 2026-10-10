# Opening Drive v2 tight end extension

`opening_drive_v2_te` applies the Opening Drive v2 boosted-tree protocol to tight
ends, with separate preseason (`te_season`) and after-Week-2 (`te_remaining`)
models. It is a research preview saved alongside the original v1/v2 bundles.
The original QB/RB/WR weights and application serving choices remain unchanged.

## Method and position relevance

The four candidates are the original raw/draft XGBoost, shallow regularized
XGBoost, provider-ablation XGBoost, and LightGBM recipes. Eight core training and
inference functions match the archived v2 source at the Python syntax-tree level.
Each target averages the two candidates selected using only the preceding three
validation years, with equal weight per year's MSE. Training starts in 2002;
validation covers 2018–2025, evaluation covers 2021–2025, and final models refit
through 2025. Targets use calendar-week rates scaled to the forecast horizon,
including missed games and reduced usage.

The fixed TE mask preserves receiving volume, target and air-yard shares, YAC,
receiver tracking, offensive snaps, receiving/rushing play summaries, red-zone and
inside-five usage, college receiving/rushing, age, physical measurements, draft
capital, team offensive context, and the common history windows. It removes
player passing, defensive and kicking fields. Team passing remains relevant
context. The mask is defined before fitting and does not depend on validation
outcomes or observed future coverage. The v2 layer then appends 23 availability
features, for 1,335 fitted inputs per horizon.

Offensive snaps are not routes run. Tracking is incomplete, particularly in early
seasons; unavailable values stay missing. There is no blocking-quality label or
claim that this measures all aspects of TE play.

The eight independent outputs are carries, rushing yards, rushing TDs, targets,
receptions, receiving yards, receiving TDs, and direct PPR points. Occasional TE
rushing is retained, including hybrid offensive roles. PPR means
`receptions + 0.1*(rushing_yards+receiving_yards) + 6*(rushing_tds+receiving_tds)`;
passing, fumbles and bonuses are excluded. Direct and component-derived points
are exposed separately. Top-player diagnostics select 12 TEs.

## Population and availability

Candidates are prior-season TE participants, TEs observed by the cutoff, and
exactly linked TE draft entrants. Future appearances never define eligibility.
The original v1 bundle has no TE panel, so this extension builds and freezes new
TE base inputs using the shared construction rules. It does not reuse WR rows or
weights and makes no matched TE v1 improvement claim.

The v2 dated transaction, weekly-roster, and timestamped injury layer is reused.
Only the latest unambiguous formal retired status excludes a player. Injured,
suspended, released and unknown players remain eligible for aggregate production.
The same policy applies to training, evaluation, baselines, and current forecasts.
Retirement announcements alone remain features. Undrafted or unlinked newcomers
can be missing, and team context reflects the last observation.

These are retrospective historical development evaluations, not a pristine
holdout. The 2026 forecasts reconstruct preseason and after-Week-2 cutoffs from
later captures. They are not forecasts updated to the current week.

## Results and validation

Pooled 2021–2025 points errors on identical eligible player outcomes, compared
with affine persistence fitted on earlier seasons:

| Horizon | Evaluation rows | Tree RMSE | Baseline RMSE | MSE reduction | High-usage MSE reduction |
|---|---:|---:|---:|---:|---:|
| After Week 2 | 721 | 30.57 | 38.74 | 37.7% | 32.7% |
| Preseason | 699 | 38.37 | 42.96 | 20.2% | 5.4% |

After-Week-2 top-12 selections realize more points than affine selections in all
five evaluation years. Preseason selections win in three years but lose in 2023
(99.7 fewer realized points) and 2024 (43.9 fewer). This supports the remaining
season model more strongly than preseason selection. These are development
results, not established prospective performance.

All four receiving outputs improve pooled MSE in both horizons. Sparse rushing
outputs are weaker: carries and rushing yards lose to affine in both horizons,
and preseason rushing TDs also lose. They are retained to preserve the explicit
PPR and hybrid-role contract, not presented as proven improvements.

Independent component outputs can be inconsistent: receptions exceed targets on
56/721 remaining-season and 55/699 preseason evaluation rows; rushing TDs exceed
carries on 138/721 and 132/699 rows. The original protocol does not constrain these
outputs jointly. Direct PPR predictions remain separate from component arithmetic.

Both model audits pass, including 512 historical candidate checkpoints, ensemble
selection, all reported metric groups, and the original base matrices. An
independent relational check verifies availability cutoffs and latest states.
Five historical remaining-season candidates and two preseason candidates are
excluded by the dated retirement rule; their outcomes never decide eligibility.

The sealed bundle contains two horizon models, 32 fitted ensemble members, and
50 verified artifacts. Its 151 after-Week-2 and 137 preseason forecasts replay
with maximum absolute difference 0.0. Real-bundle API checks cover both models,
metrics, forecasts, player filters, invalid cutoffs, and manifest pins. Browser
checks cover both horizons, TE filtering, search, and component details.
The final 120-test regression suite, lint, and whitespace checks pass.

[TE player rankings](../../data/outputs/opening_drive_v2_te_rankings_2026/index.html) ·
[Full evaluation](../../experiments/future_player_lab/runs/opening_drive_v2_te_review_20261010_001/report.md) ·
[Audit](../../experiments/future_player_lab/runs/opening_drive_v2_te_review_20261010_001/audit.json) ·
[Portable metrics](assets/opening_drive_v2_te_2026-10-10_metrics.csv) ·
[Provenance](assets/opening_drive_v2_te_2026-10-10_provenance.json)

## Reproduction and access

See the [training, audit, replay, and rankings commands](../../experiments/future_player_lab/campaigns/README.md#tight-end-extension-of-v2).
The local campaign API exposes:

```text
GET /api/chairstats/campaigns/opening_drive_v2_te
GET /api/chairstats/campaigns/opening_drive_v2_te/models/te_season
GET /api/chairstats/campaigns/opening_drive_v2_te/models/te_remaining
GET /api/chairstats/campaigns/opening_drive_v2_te/models/te_remaining/forecasts?season=2026&origin_week=2
```

Model weights, feature contracts, historical predictions, current forecasts and
source receipts remain in the local research run and sealed campaign store.

The accepted run is `opening_drive_v2_te_003`. Its checksummed `source_snapshot/`
retains the exact code used for fitting, including the original entrant rules.
A concurrent workspace change to draft-entry recovery was not incorporated into
this TE experiment. The training source copies in the sealed bundle are authoritative;
replaying saved matrices does not require the live workspace to match those files.
