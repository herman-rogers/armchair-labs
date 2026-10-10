# Opening Drive v2 dated availability experiment

The campaign was subsequently corrected **under the same `opening_drive_v2` ID**.
The current manifest is `ce90935b2f03d84f593955cb14a30e1813b2ca3cd29a9cb61ffefb46adfe5fa7`;
the original `43ecaf34…` bundle is retained under
`data/campaign_revisions/opening_drive_v2/`. The original experiment below remains
a record of that earlier revision.

The [QB/rookie correction readout](../../experiments/future_player_lab/runs/opening_drive_v2_update_20261010_001/export/readout.html)
records the subsequent bounded experiment and its limitations. Exact-ID draft
facts recover 43 preseason candidates and 14 Week 2 candidates, including eight
QBs at each cutoff. All historical rows and existing current feature values are
preserved. The saved 2026 preseason passing policies remain unchanged; Week 2
selects trees with 46 explicit QB inputs for attempts and an attempts × blended
efficiency forecast for yards, using only 2023–2025 selection evidence.

Historical 2021–2025 Week 2 yardage RMSE improves from 749.31 to 745.82, but MAE
worsens from 529.65 to 530.21 and high-prior-usage results worsen modestly. This is
mixed development evidence, not an across-cohort win. Seven other QB target
weights, including direct fantasy points, and all RB/WR weights are preserved.
All 1,048 revised forecasts replay exactly from the 220-artifact bundle. The
[rankings page](../../data/outputs/opening_drive_rankings_2026/index.html) now uses
this revision; actual ranks remain standard PPR through Week 4.

## Original availability experiment

Opening Drive v2 adds dated availability information to the six Opening Drive
position/horizon models. It corrects the Drew Brees eligibility failure and lowers
paired fantasy-point MSE in all six packages, with the largest changes in preseason
RB and WR. Top-player selection remains mixed, and QB preseason still loses to
affine persistence in four of five evaluation years. The campaign is saved as a
separate research preview; application serving selections and v1 artifacts remain
unchanged.

## Controlled comparison

The four recipes, estimator settings, scoring contracts, calendar-week targets,
expanding historical training, and preceding-three-year ensemble selection are
unchanged from [v1](opening_drive_2026-10-09.md). Eight core functions match the
archived v1 source at the Python syntax-tree level. All retained original feature
and outcome columns match the frozen v1 panels exactly; 23 availability features
are appended.

Training starts in 2002. Historical candidate forecasts cover 2018–2025; evaluation
covers 2021–2025. Every historical forecast trains on earlier seasons, and its two
recipes are selected using only the preceding three validation seasons. Final
2026 models refit through 2025. No current-season-only training or future-appearance
eligibility filter is introduced.

The main comparison pairs v1 and v2 on the same v2-eligible player outcomes. A
separate selection diagnostic applies the new eligibility rule to frozen v1
predictions without retraining. This distinguishes the eligibility effect from
the combined effect of retraining and adding features. It does not identify a
causal effect for any single new feature. These are reused historical development
years, with no pristine holdout or prospective-performance claim.

## Availability policy

The layer uses dated transaction reports, completed-week roster states, and
injury reports with modification timestamps. Features describe latest recorded
status, evidence age, conflicts, observed recent transactions, team changes, and
injury-report status and age. A historical active record is not proof of current
availability; evidence age and coverage limitations remain material.

Only the latest unambiguous formal retired status excludes a player. Retirement
announcements remain features because the archive can omit a later return
announcement. Injury, reserve, suspension, practice-squad, release, and unknown
states remain eligible for aggregate future production. Same-time conflicting
states stay unknown. Unique normalized identity names link transaction clauses;
ambiguous names are withheld, and unrelated sentences do not inherit an action.
There are no player-specific eligibility overrides.

Date-grain transactions become usable the following day at 12:00 UTC. Weekly
rosters become usable the day after that week's final game at 12:00 UTC. Preseason
uses opening-game day at 12:00 UTC, before kickoff; remaining season uses the day
after the final Week 2 game at 12:00 UTC. Week 1 roster records never supply
preseason status. Injury reports modified after the cutoff are withheld.

The capture contains 44,177 ESPN reports across 57 pages, yielding 88,411 normalized
transaction observations. Combined with weekly rosters, the state archive contains
909,708 observations. The 2003 transaction archive was unavailable in this capture.
Raw responses, parsed evidence, matching exclusions, dates, and checksums are
retained locally. Source dates and roster vintages are retrospective, not archived
publication-time snapshots.

Across the historical panels, the policy excludes 22 player/cutoff rows, including
20 evaluation rows. All have zero recorded production for their forecast period;
that outcome check is descriptive and never determines eligibility. V1's starting
candidate universe and team-context features remain fixed. Undrafted or unlinked
newcomers absent from v1 remain absent, and stale team context is flagged rather
than rebuilt. Expanding the population and rebuilding team context require a
separate controlled experiment.

The methodological motivation is consistent with weekly fantasy research that
uses an eligible player list and distinguishes availability from conditional
performance, including the [NFL DFS methods paper](https://arxiv.org/html/2309.15253v1)
and the availability ablation reported in a
[Princeton senior thesis](https://theses-dissertations.princeton.edu/entities/publication/d6981488-f30c-47ce-8f06-b6c3e98f2578).
These sources motivate the experiment; they do not establish that this particular
layer or training window is optimal.

## Paired results

Fantasy-point horizon totals on identical eligible rows. Positive MSE reduction
means improvement over frozen v1 on those rows. High-usage comparisons share
thresholds learned from the original earlier-season training populations.

| Position | Horizon | Paired rows | V1 RMSE | V2 RMSE | MSE reduction | High-usage MSE reduction |
|---|---|---:|---:|---:|---:|---:|
| QB | After Week 2 | 468 | 57.88 | 57.78 | 0.35% | -1.25% |
| QB | Preseason | 456 | 77.80 | 77.76 | 0.12% | -0.01% |
| RB | After Week 2 | 901 | 46.16 | 46.14 | 0.11% | 1.41% |
| RB | Preseason | 862 | 57.45 | 56.31 | 3.95% | 0.76% |
| WR | After Week 2 | 1371 | 41.20 | 41.14 | 0.29% | -0.14% |
| WR | Preseason | 1325 | 50.16 | 48.90 | 4.99% | 3.63% |

37 of 50 position/horizon/target combinations improve ensemble MSE against v1 on
the matched rows. The point-model gains after Week 2 and for preseason QB are all
below 0.4%, so they should not be interpreted as established practical advantages.
High-usage QB remaining, QB preseason, and WR remaining errors slightly worsen.

Against newly fitted affine persistence on the same v2 population, all six point
models improve pooled MSE, by 17.1%–27.1%. Across all component/point targets,
48 of 50 improve against that baseline. This preserves the distinction between
beating a simple baseline and improving on v1.

## Selection and the Brees failure

In 2021 preseason, frozen v1's top 12 realized 3,063.90 points. Applying the dated
eligibility rule alone removes Brees and admits Brady, increasing that total to
3,444.64 points: exactly 380.74 more. The retrained v2 selection realizes 3,482.68
points, 38.04 above the affine selection. Brees is also excluded after Week 2.

The captured ESPN record dates Brees's formal retirement move to June 6, while the
[NFL transaction archive](https://www.nfl.com/transactions/league/reserve-list/2021/6)
dates it to June 11. Both precede the September forecast cutoff. This discrepancy
is retained as a source-quality limitation, without a named data override.
Brady's 2022 retirement announcement remains an uncertainty feature and does not
exclude him.

The broader selection results do not justify replacing every v1 model:

- QB preseason beats affine selection in 2021 but loses in 2022–2025. Its
  high-prior-usage MSE remains about 5.0% worse than affine persistence.
- RB remaining-season top-24 selection now loses to affine in 2021 and 2024,
  whereas v1 won in every evaluation year.
- WR preseason top-24 selection loses to affine in 2023 and 2024, versus one
  losing year for v1. It trails v1's eligible selection in three of five years
  despite the 4.99% pooled MSE improvement.
- RB preseason and WR remaining-season selections still beat affine in all five
  years. QB remaining-season selection loses in one year, versus two for v1.

Keep the dated layer and the frozen-v1-with-eligibility control available for
further evaluation. Preseason RB and WR show the clearest average-error gains,
but model choice must still account for selection and high-usage performance.
No universal v2 serving promotion follows from these development results.
Independent component inconsistencies and differences between direct and
component-derived points remain disclosed in the full baseline report.

## Validation and artifacts

All six cells passed independent artifact, chronological-selection, checkpoint,
and metric audits. The comparison recomputed all 1,600 v2 candidate checkpoints
and rechecked the preserved v1 cells. An independent relational audit reconstructs
latest availability states and verifies forecast calendars. The 97-test regression
suite, lint, and whitespace checks passed.

The local `opening_drive_v2` bundle contains 144 verified artifacts and 100 final
ensemble members. Its 991 player/horizon forecasts replay with maximum absolute
prediction difference 0.0. Eighteen actual-bundle API requests passed; unsupported
origins return 404 and invalid manifest pins return 409. Both v1 and v2 bundles
pass integrity verification. Current forecasts remain reconstructed cutoffs from
later captures, not forecasts issued before those games.

[Full matched-player comparison](../../experiments/future_player_lab/runs/opening_drive_v2_review_20261010_001/report.md) ·
[Baseline and coherence report](../../experiments/future_player_lab/runs/opening_drive_v2_review_20261010_001/baseline/report.md) ·
[Portable target metrics](assets/opening_drive_v2_2026-10-10_metrics.csv) ·
[Artifact provenance](assets/opening_drive_v2_2026-10-10_provenance.json) ·
[Reproduction workflow](../../experiments/future_player_lab/campaigns/README.md#version-2-availability-experiment)
