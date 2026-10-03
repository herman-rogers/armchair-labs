# RB usage residual correction: completed experiment

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

The proposed correction does **not** establish an improvement over the usage
baseline. Its tuned selection gain is +0.362 next-four-week points per pick,
with a season-bootstrap 95% interval of [-0.782, +1.516]. Point MAE worsens
slightly, and the small-prior-sample selections lose points. Keep the usage
baseline; this result does not support advancing this candidate as specified.

[Runner](../../research/rb_residual_correction.py) ·
[Fixed protocol](../../research/rb_residual_protocol.json) ·
[Saved report](../../data/research/rb_residual_20260923_r1/report.json) ·
[Manifest](../../data/research/rb_residual_20260923_r1/manifest.json)

## Dashboard status

Neither the preceding incremental-information audit nor this new residual
experiment is connected to a dashboard API/UI view. Both are viewable in the
repository as research notes and versioned artifacts. The existing Intelligence
workspace has the earlier player outlook, college and historical evidence views.

There is no pending code/data verification blocking a research-only display of
these results. The missing step is dashboard integration. Displaying a completed
negative experiment is useful and does not require changing the default model.
Forecast promotion is a separate question: this candidate fails the exploratory
advancement checks, and historical reanalysis would not replace prospective
validation even if those checks passed.

## Design fixed before this run

This implements the first experiment proposed in the
[baseline research note](incremental_information_2026-09-22.md). The protocol was
written before running the new candidate. It is not a preregistered independent
confirmation: the underlying 2019–2025 history has already been inspected.

- **Unchanged baseline:** the accepted usage ridge, with its existing seven
  features, training population, standardization and penalty of 10.
- **Residual target:** actual next-four-week points minus the usage prediction
  generated using only earlier seasons. Baseline out-of-fold residuals begin in
  2015, after two prior training seasons. They are not in-sample fitted errors.
- **Correction:** ridge on the baseline prediction, snap change, snap observation
  count, prior-season appearances and the original small-prior-sample indicator.
  All correction scaling and coefficients use earlier residual seasons only.
- **Control:** residual recalibration using only the baseline prediction, with
  the same shrinkage choices available to both correction models.
- **Shrinkage:** `baseline + weight × n/(n + half_life) × residual_prediction`,
  where `n = prior_season_games + observed_games`. This is a cutoff-known sample
  proxy, not a count of independent observations or an injury estimate.
- **Nested tuning:** weights 0, 0.25, 0.5 and 1; half-lives 0, 10 and 30. Zero is
  represented once, for ten configurations. Choose on equal-season selection
  points in earlier validation seasons; break ties toward smaller weights, then
  larger half-lives. Each validation prediction itself uses earlier training.
  No evaluated season chooses its own parameters.
- **Evaluation:** original 2019–2025 decision Weeks 3–13, original eligibility,
  three picks per window, next four calendar weeks. The main cohort has 77
  windows/231 selections. The original small-prior-sample cohort has 69/207.
  Small sample means 1–9 previous-season raw appearances, not specifically rookies.

The 2019 fold has residual training seasons 2015–2018 and tuning seasons
2017–2018. The 2025 fold has residual training seasons 2015–2024 and tuning
seasons 2017–2024. All training outcome windows end by Week 17 of an earlier
season, before the subsequent season's decision cutoff.

The provisional advancement criteria require at least +1 point per selection in
the main cohort, positive interval lower endpoints versus usage and recalibration,
non-worsening main-cohort MAE, and no negative selection gain in the secondary
cohort. The one-point threshold is a working research threshold, not estimated
waiver profit. All five checks fail; the conclusion also holds without that
arbitrary materiality threshold because the gain is uncertain and MAE worsens.

## Selection results

Differences are equal-weight season means versus usage. Changed picks count
player/window replacements, not unique players. Intervals use 10,000 seeded
season-block resamples; the cohorts overlap and are not independent experiments.

| Cohort | Policy | Changed picks | Four-week points gained per selection | 95% interval |
| --- | --- | ---: | ---: | ---: |
| All late-price RBs | Original role ridge | 43 / 231 | +0.652 | -1.331 to +2.533 |
| All late-price RBs | Tuned recalibration | 0 / 231 | 0.000 | 0.000 to 0.000 |
| All late-price RBs | Full residual correction, no shrinkage | 27 / 231 | +0.774 | -0.680 to +2.106 |
| All late-price RBs | **Tuned residual correction** | **22 / 231** | **+0.362** | **-0.782 to +1.516** |
| Small prior sample | Original role ridge | 8 / 207 | -0.200 | -0.421 to -0.021 |
| Small prior sample | Tuned recalibration | 0 / 207 | 0.000 | 0.000 to 0.000 |
| Small prior sample | Full residual correction, no shrinkage | 7 / 207 | -0.229 | -0.497 to -0.021 |
| Small prior sample | **Tuned residual correction** | **6 / 207** | **-0.224** | **-0.497 to -0.012** |

The primary tuned policy is positive in four of seven seasons; excluding 2019
turns its remaining average gain slightly negative. The small-sample policy has
four negative seasons and three ties. Its six changed picks do not justify a
general claim about all sparse-history players.

The unshrunk diagnostic has a larger point estimate, but its interval still
includes zero. Selecting it after seeing the outer-fold results would be another
historical selection step, not validation of a better candidate.

## Forecast errors

These errors cover **all eligible candidates in the evaluated windows**, not just
the three selected players: 2,164 main-cohort player/windows and 325 small-sample
player/windows. Every policy uses the same rows. MAEs below pool player rows;
the paired improvement gives each season equal weight.

| Cohort | Usage MAE | Tuned correction MAE | Equal-season MAE improvement | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| All late-price RBs | 11.295 | 11.312 | -0.017 | -0.088 to +0.050 |
| Small prior sample | 12.209 | 12.256 | -0.091 | -0.398 to +0.210 |

Equal-season MSE also worsens: improvement is -0.495 squared points in the main
cohort and -2.084 in the small-sample cohort. Residual learning has not uncovered
a useful correction hidden by the original full-target fit.

## What tuning chose

| Evaluation season | Role correction weight | Sample half-life | Recalibration weight |
| --- | ---: | ---: | ---: |
| 2019 | 1.0 | 0 | 0 |
| 2020 | 1.0 | 0 | 0 |
| 2021 | 1.0 | 0 | 0 |
| 2022 | 0.5 | 10 | 0 |
| 2023 | 0.5 | 10 | 0 |
| 2024 | 0.5 | 10 | 0 |
| 2025 | 0.5 | 10 | 0 |

Full corrections won the earliest inner comparisons; later comparisons selected
a smaller correction with additional sample shrinkage. The selected policy still
fails to generalize reliably. Recalibration always selects zero. Because tuning
optimizes selection points, it cannot reward a monotonic recalibration that leaves
the ordering unchanged. This does **not** establish that recalibration could never
improve point calibration under an MAE-focused experiment.

Every configuration's earlier-season scores are retained in the report. There was
one real-data experiment run; no outer-result-driven retuning followed it.

## Interpretation and next decision

This is useful negative evidence about the particular role variables and model
tested. It supports the earlier concern that these inputs largely repeat the
usage baseline. It does not prove that baseline errors are intrinsically
unpredictable or that every form of residual learning must fail.

Retain usage for this task and retain the negative result in the research record.
Do not keep expanding penalties and shrinkage grids on these same seven seasons.
A subsequent role experiment needs a concrete source of information that the
usage baseline lacks—for example dated changes in teammate availability or role
allocation—with cutoff-safe historical evidence or newly frozen prospective
observations. The college-prior/usage experiment remains a separate, unrun study.

Preseason ADP is still a cost stratum rather than an executable waiver/FAAB price.
Inactive players are outside the original pilot's candidate pool. Reconstructed
historical statistics, snap coverage and the original missing-data conventions
remain limitations. The season bootstrap does not eliminate dependence between
players across seasons or fully quantify uncertainty from training and tuning.
No interval is multiple-comparison adjusted; no 2026 outcomes are graded.

## Reproduction and verification

```sh
.venv/bin/python research/rb_residual_correction.py \
  --history data/research/historical_v2_20260922_r5 \
  --version YOUR_NEW_RESIDUAL_VERSION
.venv/bin/pytest research/test_rb_residual_correction.py \
  research/test_incremental_information.py research/test_research_audits.py
```

The runner checks accepted history and original pilot hashes, snapshots its
implementation and protocol, and writes a new version only. It verifies:

- Exact reproduction of all 6,946 saved rows' original role features, price,
  outcomes, usage forecasts and role forecasts; maximum absolute differences zero.
- Exact reproduction of 876 usage/role selection records across both cohorts.
- All 28 recorded source hashes and 12 protected artifact hashes unchanged.
- 29 targeted tests pass, including outer-outcome perturbation, future-row
  exclusion, chronological residual targets, tuning chronology, identical
  candidate pools, tie-to-zero behavior and missing/duplicate data guards.
- Ruff checks and formatting pass.

Passing implementation checks means the experiment was run consistently. It
does not turn its inconclusive/negative predictive result into a promoted model.
