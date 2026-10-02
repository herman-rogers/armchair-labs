# Research evidence: dataset retention and tentative rookie forecasts

## What changed in the interface

Research & Evidence now shares player-model names and descriptions with Player
rankings. Comparisons, the ranker leaderboard, and model weights default to the
short player-forecast list; components and other experiments remain available via
**All research outputs (advanced)**. Only saved models in the selected slice appear.
Changing a population or hiding advanced options resolves the displayed selection
to an available model, rather than leaving a stale selection or calling a baseline
a model. Weight summaries explain why product forecasts may have no coefficients.

The evidence-window list defaults to modern, long history, and the original market
archive. Recent-era and extended/backfilled market windows remain under advanced.
Window labels describe their purpose, not a purported quality grade. The page
includes a dataset-quality/retention guide. This is presentation only: no predictions,
saved reports, source snapshots, or production policies were changed.

## Dataset versions are different from evidence windows

The screenshot's five options are overlapping evaluation windows of the same
selected dataset. They do not represent five generations of repaired data. Switching
one filters saved evaluations; it does not change training inputs or retrain models.
They also are not five independent replications of a result.

The live historical research catalog currently contains only:

- `historical_v2_20260922_r5`: accepted repaired research; the default.
- `legacy`: saved research before repairs; reference only.

The repaired version passes 12 recorded integrity gates. Its earlier audit confirms
all 47 identified false-zero outcomes were repaired. This is evidence of improved
data handling, not evidence that more complicated models now beat the market.
Historical publication vintages and context coverage remain incomplete. Changing
the eligible player universe also means legacy-versus-repaired performance changes
cannot be attributed to an individual repair.

### Coverage differences within the repaired history

Computed directly from `historical_v2_20260922_r5/outputs/metric_backtest_predictions.parquet`,
excluding pending 2026 forecasts. A row here is a candidate player-season, not a
unique player. Market coverage uses all candidates; the two context columns use
returners only. Presence is not a guarantee of correctness or publication vintage.

| Evaluation window | Candidate rows | Position ECR present | Returners | Observed cutoff state | Dropback on-field proxy present |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long history, 2004–2025 | 17,450 | 8,284 | 12,677 | 1,450 | 5,279 |
| Modern, 2019–2025 | 5,955 | 4,519 | 4,244 | 1,450 | 4,194 |
| Recent, 2020–2025 | 5,114 | 3,981 | 3,657 | 1,450 | 3,630 |
| Original market archive, 2021–2025 | 4,321 | 3,469 | 3,071 | 1,301 | 3,063 |
| Extended market, 2011–2025 | 12,351 | 8,284 | 8,801 | 1,450 | 5,279 |

For example, the modern window has observed cutoff state for only **34.2%** of
returners despite having the on-field proxy for **98.8%**. The long window has that
proxy for **41.6%**. Thus “modern enriched” was not an adequate quality description:
quality and coverage are source-specific. The on-field metric is not verified routes
run, and missing state is not confirmed active status.

### What to retain

| Asset | Decision | Reason |
| --- | --- | --- |
| Accepted historical r5, copied inputs, manifests and acceptance records | Keep active | Primary audited historical research; dependencies of outlook, rookie and college studies |
| `college_source_20260922_r1` and `college_nfl_v1_20260922_r5` | Keep active | Pinned college source and current linked translation study; complementary to NFL history |
| `player_outlook_v1_20260922_r5` and saved rookie-watch observations | Keep active | In-season forecasts have a different horizon and purpose from preseason forecasts |
| V1/draft archive and frozen prospective forecasts | Keep protected | Reproduce the user's actual draft and forecasts made before outcomes |
| Legacy pre-repair research | Keep archived/reference-only | Regression comparisons and audit trail; not the default for current conclusions |
| Failed historical builds: base version and r1–r4 | Archive candidates | All five manifests say failed; none is research-accepted. Their `du` footprints total approximately 2.4 GiB, not guaranteed reclaimable physical space |
| College r1–r2 incomplete; r3–r4 superseded; outlook predecessors | Archive candidates | Preserve provenance/debugging if useful; current pointers select r5. Review dependencies and backups before removal |
| All five evidence windows | Keep calculations; declutter controls | Negligible benefit from deleting slices; older eras and alternate starts are useful sensitivity checks |

No files were deleted or moved. The 13 protected artifact hashes still match the
historical rebuild's preservation manifest. “Latest” should not replace acceptance,
provenance, coverage, and held-out performance as a selection criterion.

## Are the rookie forecasts useful?

Yes—as tentative priors, with current usage considered separately. The linked
college study can score **134 of 157** audited 2026 rookie candidates; unscored
players are unknown, not zero-value. The college model is not yet incorporated into
the existing saved Next-gen season model. These are separate research outputs.

### Ballpark preseason full-season forecasts

The following are the new college-plus-draft model's expected rookie-season fantasy
points under the application's league scoring, rounded to about five points. These
are **preseason full-season expectations**, not rest-of-season totals, ceilings,
validated confidence intervals, or trade values. They include the possibility of
limited NFL involvement; they are not estimates conditional on becoming a starter.

| Position | Tentative order among these candidates | Full-season points |
| --- | --- | ---: |
| WR | Carnell Tate | 175 |
| WR | Jordyn Tyson | 150 |
| WR | Makai Lemon | 115 |
| WR | KC Concepcion | 110 |
| WR | Omar Cooper Jr. | 95 |
| WR | De'Zhaun Stribling | 90 |
| WR | Denzel Boston | 90 |
| WR | Germie Bernard | 80 |
| RB | Jeremiyah Love | 225 |
| RB | Jadarian Price | 105 |
| RB | Kaelon Black | 75 |
| RB | Jonah Coleman | 70 |
| RB | Emmett Johnson | 55 |
| TE | Kenyon Sadiq | 90 |
| TE | Eli Stowers | 65 |
| TE | Max Klare | 55 |
| TE | Sam Roush | 55 |

Draft capital accounts for most of the separation at the top: Tate's draft-only
estimate is 174.1 versus 174.7 with college data, while Boston is 88.5 versus 88.0.
The new college inputs are **not** the main reason Tate rates above Boston.
College-plus-draft does not outperform draft-only for RB/QB rookie points, so these
experimental estimates do not justify replacing those positions' simpler baselines.

### Current four-week sanity check, saved through Week 2

For the same Weeks 3–6 horizon, a simple median of the outlook, usage-only baseline,
and rookie-analog estimate gives the following ballparks. This is a read-only
calculation for this review, **not a newly validated or published ensemble**. The
inputs overlap, so agreement among them is not three independent confirmations.
No preseason season-total number is averaged into this four-week calculation.

| Player | Outlook | Usage baseline | Rookie analog | Rough four-week midpoint |
| --- | ---: | ---: | ---: | ---: |
| Denzel Boston | 43.0 | 47.0 | 42.5 | 43 |
| KC Concepcion | 37.0 | 39.2 | 29.6 | 37 |
| Carnell Tate | 32.7 | 35.1 | 31.4 | 33 |
| Makai Lemon | 22.3 | 24.9 | 25.5 | 25 |
| Jeremiyah Love | 36.7 | 36.5 | 49.1 | 37 |
| Jadarian Price | 32.5 | 31.9 | 36.9 | 33 |
| Kaelon Black | 27.1 | 26.4 | 25.0 | 26 |
| Jonah Coleman | 21.6 | 21.9 | 29.3 | 22 |
| Kenyon Sadiq | 21.4 | 21.9 | 16.4 | 21 |

These are saved observation-based estimates, not live injury-adjusted start advice.
Recheck current availability before acting. Boston's two-week snap average is 92.5%
versus Tate's 81.5%; both have 11 targets. Current opportunity can legitimately
reverse their preseason order. Boston's 88-point preseason estimate is not a cap
on a season in which his role has developed favorably.

### How much actual predictive ability?

Chronological held-out rookie-year results for 2018–2025, restricted to identical
eligible rows for each comparison. Lower mean absolute error (MAE) is better.

| Position | Held-out candidates | Position-mean MAE | Draft-only MAE | College-only MAE | College + draft MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB | 74 | 73.19 | 45.63 | 64.30 | 48.16 |
| RB | 217 | 49.01 | 31.72 | 40.83 | 31.86 |
| WR | 329 | 43.86 | 24.12 | 32.89 | 23.36 |
| TE | 150 | 30.09 | 18.14 | 22.40 | 17.27 |

College-only improves on the position-mean baseline, but draft capital is stronger.
Adding college features lowers rookie-point MAE about **3.1% for WR** and **4.8% for
TE**; both season-bootstrap improvement intervals include zero in the main cohort.
RB is essentially unchanged/slightly worse and QB is worse. These aggregate errors
include low-output players and do not by themselves prove we can find breakout
stars or profitable waiver targets. Identity-sensitivity results also vary.

More encouraging, though still exploratory:

- WR rookie games: MAE **3.79 → 3.60**, with season-bootstrap improvement interval
  **+0.13 to +0.26 games**.
- TE rookie games: **3.67 → 3.49**, interval **+0.11 to +0.32 games**.
- TE first-three-year points: **50.47 → 46.45** on 91 eligible players, about an
  **8%** MAE reduction; interval **+0.24 to +5.66 points**. RMSE is essentially
  unchanged, so large misses have not clearly improved.

These intervals describe average model-error differences, not individual player
ranges. Games capture participation, not a calibrated injury or starting probability.
Many positions, targets and variants were examined; none of these findings is a
multiple-comparison-adjusted, independent confirmation of a market-beating edge.

Recommendation: retain college-plus-draft as an exploratory preseason/longer-term
prior, use recent usage as the in-season baseline, and test an integrated
college-prior/usage-update model prospectively before publishing confidence scores
or making it the default ranking.

## Evidence files and verification

- `data/research/historical_v2_20260922_r5/acceptance.json`
- `data/research/historical_v2_20260922_r5/outputs/metric_report.json`
- `data/research/college_nfl_v1_20260922_r5/report.json`
- `data/research/college_nfl_v1_20260922_r5/nfl_predictions.parquet`
- `data/research/player_outlook_v1_20260922_r5/outlook.json`
- `data/outputs/rookie_watch_2026.json`

Frontend build and deterministic browser regression passed; the live repaired report
was also checked. Relevant API dataset/version tests passed. Existing unrelated lint
warnings remain; Vite warns that Node 21.7.3 is outside its supported Node versions,
although the build succeeds.
