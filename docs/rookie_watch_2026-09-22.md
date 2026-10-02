# Rookie intelligence from early NFL production

Rookies already existed in the preseason research. The corrected r5 dataset contains
157 eligible 2026 rookies; the older live research has a different, unrepaired pool.
The production returner model is not a rookie model, and rookies in the published
board can use market fallbacks. The new **Intelligence → Rookie watch** view adds
in-season comparisons without replacing either the archived draft board or the
frozen prospective forecasts. Population filtering also works when comparing
current league observations in Player rankings and Mispricing.

NFL season **2026** includes the games played in early 2027. Hunter's rookie season
was 2025: he can be a historical comparison, but is not a current rookie candidate.

## Initial publication

Built from completed 2026 Weeks 1–2, forecasting total league points in Weeks 3–6.
All 157 rookies remain visible; 56 have sufficient observed production or offensive
usage and historical support for a forecast. Unobserved players are not assigned a
prediction. Zero-only box-score entries without offensive snaps do not establish
eligibility. Missing box-score entries are never described as games played.

| Player | Points through Week 2 | Next-four-week estimate | Historical 10th–90th percentile |
| --- | ---: | ---: | ---: |
| Denzel Boston | 39.4 | 42.5 | 18.4–70.3 |
| Carnell Tate | 13.5 | 31.4 | 4.6–61.5 |
| Kaelon Black | 10.5 | 25.0 | 1.5–49.6 |

Boston and Tate are first and second in the current rookie WR forecast. Their league
availability is joined live by ESPN ID. Ownership missing from the fetched pool is
**unknown**, not free agency. These are four-week point forecasts, not acquisition
prices or validated recommendations to replace a specific starter.

The displayed range describes actual outcomes of comparable historical rookies.
It is not a calibrated prediction interval. Five closest comparisons are displayed,
including unsuccessful outcomes: for example, Boston's matches include Sterling
Shepard, Emeka Egbuka, Corey Coleman, Jahan Dotson, and DK Metcalf.

## Method and validation

Historical cohort: audited r5 rookie populations from 2013–2025, preserving their
historical fantasy positions. At the same completed calendar-week cutoff, compare
each player only with earlier-season rookies at the same position. Eligibility,
features, scaling, and neighbor selection never use future production. Historical
injuries, inactive weeks, and byes contribute zero points to the four-week outcome;
poor outcomes remain in the cohort.

Inputs: league scoring pace, targets/carries per elapsed week, latest-week targets/
carries, observed offensive snap share, and NFL draft capital. The fixed model uses
the mean outcome of the nearest 25 players, with a minimum historical pool of 20.
Feature scales and missing-value imputation use earlier seasons only. Undrafted or
unknown draft capital has an explicit missing indicator. Missing snap shares are
not fabricated as zero. Snap share is not a charted route measure.

Week-2 walk-forward evaluation uses 2018–2025 test seasons and identical scored
players for all baselines. QB history only supports forecasts in 2021–2025.
Mean absolute errors below are **four-week totals**, not per-week errors:

| Position | Scored / eligible rookies | Analog error | Current scoring pace error | Earlier rookie mean error |
| --- | ---: | ---: | ---: | ---: |
| QB | 18 / 28 | 22.74 | 19.15 | 23.12 |
| RB | 129 / 129 | 14.69 | 14.82 | 19.54 |
| WR | 197 / 197 | 11.21 | 12.57 | 15.55 |
| TE | 89 / 89 | 7.53 | 7.98 | 10.56 |

WR error is about 11% lower than carrying forward scoring pace, with improvement in
five of eight seasons. RB is nearly tied; QB is worse. These exploratory results do
not establish an advantage over current consensus, waiver profit, or calibrated
uncertainty. No parameter search was performed. The final eligibility definition
excludes zero-only stat lines without offensive snaps; this source-quality correction
was made during implementation, so this is development evidence, not a locked
prospective validation.

Current injuries, upcoming opponents, and individual future byes are not modeled.
The UI shows current ESPN health separately. Historical player metadata and feeds
may have retrospective revisions; these are not original Tuesday publication
snapshots. New point/usage snapshots and complete reports are archived prospectively.

## Rebuild

```sh
just rookie-watch data/research/historical_v2_20260922_r5
# Optional replay at an earlier completed cutoff:
just rookie-watch data/research/historical_v2_20260922_r5 --through-week 1
```

Run after each completed NFL week. No background refresh is configured. The runner
checks that the explicit historical version passed its integrity audit, that source
hashes still match, and that historical and current scoring rules agree. It uses
only consecutive finished weeks with weekly-stat coverage for every scheduled team,
and rejects a requested cutoff without full coverage. Cutoffs are limited to Weeks
1–13 so the four-week horizon also exists in historical 17-week schedules.

Outputs: `data/outputs/rookie_watch_2026.json`, plus timestamped input/report copies
in `data/outputs/rookie_snapshots/`. The JSON records source hashes, implementation
hashes, historical version, cutoff, cohort sizes, fold errors, and limitations.
API: `GET /api/research/rookies`. It remains readable without private ESPN access,
with ownership explicitly unknown. The page exposes cutoff and stale-data notices.

Official source timing: [nflverse update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).
Public participation is unavailable live for recent seasons; this model uses
weekly box scores and PFR offensive snaps instead.

Tests cover future-data invariance, training-season isolation, missing source
coverage, duplicate keys, no-observation behavior, shared backtest populations, and
unknown versus free-agent ownership. Browser checks exercise availability and
position filters against the local API.
