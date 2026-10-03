# Player outlook and uncertainty repairs — 2026-09-22

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

## Available in the app

Intelligence → **Player outlook** combines four-week expected points, a simple
recent-usage comparator, opportunity change, offensive role evidence, participation,
and an explicitly gated outcome range. The view supports position, ownership,
rookie/small-prior-sample, scored/unscored, and name filters. Open a player for
sample sizes, recent targets/carries, snap-share variation, concentrated scoring,
long-TD bonus dependence, and the assumptions behind the numbers.

The current release is `player_outlook_v1_20260922_r5`, based on accepted repaired
history `historical_v2_20260922_r5`. Observations end at 2026 Week 2; estimates
cover Weeks 3–6. There are 683 candidates with a resolvable team schedule, of
which 422 have observed offensive activity and a forecast. Other players remain
unscored, not zero-valued. Players with no resolvable team schedule are outside
this release's candidate coverage.

**No player outcome ranges or confidence scores are currently published.**
Point estimates and participation/role models remain exploratory, with baselines
and historical errors visible. This is not a promoted production forecast,
remaining-season trade value, medical injury model, or demonstrated market edge.

## The three repairs

1. **Expose historical uncertainty checks.** Stats & evidence now shows the
   accepted report's nominal versus observed season-point interval coverage,
   sample sizes, and season-by-season residual bands. Filters follow the evidence
   window and position. These remain population-level diagnostics, not individual
   player confidence scores or weekly floors.
2. **Describe participation honestly.** Route-opportunity fields are displayed
   as dropback on-field proxies, including report catalog/result labels and player
   details. They do not establish that a player ran a route: backs and tight ends
   can be blocking. Frozen field IDs, values, and report files were not rewritten.
3. **Remove duplicate availability discounts.** Static board VOR still determines
   roster power order, but is no longer resampled under an availability draw.
   “Scenario VOR” is replaced by **Availability downside**: the 25th percentile of
   conditional expected lineup points across availability scenarios. Its spread
   excludes ordinary scoring variance and is not a calibrated weekly floor.
   Deprecated `expected_lineup_vor`, `lineup_vor_risk`, and `risk_adjusted_total`
   API fields are null. ESPN fallback season totals are also converted to
   active-game means before applying availability, avoiding a second discount.

## Method and data boundaries

- Historical features use repaired league-exact weekly points, frozen passing
  attempts, offensive snaps, schedules, and audited preseason candidate metadata.
  xFP, synthetic depth, current injury status, and undated context are not inputs.
- Current all-player observations come from the fingerprinted NFL snapshot
  captured by `research/rookie_watch.py`. The outlook runner itself is offline.
- Historical features cover 2013–2025, at decision Weeks 2, 4, 8, and 12, plus
  the current decision week if different. Each outcome covers the next four
  calendar weeks, including absences and byes. Supported decision weeks are 1–13.
- Eligibility, team, position, role, and usage never use observations after the
  decision cutoff. Full scoring outcomes stay keyed by player ID. Later-only
  appearances do not create candidates at an earlier decision.
- Missing snap identities/feed coverage are unknown. Verified absent snap rows
  can count as no observed offense, not as proof of injury. Feed completeness is
  checked at the scheduled-team/week level; provider row completeness and
  retrospective revisions remain limitations.
- Conflicting player/week keys fail closed. Six colliding defensive snap records
  found during the initial run are outside the skill-position study, not arbitrarily
  deduplicated. The final build records 217 unmapped historical raw snap records and
  222,880 mapped non-skill-position records excluded from modeling.
- Recent usage uses up to three calendar weeks, per scheduled team game.
  Opportunity change compares that window with the preceding two weeks. Role
  change requires at least two recent and two earlier share observations; scoring
  and snap variation require at least three observations. Thus Week 2 does not
  support a durable trend/stability claim.
- The point model is fixed-penalty ridge regression, separately by position and
  decision week. Imputation, missingness indicators, and scaling are training-only.
  Features add prior participation, scoring dispersion, bonus concentration,
  usage change, and snap stability to a simple recent-usage model.
- Participation predicts weeks with any offensive snaps; conditional role predicts
  snap share when on offense. Neither is an injury probability or job-security
  score. Participation estimates cannot exceed the cutoff team's scheduled games.
  The participation baseline carries forward the recent three-week fraction;
  the role baseline holds recent active snap share.
- Points already include absences. Participation is a separate diagnostic, never
  another multiplier applied to the point forecast.
- Outcome ranges use earlier-season forecast errors to learn residual scale,
  then two separate earlier seasons to calibrate the residual quantiles. Point
  and scale models never fit those calibration outcomes. For 2026, model training
  ends in 2023 and calibration uses 2024–2025; this clean separation sacrifices
  some training recency.
- Current saved ESPN health and ownership are display-only context joined by
  identifier. They do not silently refit or injury-adjust the saved forecast.

## Historical validation

There are 36,907 historical feature rows and 20,292 candidate decision rows in
walk-forward tests over 2019–2025. At the current Week 2 cutoff, 2,961 forecasts
have comparable outcomes. Errors below are mean absolute error for **four-week
totals**, not weekly errors; all three methods use identical comparison rows.

| Position | Forecasts | Outlook error | Recent-usage error | Recent-points pace error |
| --- | ---: | ---: | ---: | ---: |
| QB | 309 | 17.85 | 18.23 | 19.39 |
| RB | 770 | 12.99 | 12.96 | 14.53 |
| WR | 1,187 | 12.73 | 12.65 | 14.70 |
| TE | 695 | 8.43 | 8.49 | 9.90 |

The expanded model improves over extrapolating points, but that is not enough:
RB/WR do not improve reliably over the simple usage model. Pooled QB/TE gains
are small. The participation model loses to the participation-pace baseline at
all four positions at Week 2. Conditional-role gains are also small/mixed.
The frontend presents the participation baseline first and explicitly reports
when the model has not improved on it.

Range publication requires complete snap coverage/identity and known prior sample,
plus all applicable position-wide, rookie, and prior-sample subgroup checks:

- At least 200 forecasts across at least five held-out seasons.
- Observed coverage between 75% and 85% for a nominal 80% interval.
- Interval score no worse than the usage-model residual range (width is penalized).
- Point MAE no worse than continuing recent scoring pace.
- A positive lower endpoint of a season-block-bootstrap 95% interval for MAE
  improvement over the usage model (5,000 resamples, equal-weight season means).

Checking the actual proposed publication subgroup matters. Experienced QBs
(≥10 prior offensive weeks) have only 185 observations and **72.4%** range
coverage, despite acceptable position-wide coverage. Experienced TE ranges
cover **80.4%**, but their incremental point-error improvement is not reliable
under the season-block check. Rookie groups are small and do not pass.
Consequently, **all current ranges are withheld**. Raw research ranges remain
in diagnostic parquet files; they are omitted from the public player payload.
Confidence scores are unconditionally withheld in this first outlook version.

These are exploratory, retrospective comparisons, not a preregistered independent
discovery test. No acquisition-price benefit has been measured. A prospective
season and stronger subgroup evidence are needed before promoting the system.

## Rebuild, serving, and preservation

Refresh the captured NFL snapshot using the existing `rookie-watch` workflow,
then build a **new** outlook version:

```sh
just player-outlook data/research/historical_v2_20260922_r5 \
  data/outputs/rookie_watch_2026.json player_outlook_v1_YYYYMMDD_r1
```

The create-only runner verifies the accepted history, snapshot hashes, and scoring
configuration; archives its input snapshot and implementation; saves features,
validation predictions, diagnostics, public report, and manifest; then atomically
updates only `data/outputs/player_outlook_2026.json` to point to the new release.
It does not download data or overwrite an existing version.

`GET /api/research/outlook` verifies the pointer, manifest, input/output hashes,
and the continued acceptance of the repaired history. Corruption returns an error,
not a silent fallback to old/unverified predictions. Private-league unavailability
does not hide public NFL research; unmatched ownership stays unknown.

All 13 protected production/reference artifacts remained byte-identical, including
the original V1 draft reference and frozen 2026 experiment. The accepted historical
build was not changed. Earlier outlook development releases are retained but are
not the current pointer target.

Verification: 508 offline tests passed, 35 network tests excluded; targeted Ruff
and mypy passed; frontend production build passed. Browser regressions cover
null legacy risk fields, the replacement metric, proxy labels, calibration,
outlook filters/details, withheld ranges, and mobile layout. The existing Node
version warning and nine unrelated frontend lint warnings remain.
