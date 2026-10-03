# Player profiles across college and NFL seasons

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

For the current corrected-gold profiles, NFL Next Gen Stats, and approved forecast
integration, see [the profile migration](profile_gold_migration_2026-09-23.md).
The remainder of this document records the original v1 release and archived UI.

The **Intelligence → Players → Career history** view follows a player across captured
seasons. The same profile opens from college careers, player details, research
rankings, and the current outlook. It combines observations and saved
forecasts without collapsing different horizons into another composite score.

Intelligence now has three tabs:

| Tab | Views |
| --- | --- |
| Players (default) | Career history, four-week outlook, and preseason rankings, with shared search, position, and population filters |
| League impact | Team strength and roster scenarios |
| Research | Model performance, market comparisons, college backtests, and college identity audits |

Select **Rookies** in the player population filter for rookie lists. Historical
rookie analogs and their evidence now appear under **Forecasts** in the shared
rookie profile, at a matching season/week cutoff. Select **College → NFL** for
college careers or NFL translation forecasts. Player selection opens the shared
profile directly; view-specific forecast diagnostics remain available underneath.

## Published coverage

Release: `player_profiles_v1_20260923_r2`, selected by
`data/outputs/player_profiles.json`.

| Evidence | Captured coverage |
| --- | --- |
| NFL profiles | 4,931 historical/current research candidates |
| Current candidates | 683 |
| Accepted college links within the NFL profile population | 3,537 |
| NFL player-week observations | 149,590; 2001 through 2026 Week 2 |
| Historical injury/practice reports | 27,582; 2009–2025 |
| Features computed before each forecast season | 18,850 player/forecast-season rows |
| College source history | Begins in 2004; individual coverage varies |

This is the union of saved NFL forecast candidates and captured current outlook
candidates, not every person who has appeared on an NFL roster. College-only and
unresolved identities can be opened through **Players → College → NFL → College careers**. Only accepted
identity links join their college and NFL records.

The release pins `historical_v2_20260922_r5`,
`college_nfl_v1_20260922_r5`, and `player_outlook_v1_20260922_r5`. The current
observations were saved on September 22, 2026 at 23:35 UTC. The builder uses these
verified local inputs without fetching new data.

## What the profile contains

- **Overview:** completed-season history, career scoring rate, recent three-season
  baseline, an interactive trajectory, observed team/role changes, and current
  opportunity when viewing the latest cutoff.
- **Season history:** every captured NFL season, teams and positions, participation,
  scoring, workload, and efficiency. College school stints, transfers, production,
  shares, and coverage stay in their own table.
- **Role & participation:** production at high, rotational, limited, and unknown
  participation levels; consecutive role periods; recorded injury/practice reports.
- **Forecasts:** existing preseason forecasts, college development forecasts,
  college-to-NFL rookie and first-three-year estimates, and the current four-week
  outlook. Each retains its own horizon and units.
- **Sources & gaps:** accepted identity evidence, historical roster evidence,
  definitions, missing coverage, and pinned source versions.

For example, Lamar Jackson has three captured college seasons, eight completed NFL
seasons, and a partial 2026 season. Across completed NFL seasons, the captured record
contains 103 high-participation weeks averaging 23.14 league points and 10
limited-participation weeks averaging 2.50. His overall completed-history rate is
20.94. Keeping the participation samples visible helps distinguish a change in
opportunity from a change in performance. These descriptive differences do not
estimate the causal effect of a starting role or isolate talent.

## Canonical measures and refactoring

`src/engine/metrics/player_profile.py` owns the observed-history formulas. Career,
season, role-band, and consecutive-period summaries all call the same aggregator.
The frontend reads its metric catalog rather than recomputing rates. The profile
replaces the separate college career renderer, and college release verification is
shared through `college_sources.py`. Existing saved model outputs retain their
original definitions and artifacts; their different denominators are not silently
renamed to match the profile.

| Measure | Definition |
| --- | --- |
| Observed week | A recorded scoring row or observed positive offensive snaps; not a start or a medical availability label |
| League points | Accepted audited scoring, including this league's touchdown bonuses; enrichment never scores the player again |
| Points / observed week | Recorded league points divided by observed weeks |
| Targets or carries / observed week | Published only when that stat covers every observed week |
| Offensive snap share | Mean of recorded shares; missing participation is unknown |
| Efficiency | Ratio of totals on weeks where both numerator and denominator exist |
| Participation bands | High ≥70%, rotational ≥35% and <70%, limited <35%; unknown kept separate |
| Recent three-season baseline | Completed seasons only, 0.55 annual decay, weighted by observed exposure |

Positive-snap weeks without a scoring row contribute zero **recorded** points;
their missing stat volumes remain unknown. They do not invent a target, carry, or
start. Missing weeks and missing college seasons remain missing. Role periods split
at team changes, season changes, participation-band changes, and calendar gaps.
Annual role-change annotations require at least four snap observations in both
adjacent seasons and a change of at least 15 percentage points.

The exported feature table provides completed-career exposure and scoring,
recorded workload totals, the recent baseline, and participation-specific samples
and scoring. Its keys are `player_id` and `forecast_season`; use both when joining
historical folds. This release supplies a consistent feature foundation. It does
not fit or promote a new prediction model, and it does not establish an improvement
over ECR. That requires a separate incremental-value test on held-out seasons.

## Cutoffs and limitations

The API filters weeks before calculating any history, role period, or feature.
Completed historical years can be selected in the UI. A partial selected season
remains separate from completed-season priors. College production is limited to
seasons before the selected NFL year. Week-zero requests omit that year's saved
forecasts because a season/week cutoff alone cannot establish when an August
forecast became available.

NFL translation outcomes remain hidden until their entire target window has
completed; three-year outcomes cannot appear in a rookie-year view. The current
outlook is available only at its exact captured cutoff. Expected points already
include absences and are not discounted a second time.

Historical source revisions and current captured identity details still exist in
these records. The selected cutoff is a reconstruction of observations, not a
claim that every field has its original publication-time vintage. Injuries are
reported statuses, not diagnoses or a complete absence ledger. Participation is
not routes run, depth-chart rank, or proof of a skill change. Earlier NFL careers
can be truncated by the 2001 source boundary; college quality flags remain visible.

## Build and API

```bash
just player-profiles player_profiles_v1_YYYYMMDD_rN
# Equivalent:
uv run python research/player_profiles.py --version player_profiles_v1_YYYYMMDD_rN
```

Each build requires a new directory. It verifies the accepted NFL history and the
college/outlook dependency chain, snapshots the implementation, hashes inputs and
outputs, checks protected artifacts, and atomically updates the profile pointer.
Changed dependencies produce an explicit API error rather than a silent fallback.

```text
GET /api/profiles?scope=current&search=Lamar&position=QB
GET /api/profiles?scope=all&limit=50&offset=50
GET /api/profiles/player?player_id=00-0034796
GET /api/profiles/player?player_id=00-0034796&season=2018&week=18
GET /api/profiles/player?college_id=<accepted-or-unresolved-college-id>
```

The version directory contains `players.parquet`, `nfl_weeks.parquet`,
`profile_features.parquet`, `injuries.parquet`, `report.json`, and the manifest.
The older college career endpoint remains available as an archive of its original
college-release data; the application uses the shared profile endpoint.

## Validation

All 557 offline Python tests pass. Profile tests cover future-data isolation,
completed/partial cutoffs, outcome maturation, ambiguous identity separation,
exposure weighting, duplicate joins, missing evidence, audited scoring ownership,
and rejection of changed artifacts. Targeted Ruff and mypy checks pass; the
frontend production build and lint complete successfully with existing toolchain
and lint warnings.

All 14,558 captured historical player-seasons reconcile to accepted audited
scoring within floating-point tolerance (maximum difference below `2e-13`). All
13 protected production/frozen artifacts match their saved hashes. Browser checks
cover the shared profile, college integration, historical selection, and desktop
and mobile rendering; mobile checks measure the expanded profile itself as well
as page overflow.
