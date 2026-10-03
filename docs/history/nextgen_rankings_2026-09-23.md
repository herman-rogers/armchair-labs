# Published NextGen rankings — September 23, 2026

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

September 24: `revalidation_review_20260924_r1` now packages these same rankings
with the [completed registry review](model_revalidation_review_2026-09-24.md).
Ranking values, cutoffs and evaluation artifacts are unchanged.

`nextgen_rankings_20260923_r6` supplies current-season rankings to the normal
dashboard. It packages research run `nextgen_rankings_20260923_r4` against corrected
gold `canonical_nextgen_20260923_r1` and the existing verified profiles. The catalog
at `data/current.json` selects the release atomically.

## What is available

- **Intelligence → NextGen rankings:** 841 QB/RB/WR/TE candidates, rest of regular
  season (Weeks 3–18) and next four weeks (Weeks 3–6), production through Week 2.
- Positional ranks, raw overall points ranks, forecast points, scheduled games,
  prior/current observed samples, evidence status, recipe and dated preseason ECR.
- Search, position and league-ownership filters. Filtering retains full-pool ranks;
  equal predictions share rank. CSV exports the entire position/search population,
  with its release, cutoff, evidence status and constraints. The export button
  explicitly states that ownership filtering is not applied to the export.
- Remaining-season rankings in League rosters/free agents and current profile
  Forecasts tabs. Historical profiles never import the current ranking.
- Dated reviewed season-ending reports suspend a player from the ranks. Jaxson
  Dart has a zero constrained estimate and no rank; the unconstrained statistical
  estimate and source remain available for audit. Ordinary OUT/IR tags do not
  imply an invented return date or automatically zero the forecast.

Overall point ranks are not positional-scarcity-adjusted draft value. Neither
horizon is a FAAB valuation or opponent-specific weekly lineup recommendation.
Ownership and injury observations refresh separately from model production data.
The dashboard displays both dates and flags a completed-week ranking lag.

## Model and historical coverage

The [protocol](../../research/nextgen_rankings_protocol.md) defines the candidate
population, targets, fixed recipes, chronological fitting/selection, multiplicity
correction and publication checks. Historical observations begin in 2001; model
candidate seasons span 2004–2026. Each test fold trains on **all earlier completed
candidate seasons**, including players with no prior production. Forecast tests
begin in 2007; the nested selection-policy evaluation begins in 2011 after four
earlier out-of-year validation seasons. Modern 2019–2025 results are an evaluation
slice, not a restriction on training data. Current 2026 outcomes are null.

Candidates are known by the forecast cutoff. Later appearances cannot admit a
player, change their team or supply predictor values. Features cover prior,
three-year and career production, current observed workload, age, draft capital,
experience and snaps. The enriched challenger adds accepted, earlier college
production and prior NFL tracking summaries; missing coverage retains the player.
Known incomplete/invalid college aggregates are not silently used.

References include prior production, current production, a prior/current blend,
and earlier-season affine calibrations of current/blended output. Calibration and
recipe choice distinguish observed opportunity from limited opportunity using
cutoff-known attempts, carries and targets. This is a descriptive workload group,
not a certified starter label. Both groups remain in the population; advanced
profile challengers train on all earlier position rows.

## Result: publish references, keep advanced challengers in research

No advanced selection policy passed all final publication checks. The serving
recipes are calibrated prior/current blends for observed-workload players and
current-production references for limited-workload players. They are labelled
**Reference**, not “validated advanced model.”

| Position | Horizon | Modern reference MAE, points | Advanced policy improvement, points | Adjusted q | Published basis |
| --- | --- | ---: | ---: | ---: | --- |
| QB | Remaining season | 33.81 | 0.73 | 0.450 | Reference |
| QB | Next four weeks | 7.64 | 0.00 | 1.000 | Reference |
| RB | Remaining season | 25.51 | 1.39 | 0.125 | Reference |
| RB | Next four weeks | 6.98 | 0.16 | 0.450 | Reference |
| WR | Remaining season | 23.86 | 0.25 | 0.333 | Reference |
| WR | Next four weeks | 6.81 | 0.00 | 1.000 | Reference |
| TE | Remaining season | 17.68 | 0.56 | 0.333 | Reference |
| TE | Next four weeks | 4.94 | 0.00 | 1.000 | Reference |

These are equal-season errors across the full eligible position population, not
starter-only errors or individual-player uncertainty intervals. The policy can
select a reference in a fold; zero improvement does not mean every individual
challenger had identical predictions. Full-history error/ranking checks, modern
season-bootstrap uncertainty, squared error, top-K point capture and rookie/
small-prior-history guards also apply. RB remaining-season improvement was
promising, but its adjusted q did not clear the declared 0.05 threshold.

The richer profiles remain useful player evidence. This experiment does **not**
establish that adding college/tracking/profile predictors improves these two
ranking horizons reliably. Nor does it establish that new unobserved roles, injury
returns, or small-sample market misses are solved. In particular, a player without
observed workload can still receive an overly low reference after a new role change.

## Preserved iterations and limitations

Runs r1–r4 and their exact executed code, protocol, panels, predictions, fold
selections and results remain immutable. r1 used weaker raw references; r2 added
calibration; r3 selected by prior sample size; r4 changed that grouping to observed
workload and calibrated by workload group. The final design was informed by these
retrospective findings. Publication thresholds were not relaxed. No prospective
confirmation or untouched historical holdout is claimed, and historical source
publication vintages remain incomplete.

The prepared r5 delivery was never published because dependency lookup failed
before catalog replacement. It is retained as an archived artifact; r6 corrects
the packaging lookup and is the published delivery.

The delivery registry archives all four research runs and preserves every older
model/data disposition. The normal API and exports enforce exact target,
position, population, horizon, allowed use and source-incident checks. There is
no legacy ranking fallback. Manifests bind all artifacts and dependencies; later
catalog publications cannot silently remove rankings.

## Rebuild

For another experiment on the current data, use new immutable version names:

```sh
just nextgen-rankings NEW_RESEARCH
just nextgen-rankings-publish NEW_RESEARCH NEW_DELIVERY
```

For a new completed week, first capture/validate new source data and build matching
gold, profiles and base analysis as described in the
[system runbook](nextgen_system_implementation_2026-09-23.md). Then:

```sh
just nextgen-rankings NEW_RESEARCH --analysis NEW_BASE_ANALYSIS
just nextgen-rankings-publish NEW_RESEARCH NEW_DELIVERY
```

Each build replays historical forecasts at the new week cutoff. Refreshing ESPN
ownership alone does not retrain or advance the ranking production cutoff.

Verification covers feature/label separation, cutoff candidate inclusion, role
grouping, byes, rank ties/filter stability, date-scoped absence constraints,
shadow/incident exclusion, CSV parity and publication continuity. Browser checks
exercise both horizons, profiles, historical cutoffs, mobile layout, and league
integration. Run `web/tests/rankings_smoke.py` against local API/frontend servers;
the test supplies local ownership fixtures and does not refresh ESPN.

Release verification: 647 offline tests passed (35 network tests deselected),
the production frontend build passed, and both the ranking and general NextGen
desktop/mobile browser suites passed. Focused Python lint/format checks passed.
Frontend lint completed with existing warnings in shared/legacy components.
