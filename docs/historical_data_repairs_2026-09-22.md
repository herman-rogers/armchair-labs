# Historical data repairs and isolated rebuild

## Scope and preservation

The implementation repairs the four defects identified in the data-integrity review,
plus the historical-position and outcome-loss bugs exposed by rebuilding the expanded
candidate universe. It does **not** promote a challenger or replace the current draft.

The rebuild is create-only and offline. Each version contains copied raw caches,
static inputs, implementation/configuration snapshots, source hashes, intermediate
inputs, a transaction review queue, refitted forecasts, and a manifest. Failed or
superseded development runs are retained separately and cannot pass the research
entry gate. The manifest checks the original boards, archived V1/configuration, and
frozen 2026 forecast for byte-for-byte preservation.

## Repairs

| Area | Changed behavior |
| --- | --- |
| Transaction direction | The announcing club is `source_team`, not automatically `to_team`. Outgoing trades resolve explicit destinations; acquiring-club announcements complement destination-less outgoing announcements. |
| Transaction interpretation | Full-name identity checks remain in place. Player-specific clauses protect initials, Jr./Sr., and numbered draft picks; actions concerning another player no longer set the matched player's status. Practice-squad exits are distinguished from practice-squad signings. |
| Same-day state | Duplicate announcements are reconciled. Conflicting teams, statuses, or reserve classifications become unknown, with retained evidence; input order never supplies an invented event chronology. |
| Missing roster evidence | A prior-season team is labeled inferred; absence of a transaction is not evidence of active status. Unknown status, team changes, and incomplete vacated-opportunity totals remain null. Known departures and context coverage remain separate diagnostics. |
| Contracts | Undated same-year contracts are excluded. Canonical money requires signing and source-availability dates before the cutoff. Remaining years also requires an explicit effective start year. Undated prior-year amounts/assumed terms are diagnostic proxies, not fitted features. Current `is_active` is ignored. |
| Route participation | Numerator and denominator use matched, deduplicated dropbacks, including scrambles. Incomplete participation coverage yields null, not zero; no clipping hides impossible ratios. Nonfinite numeric signals become null. Recent-window descriptors average observed values rather than manufacturing zero observations. |
| Positions | FB/HB map to RB consistently in research. Historical weekly roster positions supersede retrospective stat labels; the existing league override for Travis Hunter is preserved. Dated rookie-year market positions can override today's rookie metadata. |
| Outcomes | Candidate eligibility and earned points are separate. All weekly points are scored by player ID, including weeks after a switch to LB/DB/LS. Outcome VOR is evaluated at the candidate's forecast position, not a later NFL position. |

Regression examples include Deebo Samuel, Brian Robinson, Skyy Moore, Aaron Rodgers,
Malik Willis, Ameer Abdullah, D'Andre Swift, and Joe Mixon. Historical-position checks
cover Jordan Thomas and Allen Bradford; the outcome change also prevents erasing
points for players such as Nick Bellore and Bo Melton.

## Additional historical safeguards

- ESPN's historical transaction dates have known defects. The canonical rebuild uses
  official club dates; ESPN remains available for diagnostic replay, not verified
  cutoff state. Official coverage does not imply complete transaction coverage.
- Legacy Week 1 depth charts have no actual preseason publication timestamp. Their
  synthetic August 31 dates are excluded; only timestamped 2025+ charts are used.
- Historical expected-opportunity model vintages are unavailable, and the documented
  2006–2020 training window overlaps early evaluation seasons. xFP and its derived
  weekly features are quarantined from the refit pending adequate vintage evidence.
- Changed derived-cache versions prevent reuse of old route/descriptor inputs.
- Research requires a completed version, a passing integrity audit, matching forecast
  and weekly-outcome hashes, and unchanged copied source files.

These are conservative historical reconstructions, **not** original publication-time
snapshots of every feed. An unknown record is contained, not magically recovered.
Current player metadata, revised historical statistics, incomplete official archives,
and historical fantasy-position eligibility still warrant coverage-aware research.
Other provider-derived quantities (for example EPA/CPOE, pass-over-expected, and NGS
expected-yardage metrics) also lack original model-vintage snapshots here. A passing
integrity audit is not a certification that every historical feature could have been
deployed with the exact same provider model at that time. Their challengers remain
exploratory; the explicit xFP quarantine addresses a documented training-window issue.

## Reproduce

Choose a new version name; existing version directories are never overwritten:

```sh
.venv/bin/python research/rebuild_history.py --version YOUR_NEW_VERSION
.venv/bin/python research/data_integrity_audit.py --data-dir data/research/YOUR_NEW_VERSION --require-clean
.venv/bin/python research/role_transition_pilot.py --data-dir data/research/YOUR_NEW_VERSION
.venv/bin/python research/value_capture_audit.py --data-dir data/research/YOUR_NEW_VERSION
```

The rebuild audits the historical inputs before fitting. The second command verifies
the exported forecasts after fitting. The final two commands refuse unaudited or
changed inputs. None writes to `data/outputs` or the original frozen artifacts.

## Validation and research results

The completed, research-accepted run is `data/research/historical_v2_20260922_r5`:

- 18,289 forecast rows: 17,450 completed historical rows and 839 pending 2026 rows.
- 279,799 weekly-panel rows; 214 copied raw cache files and 227 raw/static source hashes.
- All 12 integrity gates pass, both before fitting and after export. Scoring mismatches, outcome mismatches, duplicate
  forecast/weekly keys, route ratios outside `[0, 1]`, nonfinite checked signals,
  transaction ordering differences, and future transaction violations are all zero.
- All **47** previously identified false-zero outcomes are repaired. For example,
  Andrew Beck's 2019 target is 24.54 points and Connor Heyward's 2022 target is 35.8.
- The SF 2016 Week 1 example now has 40 matched dropbacks in its denominator, including
  scrambles: the fully participating receiver is 40/40, not 40/35.
- No unverified contract money, quarantined xFP model inputs, or pre-2025 depth proxies
  enter the rebuilt fits. Valid signed air-yard shares are **not** clipped to `[0, 1]`.
- The final combined application/research test run passed 463 tests, with 35 network
  tests deselected. Targeted mypy, Ruff, and diff-whitespace checks pass.
- All 13 protected artifacts remain byte-identical, including V1 and the frozen 2026
  forecast. The 227 copied source hashes and nine recorded build-output hashes match.

Coverage is intentionally visible. Among all forecast rows, 1,656 returner cutoff
states are observed, 11,255 retain only an inferred prior team, 370 have an unresolved
action, and 13 have conflicting same-day evidence. The other 4,995 rows have no
returner transaction feature row. There are still 101 unmatched skill-position snap
records out of 87,479 raw records; these are not repaired through speculative name
matching. This is not a claim of complete historical roster coverage.

The [availability follow-up](historical_availability_2026-09-23.md) addresses public
absence facts missing from these inputs, beginning with Hunt's 2019 suspension.
It adds a dated evidence ledger, deterministic games ceilings, and a separate
coverage/review report. The accepted `r5` artifacts described here remain unchanged.

The candidate universe and fitting contracts also incorporate the earlier review
fixes: completed rows include 1,245 entries absent from the legacy retained table,
while 49 legacy entries do not survive the revised historical eligibility policy.
Performance changes must not be attributed solely to one data repair.

### Completed rebuild and acceptance

The walk-forward refit and both research reruns completed. The
[acceptance record](../data/research/historical_v2_20260922_r5/acceptance.json) binds
the final integrity audit, build manifest, and research outputs by hash. The build
manifest retains its original `rebuilt_pending_audit` completion state; the separate
final audit and acceptance record establish acceptance without rewriting research
provenance. This version is accepted **for exploratory research, not promotion**.
No 2026 outcomes were graded and no live model or draft board was replaced.

The report's stale legacy source descriptions were corrected after fitting. The
manifest records that wording-only correction and before/after report hashes;
numerical predictions, coefficients, and scores were not changed by it.

### Price-aware RB role transitions

The repaired pilot evaluates 2019–2025 using prior-season-only model training,
three late-preseason-price RB selections per decision window, and the following
four calendar weeks of outcomes. The main cohort contains 77 windows/231 selections;
the small-prior-sample cohort contains 69 windows/207 selections. Small sample means
1–9 prior raw appearances at any position, not rookies.

| Comparison | Next-four-week points per selection: role-model difference | Season-bootstrap 95% interval |
| --- | ---: | ---: |
| All late-price RBs: versus trailing points | +3.93 | +1.69 to +6.34 |
| All late-price RBs: versus recent snaps | +1.12 | -0.69 to +2.75 |
| All late-price RBs: versus usage-only ridge | +0.65 | -1.33 to +2.53 |
| Small prior sample: versus trailing points | -0.45 | -1.26 to +0.43 |
| Small prior sample: versus usage-only ridge | -0.20 | -0.42 to -0.02 |

Role features improve on a points-only baseline, but incremental value over stronger
usage baselines is unproven. The small-sample result does not support promoting this
challenger. These are exploratory intervals from seven seasons, not independent
confirmatory trials or multiple-comparison-adjusted discoveries. Preseason ADP is a
price proxy, not a same-week executable waiver or FAAB price.

### Market-relative value capture

All 288 sensitivity comparisons were rerun. In the 2019–2025 returner/top-60/ECR
comparison, the standalone season-points, adaptive, and next-generation rankings
still lose to consensus on the study's value target. Their average net differences
per season are -12.64, -11.81, and -11.37 respectively, with all three bootstrap
intervals below zero. These units are differences in summed positive VOR weighted
by games/17, not raw fantasy points, draft win probability, or monetary returns.

An 80% consensus/20% next-generation blend produces +1.56, but its interval spans
-0.55 to +3.12. This does not establish a reliable market-beating edge, particularly
after inspecting many variants. Historical ECR and final-preseason ADP also have
different observation timing. No challenger is promoted on these results.

### Research priorities from the repaired evidence

1. Resolve the documented transaction review queue and missing snap identities using
   dated, player-specific evidence; retain unknowns until verified.
2. Make simple recent usage and usage-only models mandatory role-model benchmarks.
   For small samples, test conservative shrinkage before adding more context.
3. Obtain synchronized, executable price/availability snapshots and reserve untouched
   future evaluation periods. Market blends and contextual features should earn their
   place through incremental, price-aware out-of-sample results.

Machine-readable results:
[integrity audit](../data/research/historical_v2_20260922_r5/outputs/data_integrity_audit_2026-09-22.json),
[RB-role pilot](../data/research/historical_v2_20260922_r5/outputs/role_transition_pilot_2026-09-22.json),
[value-capture audit](../data/research/historical_v2_20260922_r5/outputs/value_research_2026-09-22.json).
