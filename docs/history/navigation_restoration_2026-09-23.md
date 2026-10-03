# Everyday analysis and league navigation

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

The primary navigation separates Intelligence, League, and Research. Players opens
shared player profiles directly. NextGen rankings now has its own Intelligence tab;
there is no separate Career profiles tab, and all-careers browsing is hidden.
Historical data is retained
for profiles and statistical comparisons.

| Workflow | Current location |
| --- | --- |
| Current remaining-season / next-four-week point rankings and evidence | Intelligence → NextGen rankings |
| Player search, production, opportunity, consistency, dated ECR, profile drilldown | Intelligence → Players |
| Statistical career comparisons, including historical players | Player Details → Similar careers |
| Rookie draft capital, current opportunity and linked college history | Intelligence → Rookies |
| Weekly actual scores, starters, bench and captured ESPN projections | League → League overview; choose week |
| Standings, roster browsing/comparison, captured free agents | League → League overview / Rosters / Free agents |
| Acquisition history and actual draft selections | League → Transactions / Draft recap |
| Evaluations, baseline forecast details/export, archived model comparisons | Research |

Previous-season baseline rank, position rank and projected points were removed from
player/rookie lists at the user's request. They are not a Next Gen Stats model.
ECR preserves FantasyPros overall/positional consensus values and their publication
dates from corrected gold. The subsequent
[ranking release](nextgen_rankings_2026-09-23.md) adds current-season point rankings
with explicit reference labels after the advanced challengers failed promotion.
It does not restore the old preseason baseline columns or archived model scores.

Search and population/position filters run over the complete source result. The UI
loads every matching API page under the pinned catalog, then sorts the complete set
before taking a display page. Player measurements, forecasts, rookies, career directory,
research inventory and individual evidence share this loading behavior. The historical
profile-directory component retains the same fix if reused, but is not exposed in Players. Sorting resets
the display page; changing filters also resets it. Nulls remain last in either direction.
League observations are already loaded in full before roster filtering/sorting.

League matchups/refresh read ESPN snapshots without loading retired model boards.
Historical lineups are the selected week's actual box score; future unrecorded values
remain unknown. Captured-at times and stale state are explicit.

A persistent Needs attention panel lists your team's injury tags, dated reviewed reports,
current-week byes, missing profile links, unfilled starting slots and exhausted FAAB.
Reviewed news is verified against the latest accepted same-season raw injury capture.
Refresh league updates ESPN, not the separately captured news. ACTIVE is not medical
clearance. Current-snapshot alerts are never applied to historical/future lineups.

Validation includes backend rank/timing/observation/attention tests and browser checks:
`web/tests/nextgen_smoke.py`, `web/tests/pagination_smoke.py`, and
`web/tests/league_attention_smoke.py`. Pagination fixtures place the leading results
beyond both the display page and the API page boundary and search from later pages.

## Similar careers

Profiles offer five nearest same-position career-stage neighbors from the verified profile
release. Compare the first N career seasons, separately by career year, using per-observed-
week workload/yardage (QB: attempts/pass yards/carries/rush yards; RB: carries/rush yards/
targets/receiving yards; WR/TE: targets/receptions/receiving yards). Every included season
requires eight observed weeks and complete metric coverage. Truncated early careers are
excluded; missing values are never filled with zero. Partial seasons and seasons after the
selected profile cutoff are excluded for both target and comparison players. Later seasons
of a historical peer do not enter an earlier-career match.

Distance is RMS standardized difference over metric/year pairs, with cohort standard
deviation giving each dimension comparable scale. Zero-variance dimensions contribute
zero. At least five peers must be eligible. The UI shows source years, sample size,
comparison-pool size, method and raw season rates. This is descriptive statistical
resemblance, not a calibrated probability, future projection or Next Gen ranking; it does
not adjust for era, age, scheme, teammates or injuries. Unit tests cover self-exclusion,
position/coverage restrictions, same-stage comparisons, and temporal boundaries.
