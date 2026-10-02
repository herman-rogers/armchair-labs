# League overview — September 24, 2026

League now opens to **League overview**, replacing the separate Standings and Matchups
tabs. It combines:

- Your record, latest captured completed result, roster forecast rank and forecast coverage.
- League-wide player alert counts, your captured FAAB, and recent transaction history.
- Standings alongside completed-game points for/against, NextGen roster totals/ranks,
  coverage, urgent flags and FAAB. Expand a team for position totals, source mix,
  availability-constraint count and a link to the individual roster forecasts.
- A compact scoreboard for the selected week. Your matchup comes first; expand any
  matchup to inspect actual starters, bench, ESPN projections, statuses and notes.

All NextGen displays use the published **rest-of-season** rankings API and its serving
policy. Team forecast = sum of the captured roster's covered QB/RB/WR/TE forecasts,
including bench/reserve. Team rank requires every captured skill player to have a
forecast; missing is never treated as zero, and tied totals share rank. A constrained
zero forecast is valid coverage. K/DST are excluded. Counts of reference forecasts
and availability constraints are explicit. Known partial totals remain inspectable.
These are descriptive roster totals, not validated team win forecasts, weekly lineup
projections or optimized lineups. Different roster sizes affect the totals.

Free agents, rosters and draft recap display NextGen **overall** rank, position rank,
remaining points, forecast basis and availability constraints. Ownership/search never
renumbers the published player ranks. Draft recap preserves actual selection order;
its NextGen columns are current forecasts, not the ranks available at draft time.
Draft identities use the verified ESPN→NFL crosswalk, including players no longer in
the current captured roster/free-agent pool. Unmatched identities stay unranked.

Scoring totals include only captured completed schedule entries before the snapshot's
current week. Actual zeros remain zero; missing opposing results remain unknown.
Future schedule placeholder zeros are excluded. ESPN capture time and forecast
publication/production cutoffs are separate. Refresh league refreshes ESPN observations,
not the published model. Forecast season mismatches and unavailable forecasts suppress
ranks without hiding the observation-based league view.

Validation: `node web/tests/league_summary_test.mjs` checks coverage, tie ranks, valid
zeros, season mismatches, global player ranks, score coverage and activity chronology.
`web/tests/league_overview_smoke.py` uses saved observations and published forecasts
(no ESPN refresh) for desktop/mobile overview, roster drilldown, free-agent/draft ranks
and future-week behavior. Backend regression tests cover dropped-draft-player identity
links and the existing observation-only boundary.
