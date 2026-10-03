# Intelligence workspace: repaired data and decision use

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

## What is connected

Intelligence defaults to the latest accepted historical research version, currently
`historical_v2_20260922_r5`. A dataset selector retains legacy research for comparison.
Player rankings, current-observation comparisons, roster scenarios, and Stats & evidence
all carry the same dataset identifier. The evidence view serves that version's report,
not whichever production report was most recently written.

Accepted versions require matching acceptance, manifest, integrity audit, predictions,
outcomes, source files, and recorded output hashes. Changed or incomplete versions fail
closed. Failed development builds are not offered as accepted datasets. Model definitions
come from the version's manifest, so newer configuration changes cannot relabel old output.

This is a read-only connection: no board, forecast, or frozen experiment is promoted or
rewritten. The frozen Adaptive experiment remains available under legacy research and is
not presented as a repaired forecast. Original V1 is still available as the draft reference.

## Which numbers to use

| Purpose | Starting point | Important limitation |
| --- | --- | --- |
| Returning-player preseason baseline | Core season forecast | Active-game PPG × expected games; not remaining-season points |
| Scoring strength when active | Core active-game PPG | Availability is separate; raw PPG is not cross-position trade value |
| Rookie and returning-player preseason comparisons | Next-gen season forecast | Combines separate population models; still experimental |
| Check the market's influence | Market-informed season forecast | Uses consensus information; not independent market-beating evidence |
| Compare automatic model selection | Adaptive selector | Selects components using earlier seasons; output is season points despite its internal PPG name |
| In-season rookie opportunity | Rookie watch | Separate saved four-week analog estimate, not a full rest-of-season model |

The short model list is for navigation, not a claim that every listed model has earned
promotion. All research outputs remain accessible under the advanced model list. Games,
return probability, two-stage/direct variants, and career/athletic/security/full-stack
outputs are mainly diagnostic components or experimental alternatives.

## Using the views during the season

- **Player rankings:** filter by position and current ownership to build a shortlist.
  Selecting Rookies while on a core returner model switches to the combined Next-gen
  model. Player details show the historical cutoff, roster-evidence quality, and prior
  production sample alongside separately labeled current saved health.
- **Market comparisons:** identifies disagreements with dated consensus or current ESPN
  draft-room order. Neither is a current executable trade price or waiver cost. Historical
  seasons compared with current ESPN order remain explicitly labeled as such.
- **League impact:** a temporary roster scenario using saved forecasts and declared
  production fallbacks, not a validated rest-of-season simulation.
- **Rookie watch:** uses its own saved in-season artifact and displays the historical
  source version, current observation cutoff, four-week horizon, freshness, sample size,
  comparables, and pace baseline. Coverage filters separate scored and unknown players.
  Current injuries/matchups are not incorporated in the analog estimate. Its historical
  spread is not a calibrated prediction interval.
- **Stats & evidence:** evaluates the selected version's historical forecasts. A passing
  data audit is not proof of profitable trades, correct current roles, or model superiority.

The rebuilt historical rookie model has scored forecasts from 2007 onward. The live
browser check confirmed 188 scored rookie forecasts for 2025 and 157 pending for 2026.
The current 2026 preseason cutoff is August 28; rebuilding on September 22 did **not**
turn these into September 22 rest-of-season projections. Rookie Watch separately uses
observations through Week 2 for Weeks 3–6 at the time of this pass.

## Remaining decision-model gaps

We do not yet have a validated, all-player rest-of-season trade-value model, synchronized
trade/FAAB prices, or a claim that experimental rankings beat consensus. Treat current
usage, health, byes, remaining opportunity, roster needs, and acquisition cost as required
checks before acting. Saved reports do not automatically refit when new games arrive.

## Verification

- 485 application/research tests passed; 35 network tests excluded.
- Frontend TypeScript/production build passed. Existing Node-version and unrelated
  frontend lint warnings remain; lint exited successfully.
- Browser fixtures cover dataset switching, advanced models, rookie-model selection,
  coverage filters, ownership, scenarios, evidence, and mobile widths.
- A real-data browser check loaded the repaired rookie rankings, matching 18,289-row
  report, and the current rookie feed without browser exceptions.
- Targeted Ruff, mypy, and diff-whitespace checks pass.
