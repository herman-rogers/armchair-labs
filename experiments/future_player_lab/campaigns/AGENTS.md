# Opening Drive releases

**Development posture (user, 2026-10-27):** this is a single-user fantasy-football dev
system. Keep earlier versions' published bundles and sealed runs unchanged, but do not
add production-style gates to new work: no promotion blockers, minimum-week evidence
requirements or manual-approval steps. Horizon release checks only warn, for every
campaign. New research runs may be rerun or overwritten. The stricter rules below
describe how v3.2/v4 were built.

**No hard-coded model lists (user, 2026-10-27).** The campaign directory
(`data/campaigns/`) is the registry: any folder with a sealed manifest is served by
the API and offered in the app. The only named campaign in application code is
`engine.chairstats.application.DEFAULT_CAMPAIGN` (currently `opening_drive_v41`;
the newest campaign is used if it is missing). Do not branch on campaign ids in
API or frontend code. Put per-campaign behavior in the manifest, the calibration
artifact or the campaign config instead.

The product goal, confirmed by the user on 2026-10-14, is that each new Opening
Drive release updates weekly forecasts **and** the remaining-season/full-season
totals, per-game averages and rankings that users see. A weekly-only upgrade is
an experiment, not a complete application release.

## Version lineage and functionality

The user explicitly clarified on 2026-10-14: **v3.2 must be v3 improved with the
injury/workload changes, not v2 with an adjustment on top. Versions must move the
agreed model forward without silently losing functionality or reverting to an
older version.** V3.1 remains paused; v3 is the agreed base for v3.2.

- Record the declared base and model lineage for every served output. Older
  versions may remain labeled comparison controls; they must not silently replace
  the base's newer forecasting approach.
- **Retrain from methods, never from frozen models** (user, 2026-10-27; first
  applied in v4.1). A new version regenerates its panel, every model stage and all
  predictions in one full run. Earlier versions contribute methods (code recipes,
  hyperparameters, selection protocols, feature ideas) and any data source, but a
  previous version's trained models or saved predictions must not be inputs, features
  or frozen components of the new version. They may only be evaluation controls.
- Preserve point prediction, weekly forecasts, variance/distributions, intervals,
  component-stat outputs, horizons, averages, rankings, lineup behavior and API/
  refresh/replay compatibility. Archiving removed outputs is not equivalent to
  retaining live functionality.
- Evaluate changes against the declared base on matched data and cutoffs. Report
  point, ranking and uncertainty results separately, plus a functionality matrix.
  Do not assume better calibration improves point means or that a new version
  number guarantees an improvement. Disclose regressions and tradeoffs.
- Do not silently change the base, roll back methodology or remove capabilities.
  Such a change requires an explicit explanation and user agreement, unless it
  has already been explicitly authorized in the conversation.
- The v2-anchored `horizon_projection` bridge is a known interim implementation
  that fails the intended v3-based extension. Its points-only current horizon
  outputs also leave a functionality gap. Do not treat it as the approved pattern
  for subsequent versions. The arithmetic release gate does not enforce lineage
  or prove feature parity or accuracy gains.

## Horizon coherence and preservation

- Build future-week estimates from information available at one forecast origin.
  Never sum later rolling forecasts that incorporate outcomes unknown at that origin.
- Remaining points must equal the sum of the future-week curve. Projected season
  points must equal accepted actual points through the cutoff plus remaining points.
  Averages use the declared scheduled-game denominator; byes contribute no game.
- The current week in that curve must equal the served weekly prediction. Refresh
  all horizons together, including pregame updates. Preserve preseason forecasts
  at their original cutoff and distinguish them from current season projections.
- Evaluate how weekly information transfers to longer horizons using strictly
  earlier selection data. Do not assume weekly accuracy gains imply season gains,
  multiply a one-week injury absence over the entire season, or add weekly interval
  widths as independent uncertainty.
- Releases that declare a `horizon_contract` are checked by
  `engine.chairstats.horizons.validate_opening_drive_release` during publication and
  replacement; problems are reported as warnings and must be fixed, but never block.
- Keep frozen research inputs, forecasts and fitted models immutable. Publish
  revisions through the archived replacement flow. Changes to numerical inference
  require a new sealed run; never rewrite old receipts to make a replay pass.

See [the shared horizon projection workflow](horizon_projection/README.md) and
[campaign operations](../../../docs/operations/model-campaigns.md).
