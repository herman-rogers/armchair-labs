# College → NFL pathways

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

Implemented a separately versioned identity/data foundation and exploratory temporal
backtests. Open **Intelligence → College → NFL**. No V1, production, Next-gen,
Adaptive, current outlook, or frozen draft artifact was overwritten.

Published study: `data/research/college_nfl_v1_20260922_r5`.
Raw source snapshot: `data/research/college_source_20260922_r1`.
Accepted NFL history: `historical_v2_20260922_r5`.

## What is connected

- 66 raw Parquet partitions: college player game boxes, season rosters and the
  independent all-division schedule for 2004–2025.
- 374,224 college player/team/season records; 144,870 source athlete IDs. These
  are source identities, not a guarantee that all aliases are distinct people.
- 10,002 college identities linked to NFL GSIS identities, across all positions.
- 318 unresolved/ambiguous college matches held for review. Six conflicting NFL
  ESPN aliases are withheld from the external-ID registry.
- Separate typed namespaces for college ESPN, NFL ESPN, GSIS and college-reference
  IDs. The registry also retains NFL-only identities for future source joins.
- Team stints remain separate. Transfers sharing the same college ID retain a
  single career; different IDs are not silently merged on a fuzzy name.
- Current NFL outlook is joined by GSIS for display, with its own forecast cutoff
  and horizon. It is not mixed into the college/preseason training data.

Match hierarchy: provider ID **plus** name/chronology checks; otherwise exact
name + birth date; otherwise unique exact name + school + entry window. Known
birth-date conflicts block an automatic match. Multiple candidates or multiple
college IDs resolving to the same NFL player require review. Explicit overrides
live in `data/static/college_identity_overrides.json`; every override requires a
reason and HTTPS evidence URL. The initial override file is empty.

## Sources and historical limits

The [sportsdataverse college data archive](https://github.com/sportsdataverse/cfbfastR-cfb-data)
publishes ESPN-derived player boxes, rosters and schedules. Its
[data dictionary](https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/DATASETS.md)
describes their grain. This capture is pinned to Git commit
`c0d9161aa61cd6d09e6c1cffe95bbb8a4dcf9bcd`, verifying upstream Git blob hashes,
file sizes and our own SHA-256 fingerprints.

2004 is the earliest available player-stat/roster partition in this source.
Schedules extend to 2001; that does **not** establish player-stat coverage in
2001–2003. Earlier college careers are left-truncated and flagged. This is not a
claim that older stats cannot exist elsewhere.

These are revised historical records, not original publication-time snapshots.
Current roster age, height/weight, status, NFL draft labels and recruiting fields
are deliberately not borrowed from historical college rosters as model features.
Birth date is used to calculate age at the forecast cutoff.

## Data checks

- Unique athlete/team/game/category keys; identical duplicate rows are counted,
  conflicting duplicate rows stop the build.
- Missing/placeholder identities stop normalization. Nonfinite or missing required
  statistics quarantine a team-game instead of becoming invented zeros.
- Some older passing rows contain unnamed `stat_1`…`stat_5` cells. We retain their
  raw records but do not guess the column order.
- Player passing and receiving team totals must reconcile in yards and TDs.
  Unreconciled games may represent provider omissions, revisions, or unusual play
  accounting. Their reported numbers remain visible, but affected team-seasons do
  not qualify as complete model inputs.
- Team-game capture is checked against an independent schedule. FBS coverage is
  much stronger than FCS/lower divisions. Raw records are retained for every
  captured division, not just future NFL players.
- College market shares use all captured players' team totals, not NFL survivors.
- College zero-stat roster players remain in the data. Stat appearances are not
  labeled games played, starts or injury outcomes.
- All completed NFL rookie point targets reconcile to the repaired weekly scorer.
- Multi-team college seasons remain inspectable but are withheld from this first
  forecasting model rather than combining ambiguous denominators.

## Forecasts and evaluation

The NFL cohort starts with all 4,272 audited rookie candidates from 2004–2026.
2,066 have eligible final college seasons after identity and quality gates.
For 2026 specifically, 134/157 candidates qualify. Coverage by class and position
is published; these are not silently reduced to successful NFL players.

NFL translation compares four models on identical eligible rows:

1. Earlier entrants' position mean.
2. Draft-capital-only baseline.
3. College-only features, excluding actual NFL draft capital.
4. College features plus NFL draft capital.

Features include final-season production and team shares, prior observed career
production, year-to-year changes, age and explicit missing/truncated-history flags.
Missing NFL draft picks combine undrafted and unknown records, using the explicit
pick-300 sentinel and known-pick indicator inherited from the baseline design.
No college targets/routes or opponent-adjusted efficiency are fabricated.

Targets are rookie-year league points, rookie-year games and first-three-NFL-year
league points. Every fold learns imputation/scaling/model weights only from
earlier **completed** outcomes. Three-year targets have a three-year maturation
embargo. The fixed ridge penalty is 100; it was not selected on the reported
held-out seasons. NFL year-one folds begin in 2009; three-year folds begin in 2011.
2026 rookie-year outcomes and immature three-year windows remain ungraded.

Separate next-college-season yardage backtests compare a college model against
carrying forward last-season passing, rushing or receiving yards. This task is
**conditional on observed roster return**; departing players are not assigned
zero college production. It does not yet model the probability of staying in
college, transferring, declaring or reaching the NFL.

The report contains all-era and 2018+ MAE/RMSE, per-year errors, season-block
bootstrap intervals, and a full refit excluding name+school-only identity links.
No individual outcome intervals, confidence scores or automatic promotion are
published. These are exploratory retrospective results, not a preregistered
market-value test.

### Initial 2018+ NFL rookie-point results

Mean absolute error, lower is better, on the same eligible players:

| Position | Players | Draft only | College + draft |
| --- | ---: | ---: | ---: |
| QB | 74 | 45.63 | 48.16 |
| RB | 217 | 31.72 | 31.86 |
| WR | 329 | 24.12 | 23.36 |
| TE | 150 | 18.14 | 17.27 |

WR/TE point estimates improve modestly, but their season-bootstrap improvement
intervals still include zero. QB worsens; RB does not improve. College-only models
beat position means but do not beat the post-draft baseline. Identity-restricted
refits also change results, which is another reason to retain the research label.

For continuing college receivers, MAE moves from 134.12 to 131.90 receiving yards;
the first passing/rushing models lose to last-season yardage. More data is not,
by itself, evidence of a better forecast.

## Using it

The UI contains NFL translation forecasts, the all-college identity ledger and
backtest evidence. Expand a player to inspect school seasons and data-quality
flags, link method, completed NFL scoring, entry-year forecasts and the separate
current NFL outlook. Select **All classes** to find returning NFL players such as
Hunter, or use the identity ledger. Boston and Tate appear in the 2026 entry class.

Read-only endpoints:

- `GET /api/research/college`
- `GET /api/research/college/players?season=2026&search=Boston`
- `GET /api/research/college/identities?search=Travis%20Hunter`
- `GET /api/research/college/career?player_id=00-0041037`
- `GET /api/research/college/career?college_id=4832800`

The API verifies the pointer, immutable study artifacts, raw source fingerprints
and continuing acceptance of the repaired NFL history. A changed artifact returns
an error, not legacy predictions. Missing identity matches remain unknown.

## Rebuild

```sh
just college-capture college_source_YYYYMMDD_r1 2025
just college-translation data/research/college_source_YYYYMMDD_r1 \
  data/research/historical_v2_20260922_r5 college_nfl_v1_YYYYMMDD_r1
```

Captures and studies are create-only. The only replaced publication artifact is
the `data/outputs/college_nfl.json` pointer after successful completion. Failed
development builds remain unpublished. Earlier complete research releases remain
available on disk; production/reference artifacts are never rebuilt by this job.

## Remaining boundaries

The foundation now supports player-level source linking and new model experiments.
It does **not** yet estimate an arbitrary college player's probability of becoming
an NFL contributor. That needs a verified college-exit population and outcome
linkage that distinguishes "never reached NFL" from "unmatched". Further work
should prioritize review-queue resolution, independent early-stat reconciliation,
recruiting/transfer histories, competition adjustment and prospective validation.

Verification: 521 offline tests pass, 35 network tests excluded. Targeted Ruff and
mypy pass; frontend production build succeeds. Existing UI regression checks and
live Boston/Hunter desktop/mobile lineage checks pass. Node 21 triggers Vite's
existing supported-version warning; existing frontend lint warnings remain.
