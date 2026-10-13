# Opening Drive interval-width investigation

This isolated experiment diagnoses where v3.1's extra interval width earns its
cost, and tests corrections that can be more selective. It never modifies the
v3/v3.1 point models, sealed research runs, served forecasts or prospective log.

## Run and review

From the repository root:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.interval_research.run
```

Default immutable output:
`experiments/future_player_lab/runs/opening_drive_width_research_20261013_002`.
Interactive review:
`data/outputs/opening_drive_width_research_2026/index.html`.
Use fresh `--output` and `--report` paths for another run. Existing paths are
rejected. Inputs, source, protocol, calibration fits, selection decisions,
predictions, diagnostics and checksums are retained. The protocol is written
before this run evaluates candidates. Original v3.1 bounds are reproduced
exactly on shared rows, and protected artifacts are hashed before and after.
The initial `_001` run is retained. `_002` corrects near-integer floating-point
rounding in empirical endpoint ranks and reruns the same candidate protocol.
Presentation can be regenerated from the sealed scores without refitting:

```sh
.venv/bin/python -m experiments.future_player_lab.campaigns.interval_research.report
```

## Questions and candidates

The two controls are original v3 and locked v3.1. Seven fixed alternatives test:

| Method | Question |
| --- | --- |
| Half v3.1 expansion | How much of the gain survives a smaller correction? |
| Signed, width-scaled | Does permitting narrowing improve over-covered groups? |
| Signed, additive | Does scaling padding by existing width add unnecessary cost? |
| Separate tails, width-scaled | Do lower and upper misses need different corrections? |
| Separate tails, additive | Can endpoint shifts improve calibration with less width? |
| Separate tails, two-year window | Is older calibration history stale? |
| Separate tails, finer low groups | Does separating projections below 1 from 1–5 help? |

Two additional policies select among those nine fixed methods using **earlier
validation seasons only**. The first minimizes interval score. The second
requires earlier equal-year coverage of at least 78% for nominal 80% intervals
and 88% for nominal 90% intervals, then minimizes score. Neither is selected by
looking at the evaluation season. If no supported method passes the coverage
guard, it falls back to v3.1 and records that decision.

These are interval adjustments, not new scoring distributions. Point forecasts,
the original quantiles and standard deviations remain unchanged. Known byes
remain zero. Negative endpoints are not clipped because the scoring contract
can produce negative points. Crossed adjusted endpoints collapse to their
midpoint; 90% intervals are expanded when needed to contain 80% intervals.

## Chronology and support

Player evaluation is 2021–2025. Selected v3 distributions from 2018 onward supply
past calibration; 2019–2020 provide the first chronological validation forecasts.
The original v3 policies themselves were selected using 2018–2020. These are
reused development years, not an untouched test.

Artificial-lineup evaluation is 2023–2025: 2021 supplies the first calibration
history and 2022 the first method-validation year. The two original fixed roster
selection designs and stacked/unstacked subsets are retained. Counts overlap and
are not independent lineups. No actual league or K/DST labels enter this search.

Default calibration uses the preceding five seasons; the recency candidate uses
two. Groups use fixed projection boundaries known before the game. At least 200
player rows or 100 lineup rows are required, with position fallback. Finer low
groups apply to players, not lineups. No correction is fitted with current-year
outcomes. The selectors use up to two earlier validation seasons, require 100
rows per method/group, and fall back to a position-level policy for unsupported
groups. Final 2026 research policies are saved but not published into the live
prospective comparison.

## How to assess the result

An interval's score decomposes into width plus penalties for outcomes below and
above the bounds. `diagnosis.csv` shows whether v3.1's added width was offset by
saved miss penalties. Positive `net_score_cost` is worse. Example player-weeks
show concrete endpoint changes and their realized costs; they are descriptive,
not training rules.

`scores.csv` includes coverage, width, interval score and separate tail misses,
by season, position, projected scoring level and history support. Starter-like
projections of at least 5 points are also separate from near-zero players. The
interactive report plots width against coverage for both 80% and 90% ranges.

Method selection averages interval scores with alpha weights (0.2 and 0.1),
then weights each validation year equally. Lineups have only the 80% interval.
The choice is made separately by position/scoring group; lineup choices are
pooled. `comparisons.csv` gives paired differences versus v3.1, season win counts
and 5,000-draw season-block bootstrap intervals. Only five player years and three
lineup years are available. These are descriptive intervals without a correction
for multiple comparisons or previous reuse of the years.

Improvement requires considering coverage **and** precision. Exact 80% coverage
is not always attainable with discrete zero-heavy scores. A method that appears
better here needs a separately versioned future trial; existing pregame captures
must not be rewritten. The experiment intentionally does not auto-promote a
winner or claim that calibration guarantees hold under temporal dependence.

Method references:
[Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222) and
[Evaluating epidemic forecasts in an interval format](https://arxiv.org/abs/2005.12881).
They motivate the endpoint corrections and score decomposition; this study is
an NFL-specific chronological adaptation.

## Verification

```sh
.venv/bin/python -m pytest experiments/future_player_lab/tests/test_interval_width_research.py
```

Tests cover future-label exclusion in fitting and selection, preservation of
means and quantiles, finite/nested bounds, byes, actual narrowing, asymmetric
tail correction, sparse-group fallback, guard behavior, and exact score-cost
decomposition. Changing 2025 outcomes cannot change any 2021–2025 forecast bound
or selected policy in the chronology test.
