# Target-count integrity finding in the consolidated data

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

The published gold `canonical_20260923_r5`, derived from enriched
`canonical_20260923_r4`, retains invalid receiving-target history for NFL source
seasons **2003–2008**. Manifest verification passes: it certifies preserved bytes
and the release's declared checks, not this previously untested football invariant.

| Source season | Recorded targets, QB/RB/WR/TE | Recorded receptions | Weekly rows with receptions > targets |
| --- | ---: | ---: | ---: |
| 2003 | 3 | 9,673 | 3,397 |
| 2004 | 5 | 9,757 | 3,441 |
| 2005 | 0 | 9,778 | 3,376 |
| 2006 | 67 | 9,792 | 3,420 |
| 2007 | 14 | 10,395 | 3,476 |
| 2008 | 17 | 10,081 | 3,460 |

These are observations in gold's `nfl_player_weeks`, not new external estimates.
There are **20,570** invalid weekly pairs and **2,538** admitted preseason rows
with annual receptions greater than targets, affecting forecast years 2004–2009.
The same weekly invariant passes for captured 2001–2002 and 2009–2025 observations.
Passing that invariant alone does not establish complete target coverage.

For example, Tiki Barber's 2006 source season has 58 receptions but ten targets.
The corresponding 2007 feature row yields 46.5 receiving yards per recorded target
and a receiving-first-down rate of 2.3. Amani Toomer's row has 32 receptions but
nine targets. These cannot be treated as ordinary low-volume receiving seasons.
The enrichment's first-down rate divides recorded first downs by this target
total; it does not establish denominator completeness. The root cause of the
captured target deficit has not been independently resolved here.

The initial family audit, `nextgen_signals_20260923_r1`, exposed the defect through
extreme early forecasts, including a 4,911.61-point RB prediction. That version
is retained only as a contaminated diagnostic. Do not use its feature gains as
selection evidence. Gold r5's conversion of NaN shares to null does not repair
the finite, incorrect target totals.

The final [signal audit](nextgen_signal_consolidation_2026-09-23.md) uses a
documented view derived from the enriched data. It detects source seasons where
aggregate receptions exceed targets and at least ten weekly rows violate the
same inequality. It then:

- Marks all weekly targets in those source seasons unknown.
- Masks affected preseason targets, target/air-yard shares, receiving-first-down
  rates and prior-season opportunity-shape summaries before deriving rates.
- Withholds incomplete career target totals instead of summing a partial history
  and presenting it as complete.
- Keeps every candidate, outcome, observation year and earlier training year.
  Receptions and audited league points are retained. No targets are invented.

The rule applies to 4,373 dated candidate rows, including rows whose fields were
already missing; 2,538 have a directly demonstrated annual inequality. The
[machine-readable diagnostic](../../data/research/nextgen_signals_20260923_r3/target_quality.json)
records the exact scope and masked fields. Its season audit concerns historical
players admitted to this research cohort; the table above examines all four
offensive positions in gold.

This is a research-local correction. **The upstream enriched/gold releases and
published profiles remain unchanged by this work.** Other models and displays
must not assume they inherit it. A canonical repair should reconstruct targets
from independently verified complete evidence, or preserve them as unknown,
then regenerate every dependent rate, share, weekly usage aggregate and profile.
Add reception/target consistency and numerator/denominator coverage checks to
release acceptance; preserve the old release and publish a new version.

This finding does not demonstrate that league-point outcomes or all V2 forecasts
are wrong. It invalidates unqualified claims about these receiving inputs and
any dependent models until their actual dependencies are examined. Later training
folds can still be affected when they retain corrupted earlier features.
