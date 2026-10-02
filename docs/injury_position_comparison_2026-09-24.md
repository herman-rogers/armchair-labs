# Recorded injury-related Out designations by position

The exploratory comparison finds fewer recorded injury-related Out seasons for
QBs than RBs, WRs, or TEs. It does not detect differences between RB, WR, and TE.
This measures captured reporting outcomes in the existing statistical cohort,
not the medical incidence of new injuries or risk at equal playing exposure.

## Population and outcome

Pinned gold: `canonical_nextgen_20260923_r1`. The analysis uses 9,964 regular-season
player-seasons from 2,634 players during 2009–2025 in `nfl_player_seasons`.
Every included player-season has at least one recorded statistical appearance.
Positions come from that season table, not present-day roster positions.

The outcome is at least one regular-season injury-report **Out** designation with
a non-null primary reason passing the recorded injury filter. Explicit illness,
COVID, non-injury, personal, inactive, other, migraine, appendix, infection and
hernia labels are excluded. This is a heuristic primary-label classification,
not medical adjudication; secondary reasons and practice-only evidence are unused.
The all-reason Out sensitivity analysis gives the same qualitative result.

Source reports are deduplicated to the latest captured update per player, season,
week and team (two duplicate rows removed). Repeated weeks cannot count as
multiple affected seasons. Missing reports mean no captured qualifying Out
designation, never confirmed health. The provider defines `report_status` as the
status for a game on the official injury report, not a count of new injuries:
[nflverse injury dictionary](https://nflreadr.nflverse.com/articles/dictionary_injuries.html).

| Position | Player-seasons | With qualifying Out | Percentage |
|---|---:|---:|---:|
| QB | 1,304 | 214 | 16.4% |
| RB | 2,844 | 683 | 24.0% |
| WR | 3,766 | 916 | 24.3% |
| TE | 2,050 | 520 | 25.4% |

## Statistical comparison

A linear probability model includes position and season fixed effects. CR1
standard errors cluster by player, allowing dependence across a player's seasons.
The joint Wald test for the three position coefficients gives chi-square 35.829
on three degrees of freedom, **p = 8.14e-8**. Six pairwise tests use Holm correction.
These are large-sample, model-based inferences for the specified observed cohort.

| Comparison | Season-adjusted difference | Pointwise 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| RB minus QB | +7.75 percentage points | +4.71 to +10.80 | 0.00000243 |
| WR minus QB | +7.82 percentage points | +4.89 to +10.76 | 0.000000852 |
| TE minus QB | +8.87 percentage points | +5.63 to +12.11 | 0.000000477 |
| RB minus WR | −0.07 percentage points | −2.49 to +2.34 | 1.00 |
| RB minus TE | −1.12 percentage points | −3.90 to +1.66 | 1.00 |
| TE minus WR | +1.05 percentage points | −1.61 to +3.70 | 1.00 |

Confidence intervals are pointwise, not simultaneous multiplicity-adjusted
intervals. Failure to detect a difference is not proof of equivalence. Relative
to the QB rate, the raw RB percentage is about 1.46 times as large, but this must
not be described as a 46% increase in medical injury incidence.

The qualitative pairwise conclusions also hold when using all Out reasons,
restricting to 2016–2025, or excluding 2025. These are exploratory sensitivities
on overlapping data, not independent replication.

## Interpretation limits and the stronger follow-up

The denominator excludes players without statistical appearances, including
some full-season absences. There are 78 QB/RB/WR/TE player-seasons with qualifying
Out evidence outside this cohort. IR/PUP absences can persist without a weekly
Out report. Some players play through injuries or miss a game after a different
designation. These percentages therefore underdescribe the full injury burden.

Backup QBs, starting QBs, rotational backs and full-time receivers have different
exposure. This model adjusts season only; it does not isolate position from age,
role, workload, prior injury or team reporting practices. It cannot establish
that position causes an injury difference. Public injury ascertainment can also
differ across player groups; an NFL ACL validation study documents that issue,
without establishing the bias in this specific dataset:
[Inclan et al.](https://pubmed.ncbi.nlm.nih.gov/34166138/).

A stronger study should build a roster-based player-week population that retains
zero-stat and IR/PUP weeks, joins actual team games to exclude byes, and audits
report coverage by team and week. Separate outcomes should include:

1. New documented injury episodes per 1,000 at-risk player-games, with explicit
   onset, recovery and recurrence rules. Report weeks cannot supply these alone.
2. Documented injury-related missed games divided by eligible scheduled games,
   including IR/PUP, with suspensions and other absences handled separately.
3. Probability of any documented injury absence over a season, stratified by
   starter/rotation role and prior workload.

Use a repeated-measures binary model for absence, or an exposure-offset count or
survival model for incident injuries. Include season, age, prior injury and
pre-injury workload where observed. Actual full-season snaps can be reduced by
injury itself, so simply controlling for final snaps can introduce bias. Snap
incidence requires identifying injuries attributable to that exposure. Report
absolute differences, relative rates, confidence intervals and corrected pairwise
tests. These are follow-up requirements, not claims that this first pass met them.

## Artifacts and checks

- [Reproduction script](../research/injury_position_comparison.py)
- [Results and all six comparisons](../data/research/injury_position_20260924_r2/results.json)
- [Coverage, exclusions and source hashes](../data/research/injury_position_20260924_r2/audit.json)
- [Analysis cohort](../data/research/injury_position_20260924_r2/cohort.parquet)

Run with a new output directory:

```sh
.venv/bin/python research/injury_position_comparison.py \
  --output-dir data/research/injury_position_YOUR_VERSION
```

Checks passed for unique cohort keys, qualifying injury-Out outcomes being a
subset of all Out outcomes, nonzero statistical appearances, row-order invariance,
and Ruff. An independent 3,000-resample player-cluster bootstrap of the raw RB-minus-QB
difference produced a 95% interval of +4.50 to +10.57 percentage points (seed
24092026), consistent with the adjusted analysis. The r2 run preserves the final
script and source hashes; r1 was the initial numerical check with identical results.
No production model, gold table or catalog was changed.
