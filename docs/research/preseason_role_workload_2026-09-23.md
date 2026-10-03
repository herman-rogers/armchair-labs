# Preseason workload when a player's job changes

> Dated research record. Findings and roster advice apply to the stated data and cutoff; follow-up ideas are not an active implementation commitment.

The next research question is **what job the player is expected to hold at the
forecast cutoff, and what workload follows from that job**. Prior PPG is a mixture
of roles. Separating that mixture is useful diagnosis, but the first experiment
below does not establish an improved forecast or market edge.

This study uses the accepted `historical_v2_20260922_r5` rebuild. Its retained
baseline reproduces the cited audit: 87 modern V2 calls outside ECR's top 60,
all with at least ten prior games; 70 fourth-year-or-later calls; zero of 13
small-prior-sample market misses recovered; two of 29 RB misses recovered.

The [runner](../../research/preseason_role_workload.py),
[temporal tests](../../research/test_preseason_role_workload.py), and
[complete results](../../data/research/preseason_role_workload_20260922_r1/report.json)
are saved separately from production. The artifact name retains the September 22
research batch date; this readout was completed September 23. No production
forecast, frozen 2026 file, or existing research report was changed.

**Lamar's diluted prior is real, but changing its representation does not fix the
forecast by itself.** Recomputed from his 2018 regular-season scoring and snaps:

| Historical view | Games | League PPG |
| --- | ---: | ---: |
| All box-score appearances | 16 | 10.03 |
| At least 50% of offensive snaps | 7 | 19.02 |
| Other appearances | 9 | 3.04 |
| Final four scheduled team games | 4 | 20.14 |

His average snap share in the final four games was 99.25%. The seven high-snap
games are an observed usage proxy, not an independently sourced starts label.
The existing repaired forecast is 11.07 active-game PPG, 11.75 games, and 130.09
season points. Those units must not be compared interchangeably.

The new annual-usage, recent-usage, and role-summary regressions predict 131.99,
143.73, and 127.10 season points respectively, against 428.68 realized. The role
model even predicts only 79.78 passing attempts, against 401 realized. Its
historical role evidence is present, but the learned mapping still fails badly
on this case. Selecting the 19.02 PPG slice and multiplying it by a full season
would instead *assume* the future role and availability; it would not validate
either forecast. It would also miss his subsequent efficiency improvement.

**The first experiment isolates increasingly detailed historical representations.**
All four challengers use fixed-penalty ridge regression, fit separately by
position. Test seasons are 2019–2025; training starts in 2014 and always ends
before the test season. Imputation, missingness indicators, scaling, and
coefficients are learned only from training rows. No thresholds or penalties
were changed after inspecting results.

| Specification | Inputs added |
| --- | --- |
| Production refit | Prior PPG, prior games, age, experience, upcoming season length |
| Annual usage | Prior passing attempts, carries, targets, offensive snaps/share, offensive games |
| Recent usage | Points and usage in the final four scheduled games of the last prior-season observed team |
| Role summary | High-usage game count, PPG in/away from that usage, final-four high-usage fraction |

High usage means at least 50% snaps for QB; at least 50% snaps and 12 carries plus
targets for RB; at least four targets for TE. These are fixed operational proxies,
not validated job definitions. TE target earning is not evidence of route-running
or blocking assignments. The RB proxy does not separately identify receiving,
goal-line, and early-down jobs.

Separate regressions predict season points and each position's workload totals:
QB attempts/carries, RB carries/targets, and TE targets/offensive snaps. The point
forecast already incorporates nonappearances. Workload estimates are diagnostics,
not extra multipliers applied to points. This is a representation test, not yet
a probability model over future jobs or a workload-times-efficiency model.

There are 4,394 feature rows, 4,388 with eligible snap coverage/identity, and 2,590
held-out forecasts: 556 QB, 1,152 RB, and 882 TE. Rookies are outside the model.
The study records 217 unmapped raw snap records and excludes 226 weekly rows with
no player ID. Every studied season has scoring and snap coverage at scheduled
team/week level; that does not certify every individual source row. Missing
identity or season-level snap coverage is unknown, never known zero.

Late windows retain zero appearances through the end of the team's schedule.
They do not move backward to the last four games the player happened to play.
This avoids concealing absences but does not distinguish injury from lost work.
Annual usage rates use nominal season length (16/17); a canceled game or a trade
can therefore affect their interpretation. All inputs are regular-season history.
No outcome-season usage, depth chart, transaction, or undated contract enters
the challenger features. Cached historical sources may contain later revisions.

**Results do not support promotion.** The table gives mean absolute error in
season fantasy points, averaging the seven season-level errors equally. Lower
is better; comparisons within each position use the same players.

| Model | QB | RB | TE |
| --- | ---: | ---: | ---: |
| Saved repaired V2 | **57.62** | **38.58** | **25.87** |
| Production refit | 63.19 | 40.81 | 26.72 |
| Annual usage | 63.34 | 40.68 | 26.77 |
| Recent usage | 61.82 | 40.47 | 26.99 |
| Role summary | 60.78 | 40.57 | 27.05 |

Saved V2 uses a different training/model design, so incremental feature claims
come from the four matched refits. The saved model remains the practical
benchmark and beats every challenger on overall point error.

QB passing-attempt error improves from 115.70 with annual usage to 113.75 with
recent usage and 112.58 with role summaries. The role-versus-annual improvement
is 3.12 attempts per season, season-bootstrap 95% interval [1.68, 4.32]. This is
a modest population improvement, not a solution for newly installed starters.
RB carries/targets and TE targets show no consistent incremental benefit from
the role summaries. TE snap error gets worse than annual usage.

In the 87 QB rows with prior late-season role growth, role summaries reduce point
error by 3.13 relative to recent usage, interval [1.24, 5.95]. Saved V2 still has
lower error in that subgroup. The analogous RB and TE role-growth groups have
only 50 and 51 rows and show no improvement. These exploratory slices are not
independent tests, and intervals do not correct for multiple comparisons.

**The acquisition comparison preserves the original player pool.** Each model
replaces only covered QB/RB/TE returner forecasts. WRs, rookies, and uncovered
players retain V2. All rankings use the repaired audit's positional replacement
convention and identical ECR comparison pool, with no outcome-based filtering.

| Ranking | Model-only hits / calls | Market-only hits | Net value / season |
| --- | ---: | ---: | ---: |
| Saved V2 | 24 / 87 | 36 | -12.64 |
| Production refit | 23 / 81 | 37 | -9.85 |
| Annual usage | 24 / 88 | 42 | -15.44 |
| Recent usage | 24 / 95 | 47 | -17.76 |
| Role summary | 26 / 94 | 48 | -18.21 |

Every challenger still recovers zero of 13 small-sample misses and two of 29 RB
misses. Eight of those 13 small-sample misses are actually modeled here; five
are WRs retained at V2. All 29 RB misses are modeled. The role-summary ranking's
two additional overall hits do not imply improved RB discovery; ranking changes
can also move an unchanged WR across the selection boundary.

Net value uses the original target: positive realized positional VOR multiplied
by actual games / 17. It is not fantasy points, acquisition profit, or a legal
roster simulation. Role-summary net value versus ECR is negative in all seven
seasons; its season-bootstrap interval is [-26.63, -9.95]. The small-sample
and RB recovery counts therefore remain unsolved under the tested approach.

**The missing evidence is sometimes available before our cutoff.** These examples
were checked after the pilot, to scope a new dataset; they were not inserted into
the fitted models or counted as validation of a future-role method.

| Player/fold | Forecast cutoff | Dated evidence available before cutoff | Interpretation |
| --- | --- | --- | --- |
| Lamar Jackson, 2019 | July 17, 2019 | Baltimore named him starter December 12, 2018, then announced Flacco's trade March 13, 2019 | Strong evidence that his backup appearances should not determine his expected job |
| Jordan Love, 2023 | August 25, 2023 | Green Bay's May 3 report identifies Love as the new QB1 and quotes LaFleur on his responsibility | A new starting job despite zero prior high-snap games in this study's prior-season window |
| Darren Waller, 2019 | July 17, 2019 | Raiders' June 11 minicamp report describes positive receiving practice observations and coaching support | Weak opportunity evidence; it does not establish a target share, route rate, or guaranteed starting job |

Sources: [Baltimore's starter announcement](https://www.baltimoreravens.com/news/joe-flacco-is-disappointed-but-handles-backup-role-like-a-pro),
[Baltimore's trade release](https://www.baltimoreravens.com/news/press-release-ravens-player-announcements),
[Green Bay's Love report](https://www.packers.com/news/matt-lafleur-on-jordan-love-this-offseason-he-s-the-guy-in-charge),
and [Oakland's minicamp report](https://www.raiders.com/news/five-observations-from-day-one-of-the-oakland-raiders-mandatory-minicamp).
These are publication dates on currently accessible pages, not archived snapshots
proving the exact historical text. A later capture date must be retained separately.

The studied misses also contain different problems. Purdy 2023 has six prior
high-snap games and 95.5% final-four snap share. Love 2023 and Waller 2019 have
zero games meeting their respective prior high-usage thresholds. Dobbins 2024
and Prescott 2025 have zero late-window snap share. Those numerical signatures
must not automatically receive the same “emerging role” adjustment; explaining
them requires dated competition and availability evidence. The current tests
do not establish that all 13 small-sample misses were role-transition failures.

**The next experiment should start with QB job probabilities**, where a starting
designation is easier to verify and workload is more concentrated. RB contingent
opportunity and TE receiving assignments need richer labels. Use the following
contract before collecting outcome-selected success stories:

1. Define the full candidate cohort at each historical cutoff, including players
   whose apparent promotion failed. Store player ID, event date, publication date,
   capture date, source URL, expected job, competing players, and unresolved
   conflicts. Use explicit `unknown` where evidence is missing. The examples above
   are data-contract cases, not a training sample.
2. Estimate the probability of each future job from that cutoff evidence. For QB,
   distinguish incumbent/new starter, open competition, and backup. For RB, model
   carries, receiving, and goal-line allocation, plus contingent promotion if a
   teammate becomes unavailable. A future injury can be a training outcome or a
   scenario, never knowledge supplied to the preseason features.
3. Estimate workload and scoring conditional on each job. Shrink efficiency toward
   position/role peers based on relevant attempts, targets, or carries, rather
   than counting a one-snap appearance as a full game of evidence. Preserve
   availability separately from assignment. For TE, obtain dated true route and
   blocking data or explicitly keep the receiving assignment uncertain.
4. Combine role scenarios using their probabilities: expected points are the
   sum of each scenario's probability times its conditional expected points.
   Within a scenario, model participation, workload, and efficiency jointly or
   preserve their dependence; multiplying unrelated marginal means can mislead.
5. Compare with saved V2, annual/recent usage, and a market-conditioned model on
   the same folds and player pool. Evaluate job-probability calibration, workload
   error, total-point error, and acquisition results separately. Track announced
   starters with small samples, failed promotions, late role growth, and missed
   time separately. Check ECR and date-admissible ADP; the market may already price
   the promotion. Require broader validation and a prospective shadow period
   before promotion, because 2019–2025 have already informed research choices.

The practical objective is to make a low-production backup with a newly announced
starting job receive a starter-conditioned workload distribution, while retaining
uncertainty about his efficiency and job retention. This pilot supplies a negative
baseline and concrete failure cases for that next experiment; it has not built or
validated that job-probability model.

Reproduce in a new output directory:

```sh
.venv/bin/python research/preseason_role_workload.py \
  --history data/research/historical_v2_20260922_r5 \
  --output data/research/preseason_role_workload_reproduction
.venv/bin/pytest research/test_preseason_role_workload.py research/test_research_audits.py tests/test_outlook.py
.venv/bin/ruff check research/preseason_role_workload.py research/test_preseason_role_workload.py
```

Validation: 29 targeted tests pass, including future-data perturbation,
training/test separation, unknown snap identity/coverage, duplicate-key rejection,
and late-window absence/postseason handling. Ruff passes. The integration run
reproduces the repaired market-call and miss counts and records source,
implementation, and output hashes. Protected artifact hashes are unchanged.
