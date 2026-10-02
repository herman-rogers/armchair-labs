# QB role transition experiment, frozen before fitting

Research only; accepted historical_v2_20260922_r5 and prior workload pilot.
Training begins in 2014; expanding tests are 2019–2025. These reused seasons
are exploratory. No hyperparameter search or outcome-based evidence selection.

Review cohort: returning QBs with fewer than ten prior high-snap games and
positional ECR <=32 at the saved cutoff. Keep every qualifying row, including
unknown evidence, injuries, failed promotions and utility QBs. Evaluate the full
QB population too. Rookies remain outside the model. Missing source evidence is
unknown, not backup. Collection is manual and incomplete; publish its coverage.

Evidence records contain player ID, forecast season, publication/event/capture
dates, source URL, job state, competitors and a short paraphrase. Event dates
are null when not independently established; publication is then the knowledge
date. Currently accessible pages are not historical text snapshots. Sources on
the cutoff date are admitted under an end-of-day convention; also run a strict
earlier-day sensitivity. Later sources never become features. Conflicting states
on the latest admissible date resolve to unknown. Explicit current starting jobs
are `starter`; probable, expected or unresolved choices are `competition`;
explicitly displaced players are `backup`. Do not infer a job from a signing.

The job target is starting QB in the team's FIRST scheduled regular-season game,
from frozen schedule home_qb_id/away_qb_id. This outcome is used only as a label.
It is distinct from injury availability, subsequent job retention and season
workload. Using each team's first game handles the 2017 Miami/Tampa postponement.
Actual high-snap weeks and offensive appearances are outcome diagnostics.

Fit logistic starter probabilities (C=1, no class weights) with training-only
median imputation, missingness indicators and scaling. Four specifications:
prior recent/role usage; plus dated evidence; plus market; plus market+evidence.
Market inputs: log positional ECR and its inverse, same frozen snapshot.
Evidence inputs: starter/competition/backup indicators; unknown is all zero.

Fit separate fixed ridge-10 experts for opening starters and nonstarters to
predict season passing attempts, carries, offensive games and league points.
The same usage/efficiency experts serve all four gates, isolating the effect of
job probabilities. Also fit four corresponding direct ridge models to test
whether mixtures help beyond simply adding the evidence. Saved V2 and the prior
pilot's recent-usage and role-summary fits remain comparison benchmarks.

Efficiency: prior three seasons' base passing points per attempt and rushing
points per carry, shrunk by 200 attempts / 60 carries toward earlier training
seasons' QB rates. Hyperparameters are fixed. Bonus/receiving points remain in
the league-points target but not these efficiency summaries. Expert point models
learn joint conditional season scoring using these summaries and prior workload;
do not multiply independent workload and efficiency means. The final forecast is
p(start)*E(points|start)+(1-p(start))*E(points|nonstart), with no extra games or
availability multiplier. Participation is separately predicted, not medical risk.

Report starter Brier/log loss/reliability, workload and point MAE separately.
Primary incremental evidence comparisons: market+evidence versus market, within
each architecture. Equal-weight season differences with fixed-seed 10,000-season
bootstrap intervals and leave-one-season-out ranges; small samples and multiple
comparisons preclude confirmation. Report review cohort, prior box games <10,
known starters, competitions, unknowns, and announced starters with <8 realized
high-snap games (outcome-only diagnostic, never an inclusion rule).

Keep the original all-player ECR top-60 pool, replacing only modeled QBs, with
V2 fallback everywhere else. Reconcile saved V2's 87 calls, zero/13 small-sample
miss recoveries and two/29 RB recoveries. No claim about solving RB/TE roles.
Audit ADP's collection-window end against each cutoff before using it; omit an
ADP comparison if no sufficient temporally admissible coverage exists.

Save create-only artifacts, source/implementation/output hashes, frozen evidence
and design copies. Test cutoff handling, unknowns/conflicts, future-data
perturbation, training separation and probability-weighted aggregation. Preserve
all production and existing research files. No production promotion from this run.
