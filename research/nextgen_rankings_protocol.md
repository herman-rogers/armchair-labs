# Current-season NextGen ranking protocol

Fixed before each ranking fit on September 23, 2026. This replaces neither
the frozen preseason experiment nor the retained exploratory multi-outcome runs.

## Question and observations

Predict remaining regular-season league points and next-four-calendar-weeks league
points at the currently published completed-week cutoff. Reconstruct the same
cutoff in every completed historical season. Include all preseason candidates and
players observed by the cutoff; never admit a player using a later appearance.
Keep all earlier training seasons (candidate years 2004 onward; history from 2001).
Use league-exact audited scoring. Unknown source statistics remain null. Prefix
and prior-season opportunity, scoring, participation, age and draft information
form the core profile. A separate enriched challenger adds correctly dated NFL
tracking and identity/coverage-checked college production. No old model outputs,
ECR or future status are predictors. Current injury news is a separately disclosed
constraint, not a historical feature reconstructed from today's knowledge.

## Fixed candidates and selection

References: prior-season points per scheduled team game, current points per elapsed
team game, and a four-game-prior blend of the two. Missing priors use earlier
training population means. These are declared reference estimates, not claims of
market advantage. Remaining scheduled games include known byes.

Revision 2, declared before its fits: also compare nonnegative affine calibrations
of the current and blended references (one-variable positive-coefficient ridge,
alpha 100, trained on earlier completed seasons only). Revision 1 exposed extreme
raw pace extrapolation for the current QB reference and failed its small-prior
guardrail. Preserve that entire run. No publication threshold is relaxed. This
additional recipe design was informed by the first run: retrospective selection
evaluation is not an untouched independent confirmation. The calibration can
change forecast levels; as a monotone transformation it does not itself establish
better within-position ordering.

Revision 3, declared before its fits: choose the recipe independently for players
with at least ten prior-season observed weeks and players with fewer than ten.
The first two runs showed that one position-wide choice can improve established
QBs while worsening limited-history QBs. This is a fixed exposure distinction
already used by the original research, not a player-name override. In every outer
year both group decisions use only earlier validation years from the same group;
training still uses all earlier candidates. Evaluate the complete combined policy,
retain the same eight primary scopes and every existing cohort/error/ranking gate.
This iteration is transparently research-informed, not fresh independent evidence.

Revision 4, declared before its fits: replace that history-only selection split
with observed workload at the cutoff. QB: >=10 attempts per elapsed team game;
RB: >=5 carries plus targets; WR/TE: >=2 targets. Missing workload remains in the
limited-workload group. This denotes recorded opportunity, not a verified starter
label. Fit the two calibrated references within these same workload groups,
falling back to all earlier position rows with fewer than 50 training cases.
All challengers still train on every earlier position candidate; the explicit
small-prior and rookie publication guardrails remain unchanged. Historical injury
returners should not receive a different raw-pace treatment merely because last
year was short. This fixes the grouping interpretation exposed in revision 3;
retain that run and acknowledge the same retrospective research limitation.

Challengers: core-profile ridge (alpha 100), core-profile histogram boosting and
enriched histogram boosting (120 iterations, learning rate .05, 15 leaves, leaf
minimum 30, L2 10, 63 bins, no early stopping). Fit per position and horizon, using
only earlier completed seasons. Preprocessing, imputation and varying-column
selection use training rows only. Retain predictions from every recipe.

Each test year chooses its reference from earlier out-of-year reference errors.
After four earlier validation years, a challenger can replace it only with at
least max(0.5 points, 1%) lower equal-season MAE, non-worse MSE and non-worse
top-K actual-point capture on those earlier validations. Choose the qualifying
challenger with lowest MAE, using its fixed name to break ties. Evaluate this
whole selection policy on the next year, not the retrospective best recipe.

## Publication decision

Eight primary scopes: four positions times two horizons. Compare the nested
selection policy with the independently selected reference policy, starting in
2011. Primary window is modern 2019–2025; full-history is a mandatory guardrail.
Publish the selected challenger for a scope only if modern equal-season MAE gain
is at least max(0.5 points, 1%), its 95% season-bootstrap interval is positive,
BH-adjusted exact two-sided season-sign-flip q <= .05 across all eight scopes,
MSE and top-K point-capture gains are nonnegative, and full-history MAE/MSE and
top-K gains are nonnegative. Require seven modern and eight full-history years.
Guard against materially worse rookie and small-prior-sample MAE (more than
max(1 point, 5%)) in cohorts with >=100 cases over >=5 modern years.
If the policy or current candidate does not qualify, publish the selected current
reference with an explicit reference status and reason. Do not erase the ranking,
silently serve a failed model, or claim that the reference is an advanced model.

Top K is QB 10, RB 20, WR 30, TE 10, chosen from the complete candidate pool per
position. Point capture is the sum of realized points among the K predicted best
players divided by the best possible sum for that same candidate pool. Save
annual MAE, MSE, point capture, population coverage, selected recipes, prediction
coverage and 10,000 season-bootstrap intervals. Historical data revisions and
previous inspection of these seasons limit claims: this is retrospective
validation, not an untouched prospective test or demonstrated market advantage.

## Product contract

Ranks order forecast points, with positional rank primary; overall points rank
is not positional scarcity, a draft recommendation or FAAB value. Ranks are
computed across the complete eligible population before UI filtering/pagination.
Show horizon, completed-week cutoff, publication date, recipe/evidence status,
scheduled games, current sample, availability annotations and dated preseason ECR
as a separate historical reference. Do not label preseason ECR as current market.

Reported season-ending absences suspend ranking eligibility with a visible source
and zero constrained expectation. Ordinary OUT/IR/questionable tags do not imply
season-ending absence and do not produce invented return dates. Injury constraints
are saved at publication, with current ESPN statuses shown separately. Scores for
unresolved new roles remain uncertain; no claim of solved starter recognition.
The analysis manifest binds all ranking artifacts, implementation and evidence;
API and CSV use the same scoped eligibility and incident checks. Old NextGen and
four-week models remain archived. Publication is atomic after tests and review.
