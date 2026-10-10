# Player ability and team context research

The system’s direction is separate player ability ratings, with a distinct set of rankings that account for the player’s team environment. Fantasy-point prediction remains an application and benchmark, rather than the definition of player quality.

## Agreed direction

Build several ability ratings rather than immediately fitting one universal quality score. An eventual overall rank may combine the ratings using an average, sum or another explicitly chosen rule. The dimensions, normalization, weights and aggregation formula remain undecided; none is implied by an existing fantasy ranking.

Keep separate team-context rankings that incorporate opportunity, teammate interactions and team relationships. Show both views so a player’s measured execution is not confused with the situation in which the player operates. Correlations can inform contextual analysis but do not alone establish causal contribution.

## Organize existing work into separate layers

| Layer | Question | Existing material |
|---|---|---|
| Observations | What happened, and over how much exposure? | Counts, rates, snaps, profiles, data-quality checks |
| Execution and ability research | What skills can we measure, and how much is context? | QB efficiency, rate pooling, tracking and RB play-context experiments |
| Opportunity and participation | What role and playing opportunities does the player receive? | Role-transition, usage, participation and absence research |
| Team context | How does the situation change expected contribution? | Team/player correlations, allocation and contextual production models |
| Development | How might the player change? | Age observations, draft/college translation and longitudinal histories |
| Production applications | What outcomes follow from ability, context and opportunity? | Component forecasts, multi-output models and ensembles |
| Fantasy applications | How do league rules score those outcomes? | Weekly points, rankings, boards and roster analysis |

No existing catalog entry is claimed to be a validated general ability model. Per-attempt efficiency still depends on teammates, opponents and selection. Predicted production still mixes ability, availability and opportunity.

## First cleanup delivered

The model library provides one discoverable index of method families, current release decisions, reusable model contracts, configurations, source modules and local run metadata. The library reads current statuses and weekly evidence from checksum-matched release artifacts; it does not introduce another approval registry.

Active documentation no longer asserts universal RB age decline, a universal predictive priority for volume, or a guaranteed future scoring floor from a historical quantile. The claim ledger records what was narrowed or deleted and links to implementation/evidence. Historical research is preserved so negative findings and failed experiments remain inspectable.

## Remaining work before new ratings

1. Define position-specific ability dimensions and observable evaluation targets. Decide which require context adjustment and what data identify that adjustment.
2. Audit candidate sources for identity, exposure, time coverage and revision history. Do not infer missing exposure as zero ability.
3. Assess reliability across samples and seasons, sensitivity to team changes, and dependence on playing opportunities. Keep uncertainty visible.
4. Fit and evaluate ability components separately from team/opportunity components. Compare against simple baselines on identical cases.
5. Decide how to normalize and combine ability ratings only after the components have interpretable scales. Preserve the ability vector alongside any summary rank.
6. Evaluate contextual rankings separately. Any fantasy conversion belongs downstream and uses explicitly named scoring rules.
7. Review the source-inventory backlog before deleting implementations. Classify supporting code, reusable candidates, failed hypotheses and immutable historical dependencies. Remove redundant active interfaces after their callers and evidence links have replacements.

The library does not retrain models, promote candidates, delete historical runs, or alter current production forecasts. Its purpose is to make the next simplification decisions explicit and evidence-based.

## Legacy feature retirement

New builds no longer apply named-player overrides, fixed age penalties or depth-chart discounts. The manual override engine and both player/team adjustment configuration files have been deleted. Custom combine speed/burst/agility scores and the legacy career-interaction generator have been removed; observed age, experience, combine measurements and exposure remain available. The individual-stat runner no longer reconstructs old projection features, and strips retired feature names copied from historical gold.

The legacy ridge artifact schema is now version 4. Existing version-3 fit artifacts must be refitted against the changed definition before reuse. Historical releases and saved RB model weights remain immutable; this source cleanup does not republish them.

The saved RB model is independent of the prepared legacy board matrix. Its registered input names are audited in the library against the retirement list. It still uses fixed history aggregations and some provider-derived observations (including WOPR); removing an input required by saved weights requires a separate retraining and evaluation. DuckDB joins, aggregations, rates, observation coverage and as-of windows remain legitimate data preparation.

Legacy component projections and other research formulas still exist as compatibility/comparison methods. They are explicitly cataloged, not presented as an ability system or wholesale replacement for the raw-model pipeline. Retiring that remaining layer requires migrating its board/report consumers; deleting the columns blindly would break those consumers.
