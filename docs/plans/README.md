# Plans and future work

These documents propose work; they are not operating instructions or a declaration
that every recommendation has shipped. Their dates identify the reviewed state.

| Plan | Scope and status |
|---|---|
| [Tracking data review and acquisition](tracking_data_review_and_acquisition_2026-10-02.md) | Acquired tracking/context datasets, public archive search, source gaps, and proposed identity, geometry and forecasting evaluations. |
| [Data reliability and full dataset training](nextgen_data_training_review_2026-10-02.md) | Audit plus proposed source completeness, sample counts, transformation repairs, research cache contracts, and full eligible tree/representation inputs. Keep its acceptance criteria separate from app serving performance. |
| [Local compute and shared data](research_compute_and_shared_data_plan_2026-10-02.md) | Proposed shared GCS/Parquet transport, verified local research cache, GPU evaluation, and experiment packaging. Local notebooks already exist; shared storage provisioning and deployment/authentication are not established by this plan. |
| [FantasyTools comparison and adoption priorities](fantasytools_competitive_review_2026-10-02.md) | Proposed product sequence for freshness, role changes, weekly decisions, and later FAAB/trade/context work. A public-product review, not a measured prediction comparison. |

## Existing foundations

The backend API, TanStack Query layer, verified in-process caches, and precomputed
serving responses already exist. The weekly refresh also exists. Use the
[architecture guide](../architecture/README.md) and [refresh runbook](../operations/refresh-data.md)
for those capabilities. They do not establish that the broader research input,
training, or shared-storage proposals above are complete.

## Maintaining status

When work ships, identify the completed scope in its plan and link to maintained
operations/architecture documentation. Preserve the dated audit and acceptance
criteria; do not silently turn the entire review into a completed checklist.
The [original engine implementation plan](../history/fantasy_engine_implementation_plan.md)
is historical. Follow-up ideas in [research reports](../research/README.md) remain
research context unless adopted into an active plan.
