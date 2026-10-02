# Revalidation queue review — September 24, 2026

Reviewed every `revalidation_required` entry in the published
`nextgen_rankings_20260923_r6` registry. The replacement analysis release is
`revalidation_review_20260924_r1`: **213 reviewed, 4 dependencies revalidated,
209 entries archived, zero pending**. No predictive model was promoted.

The [complete decision ledger](../data/research/revalidation_review_20260924_r1/revalidation_review.json)
records each original entry, decision, reason, evidence details and condition for
reconsideration. The [resulting registry](../data/research/revalidation_review_20260924_r1/registry.json)
is the dashboard's inventory. Historical releases retain their original statuses.

| Entry type | Reviewed | Revalidated | Archived |
| --- | ---: | ---: | ---: |
| Legacy model recipes | 51 | 0 | 51 |
| Individual-stat predictor candidates | 99 | 0 | 99 |
| Study/product/source snapshots | 56 | 3 | 53 |
| Historical gold releases | 3 | 1 | 2 |
| Unimplemented forecast claims | 4 | 0 | 4 |

## Useful dependencies retained

These four entries are now `verified` **as data dependencies**, with no new
predictive serving permissions:

- `nextgen_products_20260923_r2_college`: current college product. Rechecked its
  corrected gold, accepted history, source and output hashes, and raw college files.
- `nextgen_products_20260923_r2_outlook`: current profile/research dependency.
  Rechecked its gold, accepted history and pinned inputs/outputs. Its old four-week
  predictions remain outside normal analysis.
- `college_source_20260922_r1`: captured source used by the current college product.
  Every raw file matches its manifest. This does not certify original historical
  publication times or college prediction quality.
- `canonical_targets_20260923_r3`: exact corrected-gold dependency of the completed
  individual-stat audit. All nine consumed tables were compared again, value for
  value, with current corrected gold and matched.

Their inventory rows retain `serving=archive` and empty `allowed_uses`: those rows
describe provenance, while the existing product routes enforce their own contracts.

## Retired models and candidates

The **51 legacy recipes** are superseded implementations, not a useful active
revalidation backlog. Their exact configured specifications are retained in the
ledger, including factor dependencies and historical `apply_live` flags. Those
flags belong to the archived board builder and do not grant NextGen approval.

- 23 composite, coalesced or adaptive recipes depend on retired fitted outputs.
- 13 participation/roster/security recipes lack current target-matched approval;
  participation proxies do not establish starting status or medical availability.
- 7 feature-stack recipes are superseded for research by the corrected stat/profile
  experiments, whose results do not approve the old fits.
- 6 older baseline recipes are replaced operationally by explicit outcome-specific
  references and separately evaluated current-season rankings.
- 2 direct market recipes require corrected matched evaluation and admissible
  cutoff-time market coverage before any new serving proposal.

No legacy recipe was refitted in this review. Archival reflects supersession and
lack of a justified current use; it is not a new statistical finding that every
recipe loses to every baseline. A useful idea can return as a new, scoped experiment.

For the **99 excluded statistics**, finite coverage was recomputed from all 17,396
completed rows of the hash-verified corrected feature matrix:

| Exclusion | Count | Decision |
| --- | ---: | --- |
| Unreconstructed historical formulas | 30 | Retire until a corrected definition and evaluation exist |
| Constants | 26 | Retire for this data vintage; reconsider if valid variation appears |
| Entirely missing valid inputs | 18 | Retire until dated input coverage exists |
| Bookkeeping/metadata | 13 | Keep as metadata; retire the predictor candidate |
| Held-out outcomes | 8 | Keep as evaluation labels; never use as their own predictors |
| Saved fitted outputs/future proxies | 4 | Require leakage-safe, nested reconstruction |

The **55 remaining artifact snapshots** are superseded builds, historical model
runs, dated advice or experiments without a current serving use. In particular,
the RB residual experiment failed all five advancement checks. The QB role work
improved job recognition without establishing reliable conditional workload or
market-relative acquisition value. Their evidence stays available for research.
Accepted historical observations remain available through the current gold chain.

The four `unavailable:*` entries are placeholders for starting status, medical
availability, longer-term development and waiver/trade value. They have no
implemented, validated model to revalidate and are now explicitly archived.

`archived` closes the input-review queue without asserting valid inputs or
statistical ineffectiveness. Existing quarantines remain quarantined. Original
files are preserved at their original paths; archive is a logical disposition.

## Promising work preserved

All previously verified measurements, scoped stat findings, reference forecasts
and shadow challengers are unchanged. In particular, the QB passing booster
remains promising research under its original protocol; it was already verified
and was never in this queue. See the
[QB review](qb_passing_forecasts_review_2026-09-24.md). Clearing obsolete entries
does not erase useful findings or approve new role/availability models.

## Verification and reproduction

The [review runner](../research/review_revalidation.py) verifies the published
source, current products and stat evidence before creating a new immutable release.
It requires one decision per pending entry, rejects predictive promotion, preserves
every unaffected registry entry, and checks all copied forecast/evaluation artifacts
against their original hashes. Publication uses the existing validated, atomic
catalog update with rollback history.

```sh
.venv/bin/python research/review_revalidation.py --version NEW_REVIEW --publish
```

This command requires a published source with pending entries; after this review
it refuses to create an empty duplicate review. Its decisions describe this reviewed
inventory, not automatic approval of arbitrary future models.

The release keeps the same gold, profiles, observation cutoff, forecasts and ranking
values. This was a dependency/retirement review, not a new fit or independent holdout.

Verification completed: **60 focused tests passed** across the review, serving
policy, publication, data pipeline and product routes. Both ranking horizons still
return 841 candidates through the API and CSV export. The QB booster remains
blocked from normal forecasts and available in research. All 38 original analysis
artifacts retain their hashes, every unaffected registry entry is identical, and
Python lint/format checks pass. The local catalog now selects the reviewed release;
its previous pointer is retained in catalog history.
