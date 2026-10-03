# Current injury capture — September 23, 2026

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

Current injury evidence is now preserved in the existing raw object store:

- Snapshot: `data/raw/snapshots/injuries_20260923_r1/manifest.json`
- Content-addressed source bytes: `data/raw/objects/<prefix>/<sha256>`
- Reviewed source list: `data/static/injury_review_20260923.json`
- Collector: `research/capture_current_injuries.py`

The snapshot contains all **493 ESPN player status records**, the exact league
snapshot they came from, **seven captured news pages**, and **six reviewed
player observations**. The NFL daily roundup is retained in full as source
evidence; its other player mentions have not all been converted into annotations.

The reviewed observations cover Jaxson Dart (two differently timed sources),
DJ Moore, Tony Pollard, Rico Dowdle, and Sam Darnold. Dart's season-ending surgery
is recorded as a **reported expectation**, with the September 23 known date and
an affected period beginning at Week 3. It does not erase his first two games.
The earlier Giants article describes continuing evaluation and is preserved as
that statement, rather than relabelled as team confirmation of surgery.

Existing historical injuries remain in the gold `nfl_injuries` table and its
raw dependency snapshots. The historical reviewed absence ledger remains
`data/static/historical_absences.json`. Current uncertain news is not inserted
into that ledger as a confirmed preseason absence. Current capture is a separate
partition in the **same raw storage system**, rather than an edit to an immutable
historical release.

`injuries/espn_statuses.parquet` is explicitly a parsed provider observation;
`injuries/review.json` and `injuries/observations.json` are reviewed annotations.
Each news asset preserves requested/final URLs, response status, retrieval time,
available publication/modification metadata, reviewed publication date, and body
hash. Missing publication times remain unknown. An `ACTIVE` tag does not certify
health, starting status, or the absence of an injury.

The manifest's `assets` map resolves logical filenames to raw object paths. It is
compatible with `engine.data.releases.load_manifest`, including hash verification.
Repeated capture requires a fresh version; source failures cannot publish an
accepted snapshot.

```bash
.venv/bin/python research/capture_current_injuries.py \
  --version injuries_YYYYMMDD_r1 \
  --review data/static/injury_review_YYYYMMDD.json
```

This is a completed raw capture, not an automatic refresh job or a republished
gold/model release. The September 23 roster review uses the evidence explicitly
as decision context; historical forecasts keep their original information cutoff.

Validation: 33 focused tests passed across current captures, historical evidence,
and absence handling. Tests cover byte preservation, source tampering, immutable
versions, failed downloads, unknown publication timestamps, and backdated news.
