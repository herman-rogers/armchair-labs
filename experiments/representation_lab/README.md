# Automated representation lab

Disposable, research-only experiments for preseason full-season league points.
Run from the repository root with the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider experiments/representation_lab/tests
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/representation_lab/run.py --run-id smoke_001 --smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/representation_lab/run.py --run-id discovery_001
```

Run IDs cannot be reused. Every write is restricted to the new run directory by
an imported Python audit guard; native Parquet writes also use explicit local
paths. Gold, the corrected inventory, source, and publication pointers are hashed.
No API, CLI, model registry, production source, or publishing integration is added.
Delete this directory to remove the experiment.

Read `protocol.md` before interpreting results. Open `runs/<id>/index.html` for
searchable comparisons and `runs/<id>/report.md` for the readout. A usable run must
have `status: complete`, `protected_inputs_unchanged: true` and
`publication_unchanged: true` in its manifest. Smoke is a plumbing check only.
