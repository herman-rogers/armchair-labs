# Documentation

Start here to distinguish how the system works today from proposed work and dated evidence.

| I want to… | Read |
|---|---|
| Start, stop, configure, or test the app | [Run locally](operations/run-local.md) |
| Refresh football data, publish forecasts, or precompute API responses | [Refresh and publish data](operations/refresh-data.md) |
| Find tracking/video archives and their data contracts | [NFL data sources](architecture/data-sources.md) |
| Run notebooks or archived experiments | [Research workflows](operations/research-workflows.md) |
| Understand the backend, storage, and frontend cache | [Architecture](architecture/README.md) |
| See proposed work and its status | [Plans](plans/README.md) |
| Read model evaluations, audits, and dated league analysis | [Research](research/README.md) |
| Find what changed in an earlier release | [History](history/README.md) |
| Look up the archived metric/board contracts | [Reference](reference/README.md) |

## Where new documentation belongs

- **Operations:** maintained instructions for running the current system. Link to code and commands; avoid pinning “current” to a dated release ID.
- **Architecture:** system boundaries, data contracts, and design decisions. Mark historical examples and distinguish implemented behavior from proposals.
- **Plans:** proposed work, remaining scope, and acceptance criteria. Update status as work ships and link to its operating guide or implementation record.
- **Research:** dated evidence with dataset, cutoff, method, and limitations. A successful experiment does not itself authorize production serving. Research questions are not automatically roadmap commitments.
- **History:** dated implementation, migration, and release records. Preserve their evidence rather than continually rewriting old results.
- **Reference:** archived contracts and original source material needed to interpret older artifacts.

Keep dates in report filenames. Keep component-specific instructions beside their code,
such as the [frontend guide](../web/README.md) and [notebook guide](../experiments/future_player_lab/notebooks/README.md),
and link to them here. Generated datasets, release manifests, and model outputs stay
in their artifact locations; documentation is not the release catalog.
