"""Seal evaluated v4 policies without promoting the application default."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from ..weekly_v31.artifacts import digest, now, seal, verify, write
from ..weekly_v32.integration import DEFAULT_INTEGRATION as V32_INTEGRATION
from .artifacts import (
    BASE,
    DEFAULT_HORIZONS,
    DEFAULT_INTEGRATION,
    DEFAULT_UNCERTAINTY,
    ROOT,
    environment,
)
from .total_evaluation import evaluate_bridge
from .uncertainty import evaluate_totals, fit_total


def integrate(
    output=DEFAULT_INTEGRATION,
    horizons=DEFAULT_HORIZONS,
    uncertainty=DEFAULT_UNCERTAINTY,
    baseline=V32_INTEGRATION,
):
    dependencies = {}
    for name, path in (
        ("horizons", horizons),
        ("uncertainty", uncertainty),
        ("baseline", baseline),
    ):
        path = Path(path).resolve()
        verify(path)
        dependencies[name] = dict(path=str(path), sha256=digest(path / "manifest.json"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    totals = pd.read_parquet(Path(horizons) / "horizon_predictions.parquet")
    totals = totals[totals.season.ge(2021)]
    write(output / "horizon_uncertainty.json", fit_total(totals[totals.method.eq("v4")], 2026))
    evaluate_totals(totals).to_csv(output / "horizon_uncertainty_scores.csv", index=False)
    bridge_scores, bridge_support = evaluate_bridge(Path(horizons) / "v32_bridge_matched.parquet")
    bridge_scores.to_csv(output / "bridge_uncertainty_scores.csv", index=False)
    bridge_support.to_csv(output / "bridge_uncertainty_support.csv", index=False)
    source = Path(__file__).parent
    sources = {}
    paths = [
        p
        for name in ("weekly_v3", "weekly_v31", "weekly_v32", "weekly_v4")
        for p in (source.parent / name).glob("*.py")
    ]
    paths.extend((ROOT / "src/engine/scoring").glob("*.py"))
    paths.append(ROOT / "src/engine/metrics/weekly_distribution.py")
    for path in sorted(paths):
        sources[str(path.relative_to(ROOT))] = digest(path)
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(
        output / "release_policy.json",
        dict(
            default_changed=False,
            base_campaign=BASE,
            point_model="Frozen v3.2 weekly; v3-based fitted future horizon residuals",
            retraining="Separate residual-update candidate; never automatically promoted",
            prospective_minimum=dict(completed_weeks=4, observed_player_weeks=2000),
            review_gates=[
                "All four positions and three horizons, components, replay and refresh verified",
                "Matched point, ranking, component and distribution scores "
                "reviewed by position/origin",
                "Any historical regressions explicitly accepted before release",
                "At least four future completed weeks and 2000 paired observed player-weeks",
                "Prospective RMSE no more than 1% above v3.2; CRPS no worse on paired rows",
                "Coverage, interval scores, tails and offense-lineup evidence reviewed separately",
                "Explicit manual promotion; passing a numeric check never changes the default",
            ],
            limitations=[
                "Historical development seasons have been reused; "
                "prospective validation is pending",
                "Direct point and component models are evaluated separately, not forced to agree",
                "Inherited v3.2 K/DST and mixed-timing lineup uncertainty limitations remain",
            ],
        ),
    )
    write(
        output / "functionality.json",
        dict(
            weekly_points="v3.2 preserved",
            weekly_distributions="chronologically selected mean-preserving v4 calibration",
            components="live newly fitted weekly/remaining/season component forecasts",
            long_horizons="v3-based fixed-origin residual models; no v2 anchor",
            long_horizon_uncertainty="empirical errors of whole-horizon totals",
            rankings="recomputed from each horizon total",
            lineup_dependence="v3.2 frozen dependence retained; matched evaluation required",
            preseason="Original v3 preseason forecasts retained at origin zero with source labels",
            refresh="paired capture after v3.2; no post-deadline forecast backfill",
            replay="sealed inputs and numerical source hashes",
            retraining="separate candidate forecasts; fixed-weight v4 is published",
        ),
    )
    return seal(
        output,
        dict(
            generated_at=now().isoformat(),
            base_campaign=BASE,
            forecast_season=2026,
            status="prospective_candidate",
            default_changed=False,
            dependencies=dependencies,
            inference_sources=sources,
            environment=environment(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--horizons", type=Path, default=DEFAULT_HORIZONS)
    parser.add_argument("--uncertainty", type=Path, default=DEFAULT_UNCERTAINTY)
    args = parser.parse_args()
    integrate(args.output, args.horizons, args.uncertainty)
    print(args.output)


if __name__ == "__main__":
    main()
