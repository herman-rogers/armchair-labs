"""Build and seal all four corrected raw-input profile methods reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dataset
from .profile_report import finalize_profiles
from .profiles import PROFILE_SPEC, build_profiles
from .run import RUNS

VARIANTS = (
    ("standard_kmeans", "kmeans", "pca"),
    ("standard_gmm", "gaussian_mixture", "pca"),
    ("balanced_kmeans", "kmeans", "group_pca"),
    ("balanced_gmm", "gaussian_mixture", "group_pca"),
)


def build_library(out, *, years=(2021, 2022, 2023, 2024, 2025), threads=4, seed=20261003):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "horizons": ["season", "remaining"],
        "evaluation_years": list(years),
        "threads": threads,
        "seed": seed,
        "variants": VARIANTS,
        "descriptive_k": [3, 5, 8],
        "chronological_k": 5,
        "geometry": PROFILE_SPEC,
        "source": "checksum-verified corrected default gpu_raw_inputs_001 cache",
    }
    (out / "build_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for horizon in config["horizons"]:
        dataset = load_dataset(horizon)
        for label, method, kind in VARIANTS:
            print(f"Building {horizon}/{label}", flush=True)
            build_profiles(
                dataset,
                out / label / horizon,
                years,
                threads=threads,
                seed=seed,
                method=method,
                representation_spec={**PROFILE_SPEC, "kind": kind},
                progress=lambda message: print(message, flush=True),
            )
    implementation = out / "implementation"
    implementation.mkdir(exist_ok=True)
    (implementation / "profile_build.py").write_bytes(Path(__file__).read_bytes())
    return finalize_profiles(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    print(
        build_library(RUNS / args.run_id, years=args.years, threads=args.threads, seed=args.seed)[
            "status"
        ]
    )
