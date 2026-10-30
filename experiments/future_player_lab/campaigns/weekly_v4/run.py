"""Run the complete isolated v4 experiment in a fresh directory."""

import argparse
from pathlib import Path

from ..weekly_v31.artifacts import digest, write
from . import horizons, integration, research_uncertainty
from .report import report


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    horizons.run(output / "horizons")
    research_uncertainty.run(output / "uncertainty")
    integration.integrate(output / "integration", output / "horizons", output / "uncertainty")
    write(
        output / "run.json",
        {
            name: dict(path=str(output / name), sha256=digest(output / name / "manifest.json"))
            for name in ("horizons", "uncertainty", "integration")
        },
    )
    report(output / "integration", output / "prospective", output / "review.html")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(run(parser.parse_args().output))
