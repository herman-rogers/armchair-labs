"""Track v3.2 and its v4 successor at the same injury/pregame cadence."""

import json

from ..rescore import REGISTRY, latest_raw_snapshot
from ..weekly_v32.shadow import track as track_baseline
from .artifacts import DEFAULT_INTEGRATION
from .prospective import track


def main():
    snapshot = latest_raw_snapshot()
    baseline = track_baseline(snapshot, REGISTRY)
    candidate = (
        track(snapshot, REGISTRY) if DEFAULT_INTEGRATION.exists() else {"status": "not_fitted"}
    )
    print(json.dumps(dict(v32=baseline, v4=candidate), indent=2))


if __name__ == "__main__":
    main()
