"""Campaign monitor: every config's ``pregame`` arm, then its ``backfill`` arm.

Which campaigns run, in what order and with which arguments is declared in
``configs/<campaign>.json`` (``tracker.pregame``); see ``rescore.run_trackers``.
Each tracker is isolated, so one failure never blocks the others.
"""

import json

from .rescore import REGISTRY, latest_raw_snapshot, run_trackers

# Backfilled origins per 15-minute pass; each takes a few minutes.
BACKFILL_PER_PASS = 2


def main():
    try:
        snapshot = latest_raw_snapshot()
    except Exception as error:  # noqa: BLE001 - trackers that need it report the failure
        snapshot = error
    context = dict(
        snapshot=snapshot, registry=REGISTRY, allow_network=True, limit=BACKFILL_PER_PASS
    )
    result = dict(pregame=run_trackers("pregame", context))
    # Pregame first; then fill a few missing earlier weeks per pass (see backfill.py).
    result["backfill"] = run_trackers("backfill", context)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
