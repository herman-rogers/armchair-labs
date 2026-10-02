"""Build a complete, versioned player-history release from verified local sources."""

import argparse

from patron.data.player_profiles import build_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    print(f"Building player profiles: {args.version}", flush=True)
    root = build_profiles(args.version, args.season)
    print(f"Published player profiles: {root}", flush=True)


if __name__ == "__main__":
    main()
