"""Compatibility command; production orchestration lives in engine.data.

Use `engine data --help` for the supported workflow.
"""

import sys

from engine.data.pipeline import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "gold":
        args[0] = "tables"
    main(args)
