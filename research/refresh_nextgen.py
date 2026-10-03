"""Compatibility command; production orchestration lives in engine.data.

Use `engine data refresh` for the supported workflow.
"""

import sys

from engine.data.pipeline import main

if __name__ == "__main__":
    main(["refresh", *sys.argv[1:]])
