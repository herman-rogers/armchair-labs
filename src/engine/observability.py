"""Logging configuration.

Under uvicorn, a library's loggers are silent by default: uvicorn installs its own
handlers and nothing attaches ours. That meant the server ran with no record of when
it fetched from ESPN, how the crosswalk join went, or when it served stale data after
a failure — precisely the events worth having when something looks wrong at 11am on a
waiver morning.

This attaches one handler to the `engine` logger tree and leaves everything else
alone, so our lines appear next to uvicorn's access log without duplicating them.
"""

from __future__ import annotations

import logging
import sys

ROOT_LOGGER = "engine"
LOG_FORMAT = "%(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO, stream=sys.stderr) -> logging.Logger:
    """Attach a handler to the `engine` logger tree. Idempotent.

    Args:
        level: Level for our loggers. Everything else keeps its own.
        stream: Where to write. Defaults to stderr, which is where uvicorn writes.

    Returns:
        The configured `engine` logger.
    """
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)

    # Never stack handlers: a reload, or a second call from the CLI, would otherwise
    # duplicate every line.
    if not any(getattr(handler, "_engine", False) for handler in logger.handlers):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._engine = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    # Our records are emitted by our own handler; letting them propagate as well would
    # print each line twice once uvicorn has configured the root logger.
    logger.propagate = False
    return logger
