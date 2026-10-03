"""Launching a browser the operating system considers a real application.

The obvious approach — let Playwright launch Chrome as a child process — produces a
window you cannot type into on macOS. Playwright's Chrome is a child of the Python
process, which is not a foreground GUI application, so the window renders but never
becomes key and the keyboard keeps talking to the terminal. That is fatal here: the
entire point is for a person to type a password into that window.

So the browser is started through the platform's own launcher (`open` on macOS, which
hands off to LaunchServices) with the remote debugging port enabled, and Playwright
attaches to it afterwards over CDP. The result is an ordinary Chrome that behaves like
one: it takes focus, autofill and password managers work, and 2FA prompts appear where
they normally would.

A side benefit worth having: Chrome started this way is not in automation mode at all,
so `navigator.webdriver` is false without needing a flag to hide it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CHROME_APP_NAME = "Google Chrome"

#: Where Chrome usually lives when it is not on PATH.
CHROME_BINARIES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

STARTUP_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.5


class BrowserUnavailableError(RuntimeError):
    """Raised when no usable browser could be started."""


@dataclass
class BrowserSession:
    """A running browser with its debugging endpoint open."""

    port: int
    profile_dir: Path
    pid: int | None = None

    @property
    def cdp_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_up(self) -> bool:
        return _probe(self.port) is not None

    def quit(self, timeout_seconds: float = 10.0) -> None:
        """Shut down the browser this session started, and only that one.

        Signals the recorded pid rather than pattern-killing, so an ordinary Chrome the
        user happens to have open can never be caught by it. Asks politely first: a
        SIGKILL leaves the profile's session state half-written, which is what a later
        run needs in order to skip the login form.
        """
        if self.pid is None:
            logger.debug("no pid recorded; leaving the browser running")
            return

        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            logger.debug("could not signal the browser", exc_info=True)
            return

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not self.is_up():
                return
            time.sleep(POLL_INTERVAL_SECONDS)

        logger.debug("browser ignored SIGTERM; forcing")
        with contextlib.suppress(OSError):
            os.kill(self.pid, signal.SIGKILL)


def free_port() -> int:
    """An unused localhost port, so a debugger already on 9222 is not disturbed."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _probe(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _find_pid(profile_dir: Path) -> int | None:
    """The pid of the browser owning `profile_dir`, if it can be identified.

    `open` hands the launch to LaunchServices and returns immediately, so the pid has
    to be looked up rather than captured. The pattern deliberately omits the leading
    dashes of `--user-data-dir`: pgrep parses a pattern starting with `-` as its own
    options, which silently matches nothing.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"user-data-dir={profile_dir}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("could not look up the browser pid", exc_info=True)
        return None

    pids = [int(line) for line in result.stdout.split() if line.isdigit()]
    # The first match is the parent; the rest are renderer and GPU helpers, which exit
    # with it.
    return pids[0] if pids else None


def _chrome_binary() -> str | None:
    for candidate in CHROME_BINARIES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("google-chrome") or shutil.which("chromium")


def _launch_args(profile_dir: Path, port: int) -> list[str]:
    return [
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        # A dedicated window, so the sign-in is not buried in the user's own tabs.
        "--new-window",
    ]


def launch(profile_dir: Path, port: int | None = None) -> BrowserSession:
    """Start Chrome with debugging enabled and wait for it to accept connections.

    Args:
        profile_dir: Persisted profile directory. Reusing it is what lets a later
            sign-in skip straight past the form.
        port: Debugging port; an unused one is chosen when omitted.

    Raises:
        BrowserUnavailableError: if Chrome is missing or never came up.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    port = port or free_port()
    args = _launch_args(profile_dir, port)

    if platform.system() == "Darwin":
        # `open` routes through LaunchServices, which is what makes the window
        # focusable. Launching the binary directly here reproduces the original bug.
        command = ["open", "-na", CHROME_APP_NAME, "--args", *args]
    else:
        binary = _chrome_binary()
        if binary is None:
            raise BrowserUnavailableError(
                "No Chrome or Chromium found. Install one, or set ESPN_S2 and "
                "ESPN_SWID in .env by hand."
            )
        command = [binary, *args]

    logger.debug("launching browser: %s", " ".join(command))
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=30)
    except FileNotFoundError as error:
        raise BrowserUnavailableError(
            f"Could not find {CHROME_APP_NAME}. Install it from "
            "https://www.google.com/chrome/, or set ESPN_S2 and ESPN_SWID in .env "
            "by hand."
        ) from error
    except subprocess.SubprocessError as error:
        raise BrowserUnavailableError(f"Could not start the browser: {error}") from error

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        version = _probe(port)
        if version is not None:
            logger.info("browser ready: %s", version.get("Browser", "unknown"))
            return BrowserSession(port=port, profile_dir=profile_dir, pid=_find_pid(profile_dir))
        time.sleep(POLL_INTERVAL_SECONDS)

    raise BrowserUnavailableError(
        f"The browser did not open a debugging port within {STARTUP_TIMEOUT_SECONDS}s. "
        "If a Chrome window opened, close it and try again."
    )
