"""Browser-based ESPN login.

ESPN's fantasy API is unofficial and has no token flow — access is granted by two
cookies that a normal browser session sets after a Disney ID login. Rather than asking
someone to open devtools and copy them by hand, this opens a real browser, waits for
the login to complete, and reads the cookies out of the session.

The browser is started by the operating system and attached to afterwards, rather than
launched as a child of this process; see `patron.espn.browser` for why that distinction
decides whether the window can accept a keystroke at all.

The profile persists in the data directory, so the first run needs a real login and
later runs usually find the session still valid and finish without interaction. That
matters because these cookies do eventually expire, and re-auth should be cheap.

Nothing is automated inside the login form. Credentials are typed by the person who
owns them, into an ordinary browser window, exactly as they would be normally — this
only watches for the resulting session.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from patron.espn import browser
from patron.espn.browser import BrowserUnavailableError
from patron.espn.credentials import EspnCredentials

logger = logging.getLogger(__name__)

#: Landing page for the login. The fantasy home shows a login prompt when signed out
#: and lands on the user's teams when signed in, so it works for both states.
LOGIN_URL = "https://www.espn.com/fantasy/football/"

#: Cookies that together grant private-league access.
REQUIRED_COOKIES = ("espn_s2", "SWID")

__all__ = ["BrowserUnavailableError", "LoginResult", "LoginTimeoutError", "login"]

DEFAULT_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 1.0


class LoginTimeoutError(RuntimeError):
    """Raised when the login did not complete before the timeout."""


@dataclass
class LoginResult:
    credentials: EspnCredentials
    #: True when the persisted profile was still signed in and nothing was typed.
    reused_session: bool
    elapsed_seconds: float


def _extract(cookies: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Pull the two cookies we need out of a Playwright cookie list.

    ESPN sets them on `.espn.com`, but the domain has moved before and could again, so
    match on name and prefer the espn.com one rather than pinning a domain.
    """
    found: dict[str, str] = {}
    for name in REQUIRED_COOKIES:
        candidates = [c for c in cookies if c.get("name") == name and c.get("value")]
        if not candidates:
            continue
        espn = [c for c in candidates if "espn.com" in (c.get("domain") or "")]
        found[name] = (espn or candidates)[0]["value"]
    return found


def login(
    profile_dir: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    on_status: object = None,
) -> LoginResult:
    """Open a browser, wait for an ESPN login, and return the session cookies.

    Args:
        profile_dir: Where to persist the browser profile between runs.
        timeout_seconds: How long to wait for the login before giving up.
        on_status: Optional callable taking a status string, for CLI progress.

    Returns:
        The harvested credentials, and whether an existing session was reused.

    Raises:
        BrowserUnavailableError: if Chrome could not be launched.
        LoginTimeoutError: if the cookies never appeared.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise BrowserUnavailableError("Playwright is not installed. Run `uv sync`.") from error

    def status(message: str) -> None:
        logger.info(message)
        if callable(on_status):
            on_status(message)

    started = time.monotonic()
    session = browser.launch(profile_dir)

    with sync_playwright() as playwright:
        connection = playwright.chromium.connect_over_cdp(session.cdp_url)
        try:
            context = connection.contexts[0] if connection.contexts else connection.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(LOGIN_URL, wait_until="domcontentloaded")

            # A persisted session usually restores immediately; give it a moment before
            # telling the user to do anything.
            time.sleep(2.0)
            found = _extract(context.cookies())
            if all(name in found for name in REQUIRED_COOKIES):
                status("Existing ESPN session found — no login needed.")
                return LoginResult(
                    credentials=EspnCredentials(espn_s2=found["espn_s2"], swid=found["SWID"]),
                    reused_session=True,
                    elapsed_seconds=time.monotonic() - started,
                )

            # Land on the login form rather than the marketing page, so there is one
            # less thing to hunt for. Best-effort: if the affordance moves, the status
            # message below still says what to do.
            opened_form = False
            try:
                page.get_by_text("Log In", exact=True).first.click(timeout=8000)
                opened_form = True
            except Exception:  # noqa: BLE001 - purely a convenience
                logger.debug("could not auto-open the login form", exc_info=True)

            status(
                "Chrome is open at the ESPN sign-in form. Sign in there. Waiting…"
                if opened_form
                else "Chrome is open. Click 'Log In' (top right) and sign in. Waiting…"
            )

            deadline = time.monotonic() + timeout_seconds
            announced = False
            while time.monotonic() < deadline:
                found = _extract(context.cookies())
                if all(name in found for name in REQUIRED_COOKIES):
                    status("Signed in — captured the session.")
                    return LoginResult(
                        credentials=EspnCredentials(espn_s2=found["espn_s2"], swid=found["SWID"]),
                        reused_session=False,
                        elapsed_seconds=time.monotonic() - started,
                    )

                # SWID alone is set for signed-out visitors; espn_s2 is the one that
                # means an actual account session. Saying so avoids the user thinking
                # it has hung when they are halfway through the form.
                if "SWID" in found and not announced:
                    status("Login page reached. Complete sign-in to finish…")
                    announced = True

                if not session.is_up():
                    raise LoginTimeoutError("The browser was closed before sign-in completed.")
                time.sleep(POLL_INTERVAL_SECONDS)

            raise LoginTimeoutError(
                f"No ESPN session appeared within {timeout_seconds}s. "
                "Re-run `patron auth login`, or pass --timeout to wait longer."
            )
        finally:
            connection.close()
            session.quit()
