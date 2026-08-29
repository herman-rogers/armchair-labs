"""Unit tests for the browser auth flow's pure parts.

The browser itself is not driven here — that needs a real sign-in. What is tested is
the cookie extraction, which is where a domain change or a partial session would
otherwise cause a confusing failure much later.
"""

from __future__ import annotations

from patron.espn.auth import REQUIRED_COOKIES, _extract


def cookie(name: str, value: str, domain: str = ".espn.com") -> dict:
    return {"name": name, "value": value, "domain": domain, "path": "/"}


class TestCookieExtraction:
    def test_finds_both_cookies(self) -> None:
        found = _extract([cookie("espn_s2", "AEB123"), cookie("SWID", "{ABC}")])
        assert found == {"espn_s2": "AEB123", "SWID": "{ABC}"}

    def test_ignores_unrelated_cookies(self) -> None:
        found = _extract([cookie("espn_s2", "A"), cookie("SWID", "{B}"), cookie("_ga", "tracking")])
        assert set(found) == set(REQUIRED_COOKIES)

    def test_a_signed_out_visitor_yields_only_swid(self) -> None:
        """ESPN sets SWID for anonymous visitors too. Only espn_s2 means a real
        account session, which is why the flow waits for both."""
        found = _extract([cookie("SWID", "{ANON}")])

        assert "SWID" in found
        assert "espn_s2" not in found

    def test_empty_values_do_not_count(self) -> None:
        found = _extract([cookie("espn_s2", ""), cookie("SWID", "{B}")])
        assert "espn_s2" not in found

    def test_espn_domain_wins_when_a_name_appears_twice(self) -> None:
        found = _extract(
            [
                cookie("espn_s2", "wrong", domain=".disney.com"),
                cookie("espn_s2", "right", domain=".espn.com"),
                cookie("SWID", "{B}"),
            ]
        )
        assert found["espn_s2"] == "right"

    def test_an_unexpected_domain_is_still_accepted(self) -> None:
        """ESPN has moved its fantasy domains before. Matching on name and preferring
        espn.com is more durable than pinning the domain outright."""
        found = _extract(
            [
                cookie("espn_s2", "value", domain=".espn-new.com"),
                cookie("SWID", "{B}", domain=".espn-new.com"),
            ]
        )
        assert found["espn_s2"] == "value"

    def test_no_cookies_yields_nothing(self) -> None:
        assert _extract([]) == {}
