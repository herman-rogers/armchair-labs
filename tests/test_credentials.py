"""Unit tests for ESPN credential storage.

The behaviour that matters: the auth flow must never flatten a `.env` the user has
been editing, and it must never leave full league access world-readable.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from engine.espn.credentials import EspnCredentials, clear, read, write


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    return tmp_path / ".env"


def sample(**overrides) -> EspnCredentials:
    return EspnCredentials(**{"espn_s2": "AEB" + "x" * 300, "swid": "{ABC-123}", **overrides})


class TestRead:
    def test_missing_file_reads_as_none(self, env_file: Path) -> None:
        assert read(env_file) is None

    def test_reads_both_cookies(self, env_file: Path) -> None:
        env_file.write_text("ESPN_S2=abc123\nESPN_SWID={XYZ}\n")
        found = read(env_file)

        assert found is not None
        assert found.espn_s2 == "abc123"
        assert found.swid == "{XYZ}"

    def test_one_cookie_alone_is_not_enough(self, env_file: Path) -> None:
        """Both are required for private-league access; half a credential is none."""
        env_file.write_text("ESPN_S2=abc123\n")
        assert read(env_file) is None

    def test_comments_and_blanks_are_ignored(self, env_file: Path) -> None:
        env_file.write_text("# a comment\n\nESPN_S2=abc\n\n  # another\nESPN_SWID={X}\n")
        assert read(env_file) is not None

    def test_quotes_are_stripped(self, env_file: Path) -> None:
        env_file.write_text("ESPN_S2=\"abc\"\nESPN_SWID='{X}'\n")
        found = read(env_file)

        assert found is not None
        assert found.espn_s2 == "abc"
        assert found.swid == "{X}"

    def test_ids_are_parsed_as_integers(self, env_file: Path) -> None:
        env_file.write_text("ESPN_S2=a\nESPN_SWID=b\nESPN_LEAGUE_ID=12345\nESPN_TEAM_ID=7\n")
        found = read(env_file)

        assert found is not None
        assert found.league_id == 12345
        assert found.team_id == 7

    def test_a_malformed_id_does_not_break_the_read(self, env_file: Path) -> None:
        """A bad league id should degrade to "not set", not take the cookies down."""
        env_file.write_text("ESPN_S2=a\nESPN_SWID=b\nESPN_LEAGUE_ID=not-a-number\n")
        found = read(env_file)

        assert found is not None
        assert found.league_id is None

    def test_environment_overrides_the_file(
        self, env_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Homelab deployments inject env vars rather than shipping a .env."""
        env_file.write_text("ESPN_S2=from_file\nESPN_SWID={FILE}\n")
        monkeypatch.setenv("ESPN_S2", "from_env")

        found = read(env_file)
        assert found is not None
        assert found.espn_s2 == "from_env"
        assert found.swid == "{FILE}"


class TestWrite:
    def test_creates_the_file(self, env_file: Path) -> None:
        write(sample(), env_file)
        assert read(env_file) == sample()

    def test_unrelated_settings_survive(self, env_file: Path) -> None:
        """It is the user's file. An auth flow has no business flattening it."""
        env_file.write_text("# my notes\nDATA_DIR=/somewhere\nESPN_S2=old\nESPN_SWID=old\n")
        write(sample(), env_file)

        text = env_file.read_text()
        assert "# my notes" in text
        assert "DATA_DIR=/somewhere" in text

    def test_existing_keys_are_replaced_in_place(self, env_file: Path) -> None:
        env_file.write_text("ESPN_S2=old\nZZZ=keep\nESPN_SWID=old\n")
        write(sample(), env_file)

        lines = [line for line in env_file.read_text().splitlines() if line]
        assert sum(1 for line in lines if line.startswith("ESPN_S2=")) == 1
        assert lines.index("ZZZ=keep") == 1, "surrounding order should be preserved"

    def test_new_keys_are_appended(self, env_file: Path) -> None:
        env_file.write_text("DATA_DIR=/somewhere\n")
        write(sample(league_id=999), env_file)

        found = read(env_file)
        assert found is not None
        assert found.league_id == 999

    def test_the_file_is_not_world_readable(self, env_file: Path) -> None:
        """These cookies grant full league access under the user's account."""
        env_file.write_text("PLACEHOLDER=1\n")
        env_file.chmod(0o644)
        write(sample(), env_file)

        mode = stat.S_IMODE(env_file.stat().st_mode)
        assert mode == 0o600, f"expected 600, got {mode:o}"

    def test_optional_ids_are_omitted_when_unset(self, env_file: Path) -> None:
        write(sample(), env_file)
        assert "ESPN_LEAGUE_ID" not in env_file.read_text()


class TestClear:
    def test_removes_only_the_cookies(self, env_file: Path) -> None:
        env_file.write_text("DATA_DIR=/x\nESPN_S2=a\nESPN_SWID=b\nESPN_LEAGUE_ID=5\n")
        removed = clear(env_file)

        assert removed == 2
        text = env_file.read_text()
        assert "DATA_DIR=/x" in text
        assert "ESPN_LEAGUE_ID=5" in text, "the league id is not a secret; keep it"
        assert "ESPN_S2" not in text

    def test_clearing_a_missing_file_is_harmless(self, env_file: Path) -> None:
        assert clear(env_file) == 0


class TestMasking:
    def test_the_full_cookie_is_never_shown(self) -> None:
        """`masked()` exists so terminal output and bug reports stay safe to share."""
        credentials = sample()
        masked = credentials.masked()

        assert credentials.espn_s2 not in masked
        assert masked.startswith("espn_s2=AEBxxx")
        assert "303 chars" in masked
