"""Shared pytest configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def league_config():
    from engine.config.league import get_league

    return get_league()


@pytest.fixture(scope="session")
def scoring_rules():
    from engine.scoring.rules import get_rules

    return get_rules()
