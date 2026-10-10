"""League scoring as DuckDB SQL expressions, generated from scoring.yaml.

The YAML stays the single source of truth; this module only translates its numbers
into SQL so the observation builder scores every week inside DuckDB.
"""

from __future__ import annotations

from engine.scoring.rules import ScoringRules, get_rules


def _c(name: str) -> str:
    return f"coalesce({name}, 0)"


def component_points_sql(rules: ScoringRules | None = None, prefix: str = "") -> str:
    """Points from the stat line: passing, rushing, receiving, fumbles, return scores."""
    rules = rules or get_rules()
    p = prefix
    terms = [
        f"{_c(p + 'passing_yards')} * {rules.passing.points_per_yard!r}",
        f"{_c(p + 'passing_tds')} * {rules.passing.touchdown!r}",
        f"{_c(p + 'passing_interceptions')} * {rules.passing.interception!r}",
        f"{_c(p + 'passing_2pt_conversions')} * {rules.passing.two_point_conversion!r}",
        f"{_c(p + 'rushing_yards')} * {rules.rushing.points_per_yard!r}",
        f"{_c(p + 'rushing_tds')} * {rules.rushing.touchdown!r}",
        f"{_c(p + 'rushing_2pt_conversions')} * {rules.rushing.two_point_conversion!r}",
        f"{_c(p + 'receptions')} * {rules.receiving.points_per_reception!r}",
        f"{_c(p + 'receiving_yards')} * {rules.receiving.points_per_yard!r}",
        f"{_c(p + 'receiving_tds')} * {rules.receiving.touchdown!r}",
        f"{_c(p + 'receiving_2pt_conversions')} * {rules.receiving.two_point_conversion!r}",
        f"({_c(p + 'rushing_fumbles_lost')} + {_c(p + 'receiving_fumbles_lost')} "
        f"+ {_c(p + 'sack_fumbles_lost')}) * {rules.misc.fumble_lost!r}",
        f"{_c(p + 'special_teams_tds')} * {rules.misc.special_teams_touchdown!r}",
    ]
    # DuckDB parses decimal literals as DECIMAL; score in DOUBLE like the Polars path.
    return "CAST(" + " + ".join(terms) + " AS DOUBLE)"


def touchdown_bonus_sql(rules: ScoringRules | None = None, column: str = "yards_gained") -> str:
    """Highest matching distance bracket only: a 55-yard score pays 3, never 2 + 3."""
    rules = rules or get_rules()
    ordered = sorted(rules.bonuses.brackets, key=lambda b: b.min_yards, reverse=True)
    clauses = []
    for bracket in ordered:
        condition = f"{column} >= {bracket.min_yards}"
        if bracket.max_yards is not None:
            condition += f" AND {column} <= {bracket.max_yards}"
        clauses.append(f"WHEN {condition} THEN {float(bracket.points)!r}")
    return "CAST(CASE " + " ".join(clauses) + " ELSE 0.0 END AS DOUBLE)"


def position_override_sql(overrides: dict[str, str], column: str = "position") -> str:
    """League-configured identity corrections, applied after roster repair."""
    if not overrides:
        return column
    clauses = " ".join(f"WHEN player_id = '{pid}' THEN '{pos}'" for pid, pos in overrides.items())
    return f"CASE {clauses} ELSE {column} END"
