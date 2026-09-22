"""Validation against the published draft board.

Phase 1's deliverable is a correctness baseline, and the thing it is measured against
is `data/static/2026_draft_list.md` — the board that was actually drafted from.

The comparison is staged rather than pass/fail on rank alone, because the pipeline is
a chain and a single number at the end cannot say which link broke. In order:

1. **Replacement baselines.** The sharpest check available, and it isolates scoring
   from ranking entirely. Because VOR is exactly `PPG - replacement`, the baselines the
   published board used are recoverable from it: every row satisfies
   `replacement = ppg - vor`, so each position's baseline can be read straight out of
   the artifact and compared against what the pipeline computes.
2. **Coverage.** How many published players the rebuild found at all. A name that fails
   to join is the §8 tax, and it must be visible rather than silently reducing the
   sample the later stages average over.
3. **Per-player PPG**, then **per-player VOR**. PPG first: if scoring is wrong, VOR is
   wrong for a reason that has nothing to do with VOR.
4. **Ranking.** Top-25 membership and rank correlation.
5. **Flags.** Whether the named judgements reproduce.

Exact reproduction is not the bar and should not be treated as one. The published board
was hand-finished on Aug 27 2026 against a play-by-play snapshot that cannot be pinned,
six rows carry manual injury judgement, and one-decimal rounding hides up to 0.05 per
column. Tolerance bands plus rank correlation is the honest test.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from patron.names import normalize

#: `123 | Player Name | POS | TEAM | 1.5 | 12.3 | flags | notes`
_ROW = re.compile(
    r"^\s*(?P<rank>\d+)\s*\|"
    r"\s*(?P<player>[^|]+?)\s*\|"
    r"\s*(?P<position>[A-Z]+)\s*\|"
    r"\s*(?P<team>[A-Z]+)\s*\|"
    r"\s*(?P<vor>-?[\d.]+)\s*\|"
    r"\s*(?P<ppg>-?[\d.]+)\s*\|"
    r"(?P<flags>[^|]*)\|"
    r"(?P<notes>.*)$"
)


def load_fixture_board(path: Path) -> pl.DataFrame:
    """Parse the published Markdown board into a frame.

    Returns columns `rank`, `player`, `position`, `team`, `vor`, `ppg`, `flags`,
    `notes`, and a normalized `key` for joining.
    """
    rows = []
    for line in path.read_text().splitlines():
        match = _ROW.match(line)
        if not match:
            continue
        data = match.groupdict()
        rows.append(
            {
                "fixture_rank": int(data["rank"]),
                "player": data["player"].strip(),
                "position": data["position"],
                "team": data["team"],
                "fixture_vor": float(data["vor"]),
                "fixture_ppg": float(data["ppg"]),
                "fixture_flags": data["flags"].strip(),
                "notes": data["notes"].strip(),
            }
        )

    if not rows:
        raise ValueError(f"no board rows parsed from {path}; has the format changed?")

    return pl.DataFrame(rows).with_columns(
        pl.struct(["player", "position"])
        .map_elements(
            lambda row: f"{normalize(row['player'])}|{row['position']}",
            return_dtype=pl.String,
        )
        .alias("key")
    )


def fixture_replacement_levels(fixture: pl.DataFrame) -> dict[str, float]:
    """Recover the baselines the published board used.

    VOR is `ppg - replacement`, so `replacement = ppg - vor` holds on every row. The
    median across a position absorbs the one-decimal rounding in the source.
    """
    levels: dict[str, float] = {}
    for position in sorted(set(fixture["position"].to_list())):
        rows = fixture.filter(pl.col("position") == position)
        implied = [
            ppg - vor
            for ppg, vor in zip(
                rows["fixture_ppg"].to_list(), rows["fixture_vor"].to_list(), strict=True
            )
        ]
        if implied:
            levels[position] = statistics.median(implied)
    return levels


def _spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation, computed without a scipy dependency."""
    if len(left) < 2:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                result[order[position]] = average
            index = stop + 1
        return result

    left_ranks, right_ranks = ranks(left), ranks(right)
    mean_left = sum(left_ranks) / len(left_ranks)
    mean_right = sum(right_ranks) / len(right_ranks)

    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    spread_left = sum((a - mean_left) ** 2 for a in left_ranks) ** 0.5
    spread_right = sum((b - mean_right) ** 2 for b in right_ranks) ** 0.5

    if spread_left == 0 or spread_right == 0:
        return float("nan")
    return covariance / (spread_left * spread_right)


@dataclass
class Stage:
    """One stage of the comparison."""

    name: str
    passed: bool
    detail: str
    rows: list[str] = field(default_factory=list)

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        lines = [f"[{mark}] {self.name}: {self.detail}"]
        lines.extend(f"        {row}" for row in self.rows)
        return "\n".join(lines)


@dataclass
class FixtureReport:
    stages: list[Stage]

    @property
    def passed(self) -> bool:
        return all(stage.passed for stage in self.stages)

    def render(self) -> str:
        body = "\n".join(stage.render() for stage in self.stages)
        verdict = "all stages passed" if self.passed else "one or more stages failed"
        return f"\nBoard validation against the published 2026 draft board\n\n{body}\n\n{verdict}\n"


def compare_to_fixture(
    board: pl.DataFrame,
    fixture: pl.DataFrame,
    replacement_levels: dict[str, float],
    tolerance_ppg: float = 0.3,
    tolerance_vor: float = 0.5,
    tolerance_baseline: float = 0.2,
    min_coverage: float = 0.90,
    min_spearman: float = 0.95,
    top_n: int = 25,
    worst_shown: int = 10,
) -> FixtureReport:
    """Run the staged comparison and return a report."""
    keyed = board.with_columns(
        pl.struct(["player_display_name", "position"])
        .map_elements(
            lambda row: f"{normalize(row['player_display_name'])}|{row['position']}",
            return_dtype=pl.String,
        )
        .alias("key")
    )
    joined = fixture.join(keyed, on="key", how="left")
    matched = joined.filter(pl.col("ppg").is_not_null())

    stages = [
        _baseline_stage(fixture, replacement_levels, tolerance_baseline),
        _coverage_stage(joined, matched, min_coverage),
    ]
    if matched.height:
        stages.extend(
            [
                _delta_stage(matched, "PPG", "fixture_ppg", "ppg", tolerance_ppg, worst_shown),
                _delta_stage(matched, "VOR", "fixture_vor", "vor", tolerance_vor, worst_shown),
                _ranking_stage(matched, fixture, keyed, top_n, min_spearman),
                _flag_stage(matched, worst_shown),
            ]
        )
    return FixtureReport(stages=stages)


def _baseline_stage(fixture: pl.DataFrame, computed: dict[str, float], tolerance: float) -> Stage:
    expected = fixture_replacement_levels(fixture)
    rows, failures = [], 0
    for position in sorted(expected):
        want = expected[position]
        got = computed.get(position)
        if got is None:
            rows.append(f"{position}: expected {want:.1f}, computed nothing")
            failures += 1
            continue
        delta = got - want
        ok = abs(delta) <= tolerance
        failures += not ok
        rows.append(
            f"{position}: expected {want:.1f}, computed {got:.1f} "
            f"({delta:+.2f}) {'ok' if ok else 'OUT OF TOLERANCE'}"
        )
    return Stage(
        name="replacement baselines",
        passed=failures == 0,
        detail=f"{len(expected) - failures}/{len(expected)} within ±{tolerance}",
        rows=rows,
    )


def _coverage_stage(joined: pl.DataFrame, matched: pl.DataFrame, minimum: float) -> Stage:
    total = joined.height
    found = matched.height
    ratio = found / total if total else 0.0
    missing = (
        joined.filter(pl.col("ppg").is_null()).select("player", "position").head(15).iter_rows()
    )
    return Stage(
        name="coverage",
        passed=ratio >= minimum,
        detail=f"{found}/{total} published players matched ({ratio:.1%})",
        rows=[f"unmatched: {name} ({position})" for name, position in missing],
    )


def _delta_stage(
    matched: pl.DataFrame,
    label: str,
    expected_column: str,
    actual_column: str,
    tolerance: float,
    worst_shown: int,
) -> Stage:
    deltas = matched.with_columns(
        (pl.col(actual_column) - pl.col(expected_column)).alias("delta")
    ).with_columns(pl.col("delta").abs().alias("abs_delta"))

    outside = deltas.filter(pl.col("abs_delta") > tolerance)
    worst = deltas.sort("abs_delta", descending=True).head(worst_shown)

    rows = [
        f"{row['player']} ({row['position']}): published {row[expected_column]:.1f}, "
        f"computed {row[actual_column]:.1f} ({row['delta']:+.2f})"
        for row in worst.iter_rows(named=True)
    ]
    median_delta = deltas["abs_delta"].median()
    median = float(median_delta) if isinstance(median_delta, (int, float)) else 0.0
    return Stage(
        name=f"per-player {label}",
        passed=outside.height == 0,
        detail=(
            f"{matched.height - outside.height}/{matched.height} within ±{tolerance}; "
            f"median |Δ| {median:.2f}"
        ),
        rows=rows,
    )


def _ranking_stage(
    matched: pl.DataFrame,
    fixture: pl.DataFrame,
    board: pl.DataFrame,
    top_n: int,
    min_spearman: float,
) -> Stage:
    published_top = set(fixture.sort("fixture_rank").head(top_n)["key"].to_list())
    computed_top = set(board.sort("rank").head(top_n)["key"].to_list())
    overlap = published_top & computed_top

    correlation = _spearman(
        matched["fixture_rank"].cast(pl.Float64).to_list(),
        matched["rank"].cast(pl.Float64).to_list(),
    )

    missed = fixture.filter(pl.col("key").is_in(list(published_top - computed_top)))
    rows = [
        f"in published top {top_n}, not computed: {row['player']} ({row['position']}) "
        f"#{row['fixture_rank']}"
        for row in missed.iter_rows(named=True)
    ]
    return Stage(
        name="ranking",
        passed=correlation >= min_spearman,
        detail=(
            f"top-{top_n} overlap {len(overlap)}/{top_n}; "
            f"Spearman {correlation:.3f} (min {min_spearman})"
        ),
        rows=rows,
    )


def _flag_stage(matched: pl.DataFrame, worst_shown: int) -> Stage:
    # BUY and TD-luck were removed from the pipeline (no repeatable next-season
    # signal; docs/v2_metrics_review.md §4). The published fixture still carries
    # them, so they are stripped from both sides: the stage now compares only the
    # flag vocabulary the board still produces.
    retired = {"BUY", "TD-luck"}

    def flag_set(value: str | None) -> set[str]:
        return {
            part
            for part in (value or "").split("/")
            if part and not part.endswith("gms") and part not in retired
        }

    disagreements = []
    agreed = 0
    for row in matched.iter_rows(named=True):
        published = flag_set(row["fixture_flags"])
        computed = flag_set(row["flags"])
        if published == computed:
            agreed += 1
        else:
            disagreements.append(
                f"{row['player']} ({row['position']}): published "
                f"{sorted(published) or '-'}, computed {sorted(computed) or '-'}"
            )

    total = matched.height
    ratio = agreed / total if total else 0.0
    return Stage(
        name="flags",
        # Advisory. Flags are thresholds on a continuous metric, so a player sitting a
        # hair either side of 4.0 TD-over-expectation will flip without anything being
        # wrong. Reported, never failed on.
        passed=True,
        detail=f"{agreed}/{total} agree ({ratio:.1%}) — advisory, not a gate",
        rows=disagreements[:worst_shown],
    )
