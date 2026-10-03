"""Build and optionally activate an immutable dedicated weekly-points product."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from patron.config.settings import get_settings
from patron.data.nextgen import load_analysis
from patron.data.releases import atomic_json, digest, identifier, load_gold, reference, write_json
from patron.metrics.current_rankings import POSITIONS, make_panel
from patron.metrics.weekly_points import add_context, choose, evaluate, fit_fold, promotion

ROOT = Path(__file__).resolve().parents[1]


def build(version, analysis=None, publish=False):
    data = get_settings().data_dir
    source, manifest = load_analysis(
        data, reference(data / "research" / identifier(analysis)) if analysis else None
    )
    gold = load_gold(data, manifest["gold"]["version"])
    root = data / "research" / identifier(version)
    if root.exists():
        raise ValueError("Weekly products are immutable; choose a new version")
    root.mkdir(parents=True)
    shutil.copy2(ROOT / "research/weekly_points_protocol.md", root / "protocol.md")
    observations = gold.manifest["current_observations"]
    season, cutoff = observations["season"], observations["through_week"]
    profiles = data / "research" / manifest["profiles"]["version"]
    weeks = pl.read_parquet(profiles / "nfl_weeks.parquet")
    schedule = pl.concat([gold.read("nfl_schedule"), gold.read("current_schedule")]).unique(
        "game_id"
    )
    inputs = (
        gold.read("preseason_features"),
        weeks,
        schedule,
        gold.read("players"),
        gold.read("college_annual"),
        gold.read("college_identity_links"),
    )
    week_rows = weeks.filter(pl.col("position").is_in(POSITIONS)).to_dicts()
    schedule_rows = schedule.to_dicts()
    # Panel is sharded by cutoff so large builds are inspectable and bounded on disk.
    for origin in range(18):
        print(f"Building weekly panel through week {origin}", flush=True)
        rows = make_panel(*inputs, season=season, through_week=origin, horizons=("next_week",))
        rows = [
            r
            for r in rows
            if r["schedule_known"]
            and r["end_week"] == origin + 1
            and (r["season"] < season or origin <= cutoff)
        ]
        add_context(rows, week_rows, schedule_rows)
        # Retain the core profile; omit enriched blocks and duplicated career aggregates.
        rows = [
            {k: v for k, v in r.items() if not k.startswith(("e_", "x_career_", "x_recent3_"))}
            for r in rows
        ]
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            root / f"panel_{origin:02}.parquet"
        )
    reports = []
    current = []
    folds = []
    for position in POSITIONS:
        rows = pl.concat(
            [
                pl.read_parquet(p).filter(pl.col("position") == position)
                for p in sorted(root.glob("panel_*.parquet"))
            ],
            how="diagonal_relaxed",
        ).to_dicts()
        earlier = []
        for year in range(2008, season + 1):
            train = [
                r
                for r in rows
                if r["season"] < year and r["actual"] is not None and r["scheduled_games"] > 0
            ]
            test = [r for r in rows if r["season"] == year]
            if not train or not test:
                continue
            reference_recipe, recipe = choose(earlier)
            print(
                f"Weekly {position} {year}: {len(train)} train / {len(test)} test, "
                f"{reference_recipe} vs {recipe}",
                flush=True,
            )
            fits = fit_fold(train, test)
            output = []
            for i, r in enumerate(test):
                row = {
                    k: r[k]
                    for k in (
                        "player_id",
                        "player_display_name",
                        "position",
                        "population",
                        "team",
                        "season",
                        "through_week",
                        "target_week",
                        "role_group",
                        "scheduled_games",
                    )
                }
                row.update(
                    actual=r["actual"],
                    reference_recipe=reference_recipe,
                    selected_recipe=recipe,
                    reference=fits[reference_recipe][i],
                    policy=fits[recipe][i],
                    **{k: v[i] for k, v in fits.items()},
                )
                if year == season:
                    row["actual"] = r["observed_actual"] if r["target_week"] <= cutoff else None
                output.append(row)
            pl.DataFrame(output, infer_schema_length=None).write_parquet(
                root / f"fold_{position}_{year}.parquet"
            )
            folds.append(
                dict(
                    position=position,
                    season=year,
                    training_last_season=max(r["season"] for r in train),
                    selection_last_season=max((r["season"] for r in earlier), default=None),
                    reference_recipe=reference_recipe,
                    selected_recipe=recipe,
                    training_rows=len(train),
                    test_rows=len(test),
                )
            )
            if year < season:
                earlier.extend(output)
            else:
                current.extend(output)
        reports.append(evaluate(earlier, position))
    promotion(reports)
    decisions = {r["position"]: r for r in reports}
    for row in current:
        decision = decisions[row["position"]]
        row["prediction"] = row["policy"] if decision["approved"] else row["reference"]
        row["recipe"] = row["selected_recipe"] if decision["approved"] else row["reference_recipe"]
        row["evidence_status"] = (
            "retrospectively_validated" if decision["approved"] else "reference"
        )
    report = dict(
        version=version,
        season=season,
        through_week=cutoff,
        target="next_calendar_week_league_points",
        generated_at=datetime.now(UTC).isoformat(),
        positions=reports,
        folds=folds,
        limitations=[
            "Retrospective source revisions are not original publication vintages.",
            "K/DST remain prior-score references, not part of the trained offensive model.",
            "No future injury or starting-lineup information is reconstructed.",
            "No demonstrated advantage over ESPN or expert projections.",
        ],
    )
    write_json(root / "report.json", report)
    pl.DataFrame(current, infer_schema_length=None).write_parquet(root / "predictions.parquet")
    for name in [
        "src/patron/metrics/weekly_points.py",
        "src/patron/metrics/current_rankings.py",
        "research/build_weekly_points.py",
    ]:
        target = root / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    product = dict(
        version=version,
        kind="nextgen_weekly_points",
        status="complete",
        season=season,
        through_week=cutoff,
        analysis=reference(source),
        gold=manifest["gold"],
        profiles=manifest["profiles"],
        generated_at=report["generated_at"],
        files={str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", product)
    if publish:
        atomic_json(data / "weekly/current.json", reference(root))
    print(
        json.dumps(
            dict(
                version=version,
                published=publish,
                positions=[
                    dict(position=r["position"], approved=r["approved"], gain=r["mean_season_gain"])
                    for r in reports
                ],
            ),
            indent=2,
        )
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--analysis")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    build(args.version, args.analysis, args.publish)
