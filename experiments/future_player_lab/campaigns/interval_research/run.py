"""Run a sealed, chronological investigation into the cost of interval width."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..weekly_v3.data import ROOT
from ..weekly_v3.run import DEFAULT_OUTPUT as V3_RUN
from ..weekly_v31 import calibration as v31
from ..weekly_v31.artifacts import digest, now, publish_directory, seal, verify, write
from ..weekly_v31.run import DEFAULT_LOG
from ..weekly_v31.run import DEFAULT_RUN as V31_RUN
from . import methods

DEFAULT_RUN = ROOT / "experiments/future_player_lab/runs/opening_drive_width_research_20261013_002"
DEFAULT_REPORT = ROOT / "data/outputs/opening_drive_width_research_2026"


def protocol():
    return dict(
        created_at=now().isoformat(),
        question="Can we retain coverage gains with less interval cost?",
        fixed_candidates=list(methods.FIXED),
        adaptive_candidates=list(methods.SELECTORS),
        player_evaluation_years=list(range(2021, 2026)),
        lineup_evaluation_years=list(range(2023, 2026)),
        calibration="Earlier out-of-time forecasts, five seasons; recent variant uses two",
        selection="Two earlier years of chronological candidate validation, equal-year scores",
        selection_score="Interval score averaged with alpha weights across 80% and 90%; lineup 80%",
        guarded_selector="Prior equal-year coverage >= nominal minus 2 percentage points per level",
        selection_support="100 validation rows per method/cohort; position then v3.1 fallback",
        fit_support="200 player rows per group; 100 lineup rows; position fallback",
        fine_groups=["under_1", "1_to_5", "5_to_10", "10_plus"],
        interpretation="Historical development; reused years and multiple comparisons; "
        "no pristine holdout, coverage guarantee, or automatic promotion",
        bootstrap="Paired season blocks, 5000 draws, descriptive 95% interval; very few seasons",
        numerical_revision="002: stabilize integer endpoint ranks against floating-point error; "
        "001 is retained as the initial run",
        preserved="Point predictions, scoring contract, v3/v3.1 models and prospective log",
        references=["https://arxiv.org/abs/1905.03222", "https://arxiv.org/abs/2005.12881"],
    )


def evaluate(frame, *, lineup=False):
    levels = (80,) if lineup else (80, 90)
    first = 2022 if lineup else 2019
    evaluation_start = 2023 if lineup else 2021
    minimum = 100 if lineup else 200
    previous, evaluated, fits, choices = [], [], [], []
    for year in range(first, 2026):
        test = frame[frame.season.eq(year)].copy()
        if test.empty:
            continue
        past = frame[frame.season.lt(year)]
        candidates = []
        for method in methods.FIXED:
            bundle = methods.fit(past, year, method, levels=levels, min_rows=minimum, lineup=lineup)
            prediction = methods.apply(test, bundle)
            prediction["method"] = method
            prediction["chosen_method"] = method
            candidates.append(prediction)
            fits.append(bundle)
        candidates = pd.concat(candidates, ignore_index=True)
        if year >= evaluation_start:
            validation = pd.concat(previous, ignore_index=True) if previous else candidates.iloc[:0]
            for guarded in (False, True):
                policy = methods.select(validation, year, levels=levels, guarded=guarded)
                choices.append(policy)
                evaluated.append(methods.apply_selection(candidates, policy))
            evaluated.append(candidates)
        previous.append(candidates)
        print(
            f"{'lineup' if lineup else 'player'} intervals {year}: {len(test):,} rows", flush=True
        )
    output = pd.concat(evaluated, ignore_index=True)
    if output.duplicated(["row_id", "method"]).any():
        raise ValueError("Duplicate forecast/method identities")
    original = frame.set_index("row_id")
    np.testing.assert_array_equal(output.prediction, original.loc[output.row_id].prediction)
    # Frozen v3 distribution quantiles remain byte-for-byte numeric equivalents.
    for column in (c for c in frame if c.startswith("q") and c[1:].isdigit()):
        np.testing.assert_array_equal(output[column], original.loc[output.row_id, column])
    final_fits = {
        method: methods.fit(
            frame[frame.season.lt(2026)],
            2026,
            method,
            levels=levels,
            min_rows=minimum,
            lineup=lineup,
        )
        for method in methods.FIXED
    }
    final_policies = [
        methods.select(pd.concat(previous, ignore_index=True), 2026, levels=levels, guarded=guarded)
        for guarded in (False, True)
    ]
    return output, dict(
        folds=fits, selection=choices, final_fits=final_fits, final_policies=final_policies
    )


def cohorts(frame):
    yield "all", frame
    if frame.position.iloc[0] == "LINEUP":
        for name, group in frame.groupby("design"):
            yield str(name), group
        for name, group in frame.groupby("has_qb_receiver_stack"):
            yield f"stack_{name}", group
    else:
        for name, group in frame.groupby(frame.prediction.map(v31.workload)):
            yield str(name), group
        yield "projected_5_plus", frame[frame.prediction.ge(5)]
        for name, group in frame.groupby(
            frame.prediction.map(lambda x: methods.group_name(x, True))
        ):
            if name in ("under_1", "1_to_5"):
                yield name, group
        if "player_history_count" in frame:
            yield "history_under_8", frame[frame.player_history_count.lt(8)]
            yield "history_32_plus", frame[frame.player_history_count.ge(32)]


def metrics(frame, *, lineup=False):
    rows = []
    for (position, method), group in frame.groupby(["position", "method"]):
        for cohort, part in cohorts(group):
            for season, subset in [("all", part), *list(part.groupby("season"))]:
                valid = subset[subset.target_points.notna()]
                if valid.empty:
                    continue
                for level in (80,) if lineup else (80, 90):
                    values = methods.components(valid, level)
                    row = dict(
                        position=position,
                        method=method,
                        cohort=cohort,
                        season=str(season),
                        level=level,
                        n=len(valid),
                        **{key: float(value.mean()) for key, value in values.items()},
                    )
                    row["coverage"] = row.pop("covered")
                    row["interval_bias"] = float(
                        (
                            (valid[f"lower_{level}"] + valid[f"upper_{level}"]) / 2
                            - valid.target_points
                        ).mean()
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def paired_diagnostics(frame, *, lineup=False):
    """Decompose each improvement versus v3.1 and retain season-cluster comparisons."""
    baseline = frame[frame.method.eq("v31")].set_index("row_id")
    summaries, cases = [], []
    for (position, method), group in frame.groupby(["position", "method"]):
        if method == "v31":
            continue
        for cohort, part in cohorts(group):
            valid = part[part.target_points.notna()]
            if valid.empty:
                continue
            old = baseline.loc[valid.row_id]
            for level in (80,) if lineup else (80, 90):
                current, before = methods.components(valid, level), methods.components(old, level)
                differences = {
                    key: current[key].astype(float) - before[key].astype(float)
                    for key in ("interval_score", "width", "below_penalty", "above_penalty")
                }
                annual = pd.DataFrame(
                    dict(season=valid.season.to_numpy(), difference=differences["interval_score"])
                )
                values = annual.groupby("season").difference.mean().to_numpy()
                rng = np.random.default_rng(20261013)
                boot = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
                summaries.append(
                    dict(
                        position=position,
                        method=method,
                        cohort=cohort,
                        level=level,
                        n=len(valid),
                        **{
                            f"delta_{key}": float(value.mean())
                            for key, value in differences.items()
                        },
                        coverage_change=float(current["covered"].mean() - before["covered"].mean()),
                        improved_rows=float((differences["interval_score"] < 0).mean()),
                        newly_covered=float((current["covered"] & ~before["covered"]).mean()),
                        lost_coverage=float((~current["covered"] & before["covered"]).mean()),
                        season_equal_delta=float(values.mean()),
                        seasons=len(values),
                        seasons_improved=int((values < 0).sum()),
                        ci95_lower=float(np.quantile(boot, 0.025)),
                        ci95_upper=float(np.quantile(boot, 0.975)),
                    )
                )
                if (
                    cohort == "all"
                    and level == 80
                    and method in ("v3", "tail_additive", "selective_guarded")
                ):
                    example = valid[
                        [
                            c
                            for c in (
                                "row_id",
                                "player_name",
                                "position",
                                "season",
                                "week",
                                "prediction",
                                "target_points",
                                "method",
                            )
                            if c in valid
                        ]
                    ].copy()
                    example["score_delta_vs_v31"] = differences["interval_score"]
                    example["v31_lower"] = old.lower_80.to_numpy()
                    example["v31_upper"] = old.upper_80.to_numpy()
                    example["candidate_lower"] = valid.lower_80.to_numpy()
                    example["candidate_upper"] = valid.upper_80.to_numpy()
                    cases.extend(
                        [
                            example.nsmallest(5, "score_delta_vs_v31"),
                            example.nlargest(5, "score_delta_vs_v31"),
                        ]
                    )
    return pd.DataFrame(summaries), pd.concat(cases, ignore_index=True)


def baseline_diagnosis(frame):
    """Explain v3.1's added cost relative to original v3, separately from model search."""
    original = frame[frame.method.eq("v3")].set_index("row_id")
    output = []
    for position, group in frame[frame.method.eq("v31")].groupby("position"):
        for cohort, part in cohorts(group):
            valid = part[part.target_points.notna()]
            if valid.empty:
                continue
            old = original.loc[valid.row_id]
            new, before = methods.components(valid, 80), methods.components(old, 80)
            extra_lower = old.lower_80.to_numpy() - valid.lower_80.to_numpy()
            extra_upper = valid.upper_80.to_numpy() - old.upper_80.to_numpy()
            output.append(
                dict(
                    position=position,
                    cohort=cohort,
                    n=len(valid),
                    coverage_v3=float(before["covered"].mean()),
                    coverage_v31=float(new["covered"].mean()),
                    added_width=float((new["width"] - before["width"]).mean()),
                    saved_below_penalty=float(
                        (before["below_penalty"] - new["below_penalty"]).mean()
                    ),
                    saved_above_penalty=float(
                        (before["above_penalty"] - new["above_penalty"]).mean()
                    ),
                    net_score_cost=float((new["interval_score"] - before["interval_score"]).mean()),
                    v3_below_miss_rate=float(before["miss_below"].mean()),
                    v3_above_miss_rate=float(before["miss_above"].mean()),
                    widened_already_covered_fraction=float(
                        (before["covered"] & ((extra_lower + extra_upper) > 0)).mean()
                    ),
                    zero_outcome_fraction=float(valid.target_points.eq(0).mean()),
                )
            )
    return pd.DataFrame(output)


def inventory(roots):
    return {
        str(path): digest(path)
        for root in roots
        for path in Path(root).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def run(output=DEFAULT_RUN, report_dir=DEFAULT_REPORT):
    from .report import report

    output, report_dir = Path(output), Path(report_dir)
    if output.exists() or report_dir.exists():
        raise FileExistsError("Use fresh research output and report directories")
    roots = [V3_RUN, V31_RUN, DEFAULT_LOG, ROOT / "data/campaigns/opening_drive_v3"]
    preserved = inventory(roots)
    verify(V31_RUN)
    input_manifest = json.loads((V3_RUN / "manifest.json").read_text())
    player_path = V3_RUN / "selected_predictions.parquet"
    if digest(player_path) != input_manifest["artifact_hashes"][player_path.name]:
        raise ValueError("V3 input forecast checksum changed")
    players = pd.read_parquet(player_path)
    players = players[players.season.le(2025)].copy()
    if not players.calibration_max_year.lt(players.season).all():
        raise ValueError("Base distributions must be out of time")
    lineups = pd.read_parquet(V3_RUN / "lineups/synthetic_lineups.parquet").rename(
        columns={"actual": "target_points"}
    )
    lineups["position"], lineups["row_id"] = "LINEUP", np.arange(len(lineups))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".width-", dir=output.parent) as tmp:
        staging = Path(tmp) / "run"
        staging.mkdir()
        write(staging / "protocol.json", protocol())  # Written before candidate evaluation.
        shutil.copytree(
            Path(__file__).parent, staging / "source", ignore=shutil.ignore_patterns("__pycache__")
        )
        players.to_parquet(staging / "input_players.parquet", index=False)
        lineups.to_parquet(staging / "input_lineups.parquet", index=False)
        scores, pairs, diagnoses = [], [], []
        for kind, frame in (("players", players), ("lineups", lineups)):
            predictions, receipt = evaluate(frame, lineup=kind == "lineups")
            predictions.to_parquet(staging / f"{kind}_predictions.parquet", index=False)
            write(staging / f"{kind}_calibration.json", receipt)
            table = metrics(predictions, lineup=kind == "lineups")
            comparisons, examples = paired_diagnostics(predictions, lineup=kind == "lineups")
            diagnosis = baseline_diagnosis(predictions)
            for value in (table, comparisons, diagnosis):
                value["kind"] = kind
            scores.append(table)
            pairs.append(comparisons)
            diagnoses.append(diagnosis)
            examples.to_csv(staging / f"{kind}_examples.csv", index=False)
            # Reproduce the locked v3.1 research forecasts on exactly the shared rows.
            locked = pd.read_parquet(
                V31_RUN / f"{'player' if kind == 'players' else 'lineup'}_predictions.parquet"
            )
            locked = locked[
                locked.interval_method.eq("position_workload" if kind == "players" else "position")
            ]
            keys = (
                ["season", "week", "player_id"]
                if kind == "players"
                else ["season", "week", "design", "team_id"]
            )
            local = predictions[predictions.method.eq("v31")].set_index(keys)
            locked = locked.set_index(keys).loc[local.index]
            for column in ("prediction", "lower_80", "upper_80"):
                np.testing.assert_array_equal(local[column], locked[column])
        tables = dict(
            scores=pd.concat(scores, ignore_index=True),
            comparisons=pd.concat(pairs, ignore_index=True),
            diagnosis=pd.concat(diagnoses, ignore_index=True),
        )
        for name, table in tables.items():
            table.to_csv(staging / f"{name}.csv", index=False)
        report(staging, **tables)
        if preserved != inventory(roots):
            raise ValueError("Protected v3/v3.1 artifacts changed during research")
        write(
            staging / "preservation.json",
            dict(files=len(preserved), unchanged=True, sha256=preserved),
        )
        seal(
            staging,
            dict(
                status="completed_research",
                version="width_investigation_1",
                completed_at=now().isoformat(),
                point_models_refit=False,
                baseline_reproduction="exact",
                protected_files=len(preserved),
                packages={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow")},
                v3_source_manifest_sha256=digest(V3_RUN / "manifest.json"),
                v31_source_manifest_sha256=digest(V31_RUN / "manifest.json"),
            ),
        )
        publish_directory(staging, output)
    report_dir.mkdir(parents=True)
    for name in (
        "index.html",
        "scores.csv",
        "comparisons.csv",
        "diagnosis.csv",
        "protocol.json",
        "manifest.json",
        "players_examples.csv",
        "lineups_examples.csv",
    ):
        shutil.copy2(output / name, report_dir / name)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(run(args.output, args.report))
