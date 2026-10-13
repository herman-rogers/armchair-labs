"""Explore v2 component combinations; distinguish hindsight fits from forward tests.

Run after opening_drive_nfl_wins.py:
    uv run python research/opening_drive_nfl_combinations.py
"""

import csv
import json
from hashlib import sha256
from html import escape
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/outputs/opening_drive_nfl_wins_2026"
OUT = SOURCE / "combinations"
GRID = {
    "rushing_td_weight": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4],
    "passing_yards_per_100_weight": [0, 0.1, 0.25, 0.5, 1],
    "rushing_yards_per_100_weight": [0, 0.1, 0.25, 0.5, 1],
    "interception_penalty": [0, 0.25, 0.5, 1, 2],
    "home_bonus": [0, 0.1, 0.25, 0.5],
}


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(score, target, weeks):
    assert len(score) == len(target) == len(weeks)
    correct = (score > 0) == target
    return dict(
        n=len(score),
        correct=int(correct.sum()),
        accuracy=float(correct.mean()),
        r_win=float(np.corrcoef(score, target)[0, 1]),
        auc=float(roc_auc_score(target, score)),
        weekly={str(w): int(correct[weeks == w].sum()) for w in sorted(set(weeks))},
        zero_scores=int((score == 0).sum()),
    )


def main():
    from itertools import product

    OUT.mkdir(parents=True, exist_ok=True)
    games = list(csv.DictReader((SOURCE / "matchups.csv").open()))
    team_rows = list(csv.DictReader((SOURCE / "team_weeks.csv").open()))
    teams = {(int(r["week"]), r["team"]): r for r in team_rows}
    target = np.array([float(g["margin"]) > 0 for g in games])
    weeks = np.array([int(g["week"]) for g in games])
    assert len(games) == len({g["game_id"] for g in games}) == 64
    assert all(float(g["margin"]) != 0 for g in games)
    assert all(int(g["origin_week"]) < int(g["week"]) for g in games)
    features = {
        s.removeprefix("edge_"): np.array([float(g[s]) for g in games])
        for s in games[0]
        if s.startswith("edge_")
    }
    efficiency = {
        "pass_yards_per_attempt": ("passing_yards", "attempts"),
        "rush_yards_per_carry": ("rushing_yards", "carries"),
        "pass_td_rate": ("passing_tds", "attempts"),
        "rush_td_rate": ("rushing_tds", "carries"),
        "interception_rate": ("passing_interceptions", "attempts"),
    }
    for name, (num, den) in efficiency.items():
        values = []
        for g in games:
            home, away = [teams[int(g["week"]), g[f"{side}_team"]] for side in ("home", "away")]
            assert float(home[f"pred_{den}"]) > 0 and float(away[f"pred_{den}"]) > 0
            values.append(
                float(home[f"pred_{num}"]) / float(home[f"pred_{den}"])
                - float(away[f"pred_{num}"]) / float(away[f"pred_{den}"])
            )
        features[name] = np.array(values)

    # Fixed tie rule: fewest departures from TD-only, then smallest departure,
    # then numeric tuple. No held-out outcomes enter parameter selection.
    parameters = list(product(*GRID.values()))
    parameters.sort(
        key=lambda t: (
            sum(v != v0 for v, v0 in zip(t, (1, 0, 0, 0, 0), strict=True)),
            abs(t[0] - 1) + sum(t[1:]),
            t,
        )
    )
    candidate_scores = np.array(
        [
            features["passing_tds"]
            + a * features["rushing_tds"]
            + b * features["passing_yards"] / 100
            + c * features["rushing_yards"] / 100
            - d * features["passing_interceptions"]
            + h
            for a, b, c, d, h in parameters
        ]
    )
    candidate_correct = (candidate_scores > 0) == target
    best = int(candidate_correct.sum(axis=1).argmax())
    baseline = features["offensive_tds"]
    simple = features["passing_tds"] + 1.5 * features["rushing_tds"]
    full_samples = {
        "Original TD rule": baseline,
        "Passing TD + 1.5 × rushing TD (hindsight example)": simple,
        "Best of 4500 combinations (hindsight)": candidate_scores[best],
    }
    in_sample = [dict(name=n, **measure(s, target, weeks)) for n, s in full_samples.items()]
    candidate_metrics = []
    for i, weights in enumerate(parameters):
        candidate_metrics.append(
            dict(
                candidate=i,
                **dict(zip(GRID, weights, strict=True)),
                correct=int(candidate_correct[i].sum()),
            )
        )
    write_csv("all_candidates.csv", candidate_metrics)

    sets = {
        "TD only + fitted threshold": ["offensive_tds"],
        "Passing and rushing TDs": ["passing_tds", "rushing_tds"],
        "TDs and interceptions": ["offensive_tds", "passing_interceptions"],
        "TDs, yards, interceptions": ["offensive_tds", "total_yards", "passing_interceptions"],
        "Seven raw components": [
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
            "passing_interceptions",
            "attempts",
            "carries",
        ],
        "Five efficiency ratios": list(efficiency),
        "TDs plus five efficiency ratios": ["offensive_tds", *efficiency],
    }
    evaluations, fold_records, predictions = [], [], []
    for mode in ("forward", "leave_one_week_out"):
        all_test = weeks >= 2 if mode == "forward" else np.ones(len(weeks), dtype=bool)
        for model in ("Original TD rule", "Grid selected on training weeks", *sets):
            scores = np.full(len(weeks), np.nan)
            for week in range(2 if mode == "forward" else 1, 5):
                train = weeks < week if mode == "forward" else weeks != week
                test = weeks == week
                assert not (train & test).any()
                if mode == "forward":
                    assert max(weeks[train]) < min(weeks[test])
                detail = dict(
                    mode=mode,
                    model=model,
                    test_week=week,
                    train_weeks=sorted(set(weeks[train].tolist())),
                    train_games=int(train.sum()),
                    test_games=int(test.sum()),
                )
                if model == "Original TD rule":
                    scores[test] = baseline[test]
                elif model == "Grid selected on training weeks":
                    chosen = int(candidate_correct[:, train].sum(axis=1).argmax())
                    scores[test] = candidate_scores[chosen, test]
                    detail["weights"] = dict(zip(GRID, parameters[chosen], strict=True))
                else:
                    x = np.column_stack([features[s] for s in sets[model]])
                    # Fixed regularization; scaling and fitting use only training rows.
                    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=0.1))
                    estimator.fit(x[train], target[train])
                    scores[test] = estimator.decision_function(x[test])
                    detail["features"] = sets[model]
                    detail["standardized_coefficients"] = estimator[-1].coef_[0].tolist()
                    detail["intercept"] = float(estimator[-1].intercept_[0])
                detail["test_correct"] = int(((scores[test] > 0) == target[test]).sum())
                fold_records.append(detail)
            assert np.isfinite(scores[all_test]).all()
            evaluations.append(
                dict(
                    mode=mode,
                    model=model,
                    **measure(scores[all_test], target[all_test], weeks[all_test]),
                )
            )
            for i in np.flatnonzero(all_test):
                predictions.append(
                    dict(
                        mode=mode,
                        model=model,
                        game_id=games[i]["game_id"],
                        week=int(weeks[i]),
                        score=float(scores[i]),
                        predicted_home_win=bool(scores[i] > 0),
                        actual_home_win=bool(target[i]),
                    )
                )

    changes = []
    for name, score in list(full_samples.items())[1:]:
        for i, g in enumerate(games):
            if (score[i] > 0) != (baseline[i] > 0):
                changes.append(
                    dict(
                        rule=name,
                        game_id=g["game_id"],
                        week=int(weeks[i]),
                        home=g["home_team"],
                        away=g["away_team"],
                        home_score=g["home_score"],
                        away_score=g["away_score"],
                        new_pick=g["home_team"] if score[i] > 0 else g["away_team"],
                        improved=bool((score[i] > 0) == target[i]),
                    )
                )
    associations = [
        dict(
            feature=k,
            r_win=float(np.corrcoef(v, target)[0, 1]),
            correct=int(((v > 0) == target).sum()),
        )
        for k, v in features.items()
    ]
    result = dict(
        source_hashes={
            n: sha256((SOURCE / n).read_bytes()).hexdigest()
            for n in ("inputs.json", "matchups.csv", "team_weeks.csv")
        },
        grid=GRID,
        candidate_count=len(parameters),
        best_weights=parameters[best],
        best_tied_candidates=int(
            (candidate_correct.sum(axis=1) == candidate_correct[best].sum()).sum()
        ),
        in_sample=in_sample,
        evaluations=evaluations,
        fold_details=fold_records,
        changed_picks=changes,
        feature_associations=associations,
        notes=[
            "Every feature comes from saved predictions, not actual game production.",
            "Grid: passing TD weight fixed at 1; vary rushing TD weight, passing and "
            "rushing yards / 100 weights, interception penalty, and home bonus. "
            "Choose higher team score; exact zero differences choose away in every method.",
            "All-game and simple-example results are hindsight-selected. Searching 4500 "
            "combinations on 64 games creates substantial selection optimism.",
            "Forward: fit Week 1 to predict Week 2, Weeks 1–2 to predict Week 3, "
            "and Weeks 1–3 to predict Week 4. Compare on the same 48 games.",
            "Leave-one-week-out uses the other three weeks, including later weeks, "
            "to predict each omitted week. It is a stability diagnostic, not a prospective test. "
            "Later forecast vintages can reflect earlier outcomes.",
            "Logistic regression uses fixed C=0.1 with StandardScaler fitted only on "
            "training rows. No hyperparameter search. Listed feature sets are exploratory.",
            "Even forward results are development evidence: the dataset has already been "
            "examined, feature families were chosen during exploration, and player models "
            "are reconstructed cutoffs. No untouched or prospective holdout exists here.",
            "The same 32 teams recur, with only four weeks. Stale team assignments, "
            "independent backup forecasts, and missing defense/kicking remain limitations.",
            "A larger rushing TD coefficient is a predictive proxy, not extra NFL scoring "
            "value or evidence that rushing touchdowns cause more wins than passing TDs.",
        ],
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_csv("evaluations.csv", evaluations)
    write_csv("held_out_predictions.csv", predictions)
    write_csv("changed_picks.csv", changes)
    render(result)
    print(
        json.dumps({k: result[k] for k in ("best_weights", "in_sample", "evaluations")}, indent=2)
    )


def render(result):
    def table(rows, fields):
        def cell(value):
            return f"{value:.3f}" if isinstance(value, float) else escape(str(value))

        return (
            "<table><tr>"
            + "".join(f"<th>{escape(label)}</th>" for _, label in fields)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell(r[k])}</td>" for k, _ in fields) + "</tr>"
                for r in rows
            )
            + "</table>"
        )

    rows = [
        dict(r, score=f"{r['correct']}/{r['n']} ({r['accuracy']:.1%})") for r in result["in_sample"]
    ]
    a, b, c, d, home = result["best_weights"]
    terms = ["passing TD"]
    for coefficient, label in (
        (a, "rushing TD"),
        (b, "passing yards / 100"),
        (c, "rushing yards / 100"),
    ):
        if coefficient:
            terms.append(f" + {coefficient:g} × ({label})")
    if d:
        terms.append(f" − {d:g} × interceptions")
    formula = "".join(terms)
    html = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Can combining v2 stats improve NFL picks?</title><style>"
        "body{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 24px;"
        "color:#172033}p,li{line-height:1.6}table{border-collapse:collapse;"
        "width:100%;margin:24px 0}td,th{padding:10px;text-align:left;"
        "border-bottom:1px solid #ddd}th{background:#edf2ff}.note{background:#fff3d6;"
        "padding:16px}section{overflow:auto}</style>"
        "<p><a href='../index.html'>Original component analysis</a></p>"
        "<h1>Can combining v2 stats improve NFL picks?</h1>"
        "<p>2026 Weeks 1–4 · 64 games · projected football components only</p>"
        "<p class='note'>Hindsight combinations improve the recorded score. "
        "Earlier-weeks-only evaluation is the stronger check; compare it with the "
        "original rule on the same 48 games. None of this is a pristine holdout.</p>"
        "<h2>Full-sample exploration</h2>"
        + table(
            rows,
            [
                ("name", "Rule"),
                ("score", "Correct winners"),
                ("r_win", "Correlation with winning"),
                ("weekly", "By week"),
            ],
        )
        + "<p>Simple example: passing TD + 1.5 × rushing TD. Best grid rule: "
        + escape(formula)
        + f". Home-team bonus: {home:g}. Apply each formula to "
        "both teams and favor the larger value. These are ranking scores, not "
        "predicted NFL points. Both rules were selected after seeing results.</p>"
    )
    for mode, heading in (
        ("forward", "Earlier-weeks-only evaluation: Weeks 2–4"),
        ("leave_one_week_out", "Leave-one-week-out stability diagnostic"),
    ):
        rows = [
            dict(r, score=f"{r['correct']}/{r['n']} ({r['accuracy']:.1%})")
            for r in result["evaluations"]
            if r["mode"] == mode
        ]
        html += (
            f"<h2>{heading}</h2><section>"
            + table(
                rows,
                [
                    ("model", "Model / selection rule"),
                    ("score", "Correct winners"),
                    ("weekly", "Correct by test week"),
                ],
            )
            + "</section>"
        )
    html += (
        "<h2>Which picks changed?</h2>"
        + table(
            result["changed_picks"],
            [
                ("rule", "Rule"),
                ("game_id", "Game"),
                ("new_pick", "New pick"),
                ("home_score", "Home score"),
                ("away_score", "Away score"),
                ("improved", "Improved over original"),
            ],
        )
        + "<h2>Method and limitations</h2><ul>"
        + "".join(f"<li>{escape(n)}</li>" for n in result["notes"])
        + "</ul>"
        "<p><a href='results.json'>Metrics, coefficients, folds and provenance</a> · "
        "<a href='all_candidates.csv'>All 4500 candidate combinations</a> · "
        "<a href='held_out_predictions.csv'>Held-out picks</a> · "
        "<a href='changed_picks.csv'>Changed picks</a></p></html>"
    )
    (OUT / "index.html").write_text(html)


if __name__ == "__main__":
    main()
