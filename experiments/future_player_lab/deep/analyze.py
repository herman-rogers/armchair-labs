"""Nested ensemble evaluation, matched production comparisons and readable artifacts."""
# ruff: noqa: E501

import argparse
import json
import shutil
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from ..data import digest
from ..evaluate import interval, paired_interval, score
from ..run import write_json
from .benchmark import EXISTING, prepare_benchmark
from .data import KEYS
from .ensemble import balanced_weights, capture_score, combine, convex_weights, select_policy
from .models import names_for
from .run import LAB, ROOT, validate_data


def load_predictions(source, data, position, horizon, capacity=None):
    protocol = json.loads((source / "protocol.json").read_text())
    config = protocol["config"]
    panel = pl.read_parquet(data / "panel.parquet")
    ix = np.flatnonzero(
        (panel["position"].to_numpy() == position) & (panel["horizon"].to_numpy() == horizon)
    )
    panel = panel[ix]
    years = panel["year"].to_numpy()
    keep = years >= config["warmup_start"]
    y = np.load(data / "targets.npy", mmap_mode="r")[ix, 0][keep]
    metadata = json.loads((data / "features.json").read_text())
    context_cols = [
        metadata["names"].index(n)
        for n in ["context:origin", "static:rookie", "context:history_count"]
    ]
    context = np.load(data / "x.npy", mmap_mode="r")[ix][:, context_cols][keep]
    panel = panel.filter(pl.Series(keep))
    baseline = np.load(data / "baseline.npy", mmap_mode="r")[ix, 0][keep]
    names = ["persistence"] + [n for path in protocol["paths"] for n in names_for(path, config)]
    p = np.full((len(y), len(names)), np.nan)
    p[:, 0] = baseline
    years = panel["year"].to_numpy()
    for year in np.unique(years):
        test = years == year
        for path in protocol["paths"]:
            file = source / "fits" / f"{position}_{horizon}" / f"{year}_{path['name']}.npz"
            diag = json.loads(file.with_suffix(".json").read_text())
            if digest(file) != diag["sha256"] or diag["train_last_year"] >= year:
                raise ValueError("Invalid chronological checkpoint")
            with np.load(file) as saved:
                for n in names_for(path, config):
                    p[test, names.index(n)] = saved[n]
    if not np.isfinite(p).all():
        raise ValueError("Incomplete prediction library")
    capacity = Path(capacity) if capacity is not None else LAB / "runs/deep_capacity_001"
    cap_manifest = json.loads((capacity / "manifest.json").read_text())
    if cap_manifest["status"] != "complete":
        raise ValueError("Adaptive tree-count experiment is not complete")
    cap_protocol = json.loads((capacity / "protocol.json").read_text())
    if cap_protocol["data_sha256"] != digest(data / "manifest.json"):
        raise ValueError("Adaptive tree counts use a different data matrix")
    for engine in cap_protocol["engines"]:
        for view in cap_protocol["views"]:
            name = f"auto_{engine}_{view}"
            column = np.full(len(y), np.nan)
            for year in np.unique(years):
                file = capacity / "fits" / f"{position}_{horizon}" / f"{year}_{name}.npz"
                diag = json.loads(file.with_suffix(".json").read_text())
                if (
                    digest(file) != diag["sha256"]
                    or diag["train_last_year"] >= year
                    or diag["inner_train_last_year"] >= diag["validation_year"]
                ):
                    raise ValueError("Invalid adaptive-capacity checkpoint")
                with np.load(file) as saved:
                    column[years == year] = saved["prediction"]
            p = np.column_stack([p, column])
            names.append(name)
    if horizon == "season":
        from .reuse import history_library

        old_panel, old_y, old_p, _, old_names = history_library(position)
        mapping = panel.join(
            old_panel.select("player_id", "year", "forecast_cutoff_date").with_row_index(
                "archive_row"
            ),
            on=["player_id", "year", "forecast_cutoff_date"],
            how="left",
            validate="1:1",
        )
        if mapping["archive_row"].null_count():
            raise ValueError("Historical library does not cover current preseason task")
        rows = mapping["archive_row"].to_numpy()
        if not np.allclose(y, old_y[rows], rtol=0, atol=1e-7):
            raise ValueError("Historical and new library labels disagree")
        p = np.column_stack([p, old_p[rows, 1:]])
        names += ["archive:" + n for n in old_names[1:]]
    return panel, y, p, context, names


def chronological(panel, y, p, context, names, config):
    """All learnable choices depend only on earlier out-of-year forecasts."""
    years, origins = panel["year"].to_numpy(), panel["origin"].to_numpy()
    exposure = panel["exposure"].to_numpy()
    if panel["position"].n_unique() != 1:
        raise ValueError("An ensemble task must have one position")
    top_k = {"QB": 10, "RB": 20, "WR": 30, "TE": 10}[panel["position"][0]]

    def task_score(actual, prediction, origins):
        return {
            **score(actual, prediction, origins),
            "capture": capture_score(actual, prediction, origins, top_k),
        }

    history, forecasts, metrics, choices, budget, diversity = {}, [], [], [], [], []
    residuals = {n: [] for n in ["policy_mse", "policy_mae", "policy_rank", "policy_capture"]}
    for year in sorted(np.unique(years)):
        if year < config["ensemble_start"]:
            continue
        prior_years = sorted(set(years[years < year]))[-config["inner_seasons"] :]
        if len(prior_years) < 3:
            raise ValueError("Ensembling needs at least three earlier forecast seasons")
        train, test = np.isin(years, prior_years), years == year
        results, details = combine(
            p[train],
            y[train],
            years[train],
            origins[train],
            p[test],
            origins[test],
            context[train],
            context[test],
            names,
            seed=config["seed"],
            top_k=top_k,
        )
        # Search-budget diagnostics are separate fixed policies, evaluated later.
        # Interleave engines and capacities so small budgets aren't one engine only.
        ordered = [0]
        for step in range(3):
            ordered += [
                j
                for j, n in enumerate(names)
                if j and (f"_t{config['tree_checkpoints'][step]}" in n)
            ]
        ordered += [j for j in range(len(names)) if j not in ordered]
        w = balanced_weights(years[train], origins[train])
        for size in [8, 16, 32, 64, len(names)]:
            take = ordered[:size]
            weights = convex_weights(p[train][:, take], y[train], w, 0.1)
            pred = p[test][:, take] @ weights
            results[f"budget_{size}"] = pred
            details[f"budget_{size}"] = {
                "candidate_count": len(take),
                "weights": {names[take[j]]: float(v) for j, v in enumerate(weights) if v > 1e-8},
            }
            budget.append(
                {"year": int(year), "candidates": len(take), **task_score(y[test], pred, origins[test])}
            )
        # Model diversity is measured using prior-season errors only.
        err = p[train] - y[train, None]
        std = err.std(axis=0)
        valid = std > 1e-8
        corr = np.corrcoef(err[:, valid], rowvar=False)
        upper = corr[np.triu_indices(len(corr), 1)]
        diversity.append(
            {
                "year": int(year),
                "prior_years": [int(v) for v in prior_years],
                "median_error_correlation": float(np.median(upper)),
                "minimum_error_correlation": float(np.min(upper)),
            }
        )
        policy_choices = {}
        if year in config["evaluation_years"]:
            for metric, policy in [
                ("mse", "policy_mse"),
                ("mae", "policy_mae"),
                ("ndcg24", "policy_rank"),
                ("capture", "policy_capture"),
            ]:
                chosen, inner, losses = select_policy(history, year, metric)
                results[policy] = results[chosen].copy()
                policy_choices[policy] = {
                    "chosen": chosen,
                    "validation_years": inner,
                    "losses": losses,
                    "composition": details[chosen],
                }
        history[int(year)] = {
            name: task_score(y[test], value, origins[test])
            for name, value in results.items()
            if not name.startswith("policy_")
        }
        choices.append(
            {
                "year": int(year),
                "combination_training_years": [int(v) for v in prior_years],
                "methods": details,
                "selected": policy_choices,
            }
        )
        for name, pred in results.items():
            radius = interval(np.asarray(residuals[name]), 0.2) if name in residuals else np.nan
            if year in config["evaluation_years"]:
                out = panel.filter(pl.Series(test)).select(*KEYS, "name", "population", "end")
                out = out.with_columns(
                    pl.lit(name).alias("model"),
                    pl.Series("actual", y[test]),
                    pl.Series("prediction", pred),
                    pl.Series("low80", pred - radius * exposure[test]).fill_nan(None),
                    pl.Series("high80", pred + radius * exposure[test]).fill_nan(None),
                )
                forecasts.append(out)
                stats = task_score(y[test], pred, origins[test])
                stats.update(
                    year=int(year),
                    model=name,
                    coverage80=float(np.mean(abs(pred - y[test]) <= radius * exposure[test]))
                    if np.isfinite(radius)
                    else None,
                    negative_predictions=int((pred < 0).sum()),
                )
                metrics.append(stats)
            if name in residuals:
                residuals[name].extend(((y[test] - pred) / exposure[test]).tolist())
        if year in config["evaluation_years"]:
            for j, name in enumerate(names):
                stats = task_score(y[test], p[test, j], origins[test])
                stats.update(
                    year=int(year),
                    model="base:" + name,
                    coverage80=None,
                    negative_predictions=int((p[test, j] < 0).sum()),
                )
                metrics.append(stats)
    return pl.concat(forecasts), metrics, choices, budget, diversity


def matched(frame, benchmark):
    if benchmark.select(KEYS).is_duplicated().any():
        raise ValueError("Duplicate published forecast")
    joined = frame.join(benchmark, on=KEYS, how="inner", validate="m:1")
    if joined.is_empty():
        return joined
    if not (joined["end"] == joined["published_end"]).all():
        raise ValueError("Forecast horizons differ")
    if not np.allclose(joined["actual"], joined["published_actual"], rtol=0, atol=1e-7):
        raise ValueError("Published and lab outcomes disagree")
    return joined


def comparison_rows(frame, seed):
    annual, summaries = [], []
    for (position, horizon, model), block in frame.partition_by(
        ["position", "horizon", "model"], as_dict=True
    ).items():
        rows = []
        for (year,), part in block.partition_by("year", as_dict=True).items():
            part = part.sort("player_id")
            a, p, b = [part[c].to_numpy() for c in ["actual", "prediction", "published"]]
            s, r = score(a, p, part["origin"].to_numpy()), score(a, b, part["origin"].to_numpy())
            k = {"QB": 10, "RB": 20, "WR": 30, "TE": 10}[position]
            denominator = np.sort(a)[::-1][:k].sum()
            capture = (
                a[np.argsort(-p, kind="stable")[:k]].sum() / denominator if denominator > 0 else 0.0
            )
            ref_capture = (
                a[np.argsort(-b, kind="stable")[:k]].sum() / denominator if denominator > 0 else 0.0
            )
            row = dict(
                position=position,
                horizon=horizon,
                model=model,
                year=year,
                n=len(a),
                mse=s["mse"],
                reference_mse=r["mse"],
                mae=s["mae"],
                reference_mae=r["mae"],
                ndcg24=s["ndcg24"],
                reference_ndcg24=r["ndcg24"],
                delta_mse=s["mse"] - r["mse"],
                delta_mae=s["mae"] - r["mae"],
                delta_ndcg24=s["ndcg24"] - r["ndcg24"],
                top_k=k,
                capture=float(capture),
                reference_capture=float(ref_capture),
                delta_capture=float(capture - ref_capture),
            )
            rows.append(row)
            annual.append(row)
        diffs = np.array([r["delta_mse"] for r in rows])
        ref = np.mean([r["reference_mse"] for r in rows])
        summaries.append(
            dict(
                position=position,
                horizon=horizon,
                model=model,
                n=sum(r["n"] for r in rows),
                years=len(rows),
                mse=float(np.mean([r["mse"] for r in rows])),
                reference_mse=float(ref),
                delta_mse_pct=float(100 * diffs.mean() / ref),
                delta_mae=float(np.mean([r["delta_mae"] for r in rows])),
                delta_ndcg24=float(np.mean([r["delta_ndcg24"] for r in rows])),
                delta_capture=float(np.mean([r["delta_capture"] for r in rows])),
                mse_gain_ci=paired_interval(-diffs, 5000, seed),
                mae_gain_ci=paired_interval([-r["delta_mae"] for r in rows], 5000, seed),
                ndcg_gain_ci=paired_interval([r["delta_ndcg24"] for r in rows], 5000, seed),
                capture_gain_ci=paired_interval([r["delta_capture"] for r in rows], 5000, seed),
                years_won=int((diffs < 0).sum()),
                mae_years_won=sum(r["delta_mae"] < 0 for r in rows),
                ndcg_years_won=sum(r["delta_ndcg24"] > 0 for r in rows),
                capture_years_won=sum(r["delta_capture"] > 0 for r in rows),
            )
        )
    return annual, summaries


def analyze_task(source_path, data_path, output_path, position, horizon, capacity_path=None):
    source, data, output = Path(source_path), Path(data_path), Path(output_path)
    config = json.loads((source / "protocol.json").read_text())["config"]
    panel, y, p, context, names = load_predictions(source, data, position, horizon, capacity_path)
    with threadpool_limits(limits=1):
        forecasts, metrics, choices, budget, diversity = chronological(
            panel, y, p, context, names, config
        )
    folder = output / f"{position}_{horizon}"
    folder.mkdir(exist_ok=False)
    forecasts.write_parquet(folder / "forecasts.parquet")
    pl.DataFrame(metrics, infer_schema_length=None).write_parquet(folder / "metrics.parquet")
    for name, value in [("choices", choices), ("budget", budget), ("diversity", diversity)]:
        write_json(folder / f"{name}.json", value)
    benchmark = pl.read_parquet(output / "published_reconciled.parquet")
    joined = matched(forecasts, benchmark)
    joined.write_parquet(folder / "matched.parquet")
    write_json(
        folder / "coverage.json",
        {
            "evaluated_candidates": forecasts.select(KEYS).unique().height,
            "matched_published_candidates": joined.select(KEYS).unique().height,
            "published_candidates": pl.read_parquet(data / "published.parquet")
            .filter(
                (pl.col("position") == position)
                & (pl.col("horizon") == horizon)
                & pl.col("year").is_in(config["evaluation_years"])
            )
            .height,
        },
    )
    # A second, explicitly matched library lets the ensemble retain production
    # when useful, rather than forcing a choice among new challengers only.
    attached = (
        panel.with_row_index("row")
        .join(benchmark, on=KEYS, how="inner", validate="1:1")
        .sort("year", "origin", "player_id")
    )
    if attached.height:
        ids = attached["row"].to_numpy()
        if not np.allclose(y[ids], attached["published_actual"], rtol=0, atol=1e-7):
            raise ValueError("Published-library labels disagree")
        extra = ["published", "published_reference"] + ["existing:" + c for c in EXISTING]
        if position == "QB":
            extra += sorted(c for c in attached.columns if c.startswith("existing:points_"))
        if position == "QB" and horizon == "next_four":
            extra.remove("existing:policy")  # Exactly the already included published policy.
        if any(attached[c].null_count() or not attached[c].is_finite().all() for c in extra):
            raise ValueError("Incomplete existing-model out-of-year library")
        augmented = np.column_stack([p[ids], attached.select(extra).to_numpy()])
        library_panel = panel[ids]
        with threadpool_limits(limits=1):
            f, m, c, b, d = chronological(
                library_panel,
                y[ids],
                augmented,
                context[ids],
                names + extra,
                config,
            )
        second = output / "published_library" / f"{position}_{horizon}"
        second.mkdir(parents=True, exist_ok=False)
        f.write_parquet(second / "forecasts.parquet")
        pl.DataFrame(m, infer_schema_length=None).write_parquet(second / "metrics.parquet")
        matched(f, benchmark).write_parquet(second / "matched.parquet")
        for name, value in [("choices", c), ("budget", b), ("diversity", d)]:
            write_json(second / f"{name}.json", value)
        write_json(second / "coverage.json", json.loads((folder / "coverage.json").read_text()))
    return f"{position}/{horizon}: ensemble methods complete"


def render(output):
    summary, comparisons, coverage, choice_counts = [], [], [], Counter()
    for folder in sorted(
        p
        for p in output.iterdir()
        if p.is_dir() and "_" in p.name and (p / "metrics.parquet").exists()
    ):
        position, horizon = folder.name.split("_", 1)
        metrics = pl.read_parquet(folder / "metrics.parquet")
        for (model,), block in metrics.partition_by("model", as_dict=True).items():
            row = dict(
                position=position,
                horizon=horizon,
                model=model,
                years=block.height,
                n=int(block["n"].sum()),
                rmse=float(np.sqrt(block["mse"].mean())),
                mse=block["mse"].mean(),
                mae=block["mae"].mean(),
                ndcg24=block["ndcg24"].mean(),
                capture=block["capture"].mean() if "capture" in block else None,
                spearman=block["spearman"].mean(),
                negative_predictions=int(block["negative_predictions"].sum()),
                coverage80=block["coverage80"].mean(),
            )
            summary.append(row)
        matched_frame = pl.read_parquet(folder / "matched.parquet")
        if matched_frame.height:
            annual, rows = comparison_rows(matched_frame, 20260925)
            pl.DataFrame(annual).write_parquet(folder / "matched_annual.parquet")
            comparisons.extend(rows)
        coverage.append(
            dict(
                position=position,
                horizon=horizon,
                **json.loads((folder / "coverage.json").read_text()),
            )
        )
        for decision in json.loads((folder / "choices.json").read_text()):
            for policy, info in decision["selected"].items():
                choice_counts[policy + ":" + info["chosen"]] += 1
    write_json(output / "summary.json", summary)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "coverage.json", coverage)
    write_json(output / "policy_choice_counts.json", dict(choice_counts))
    pl.DataFrame(summary).write_csv(output / "summary.csv")
    lines = [
        "# Deep ensemble search",
        "",
        "Retrospective research; no production promotion.",
        "",
        "Primary policies choose ensemble methods using three earlier out-of-year evaluations.",
        "Weights and features never fit on the year being evaluated.",
        "",
        "## Matched comparison against the published policy",
        "",
        "Negative error changes are better. MSE percentages are not MAE percentages.",
        "",
        "| Position | Horizon | Policy | n | Δ MSE | Δ MAE | Δ NDCG@24 | Δ Top-K capture | MSE years won |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        if r["model"].startswith("policy_"):
            lines.append(
                f"| {r['position']} | {r['horizon']} | {r['model']} | {r['n']} | "
                f"{r['delta_mse_pct']:+.2f}% | {r['delta_mae']:+.3f} | "
                f"{r['delta_ndcg24']:+.4f} | {r['delta_capture']:+.4f} | {r['years_won']}/{r['years']} |"
            )
    lines += [
        "",
        "Matching checks player identity, position, season, origin, horizon end and actual points.",
        "Coverage counts are saved separately; the lab retains its preseason candidate universe.",
        "The published QB next-four policy is the approved integration; other scopes use their references.",
        "Only next-four and remaining at week 2 have published comparators.",
        "",
        "All historical seasons are reused research. Bootstrap intervals are descriptive and unadjusted.",
        "No claim of exhaustive mathematical optimization or prospective superiority is made.",
    ]
    audit_path = output / "published_target_audit.json"
    if output.name == "published_library":
        audit_path = output.parent / "published_target_audit.json"
    audit_note = ""
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        modern = sum(r["year"] >= 2019 for r in audit["changes"])
        audit_note = (
            f"The comparison re-scores saved forecasts against canonical player points. "
            f"{audit['changed_rows']} historical outcome rows differ because of the original "
            f"weekly-position filter; {modern} are in 2019–2025. Forecast values are unchanged."
        )
        lines += ["", audit_note]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    payload = json.dumps({"scores": summary, "comparisons": comparisons}).replace("</", "<\\/")
    page = """<!doctype html><meta charset="utf-8"><title>Deep ensemble research</title>
<style>body{font:16px system-ui;background:#101824;color:#e6edf7;margin:32px}h1{font-size:30px}
select,input{padding:10px;margin:6px;background:#233247;color:inherit;border:1px solid #53657c}
table{border-collapse:collapse;width:100%;margin-top:20px}td,th{padding:9px;text-align:right;border-bottom:1px solid #344357}
td:first-child,th:first-child{text-align:left}th{cursor:pointer}p{max-width:1000px;line-height:1.5}a{color:#84c4ff}</style>
<h1>Deep ensemble research</h1><p>Chronological feature and ensemble search. Research only.
All controls use the same evaluated players. Select “Published comparison” for matched production evidence.
Negative error changes are better. No prospective confirmation is claimed.</p>
<select id="view"><option value="scores">Model scores</option><option value="comparisons">Published comparison</option></select>
<select id="position"><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select>
<select id="horizon"><option value="season">Preseason</option><option value="next_week">Next week</option>
<option value="next_four">Next four weeks</option><option value="remaining">Remaining season</option></select>
<input id="query" placeholder="Filter model" value="policy_"><p id="count"></p><table><thead id="head"></thead><tbody id="body"></tbody></table>
<p><a href="report.md">Readout</a> · <a href="summary.csv">All model scores</a> · <a href="comparisons.json">Matched comparisons</a> ·
<a href="coverage.json">Population coverage</a></p><script>
const data=PAYLOAD;let sort='mae',direction=1;
function render(){const mode=document.getElementById('view').value;let rows=data[mode].filter(r=>r.position===document.getElementById('position').value&&r.horizon===document.getElementById('horizon').value&&r.model.includes(document.getElementById('query').value));
const keys=mode==='scores'?['model','n','years','rmse','mae','ndcg24','capture','spearman','coverage80','negative_predictions']:['model','n','years','delta_mse_pct','delta_mae','delta_ndcg24','delta_capture','years_won','capture_years_won'];
if(!keys.includes(sort)){sort=mode==='scores'?'mae':'delta_mse_pct';direction=1}
const labels={model:'Model',n:mode==='scores'?'Evaluated rows':'Matched rows',years:'Seasons',rmse:'RMSE (points)',mae:'MAE (points)',ndcg24:'NDCG@24',capture:'Top-K point capture',spearman:'Rank correlation',coverage80:'80% interval coverage',negative_predictions:'Negative forecasts',delta_mse_pct:'MSE change (%)',delta_mae:'MAE change (points)',delta_ndcg24:'NDCG change',delta_capture:'Top-K capture change',years_won:'MSE seasons won',capture_years_won:'Capture seasons won'};
rows.sort((a,b)=>direction*(typeof a[sort]==='string'?a[sort].localeCompare(b[sort]):(a[sort]??Infinity)-(b[sort]??Infinity)));
document.getElementById('count').textContent=mode==='comparisons'&&['season','next_week'].includes(document.getElementById('horizon').value)?'No matching published forecast for this horizon. Choose next four weeks or remaining season.':rows.length+' models';const head=document.getElementById('head'),body=document.getElementById('body');head.replaceChildren();body.replaceChildren();let tr=document.createElement('tr');for(const k of keys){let th=document.createElement('th');th.textContent=labels[k]??k;th.onclick=()=>{direction=sort===k?-direction:1;sort=k;render()};tr.append(th)}head.append(tr);for(const row of rows){tr=document.createElement('tr');for(const k of keys){let td=document.createElement('td');td.textContent=row[k]==null?'—':typeof row[k]==='number'?row[k].toFixed(['n','years','years_won','capture_years_won','negative_predictions'].includes(k)?0:3):row[k];tr.append(td)}body.append(tr)}}
document.querySelectorAll('select,input').forEach(e=>e.addEventListener('input',render));render();</script>"""
    navigation = ""
    if (output / "published_library").exists():
        navigation = '<p><a href="published_library/index.html">Ensembles including existing and published models</a></p>'
    elif output.name == "published_library":
        navigation = '<p><a href="../index.html">New model library</a></p>'
        page = page.replace('<option value="season">Preseason</option>', "")
        page = page.replace('<option value="next_week">Next week</option>', "")
    page = page.replace(
        '<select id="view">', navigation + "<p>" + audit_note + '</p><select id="view">'
    )
    (output / "index.html").write_text(page.replace("PAYLOAD", payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="deep_search_002")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capacity", default="deep_capacity_001")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or any(
        Path(s).name != s or s in {".", ".."} for s in [args.source, args.run_id, args.capacity]
    ):
        raise ValueError("Invalid run name")
    source, output = LAB / "runs" / args.source, LAB / "runs" / args.run_id
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "fits_complete":
        raise ValueError("Base fits are not complete")
    protocol = json.loads((source / "protocol.json").read_text())
    data = LAB / "runs" / protocol["data_run"]
    if validate_data(data) != protocol["data_manifest"]:
        raise ValueError("Prepared data changed since base fitting")
    output.mkdir(exist_ok=False)
    prepare_benchmark(ROOT, data, output)
    input_hashes = {
        str(source / "manifest.json"): digest(source / "manifest.json"),
        str(data / "manifest.json"): digest(data / "manifest.json"),
    }
    audit = json.loads((output / "published_target_audit.json").read_text())
    input_hashes.update(
        {str(ROOT / name): sha for name, sha in audit["existing_library_hashes"].items()}
    )
    for path in [
        LAB / "runs" / args.capacity / "manifest.json",
        LAB / "runs/history_001/manifest.json",
        LAB.parent / "representation_lab/runs/discovery_complete_001/manifest.json",
    ]:
        input_hashes[str(path)] = digest(path)
    for name, sha in manifest["artifacts"].items():
        if digest(source / name) != sha:
            raise ValueError("Saved fit changed")
    sources = [
        Path(__file__),
        Path(__file__).with_name("ensemble.py"),
        Path(__file__).with_name("reuse.py"),
        Path(__file__).with_name("benchmark.py"),
        LAB / "evaluate.py",
    ]
    source_hashes = {str(p): digest(p) for p in sources}
    (output / "source").mkdir()
    for p in sources:
        shutil.copy2(p, output / "source" / p.name)
    record = dict(
        status="running",
        started=datetime.now(UTC).isoformat(),
        source=args.source,
        capacity_source=args.capacity,
        data_source=protocol["data_run"],
        inputs=input_hashes,
        implementation=source_hashes,
        research_only=True,
        workers=args.workers,
    )
    write_json(output / "manifest.json", record)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    analyze_task,
                    str(source),
                    str(data),
                    str(output),
                    p,
                    h,
                    str(LAB / "runs" / args.capacity),
                )
                for p in protocol["config"]["positions"]
                for h in protocol["config"]["horizons"]
            ]
            for future in as_completed(futures):
                print(future.result(), flush=True)
        render(output)
        if (output / "published_library").exists():
            render(output / "published_library")
        if validate_data(data) != protocol["data_manifest"]:
            raise ValueError("Prepared data changed during ensemble evaluation")
        if any(digest(Path(p)) != sha for p, sha in {**input_hashes, **source_hashes}.items()):
            raise ValueError("Analysis inputs or implementation changed")
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        write_json(output / "manifest.json", record)
        raise
    record.update(
        status="complete",
        completed=datetime.now(UTC).isoformat(),
        artifacts={
            str(p.relative_to(output)): digest(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write_json(output / "manifest.json", record)


if __name__ == "__main__":
    main()
