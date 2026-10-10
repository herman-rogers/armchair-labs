"""Full-data RB descriptions and separately fitted historical role cohorts.

Full-data profiles describe the available sample; their future production summaries
are associations, not validation results. Historical cohort assignments are fitted
on earlier input rows and never use outcomes or full-data preprocessing.
"""

# Embedded standalone HTML/JavaScript keeps its own line layout.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .representations import fit_representation

KEYS = ["horizon", "player_id", "year", "origin"]
PROFILE_SPEC = {"kind": "pca", "n_components": 32, "missing_indicators": True, "clip": 8}


def _cluster(z, k, seed, method="kmeans"):
    if len(z) < k:
        raise ValueError("Role clustering needs at least k training observations")
    if method == "gaussian_mixture":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=seed,
                n_init=2,
                max_iter=200,
                reg_covar=1e-4,
            ).fit(z)
        model.optimization_warnings_ = [str(w.message) for w in caught]
        model.labels_ = model.predict(z)
        model.cluster_centers_ = model.means_
        return model
    if method != "kmeans":
        raise ValueError(f"Unknown role clustering method: {method}")
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=3,
        max_iter=200,
        batch_size=min(1024, len(z)),
        reassignment_ratio=0,
    ).fit(z)


def fit_role_profiles(
    x_train,
    names,
    x_test,
    *,
    k=5,
    seed=20261003,
    method="kmeans",
    representation_spec=None,
):
    """Return labels and fitted artifacts using exclusively the supplied training X."""
    representation = fit_representation(
        x_train,
        names,
        representation_spec or PROFILE_SPEC,
        seed=seed,
    )
    train = representation.transform(x_train)
    model = _cluster(train, k, seed, method)
    test = representation.transform(x_test)
    return {
        "train_labels": model.labels_.copy(),
        "test_labels": model.predict(test),
        "test_distance": np.linalg.norm(test - model.cluster_centers_[model.predict(test)], axis=1),
        "representation": representation,
        "clusterer": model,
    }


def _key_frame(panel):
    columns = [
        c
        for c in [*KEYS, "player_display_name", "team", "candidate_basis", "age"]
        if c in panel.columns
    ]
    missing = set(KEYS) - set(columns)
    if missing:
        raise ValueError(f"Missing profile keys: {sorted(missing)}")
    return panel.select(columns)


def chronological_profiles(
    dataset,
    years,
    *,
    seed=20261003,
    k=5,
    save_dir=None,
    method="kmeans",
    representation_spec=None,
):
    """Fit each role vocabulary on years before Y; profile IDs are fold-specific."""
    season = dataset.panel["year"].to_numpy()
    keys = _key_frame(dataset.panel)
    assignments, summaries = [], []
    for year in sorted(set(years)):
        train, test = season < year, season == year
        if train.sum() < k or not test.any():
            continue
        result = fit_role_profiles(
            dataset.x[train],
            dataset.names,
            dataset.x[test],
            k=k,
            seed=seed,
            method=method,
            representation_spec=representation_spec,
        )
        labels = result["test_labels"]
        assignments.append(
            keys.filter(pl.Series(test)).with_columns(
                pl.Series("profile", labels.astype(int)),
                pl.Series("profile_distance", result["test_distance"]),
                pl.lit(k).alias("k"),
                pl.lit(int(season[train].max())).alias("training_max_year"),
                pl.lit("earlier_years_only").alias("profile_fit_scope"),
                pl.lit(method).alias("profile_method"),
            )
        )
        summaries.append(
            {
                "year": int(year),
                "k": k,
                "method": method,
                "converged": getattr(result["clusterer"], "converged_", None),
                "iterations": int(result["clusterer"].n_iter_),
                "optimization_warnings": getattr(result["clusterer"], "optimization_warnings_", []),
                "training_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "training_max_year": int(season[train].max()),
                "training_profile_sizes": np.bincount(result["train_labels"], minlength=k).tolist(),
                "test_profile_sizes": np.bincount(labels, minlength=k).tolist(),
                "representation": result["representation"].summary(),
            }
        )
        if save_dir is not None:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {name: result[name] for name in ["representation", "clusterer"]},
                path / f"roles_before_{year}.joblib",
                compress=3,
            )
    if not assignments:
        raise ValueError("No eligible chronological profile folds")
    return pl.concat(assignments), summaries


def _characteristics(base, names, labels, k):
    """Standardized mean differences are descriptive and use no target variables."""
    center = np.mean(base, axis=0, dtype=np.float64)
    scale = np.std(base, axis=0, dtype=np.float64)
    scale[scale == 0] = 1
    means, descriptions = [], []
    for label in range(k):
        subset = base[labels == label]
        if not len(subset):
            means.append(np.full(base.shape[1], np.nan))
            descriptions.append([])
            continue
        mean = np.mean(subset, axis=0, dtype=np.float64)
        means.append(mean)
        differences = (mean - center) / scale
        ranked = np.argsort(-np.abs(differences), kind="stable")
        # Show source values and coverage signals separately without changing fits.
        values = [i for i in ranked if not names[i].startswith("missing:")][:8]
        missing = [i for i in ranked if names[i].startswith("missing:")][:4]
        descriptions.append(
            [
                {
                    "feature": str(names[i]),
                    "mean": float(mean[i]),
                    "global_mean": float(center[i]),
                    "standardized_difference": float(differences[i]),
                }
                for i in [*values, *missing]
            ]
        )
    return np.asarray(means), descriptions


def descriptive_profiles(
    dataset,
    *,
    seed=20261003,
    ks=(3, 5, 8),
    method="kmeans",
    representation_spec=None,
):
    """Discover roles in every available input row; outcomes only annotate profiles."""
    representation = fit_representation(
        dataset.x,
        dataset.names,
        representation_spec or PROFILE_SPEC,
        seed=seed,
    )
    z = representation.transform(dataset.x)
    raw = fit_representation(dataset.x, dataset.names, {"kind": "raw"}, seed=seed)
    base, names = raw.transform(dataset.x), raw.get_feature_names_out()
    keys = _key_frame(dataset.panel)
    summaries, assignments, centroids, models, coordinates = [], [], [], {}, {}
    for k in ks:
        fits = [_cluster(z, k, seed + offset, method) for offset in (0, 137, 271)]
        model, labels = fits[0], fits[0].labels_
        models[str(k)] = model
        means, descriptions = _characteristics(base, names, labels, k)
        stability = [
            float(adjusted_rand_score(a.labels_, b.labels_)) for a, b in combinations(fits, 2)
        ]
        count = len(np.unique(labels))
        silhouette = (
            float(silhouette_score(z, labels, sample_size=min(1500, len(z)), random_state=seed))
            if 1 < count < len(z)
            else None
        )
        distance = np.linalg.norm(z - model.cluster_centers_[labels], axis=1)
        frame = keys.with_columns(
            pl.Series("profile", labels.astype(int)),
            pl.Series("profile_distance", distance),
            pl.lit(k).alias("k"),
            pl.lit("all_available_inputs_descriptive").alias("profile_fit_scope"),
            pl.lit(method).alias("profile_method"),
            pl.Series("pca1", z[:, 0]),
            pl.Series("pca2", z[:, 1] if z.shape[1] > 1 else np.zeros(len(z))),
        )
        assignments.append(frame)
        profiles = []
        for label in range(k):
            mask = labels == label
            if not mask.any():
                continue
            members = np.flatnonzero(mask)
            nearest = members[np.argsort(distance[members])[:5]]
            future = dataset.targets[mask]
            with np.errstate(invalid="ignore"):
                counts = np.isfinite(future).sum(axis=0)
                totals = np.where(np.isfinite(future), future, 0).sum(axis=0)
                target_means = np.divide(
                    totals, counts, out=np.full(future.shape[1], np.nan), where=counts > 0
                )
            profiles.append(
                {
                    "profile": label,
                    "rows": int(mask.sum()),
                    "unique_players": int(
                        dataset.panel.filter(pl.Series(mask))["player_id"].n_unique()
                    ),
                    "year_min": int(dataset.panel.filter(pl.Series(mask))["year"].min()),
                    "year_max": int(dataset.panel.filter(pl.Series(mask))["year"].max()),
                    "exemplars": keys[nearest.tolist()].to_dicts(),
                    "characteristics": descriptions[label],
                    "descriptive_future_stat_means": [
                        float(v) if np.isfinite(v) else None for v in target_means
                    ],
                }
            )
            centroids.append(
                pl.DataFrame(
                    {
                        "k": [k] * len(names),
                        "profile": [label] * len(names),
                        "feature": names.tolist(),
                        "mean": means[label].tolist(),
                    }
                )
            )
        summaries.append(
            {
                "k": k,
                "method": method,
                "effective_clusters": count,
                "silhouette": silhouette,
                "cross_seed_adjusted_rand": stability,
                "cross_seed_mean_adjusted_rand": float(np.mean(stability)),
                "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
                "profiles": profiles,
                "bic": float(model.bic(z)) if method == "gaussian_mixture" else None,
                "converged": [bool(m.converged_) for m in fits]
                if method == "gaussian_mixture"
                else None,
                "iterations": [int(m.n_iter_) for m in fits],
                "optimization_warnings": [getattr(m, "optimization_warnings_", []) for m in fits],
            }
        )
        coordinates[str(k)] = frame.to_dicts()
    return {
        "assignments": pl.concat(assignments),
        "centroids": pl.concat(centroids),
        "summary": summaries,
        "representation": representation,
        "models": models,
        "coordinates": coordinates,
    }


def _write_html(path, descriptive, metadata):
    payload = json.dumps(
        {"points": descriptive["coordinates"], "summary": descriptive["summary"]},
        default=str,
        allow_nan=False,
    ).replace("<", "\\u003c")
    title = html.escape(
        f"RB {metadata['horizon']} {metadata.get('method', 'kmeans')} "
        f"{metadata.get('geometry', PROFILE_SPEC)['kind']} input profiles"
    )
    scaling = (
        "training mean/std scaling and inverse-square-root source-family weights"
        if metadata.get("geometry", PROFILE_SPEC)["kind"] == "group_pca"
        else "robust median/IQR scaling (standard-deviation fallback)"
    )
    page = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1150px;color:#172839;background:#f7f9fc}
h1{font-size:27px}p{line-height:1.5}canvas{width:100%;height:500px;background:white;border:1px solid #ccc}
table{width:100%;border-collapse:collapse;background:white}th,td{padding:9px;border-bottom:1px solid #ddd;
text-align:left;vertical-align:top}small{color:#586a7b}select{padding:7px}#tip{min-height:25px}</style>
</head><body><h1>__TITLE__</h1><p>These profiles describe all available RB input histories.
All eligible raw/transformed inputs enter training-fitted median imputation, missing flags,
__SCALING__, clipping at ±8 and PCA (up to 32 components), followed by clustering.
They describe player-seasons and source coverage together; cluster numbers are arbitrary.
Future stat means only describe associations. These full-data groups are not forecast validation.</p>
<label>Number of profiles <select id="k"></select></label><p id="stats"></p>
<canvas id="plot" width="1100" height="500"></canvas><div id="tip"></div>
<small>First two PCA coordinates; the clustering uses all retained components.
Hover over a point to inspect its player-season. Extreme coordinates are clipped visually only.</small>
<div id="table"></div><script>const data=__DATA__;
const colors=['#2563eb','#db2777','#16a34a','#ea580c','#7c3aed','#0891b2','#a16207','#475569'];
const select=document.querySelector('#k'), canvas=document.querySelector('#plot');
const ctx=canvas.getContext('2d'); let plotted=[];
function escape(s){return String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
'\"':'&quot;',"'":'&#39;'}[c]));}
Object.keys(data.points).forEach(k=>select.add(new Option(k,k))); select.value='5';
function draw(){const k=select.value, points=data.points[k], info=data.summary.find(d=>String(d.k)===k);
const xs=points.map(d=>d.pca1).sort((a,b)=>a-b), ys=points.map(d=>d.pca2).sort((a,b)=>a-b);
const quant=(v,q)=>v[Math.floor((v.length-1)*q)], xmin=quant(xs,.005),xmax=quant(xs,.995),
ymin=quant(ys,.005),ymax=quant(ys,.995);ctx.clearRect(0,0,canvas.width,canvas.height);plotted=[];
for(const d of points){let x=30+Math.max(0,Math.min(1,(d.pca1-xmin)/(xmax-xmin||1)))*1040,
y=470-Math.max(0,Math.min(1,(d.pca2-ymin)/(ymax-ymin||1)))*440;ctx.fillStyle=colors[d.profile%8];
ctx.globalAlpha=.55;ctx.beginPath();ctx.arc(x,y,2.7,0,Math.PI*2);ctx.fill();plotted.push({x,y,d});}
ctx.globalAlpha=1;document.querySelector('#stats').textContent=`${points.length.toLocaleString()} player-seasons · `+
`silhouette ${info.silhouette===null?'unavailable':info.silhouette.toFixed(3)} · `+
`cross-seed adjusted Rand ${info.cross_seed_mean_adjusted_rand.toFixed(3)} (higher is more stable)`;
let rows=info.profiles.map(p=>`<tr><td style="color:${colors[p.profile%8]}"><b>${p.profile}</b></td>`+
`<td>${p.rows}<br><small>${p.unique_players} players<br>${p.year_min}–${p.year_max}</small></td>`+
`<td>${p.exemplars.map(e=>escape((e.player_display_name||e.player_id)+' '+e.year)).join('<br>')}</td>`+
`<td>${p.characteristics.slice(0,8).map(c=>escape(c.feature)+': '+(c.standardized_difference>=0?'+':'')+
c.standardized_difference.toFixed(2)+' SD').join('<br>')}</td></tr>`).join('');
document.querySelector('#table').innerHTML='<table><tr><th>Profile</th><th>Members</th><th>Examples nearest center</th>'+
'<th>Characteristic observed inputs</th></tr>'+rows;}
canvas.addEventListener('mousemove',event=>{const r=canvas.getBoundingClientRect(),
x=(event.clientX-r.left)*canvas.width/r.width,y=(event.clientY-r.top)*canvas.height/r.height;
let best=null,dist=64;for(const p of plotted){let d=(p.x-x)**2+(p.y-y)**2;if(d<dist){dist=d;best=p.d;}}
document.querySelector('#tip').textContent=best?`${best.player_display_name||best.player_id} · ${best.year} · `+
`${best.team||''} · profile ${best.profile}`:'';});select.addEventListener('change',draw);draw();
</script></body></html>"""
    path.write_text(
        page.replace("__TITLE__", title)
        .replace("__SCALING__", scaling)
        .replace("__DATA__", payload)
    )


def build_profiles(
    dataset,
    out,
    years,
    *,
    seed=20261003,
    threads=4,
    progress=print,
    method="kmeans",
    representation_spec=None,
):
    """Save full-data profiles and independent chronological cohorts beside a run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    horizon = dataset.panel["horizon"][0]
    progress(f"Descriptive full-data RB profiles: {horizon}")
    with threadpool_limits(limits=threads):
        descriptive = descriptive_profiles(
            dataset,
            seed=seed,
            method=method,
            representation_spec=representation_spec,
        )
        progress(f"Earlier-year role assignments: {horizon}")
        chronological, folds = chronological_profiles(
            dataset,
            years,
            seed=seed,
            save_dir=out / "models",
            method=method,
            representation_spec=representation_spec,
        )
    metadata = {
        "horizon": horizon,
        "method": method,
        "seed": seed,
        "input_rows": len(dataset.x),
        "input_features": len(dataset.names),
        "full_data_fit_scope": "descriptive_only",
        "chronological_fit_scope": "Strictly earlier years, no outcomes; fixed k=5",
        "profile_ids": "Arbitrary labels, scoped to each horizon/fold/k; not aligned across years",
        "geometry": representation_spec or PROFILE_SPEC,
        "mixture_parameters": {
            "covariance_type": "diag",
            "reg_covar": 1e-4,
            "n_init": 2,
            "max_iter": 200,
        }
        if method == "gaussian_mixture"
        else None,
        "descriptive": descriptive["summary"],
        "full_representation": descriptive["representation"].summary(),
        "chronological_folds": folds,
        "descriptive_target_order": [
            "carries",
            "rushing_yards",
            "rushing_tds",
            "targets",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        ],
        "limitations": [
            "Profiles combine playing role, career stage, era and source coverage.",
            "Seasonal rows repeat players; silhouette and stability are descriptive.",
            "PCA emphasizes predictor variance, not necessarily predictive information.",
            "Full-data profiles must not select or calibrate chronological forecasts.",
        ],
    }
    descriptive["assignments"].write_parquet(out / "profile_assignments.parquet")
    descriptive["centroids"].write_parquet(out / "profile_centroids.parquet")
    chronological.write_parquet(out / "chronological_profile_assignments.parquet")
    joblib.dump(
        {"representation": descriptive["representation"], "models": descriptive["models"]},
        out / "models" / "full_data_descriptive.joblib",
        compress=3,
    )
    (out / "profiles.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    _write_html(out / "index.html", descriptive, metadata)
    return metadata


def summarize_conditional_errors(predictions, assignments, out=None, *, distributions=None):
    """Matched skill and coverage within roles fitted strictly before each test year.

    A profile number is local to its annual fit. Keep ``year`` in every grouping
    instead of implying that profile 0 is the same role in different years.
    """
    if assignments.select(KEYS).is_duplicated().any():
        raise ValueError("Chronological role assignments must have unique forecast keys")
    if (
        "training_max_year" not in assignments.columns
        or assignments.filter(
            pl.col("training_max_year").is_null() | (pl.col("training_max_year") >= pl.col("year"))
        ).height
    ):
        raise ValueError("Conditional prediction analysis requires strictly earlier profile fits")
    if (
        "profile_fit_scope" in assignments.columns
        and assignments.filter(pl.col("profile_fit_scope") != "earlier_years_only").height
    ):
        raise ValueError("Full-data descriptive profiles cannot define historical forecast cohorts")
    forecast_key = [*KEYS, "target"]
    if predictions.select(*forecast_key, "recipe").is_duplicated().any():
        raise ValueError("Duplicate forecast keys in conditional analysis")
    baseline = predictions.filter(pl.col("recipe") == "persistence").select(
        *forecast_key,
        pl.col("prediction").alias("baseline_prediction"),
        pl.col("actual").alias("baseline_actual"),
    )
    rows = predictions.join(
        assignments.select(*KEYS, "profile"), on=KEYS, how="inner", validate="m:1"
    )
    rows = rows.join(baseline, on=forecast_key, how="inner", validate="m:1").filter(
        pl.col("actual").is_finite()
        & pl.col("prediction").is_finite()
        & pl.col("baseline_actual").is_finite()
        & pl.col("baseline_prediction").is_finite()
    )
    if rows.filter((pl.col("actual") - pl.col("baseline_actual")).abs() > 1e-8).height:
        raise ValueError("Model and persistence outcomes disagree on matched cohort rows")
    if distributions is not None:
        uncertainty = distributions.select(*forecast_key, "recipe", "lower80", "upper80")
        rows = rows.join(uncertainty, on=[*forecast_key, "recipe"], how="left", validate="1:1")
    rows = rows.with_columns(
        (pl.col("prediction") - pl.col("actual")).alias("error"),
        (pl.col("baseline_prediction") - pl.col("actual")).alias("baseline_error"),
    )
    metrics = [
        pl.len().alias("n"),
        pl.col("actual").mean().alias("actual_mean"),
        pl.col("actual").var(ddof=0).alias("actual_variance"),
        pl.col("prediction").mean().alias("predicted_mean"),
        pl.col("error").mean().alias("bias"),
        pl.col("error").abs().mean().alias("mae"),
        (pl.col("error") ** 2).mean().alias("mse"),
        pl.col("baseline_error").abs().mean().alias("baseline_mae"),
        (pl.col("baseline_error") ** 2).mean().alias("baseline_mse"),
    ]
    if {"lower80", "upper80"} <= set(rows.columns):
        known = pl.col("lower80").is_finite() & pl.col("upper80").is_finite()
        metrics += [
            known.sum().alias("interval_rows"),
            pl.when(known)
            .then(pl.col("actual").is_between(pl.col("lower80"), pl.col("upper80")))
            .otherwise(None)
            .mean()
            .alias("coverage80"),
            pl.when(known)
            .then(pl.col("upper80") - pl.col("lower80"))
            .otherwise(None)
            .mean()
            .alias("width80"),
        ]
    summary = rows.group_by("horizon", "year", "profile", "target", "recipe").agg(metrics)
    summary = summary.with_columns(
        pl.col("mse").sqrt().alias("rmse"),
        pl.col("baseline_mse").sqrt().alias("baseline_rmse"),
        pl.when(pl.col("baseline_mse") > 0)
        .then(100 * (1 - pl.col("mse") / pl.col("baseline_mse")))
        .otherwise(None)
        .alias("mse_improvement_pct"),
        pl.when(pl.col("actual_variance") > 0)
        .then(1 - pl.col("mse") / pl.col("actual_variance"))
        .otherwise(None)
        .alias("r2"),
    ).sort("horizon", "year", "target", "recipe", "profile")
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        summary.write_parquet(out / "conditional_errors.parquet")
        selected = summary.filter(
            pl.col("recipe").is_in(["selected", "ensemble_top3", "persistence_calibrated"])
        )
        payload = json.dumps(selected.to_dicts(), default=str, allow_nan=False).replace(
            "<", "\\u003c"
        )
        page = """<!doctype html><meta charset="utf-8"><title>RB conditional predictability</title>
<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;color:#172839;background:#f7f9fc}
select{padding:6px;margin:8px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccd}
table{border-collapse:collapse;width:100%;background:white}.bar{height:14px;display:inline-block}
p{line-height:1.5}</style><h1>Conditional RB predictability</h1>
<p>Profiles use only inputs observed before each evaluation year. Their numeric labels are local
to each year and cannot be compared across years. Skill compares model and persistence forecasts
on identical player rows. Small cohorts and repeated historical comparisons limit interpretation.</p>
<div id="controls"></div><div id="table"></div><script>const rows=__DATA__;
const controls=document.querySelector('#controls'), fields=['horizon','year','target','recipe'];
for(const field of fields){const label=document.createElement('label');label.textContent=field+' ';
const select=document.createElement('select');select.id=field;[...new Set(rows.map(r=>r[field]))]
.sort().forEach(v=>select.add(new Option(v,v)));select.onchange=draw;label.appendChild(select);controls.appendChild(label);}
function fmt(v){return v===null||v===undefined?'—':typeof v==='number'?v.toFixed(2):String(v);}
function draw(){const filtered=rows.filter(r=>fields.every(f=>String(r[f])===document.getElementById(f).value));
const cols=['profile','n','rmse','baseline_rmse','mse_improvement_pct','bias','coverage80'];
let body='<table><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'<th>MSE skill</th></tr>';
for(const r of filtered){const skill=r.mse_improvement_pct;body+='<tr>'+cols.map(c=>'<td>'+fmt(r[c])+'</td>').join('');
body+='<td><span class="bar" style="background:'+(skill>=0?'#16a34a':'#dc2626')+';width:'+
Math.min(150,Math.abs(skill||0))+'px"></span></td></tr>';}
document.querySelector('#table').innerHTML=body+'</table>';}
draw();</script>"""
        (out / "index.html").write_text(page.replace("__DATA__", payload))
    return summary
