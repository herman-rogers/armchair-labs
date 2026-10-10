"""Analyze one fully completed horizon from an otherwise running frozen campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from threadpoolctl import threadpool_limits

from .analyze import analyze, seal_analysis
from .diagnostics import controlled_comparisons, known_usage_cohorts
from .run import RUNS, digest, json_write, verify_checkpoint


def derive_preview(main, out, *, horizon="season", threads=2):
    main, out = Path(main), Path(out)
    config = json.loads((main / "config.json").read_text())
    if horizon not in config["horizons"]:
        raise ValueError("The requested horizon is not in the frozen main campaign")
    checkpoint_records = []
    for recipe in config["recipes"]:
        worker = "cpu" if recipe["device"] == "cpu" else "gpu"
        for year in config["years"]:
            checkpoint = main / "workers" / worker / horizon / "fits" / recipe["id"] / str(year)
            verify_checkpoint(checkpoint)
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            if metadata["recipe"] != recipe or metadata["year"] != year:
                raise ValueError("A completed checkpoint differs from the frozen recipe/fold")
            if [fit["stat"] for fit in metadata["fits"]] != config["targets"]:
                raise ValueError("A completed checkpoint has incomplete or reordered targets")
            checkpoint_records.append(
                {
                    "worker": worker,
                    "recipe": recipe["id"],
                    "year": year,
                    "target_fits": len(metadata["fits"]),
                    "source": str(checkpoint.resolve()),
                    "receipt_sha256": digest(checkpoint / "complete.json"),
                }
            )
    input_names = ("inputs.json", "baselines.parquet", "target_audit.parquet")
    workers = sorted({r["worker"] for r in checkpoint_records})
    for name in input_names:
        values = {digest(main / "workers" / worker / horizon / name) for worker in workers}
        if len(values) != 1:
            raise ValueError(f"Workers disagree on the {horizon} {name} input contract")
    out.mkdir(parents=True, exist_ok=False)
    (out / horizon / "fits").mkdir(parents=True)
    (out / "provenance").mkdir()
    json_write(out / "config.json", {**config, "horizons": [horizon]})
    for name in ("config.json", "signature.json"):
        shutil.copyfile(main / name, out / "provenance" / f"main_{name}")
    shutil.copytree(main / "implementation", out / "implementation")
    shutil.copyfile(__file__, out / "provenance" / "preview_source.py")
    shutil.copyfile(
        Path(__file__).parent / "diagnostics.py", out / "provenance" / "diagnostics_source.py"
    )
    for name in input_names:
        shutil.copyfile(main / "workers" / workers[0] / horizon / name, out / horizon / name)
    for row in checkpoint_records:
        target = out / horizon / "fits" / row["recipe"] / str(row["year"])
        shutil.copytree(row["source"], target, copy_function=os.link)
        verify_checkpoint(target)
    derived = {
        "purpose": (
            "Completed-horizon historical analysis preview; "
            "not a complete campaign or prospective validation"
        ),
        "source_main_run": str(main.resolve()),
        "horizon": horizon,
        "source_configuration_sha256": digest(main / "config.json"),
        "source_signature_sha256": digest(main / "signature.json"),
        "derived_configuration_sha256": digest(out / "config.json"),
        "completed_fold_checkpoints": len(checkpoint_records),
        "completed_target_fits": sum(r["target_fits"] for r in checkpoint_records),
        "checkpoint_storage": (
            "Hardlinks to verified immutable completed checkpoint files; analysis writes separately"
        ),
        "source_checkpoints": checkpoint_records,
        "new_model_fits": 0,
        "trial_or_selection_changes": False,
    }
    json_write(out / "derived_provenance.json", derived)
    json_write(out / "manifest.json", {"status": "analyzing_complete_horizon_preview", **derived})
    print(
        f"Verified {derived['completed_target_fits']} {horizon} target fits; "
        "analyzing frozen forecasts",
        flush=True,
    )
    preview_config = {**config, "horizons": [horizon]}
    with threadpool_limits(limits=threads):
        analyze(out)
        controlled_comparisons(out, preview_config)
        known_usage_cohorts(out, preview_config)
    disclaimer = (
        "Completed preseason analysis preview from the frozen main campaign. "
        "The other horizon is still running. Historical development evidence, with no new fits, "
        "prospective validation, or trial/selection changes."
    )
    readout = out / "analysis" / "readout.md"
    readout.write_text(
        "# Completed preseason analysis preview\n\n" + disclaimer + "\n\n" + readout.read_text()
    )
    page = out / "analysis" / "index.html"
    text = page.read_text().replace(
        "<h1>Running back GPU research</h1>",
        "<h1>Completed preseason analysis preview</h1><p>" + disclaimer + "</p>",
    )
    # A horizon-specific preview must not present the unfinished horizon as an empty result.
    text = text.replace("<option>remaining</option>", "")
    text = text.replace(
        'href="../profiles/index.html"', f'href="../../{main.name}/profiles/index.html"'
    )
    page.write_text(text)
    seal_analysis(out, status="complete")
    json_write(
        out / "manifest.json",
        {
            "status": "complete_preseason_analysis_preview",
            **derived,
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p != out / "manifest.json"
            },
        },
    )
    return out / "analysis"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run-id", default="rb_gpu_20261003_001")
    parser.add_argument("--run-id", default="rb_gpu_preseason_preview_001")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(derive_preview(RUNS / args.main_run_id, RUNS / args.run_id, threads=args.threads))
