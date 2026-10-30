"""Attach a completed experiment to a local copy of its selected GPU run.

This prepares a research run for publication. Original trials and immutable
snapshots are never modified. Choose the first measured six-month GPU run,
independently of its runtime, and copy its manifest/tables locally beforehand.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil

import pandas as pd

from mobility.io import atomic_json, read_json, sha256_file
from mobility.pipeline import publish_run, validate_outputs
from summarize_pipeline_scaling import build_report


def attach(run: Path, experiment: Path, historical_run: Path | None = None):
    run, experiment = run.resolve(), experiment.resolve()
    if run.is_relative_to(experiment):
        raise ValueError("Attach only to a separate mutable copy, outside the original experiment.")
    report = build_report(experiment)
    candidate = report["publication_candidate"]
    if report["experiment_status"] != "passed" or report["failed_trials"] or not candidate:
        raise ValueError("Publication requires a complete validated CPU/GPU experiment.")
    manifest = read_json(run / "manifest.json")
    if "snapshot_id" in manifest or run.parent.name == "snapshots":
        raise ValueError("Immutable published snapshots cannot be attachment targets.")
    if candidate.get("retained_run") and run == Path(candidate["retained_run"]).resolve():
        raise ValueError(
            "Copy the retained trial before attaching results; originals are immutable."
        )
    trial_dir = experiment / "trials" / candidate["trial_id"]
    measured = read_json(trial_dir / "run_manifest.json")
    if manifest["run_id"] != measured["run_id"] or manifest["backend"] != "gpu":
        raise ValueError("The supplied run is not the fixed six-month GPU publication candidate.")
    if manifest["models"]["graph"].get("gpu_algorithms_executed") is not True:
        raise ValueError("This run did not execute its graph algorithms on the GPU.")
    if manifest["models"]["clustering"].get("backend") != "gpu":
        raise ValueError("This run did not fit its trip model on the GPU.")
    if manifest != measured:
        raise ValueError("The local manifest must be an unchanged copy of the selected trial.")
    checksums = read_json(trial_dir / "artifact_checksums.json")
    if "manifest.json" not in checksums:
        raise ValueError("Selected trial has no recorded manifest checksum.")
    for relative, expected in checksums.items():
        artifact = (run / relative).resolve()
        if not artifact.is_relative_to(run) or not artifact.is_file():
            raise ValueError(f"Missing or unsafe selected-trial artifact: {relative}")
        if sha256_file(artifact) != expected:
            raise ValueError(f"Selected-trial artifact checksum differs: {relative}")
    comparison = read_json(trial_dir / "comparison.json")
    if comparison.get("status") != "passed":
        raise ValueError("The selected trial did not pass its post-timing output comparison.")
    validate_outputs(run / "tables", manifest["validation"]["counts"])
    prior = None
    historical_files = {}
    if historical_run is not None:
        historical_run = historical_run.resolve()
        if historical_run == run:
            raise ValueError("Historical benchmark sources must be separate from the target run.")
        prior = read_json(historical_run / "manifest.json")
        if not {"benchmark", "benchmark_environment", "tables"} <= prior.keys():
            raise ValueError("Historical run lacks its benchmark metadata or table declarations.")
        for name in ["benchmark_runs", "benchmark_summary"]:
            artifact = (historical_run / "tables" / prior["tables"][name]).resolve()
            if not artifact.is_relative_to(historical_run / "tables") or not artifact.is_file():
                raise ValueError(f"Missing or unsafe historical benchmark table: {name}")
            historical_files[name] = artifact
    manifest = copy.deepcopy(manifest)
    output = run / "tables"
    trials = pd.read_parquet(experiment / "benchmark_runs.parquet")
    # Add sampling diagnostics from the corresponding recorded monitor output.
    # These columns do not replace or recalculate any elapsed-time measurement.
    for index, row in trials.iterrows():
        resources = read_json(experiment / "trials" / row.trial_id / "resources.json")
        for kind in ["rss", "gpu"]:
            trials.loc[index, f"{kind}_max_observed_gap_s"] = resources[kind]["max_observed_gap_s"]
            trials.loc[index, f"{kind}_sampling_errors"] = resources[kind]["sampling_errors"]
    trials.to_parquet(output / "pipeline_benchmark_runs.parquet", index=False)
    shutil.copy2(
        experiment / "benchmark_summary.parquet", output / "pipeline_benchmark_summary.parquet"
    )
    environment = read_json(experiment / "benchmark_environment.json")
    addendum = experiment / "environment_addendum.json"
    if addendum.exists():
        hardware = read_json(addendum)
        environment["gpu"] = hardware["gpu"]
        environment.setdefault("packages", {}).update(hardware["packages"])
        environment["hardware_recorded_at"] = hardware["recorded_at"]
        environment["hardware_record_method"] = hardware["method"]
    manifest["pipeline_benchmark_environment"] = environment
    manifest["tables"].update(
        {
            "pipeline_benchmark_runs": "pipeline_benchmark_runs.parquet",
            "pipeline_benchmark_summary": "pipeline_benchmark_summary.parquet",
        }
    )
    manifest["pipeline_benchmark"] = {
        "experiment_id": report["experiment_id"],
        "status": report["experiment_status"],
        "original_trials_sha256": sha256_file(experiment / "benchmark_runs.parquet"),
        "resource_enrichment": "Original timings plus recorded monitor gaps/error counts.",
        "warmups_per_backend_size": environment["warmups"],
        "measured_repetitions_per_backend_size": environment["repeats"],
        "requested_month_counts": environment["requested_month_counts"],
    }
    manifest["publication_selection"] = {
        "trial_id": candidate["trial_id"],
        "rule": "First measured six-month GPU repetition (repeat 0), independent of runtime.",
        "cross_backend_comparison": comparison,
        "canonical_storage": "Original canonical files remain with the retained server trial.",
        "verified_trial_artifact_sha256": checksums,
    }
    if prior is not None:
        for name, artifact in historical_files.items():
            relative = f"{name}.parquet"
            shutil.copy2(artifact, output / relative)
            manifest["tables"][name] = relative
        manifest["benchmark_environment"] = prior["benchmark_environment"]
        manifest["benchmark"] = prior["benchmark"]
        manifest["historical_benchmark_note"] = (
            "Earlier stage experiments retain their original inputs, policy and timings. "
            "The separate full-pipeline experiment uses this implementation."
        )
    atomic_json(run / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    manifest = attach(args.run, args.experiment, args.historical_run)
    if args.publish:
        print(publish_run(args.run, args.run.resolve().parent.parent))
    else:
        print(json.dumps(manifest["publication_selection"], indent=2))


if __name__ == "__main__":
    main()
