"""Attach only verified copies; preserve original experiment timings and files."""

import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from mobility.io import sha256_file
from mobility.pipeline import run_pipeline
from test_pipeline import config_for, install_source, install_zones, raw_fixture
from test_scaling_report import experiment


@pytest.fixture
def helper(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "attach_pipeline_results", scripts / "attach_pipeline_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path):
    measurements = experiment(tmp_path / "experiment")
    project = tmp_path / "source"
    install_source(project, raw_fixture(rows=100))
    install_zones(project)
    source_run = run_pipeline(project, config_for(), fit_models=False)
    candidate = measurements / "trials/months-6-measured-0-gpu"
    # Synthetic transport-contract fixture, not GPU execution or benchmark evidence.
    modeled = json.loads((candidate / "run_manifest.json").read_text())["models"]
    modeled["graph"]["gpu_algorithms_executed"] = True
    modeled["clustering"]["backend"] = "gpu"
    manifest = json.loads((source_run / "manifest.json").read_text())
    manifest.update(backend="gpu", models=modeled, execution_mode="legacy")
    (source_run / "manifest.json").write_text(json.dumps(manifest))
    shutil.copy2(source_run / "manifest.json", candidate / "run_manifest.json")
    checksums = {
        str(path.relative_to(source_run)): sha256_file(path)
        for path in (source_run / "tables").rglob("*")
        if path.is_file()
    }
    checksums["manifest.json"] = sha256_file(source_run / "manifest.json")
    (candidate / "artifact_checksums.json").write_text(json.dumps(checksums))
    (candidate / "comparison.json").write_text(json.dumps({"status": "passed"}))
    env_path = measurements / "benchmark_environment.json"
    environment = json.loads(env_path.read_text())
    environment.update(warmups=1, packages={})
    env_path.write_text(json.dumps(environment))
    target = tmp_path / "artifacts/runs/copy"
    shutil.copytree(source_run / "tables", target / "tables")
    shutil.copy2(source_run / "manifest.json", target / "manifest.json")
    return measurements, target, source_run


def test_attach_needs_no_canonical_files_and_preserves_original_evidence(prepared, helper):
    measurements, target, source_run = prepared
    trial_path = measurements / "benchmark_runs.parquet"
    original_trials_hash = sha256_file(trial_path)
    original_manifest_hash = sha256_file(source_run / "manifest.json")
    manifest = helper.attach(target, measurements)
    assert not (target / "canonical").exists()
    assert sha256_file(trial_path) == original_trials_hash
    assert sha256_file(source_run / "manifest.json") == original_manifest_hash
    assert manifest["publication_selection"]["trial_id"] == "months-6-measured-0-gpu"
    assert manifest["pipeline_benchmark"]["original_trials_sha256"] == original_trials_hash
    saved = pd.read_parquet(target / "tables/pipeline_benchmark_runs.parquet")
    original = pd.read_parquet(trial_path)
    pd.testing.assert_frame_equal(saved[original.columns], original)
    assert saved.gpu_sampling_errors.eq(1).all()
    assert saved.rss_max_observed_gap_s.eq(0.015).all()


def test_changed_table_fails_before_any_attachment_writes(prepared, helper):
    measurements, target, _ = prepared
    path = target / "tables/trip_histogram.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "count"] += 1
    changed.to_parquet(path, index=False)
    before = sha256_file(target / "manifest.json")
    with pytest.raises(ValueError, match="checksum differs"):
        helper.attach(target, measurements)
    assert sha256_file(target / "manifest.json") == before
    assert not (target / "tables/pipeline_benchmark_runs.parquet").exists()


def test_original_experiment_and_snapshot_targets_are_rejected(prepared, helper):
    measurements, target, _ = prepared
    original = measurements / "scratch/original"
    shutil.copytree(target, original)
    with pytest.raises(ValueError, match="separate mutable copy"):
        helper.attach(original, measurements)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["snapshot_id"] = "immutable"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Immutable published snapshots"):
        helper.attach(target, measurements)


def test_historical_benchmarks_stay_separate_with_original_values(prepared, helper, tmp_path):
    measurements, target, _ = prepared
    prior = tmp_path / "historical"
    (prior / "tables").mkdir(parents=True)
    historical = pd.DataFrame({"backend": ["cpu", "gpu"], "elapsed_s": [1.2, 0.4]})
    historical.to_parquet(prior / "tables/old_feature_trials.parquet", index=False)
    historical.to_parquet(prior / "tables/old_feature_summary.parquet", index=False)
    metadata = {
        "tables": {
            "benchmark_runs": "old_feature_trials.parquet",
            "benchmark_summary": "old_feature_summary.parquet",
        },
        "benchmark": {"experiment_id": "older-stage-experiment"},
        "benchmark_environment": {"timing_scope": "loaded_compute"},
    }
    (prior / "manifest.json").write_text(json.dumps(metadata))
    manifest = helper.attach(target, measurements, prior)
    pd.testing.assert_frame_equal(
        pd.read_parquet(target / "tables/benchmark_runs.parquet"), historical
    )
    assert len(pd.read_parquet(target / "tables/pipeline_benchmark_runs.parquet")) == 6
    assert manifest["benchmark"]["experiment_id"] == "older-stage-experiment"
    assert manifest["pipeline_benchmark"]["experiment_id"] == "fixture"


def test_native_graph_needs_external_validation_and_preserves_trial_evidence(prepared, helper):
    from mobility.benchmarks.pipeline_scaling import BULK_STAGES, SCOPE, STAGE

    measurements, target, _ = prepared
    candidate = measurements / "trials/months-6-measured-0-gpu"
    # Transport fixture only: no synthetic timings are presented as GPU evidence.
    environment_path = measurements / "benchmark_environment.json"
    environment = json.loads(environment_path.read_text())
    environment.update(execution_mode="native", timing_scope=SCOPE)
    environment_path.write_text(json.dumps(environment))
    trials_path = measurements / "benchmark_runs.parquet"
    trials = pd.read_parquet(trials_path)
    trials["stage"], trials["timing_scope"] = STAGE, SCOPE
    trials.to_parquet(trials_path, index=False)
    for row in trials.to_dict("records"):
        directory = measurements / "trials" / row["trial_id"]
        manifest = json.loads((directory / "run_manifest.json").read_text())
        manifest.update(
            execution_mode="native",
            execution_plan={
                stage: {"backend": row["backend"], "implementation": "fixture", "rows": 100}
                for stage in BULK_STAGES
            },
        )
        manifest["models"]["graph"].update(
            cpu_reference_elapsed_s=None,
            validation={"status": "awaiting_external_reference"},
        )
        (directory / "run_manifest.json").write_text(json.dumps(manifest))
        (directory / "worker_result.json").write_text(
            json.dumps(
                {
                    "loaded_gpu_modules": [] if row["backend"] == "cpu" else ["cupy"],
                    "cpu_import_guard_enabled": row["backend"] == "cpu",
                }
            )
        )
        (directory / "comparison.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "graph_validation": {"status": "passed"},
                }
            )
        )
    manifest = json.loads((candidate / "run_manifest.json").read_text())
    saved_graph = target / "tables/graph_analysis.json"
    saved_graph.write_text(json.dumps(manifest["models"]["graph"]))
    for path in [target / "manifest.json", candidate / "run_manifest.json"]:
        path.write_text(json.dumps(manifest))
    checksum_path = candidate / "artifact_checksums.json"
    checksums = json.loads(checksum_path.read_text())
    checksums["manifest.json"] = sha256_file(target / "manifest.json")
    checksums["tables/graph_analysis.json"] = sha256_file(saved_graph)
    checksum_path.write_text(json.dumps(checksums))
    original_trial_hash = sha256_file(candidate / "run_manifest.json")
    with pytest.raises(ValueError, match="post-timing graph validation"):
        helper.attach(target, measurements)
    assert sha256_file(target / "manifest.json") == checksums["manifest.json"]
    assert not (target / "tables/pipeline_benchmark_runs.parquet").exists()

    diagnostic = {"pagerank_max_abs_error": 1e-10, "modularity": 0.4, "community_count": 4}
    (candidate / "comparison.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "graph_validation": {"status": "passed", "actual": diagnostic},
            }
        )
    )
    attached = helper.attach(target, measurements)
    validation = attached["models"]["graph"]["validation"]
    assert validation["status"] == "passed"
    assert validation["external_reference"] == diagnostic
    assert "after production timing" in validation["reference"]
    assert sha256_file(candidate / "run_manifest.json") == original_trial_hash
    assert sha256_file(saved_graph) == checksums["tables/graph_analysis.json"]
