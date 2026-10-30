"""Full production work is timed; safety and comparison failures stay observable."""

import json

import pandas as pd
import pytest

from mobility.benchmarks.pipeline_scaling import (
    BULK_STAGES,
    CPUOnlyImports,
    MARKER,
    compare_outputs,
    protocol_hash,
    remove_owned_directory,
    run_pipeline_scaling,
    source_fingerprint,
    trial_config,
    timing_identity,
    validate_execution_plan,
    compare_graph,
)
from test_pipeline import config_for, install_source, install_zones, raw_fixture
from mobility.config import ProjectConfig


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_processing_rejects_invalid_cpu_thread_limits(threads):
    with pytest.raises(ValueError):
        ProjectConfig.model_validate({"processing": {"cpu_threads": threads}})


def test_nondefault_eight_threads_reaches_actual_clustering_pools(tmp_path, monkeypatch):
    import numpy as np
    from threadpoolctl import threadpool_info
    from mobility.models import clustering
    from test_models import trip_fixture

    original = config_for()
    options = trial_config(original, [1], "cpu", 8)
    validated = ProjectConfig.model_validate(options).model_dump(mode="json")
    assert validated["processing"]["cpu_threads"] == 8
    assert validated["benchmark"]["cpu_threads"] == 8
    assert original["processing"]["cpu_threads"] == 4
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    trips = trip_fixture()
    trips["source_token"] = np.uint64(123)
    trips["source_row"] = np.arange(len(trips), dtype=np.uint64)
    trips.to_parquet(canonical / "part.parquet", index=False)
    observed = []
    actual_fit = clustering._fit_model

    def inspect_fit(*args, **kwargs):
        pools = threadpool_info()
        assert pools and all(pool["num_threads"] == 8 for pool in pools)
        observed.extend(pool["num_threads"] for pool in pools)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(clustering, "_fit_model", inspect_fit)
    result = clustering.fit_clusters(canonical, validated, tmp_path / "models", "cpu")
    assert observed and result["metadata"]["prediction_rows"] == 120


def test_protocol_pairing_changes_only_backend_execution_settings():
    original = config_for((1, 2, 3))
    cpu = trial_config(original, [1], "cpu", 4)
    gpu = trial_config(original, [1], "gpu", 4)
    assert protocol_hash(cpu) == protocol_hash(gpu)
    assert not cpu["processing"]["verify_gpu_batches"]
    assert gpu["processing"]["custom_cuda"]
    assert original["processing"]["verify_gpu_batches"]
    assert original["data"]["months"] == [1, 2, 3]
    gpu["clustering"]["training_max_rows"] += 1
    assert protocol_hash(cpu) != protocol_hash(gpu)
    with pytest.raises(ValueError, match="no automatic fallback"):
        trial_config(original, [1], "auto", 4)


def test_native_scope_does_not_match_historical_hybrid_experiments():
    native = trial_config(config_for(), [1], "gpu", 4)
    native["processing"]["execution_mode"] = "native"
    legacy = trial_config(native, [1], "gpu", 4)
    legacy["processing"]["execution_mode"] = "legacy"
    assert timing_identity(native)[0] == "native_production_pipeline"
    assert timing_identity(legacy)[0] == "production_pipeline"
    assert timing_identity(native)[1] != timing_identity(legacy)[1]
    assert protocol_hash(native) != protocol_hash(legacy)


def test_cpu_import_guard_blocks_optional_cuda_imports():
    import importlib
    import sys

    guard = CPUOnlyImports()
    sys.meta_path.insert(0, guard)
    try:
        with pytest.raises(RuntimeError, match="CPU-only worker forbids"):
            importlib.import_module("cupy.forbidden_probe")
    finally:
        sys.meta_path.remove(guard)


def test_native_gpu_requires_device_backend_for_every_bulk_stage():
    plan = {
        name: {"backend": "gpu", "implementation": "device_test", "rows": 12}
        for name in BULK_STAGES
    }
    manifest = {"execution_mode": "native", "execution_plan": plan}
    assert validate_execution_plan(manifest, "gpu") == plan
    plan["scaler_fit"]["backend"] = "cpu"
    with pytest.raises(ValueError, match="scaler_fit"):
        validate_execution_plan(manifest, "gpu")
    plan["scaler_fit"]["backend"] = "gpu"
    manifest["models"] = {"graph": {"cpu_reference_elapsed_s": 0.1}}
    with pytest.raises(ValueError, match="CPU reference inside"):
        validate_execution_plan(manifest, "gpu")
    manifest["models"]["graph"]["cpu_reference_elapsed_s"] = None
    del plan["normalization"]
    with pytest.raises(ValueError, match="normalization"):
        validate_execution_plan(manifest, "gpu")


def test_source_matching_uses_checksums_not_only_row_counts():
    source = {
        "source_id": "a",
        "nominal_month": "2024-01",
        "sha256": "abc",
        "parquet_row_count": 30,
    }
    relocated = source | {"local_path": "/different/path"}
    assert source_fingerprint([source]) == source_fingerprint([relocated])
    assert source_fingerprint([source]) != source_fingerprint([source | {"sha256": "def"}])


def test_cleanup_rejects_unowned_paths_and_does_not_follow_data_links(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "keep.txt").write_text("source data")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="unowned"):
        remove_owned_directory(scratch, tmp_path, "experiment")
    (scratch / MARKER).write_text("experiment")
    (scratch / "data").symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        remove_owned_directory(scratch, original, "experiment")
    remove_owned_directory(scratch, tmp_path, "experiment")
    assert (original / "keep.txt").read_text() == "source data"


def test_real_cpu_subprocess_pipeline_records_resources_and_preserves_inputs(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    source, metadata = install_source(root, raw_fixture(rows=30))
    install_zones(root)
    config = config_for()
    config["processing"]["execution_mode"] = "legacy"
    output = tmp_path / "measurements"
    result = run_pipeline_scaling(
        root,
        output,
        config,
        month_counts=[1],
        backends=["cpu"],
        repeats=2,
        warmups=1,
        cpu_threads=1,
    )
    assert result["status"] == "passed"
    trials = pd.read_parquet(output / "benchmark_runs.parquet")
    assert len(trials) == 3
    measured = trials.loc[trials.phase.eq("measured")]
    assert measured.validation.eq("passed").all()
    assert measured.input_rows.eq(30).all()
    assert measured.elapsed_s.gt(0).all()
    sampled = measured.loc[measured.rss_samples.gt(0)]
    assert sampled.peak_host_rss_sampled_bytes.gt(0).all()
    assert trials.loc[trials.phase.eq("warmup"), "validation"].tolist() == ["warmup"]
    summary = pd.read_parquet(output / "benchmark_summary.parquet")
    assert summary.repetitions.tolist() == [2]
    assert not (output / "scratch").exists()
    assert source.exists() and source.stat().st_size == metadata["byte_count"]
    assert not (root / "artifacts").exists()  # No run or active-publication mutation.
    for trial_id in trials.trial_id:
        directory = output / "trials" / trial_id
        resources = json.loads((directory / "resources.json").read_text())
        assert resources["rss"]["requested_interval_s"] == 0.01
        assert resources["rss"]["samples"] > 0 or resources["rss"]["last_error"]
        assert resources["gpu"]["requested_interval_s"] == 0.02
        manifest = json.loads((directory / "run_manifest.json").read_text())
        assert manifest["models"]["clustering"]["prediction_rows"] == 30
        assert "duration_anomalies" in manifest["models"]
        assert "graph" in manifest["models"]
        worker = json.loads((directory / "worker_result.json").read_text())
        assert worker["loaded_gpu_modules"] == []
        assert worker["cpu_import_guard_enabled"] is True


def test_output_comparison_rejects_equal_totals_with_wrong_zone_counts(tmp_path):
    reference = tmp_path / "reference"
    actual = tmp_path / "run" / "tables"
    for directory in (reference, actual):
        (directory / "od_hour").mkdir(parents=True)
        pd.DataFrame(
            {
                "dataset_type": ["yellow", "yellow"],
                "zone_id": [1, 2],
                "hour": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "pickup_count": [4, 7],
                "duration_sum_seconds": [10.0, 20.0],
            }
        ).to_parquet(directory / "zone_hour.parquet", index=False)
    corrupted = pd.read_parquet(actual / "zone_hour.parquet")
    corrupted["pickup_count"] = [5, 6]  # Same citywide total, incorrect per-zone work.
    corrupted.to_parquet(actual / "zone_hour.parquet", index=False)
    with pytest.raises(AssertionError):
        compare_outputs(actual.parent, reference)


def test_existing_experiment_is_never_overwritten(tmp_path):
    existing = tmp_path / "experiment"
    existing.mkdir()
    (existing / "keep.txt").write_text("prior work")
    with pytest.raises(ValueError, match="new empty"):
        run_pipeline_scaling(tmp_path, existing, config_for(), month_counts=[1])
    assert (existing / "keep.txt").read_text() == "prior work"


def test_gpu_failure_is_saved_and_does_not_become_a_cpu_speedup(tmp_path, monkeypatch):
    from mobility.benchmarks import pipeline_scaling

    root = tmp_path / "project"
    root.mkdir()
    install_source(root, raw_fixture())
    install_zones(root)
    real_launch = pipeline_scaling._launch

    def fail_gpu(trial_root, config, trial_dir, project_root):
        if config["processing"]["backend"] == "gpu":
            raise RuntimeError("Deliberate unavailable-GPU test")
        return real_launch(trial_root, config, trial_dir, project_root)

    monkeypatch.setattr(pipeline_scaling, "_launch", fail_gpu)
    output = tmp_path / "experiment"
    with pytest.raises(RuntimeError, match="pipeline trials failed"):
        config = config_for()
        config["processing"]["execution_mode"] = "legacy"
        run_pipeline_scaling(
            root, output, config, month_counts=[1], repeats=1, warmups=0, cpu_threads=1
        )
    trials = pd.read_parquet(output / "benchmark_runs.parquet")
    assert trials.loc[trials.backend.eq("gpu"), "validation"].tolist() == ["failed"]
    assert trials.loc[trials.backend.eq("gpu"), "elapsed_s"].isna().all()
    assert trials.loc[trials.backend.eq("cpu"), "validation"].tolist() == ["incomplete_pair"]
    assert pd.read_parquet(output / "benchmark_summary.parquet").empty
    assert not (output / "scratch").exists()


def test_external_graph_check_allows_relabeling_but_rejects_wrong_scores(tmp_path):
    import shutil
    from mobility.models.graph_analysis import analyze_graph
    from test_graph_analysis import graph_inputs

    edges, zones = graph_inputs()
    reference, actual = tmp_path / "reference", tmp_path / "actual"
    analyze_graph(edges, zones, {}, reference, backend="cpu")
    shutil.copytree(reference, actual)
    path = actual / "graph_nodes.parquet"
    nodes = pd.read_parquet(path)
    nodes["community"] += 100
    nodes.to_parquet(path, index=False)
    result = compare_graph(actual, reference)
    assert result["status"] == "passed"
    assert result["actual"]["pagerank_max_abs_error"] < 1e-7
    nodes.loc[0, "pagerank"] += 0.05
    nodes.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="PageRank mass"):
        compare_graph(actual, reference)


def test_scaler_roundoff_is_tolerated_but_changed_model_inputs_are_not(tmp_path):
    import shutil
    from mobility.pipeline import run_pipeline

    project = tmp_path / "project"
    install_source(project, raw_fixture())
    install_zones(project)
    config = config_for()
    config["processing"]["execution_mode"] = "legacy"
    run = run_pipeline(project, config)
    reference = tmp_path / "reference"
    shutil.copytree(run / "tables", reference)
    path = run / "tables/cluster_parameters.json"
    parameters = json.loads(path.read_text())
    parameters["mean"][0] += 1e-13
    path.write_text(json.dumps(parameters))
    comparison = compare_outputs(run, reference)
    assert comparison["scaler_tolerance"] == {"rtol": 1e-12, "atol": 1e-12}
    assert comparison["graph_validation"]["status"] == "passed"
    parameters["mean"][0] += 1e-5
    path.write_text(json.dumps(parameters))
    with pytest.raises(AssertionError, match="mean"):
        compare_outputs(run, reference)


def test_detector_comparison_keeps_alert_decisions_exact(tmp_path):
    from mobility.benchmarks.pipeline_scaling import compare_numeric_table

    baseline = pd.DataFrame(
        {
            "dataset_type": ["yellow"],
            "zone_id": [1],
            "hour": pd.to_datetime(["2024-05-01"]),
            "score": [4.0],
            "is_alert": [True],
            "history_count": [13],
        }
    )
    expected, actual = tmp_path / "expected.parquet", tmp_path / "actual.parquet"
    baseline.to_parquet(expected, index=False)
    baseline["score"] += 1e-12
    baseline.to_parquet(actual, index=False)
    keys = ["dataset_type", "zone_id", "hour"]
    compare_numeric_table(actual, expected, keys, floating=["score"])
    baseline["is_alert"] = False
    baseline.to_parquet(actual, index=False)
    with pytest.raises(AssertionError):
        compare_numeric_table(actual, expected, keys, floating=["score"])


def test_native_cpu_subprocess_has_complete_execution_evidence_without_cuda(tmp_path):
    project = tmp_path / "project"
    install_source(project, raw_fixture())
    install_zones(project)
    config = config_for()
    config["processing"]["execution_mode"] = "native"
    output = tmp_path / "native-experiment"
    run_pipeline_scaling(
        project,
        output,
        config,
        month_counts=[1],
        backends=["cpu"],
        repeats=1,
        warmups=0,
        cpu_threads=1,
    )
    trials = pd.read_parquet(output / "benchmark_runs.parquet")
    assert trials.stage.tolist() == ["native_production_pipeline"]
    assert trials.validation.tolist() == ["passed"]
    assert json.loads(trials.loaded_gpu_modules_json.iloc[0]) == []
    plan = json.loads(trials.execution_plan_json.iloc[0])
    assert set(plan) >= BULK_STAGES
    assert all(plan[name]["backend"] == "cpu" for name in BULK_STAGES)
    comparison = json.loads((output / "trials/months-1-measured-0-cpu/comparison.json").read_text())
    assert comparison["graph_validation"]["status"] == "passed"
    assert "duration_scores.parquet" in comparison["aggregate_files"]
