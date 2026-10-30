"""Full production work is timed; safety and comparison failures stay observable."""

import json

import pandas as pd
import pytest

from mobility.benchmarks.pipeline_scaling import (
    MARKER,
    compare_outputs,
    protocol_hash,
    remove_owned_directory,
    run_pipeline_scaling,
    source_fingerprint,
    trial_config,
)
from test_pipeline import config_for, install_source, install_zones, raw_fixture
from mobility.config import ProjectConfig


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_processing_rejects_invalid_cpu_thread_limits(threads):
    with pytest.raises(ValueError):
        ProjectConfig.model_validate({"processing": {"cpu_threads": threads}})


def test_nondefault_eight_threads_reaches_actual_clustering_pools(tmp_path, monkeypatch):
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
    trip_fixture().to_parquet(canonical / "part.parquet", index=False)
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
        run_pipeline_scaling(
            root, output, config_for(), month_counts=[1], repeats=1, warmups=0, cpu_threads=1
        )
    trials = pd.read_parquet(output / "benchmark_runs.parquet")
    assert trials.loc[trials.backend.eq("gpu"), "validation"].tolist() == ["failed"]
    assert trials.loc[trials.backend.eq("gpu"), "elapsed_s"].isna().all()
    assert trials.loc[trials.backend.eq("cpu"), "validation"].tolist() == ["incomplete_pair"]
    assert pd.read_parquet(output / "benchmark_summary.parquet").empty
    assert not (output / "scratch").exists()
