"""The two native production paths preserve results and execution boundaries."""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mobility.benchmarks.pipeline_scaling import compare_outputs, validate_execution_plan
from mobility.io import read_json
from mobility.models.common import PopulationScaler, PrioritySample
from mobility.pipeline import run_pipeline
from test_pipeline import config_for, install_source, install_zones, raw_fixture


def test_numeric_priority_sampling_is_stable_across_file_and_batch_order():
    source = pd.DataFrame(
        {
            "source_token": np.repeat(np.array([2**64 - 1, 9], dtype=np.uint64), 250),
            "source_row": np.tile(np.arange(250, dtype=np.uint64), 2),
            "record_id": [f"fixture:{i}" for i in range(500)],
        }
    )
    expected, actual = PrioritySample(31, 42), PrioritySample(31, 42)
    expected.add(source)
    shuffled = source.sample(frac=1, random_state=7)
    for positions in np.array_split(np.arange(len(shuffled)), 7):
        actual.add(shuffled.iloc[positions])
    pd.testing.assert_frame_equal(expected.result(), actual.result())
    assert expected.metadata() == actual.metadata()


def test_native_scaler_keeps_nonbinary_constant_features_constant():
    values = np.column_stack([np.full(127, np.sin(2 * np.pi / 24)), np.arange(127)])
    scaler = PopulationScaler()
    result = scaler.fit_transform(values)
    assert np.array_equal(result[:, 0], np.zeros(127))
    assert scaler.scale_[0] == 1
    assert np.isclose(result[:, 1].std(), 1)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1", reason="Requires actual NVIDIA GPU"
)
def test_native_full_pipeline_matches_cpu_without_cpu_trip_work(tmp_path, monkeypatch):
    for month in range(1, 7):
        install_source(tmp_path, raw_fixture(month, rows=80), month)
    install_zones(tmp_path)
    config = config_for(range(1, 7))
    cpu_run = run_pipeline(tmp_path, config)
    cpu_manifest = read_json(cpu_run / "manifest.json")
    validate_execution_plan(cpu_manifest, "cpu")
    gpu_config = deepcopy(config)
    gpu_config["processing"].update(backend="gpu", custom_cuda=True, verify_gpu_batches=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("A CPU trip-processing or reference-model function was called")

    for target in (
        "mobility.pipeline.normalize_yellow",
        "mobility.pipeline.aggregate_cpu",
        "mobility.models.clustering.fit_clusters",
        "mobility.models.anomalies.fit_anomalies",
        "mobility.models.duration_anomalies.fit_duration_anomalies",
        "mobility.models.graph_analysis._cpu_algorithms",
    ):
        monkeypatch.setattr(target, forbidden)
    gpu_run = run_pipeline(tmp_path, gpu_config)
    manifest = read_json(gpu_run / "manifest.json")
    validate_execution_plan(manifest, "gpu")
    assert manifest["raw_rows"] == 480
    assert manifest["models"]["graph"]["cpu_reference_elapsed_s"] is None
    comparison = compare_outputs(gpu_run, cpu_run / "tables")
    assert comparison["status"] == "passed"
    assert comparison["graph_validation"]["actual"]["pagerank_max_abs_error"] < 1e-7
