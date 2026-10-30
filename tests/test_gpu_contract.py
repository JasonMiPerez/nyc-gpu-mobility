"""CPU contract tests always run; actual CUDA tests require --run-gpu or an env flag."""

import os

import numpy as np
import pandas as pd
import pytest

from mobility.backends.features import NAT, reference_features, validate_inputs
from mobility.benchmarks.runner import run_benchmarks, summarize_trials


def feature_fixture(n=513):
    inputs = {
        "pickup_ns": np.full(n, 1705312800000000000, dtype=np.int64),
        "dropoff_ns": np.full(n, 1705313700000000000, dtype=np.int64),
        "distance": np.full(n, 3.0, dtype=np.float64),
        **{
            key: np.ones(n, dtype=np.uint8)
            for key in [
                "pickup_valid",
                "dropoff_valid",
                "time_allowed",
                "pickup_zone_valid",
                "dropoff_zone_valid",
            ]
        },
    }
    if n > 8:
        inputs["pickup_ns"][1] = NAT
        inputs["pickup_valid"][1] = 0
        inputs["dropoff_ns"][2] = inputs["pickup_ns"][2]
        inputs["dropoff_ns"][3] = inputs["pickup_ns"][3] - 1_000_000_000
        inputs["distance"][4] = np.inf
        inputs["time_allowed"][5] = 0
        inputs["dropoff_zone_valid"][6] = 0
        inputs["distance"][7] = 0
        inputs["distance"][8] = -1
    return inputs


@pytest.mark.parametrize("n", [0, 1, 513])
def test_reference_handles_empty_and_tails(n):
    result = reference_features(feature_fixture(n))
    assert all(len(values) == n for values in result.values())
    if n:
        assert result["duration_seconds"][0] == 900
        assert result["mean_speed_mph"][0] == 12


def test_reference_flags_are_task_specific():
    result = reference_features(feature_fixture())
    assert result["feature_flags"][:9].tolist() == [0, 1, 8, 8, 16, 4, 64, 32, 16]
    assert result["feature_eligible"][:9].tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 0]
    assert result["duration_seconds"][4] == 900  # Bad distance does not erase duration.
    assert result["mean_speed_mph"][6] == 12  # Bad zone does not change a numeric speed.
    assert np.isnan(result["duration_seconds"][5])  # Explicit time-policy exclusion.


def test_reference_does_not_overflow_int64_difference():
    values = feature_fixture(1)
    values["pickup_ns"][0] = np.iinfo(np.int64).min + 1
    values["dropoff_ns"][0] = np.iinfo(np.int64).max
    expected = (int(values["dropoff_ns"][0]) - int(values["pickup_ns"][0])) / 1e9
    assert reference_features(values)["duration_seconds"][0] == pytest.approx(expected, rel=1e-15)


def test_raw_pointer_contract_rejects_bad_shapes_and_dtypes():
    values = feature_fixture()
    values["distance"] = values["distance"].astype("float32")
    with pytest.raises(ValueError, match="float64"):
        validate_inputs(values)
    values = feature_fixture()
    values["distance"] = values["distance"][::2]
    with pytest.raises(ValueError):
        validate_inputs(values)


def test_speedups_require_matching_scope_and_passed_validation():
    trials = pd.DataFrame(
        [
            dict(
                backend="cpu",
                stage="features",
                timing_scope="loaded_compute",
                input_rows=100,
                elapsed_s=8.0,
                validation="passed",
            ),
            dict(
                backend="gpu",
                stage="features",
                timing_scope="loaded_compute",
                input_rows=100,
                elapsed_s=2.0,
                validation="passed",
            ),
            dict(
                backend="gpu",
                stage="features",
                timing_scope="kernel_only",
                input_rows=100,
                elapsed_s=0.1,
                validation="passed",
            ),
            dict(
                backend="gpu",
                stage="features",
                timing_scope="loaded_compute",
                input_rows=100,
                elapsed_s=0.001,
                validation="failed",
            ),
        ]
    )
    summary = summarize_trials(trials)
    gpu = summary[summary.backend == "gpu"].set_index("timing_scope")
    assert gpu.loc["loaded_compute", "speedup_vs_cpu"] == 4
    assert np.isnan(gpu.loc["kernel_only", "speedup_vs_cpu"])


def test_cpu_benchmark_uses_actual_size_and_writes_raw_trials(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    frame = pd.DataFrame(
        {
            "pickup_local": pd.date_range("2024-01-01", periods=20, freq="h"),
            "dropoff_local": pd.date_range("2024-01-01 00:15", periods=20, freq="h"),
            "trip_distance_miles": 3.0,
            "eligible_duration": True,
            "eligible_pickup": True,
            "eligible_dropoff": True,
        }
    )
    frame.to_parquet(canonical / "part-000.parquet", index=False)
    output = tmp_path / "benchmark"
    result = run_benchmarks(
        canonical,
        output,
        {"benchmark": {"sizes": [10, 100], "repeats": 2, "warmups": 0, "include_extended": False}},
    )
    assert result["actual_sizes"] == [10, 20]
    trials = pd.read_parquet(output / "benchmark_runs.parquet")
    assert set(trials.backend) == {"cpu"}
    assert (trials.validation == "passed").all()
    assert len(trials) == 8
    assert result["gpu_measured"] is False


@pytest.mark.skipif(
    os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1",
    reason="Set MOBILITY_RUN_GPU_TESTS=1 on the NVIDIA server",
)
@pytest.mark.parametrize("n", [0, 1, 513])
def test_cuda_library_and_custom_match_reference(n):
    from mobility.backends.cuda_features import (
        require_cupy,
        to_device,
        fused_features,
        library_features,
    )

    cp = require_cupy()
    values = feature_fixture(n)
    expected = reference_features(values)
    for operation in [library_features, fused_features]:
        actual = operation(to_device(values))
        cp.cuda.runtime.deviceSynchronize()
        for name in expected:
            if expected[name].dtype.kind in "ui":
                np.testing.assert_array_equal(cp.asnumpy(actual[name]), expected[name])
            else:
                np.testing.assert_allclose(
                    cp.asnumpy(actual[name]), expected[name], rtol=1e-10, atol=1e-9, equal_nan=True
                )


@pytest.mark.skipif(
    os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1", reason="NVIDIA server required"
)
@pytest.mark.parametrize("scenario", ["mixed", "empty", "all_invalid"])
def test_gpu_aggregation_matches_cpu_with_independent_endpoints(scenario):
    from mobility.aggregate import aggregate_cpu, ZONE_KEYS, OD_KEYS
    from mobility.data.quality import normalize_yellow
    from mobility.backends.gpu import to_device, aggregate, derive_features, synchronize

    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01 23:50", None, "2024-01-02 01:00", "2024-01-02 01:10"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2024-01-02 00:10", "2024-01-02 01:20", None, "2024-01-02 01:20"]
            ),
            "PULocationID": [1, 1, 1, 2],
            "DOLocationID": [2, 2, 2, 999],
            "trip_distance": [3.0, 2.0, np.nan, 0.0],
        }
    )
    if scenario == "empty":
        raw = raw.iloc[:0]
    elif scenario == "all_invalid":
        raw[["tpep_pickup_datetime", "tpep_dropoff_datetime"]] = pd.NaT
        raw[["PULocationID", "DOLocationID"]] = 999
    source = {"sha256": "test", "source_id": "test", "nominal_month": "2024-01"}
    canonical, _ = normalize_yellow(raw, source, {1, 2})
    expected = aggregate_cpu(canonical)
    for custom in [False, True]:
        device = derive_features(to_device(canonical), {"processing": {"custom_cuda": custom}})
        actual = aggregate(device)
        synchronize()
        for gpu, cpu, keys in zip(actual, expected, [ZONE_KEYS, OD_KEYS]):
            actual_rows = gpu.to_pandas().sort_values(keys).reset_index(drop=True)
            expected_rows = cpu.sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(
                actual_rows[expected_rows.columns],
                expected_rows,
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-7,
            )
