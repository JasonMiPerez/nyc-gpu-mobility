"""Matched clustering workloads must preserve inputs and temporal preprocessing."""

import json

import numpy as np
import pandas as pd
import pytest

from mobility.benchmarks.clustering import _prepare_inputs, run_clustering_benchmarks


def trips():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "record_id": [f"bench:{i}" for i in range(80)],
            "dataset_type": "yellow",
            "pickup_local": list(pd.date_range("2024-01-01", periods=60, freq="h"))
            + list(pd.date_range("2024-05-01", periods=20, freq="h")),
            "trip_distance_miles": rng.lognormal(1, 0.4, 80),
            "duration_seconds": rng.lognormal(6, 0.4, 80),
            "eligible_ml": True,
        }
    )


def config():
    return {
        "clustering": {"k_values": [2], "training_max_rows": 30},
        "benchmark": {"warmups": 1, "repeats": 2, "cpu_threads": 1, "seed": 42},
    }


def test_matched_cpu_trials_report_actual_rows_and_separate_fit_prediction():
    frame = trips()
    original = frame.copy(deep=True)
    rows = run_clustering_benchmarks(frame, config(), False, "test")
    pd.testing.assert_frame_equal(frame, original)
    assert len(rows) == 4
    assert all(row["validation"] == "passed" for row in rows)
    assert all(row["elapsed_s"] > 0 for row in rows)
    assert {row["input_rows"] for row in rows if row["stage"] == "kmeans_fit"} == {30}
    assert {row["input_rows"] for row in rows if row["stage"] == "kmeans_prediction"} == {80}
    assert all(
        row["prediction_label_mismatches"] == 0
        for row in rows
        if row["stage"] == "kmeans_prediction"
    )
    metadata = json.loads(rows[0]["model_parameters_json"])
    assert metadata["n_init"] == 1
    assert metadata["training_rows_available"] == 60
    assert metadata["feature_dtype"] == "float32"


def test_benchmark_scaler_and_initial_centers_do_not_use_future_rows():
    first = trips()
    second = first.copy()
    second.loc[
        second.pickup_local.ge("2024-05-01"), ["trip_distance_miles", "duration_seconds"]
    ] = 1e8
    train_a, pred_a, centers_a, meta_a = _prepare_inputs(first, config())
    train_b, pred_b, centers_b, meta_b = _prepare_inputs(second, config())
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(centers_a, centers_b)
    assert meta_a["scaler_mean"] == meta_b["scaler_mean"]
    assert not np.array_equal(pred_a, pred_b)


def test_benchmark_does_not_substitute_future_training_data():
    frame = trips().iloc[60:]
    with pytest.raises(ValueError, match="no future rows"):
        run_clustering_benchmarks(frame, config(), False, "test")
