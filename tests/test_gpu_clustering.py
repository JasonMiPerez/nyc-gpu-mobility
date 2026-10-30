"""Device-native contracts, including real GPU tests enabled on the server."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from mobility.models import gpu_clustering as gpu
from mobility.models.clustering import fit_clusters
from mobility.models.common import (
    PopulationScaler,
    PrioritySample,
    feature_matrix,
    initial_center_indices,
    ml_filter_parameters,
    ml_row_masks,
    priority_values,
)


def trip_fixture():
    rng = np.random.default_rng(12)
    times = (
        list(pd.date_range("2024-01-04", periods=60, freq="h"))
        + list(pd.date_range("2024-04-04", periods=30, freq="h"))
        + list(pd.date_range("2024-05-04", periods=30, freq="h"))
    )
    distances = rng.uniform(0.5, 12, len(times))
    durations = distances * rng.uniform(120, 300, len(times)) + 120
    return pd.DataFrame(
        {
            "record_id": [f"source:{index}" for index in range(len(times))],
            "dataset_type": "yellow",
            "pickup_local": times,
            "pickup_zone_id": 1,
            "dropoff_zone_id": 2,
            "trip_distance_miles": distances,
            "duration_seconds": durations,
            "eligible_ml": True,
            "source_token": np.full(len(times), 0xABCDEF1234567890, dtype=np.uint64),
            "source_row": np.arange(len(times), dtype=np.uint64),
        }
    )


def config():
    return {
        "processing": {"execution_mode": "native", "batch_rows": 17, "cpu_threads": 1},
        "clustering": {
            "k_values": [2, 3],
            "training_max_rows": 40,
            "evaluation_max_rows": 20,
            "seed": 42,
        },
        "dashboard": {"max_points": 10},
    }


def test_splitmix_contract_matches_python_wraparound_and_shared_cpu():
    tokens = np.array([0, 1, 2**64 - 1, 0xABCDEF1234567890], dtype=np.uint64)
    rows = np.array([0, 9, 7, 1000000], dtype=np.uint64)
    mask = 2**64 - 1

    def reference(token, row, seed):
        value = ((int(token) ^ int(row) ^ (seed & mask)) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    for seed in (0, 42, 43, -1):
        actual = gpu.priority_values(tokens, rows, seed, np)
        expected = np.array(
            [reference(token, row, seed) for token, row in zip(tokens, rows)], dtype=np.uint64
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, priority_values(tokens, rows, seed))


def test_gpu_failure_never_falls_back_to_cpu(monkeypatch, tmp_path):
    def unavailable():
        raise RuntimeError("GPU unavailable; no CPU fallback was used")

    monkeypatch.setattr(gpu, "_libraries", unavailable)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        gpu.fit_clusters_gpu(tmp_path / "canonical", config(), tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_native_gpu_reader_requires_numeric_provenance_before_reading_rows(tmp_path):
    trip_fixture().drop(columns=["source_token", "source_row"]).to_parquet(
        tmp_path / "old.parquet", index=False
    )

    class NoRead:
        def read_parquet(self, *args, **kwargs):
            raise AssertionError("Read attempted before canonical schema validation")

    with pytest.raises(ValueError, match="Rebuild canonical data in native mode"):
        list(gpu.canonical_gpu_batches(tmp_path, 100, NoRead()))


@pytest.fixture
def gpu_libraries():
    if os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1":
        pytest.skip("NVIDIA server required")
    return gpu._libraries()


@pytest.mark.gpu
def test_device_samples_masks_features_scaler_and_centers_match_cpu(gpu_libraries):
    cp, cudf, *_ = gpu_libraries
    data = trip_fixture()
    data.loc[0, ["trip_distance_miles", "duration_seconds"]] = [100125.14, 780]
    data.loc[1, "trip_distance_miles"] = 0
    data.loc[2, "duration_seconds"] = 180 * 60 + 1
    data.loc[3, "eligible_ml"] = False
    data.loc[4, "trip_distance_miles"] = np.nan
    data.loc[5, "pickup_local"] = pd.NaT
    device = cudf.from_pandas(data)
    expected_masks = ml_row_masks(data)
    actual_masks = gpu.gpu_eligibility_masks(device, ml_filter_parameters(), cp)
    for name in expected_masks:
        assert isinstance(actual_masks[name], cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(actual_masks[name]), expected_masks[name].to_numpy()
        )
    eligible = data.loc[expected_masks["model_eligible"]].copy()
    cpu_sample = PrioritySample(32, 42)
    cpu_sample.add(eligible)
    sample = gpu.GpuPrioritySample(32, 42, cp, cudf)
    for block in (eligible.iloc[60:], eligible.iloc[:20], eligible.iloc[20:60]):
        sample.add(cudf.from_pandas(block))
    assert sample.metadata() == cpu_sample.metadata()
    assert (
        sample.result()["record_id"].to_arrow().to_pylist()
        == cpu_sample.result()["record_id"].tolist()
    )
    features = gpu.gpu_feature_matrix(sample.result(), cp)
    assert isinstance(features, cp.ndarray) and features.dtype == cp.float64
    host_features = feature_matrix(cpu_sample.result(), np.float64)
    np.testing.assert_allclose(cp.asnumpy(features), host_features, rtol=1e-12, atol=1e-12)
    scaler = PopulationScaler()
    expected_matrix = scaler.fit_transform(host_features).astype(np.float32)
    mean, scale = gpu.gpu_fit_scaler(features, cp)
    matrix = gpu.gpu_scale(features, mean, scale, cp)
    assert isinstance(matrix, cp.ndarray) and matrix.dtype == cp.float32
    np.testing.assert_allclose(cp.asnumpy(mean), scaler.mean_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(scale), scaler.scale_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(matrix), expected_matrix, rtol=2e-6, atol=2e-6)
    for seed in (42, 43, 44):
        centers = gpu.gpu_initial_centers(sample.result(), matrix, 3, seed, cp)
        expected = expected_matrix[initial_center_indices(cpu_sample.result(), 3, seed)]
        np.testing.assert_allclose(cp.asnumpy(centers), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.gpu
def test_device_priority_ties_and_constant_features_are_explicit(gpu_libraries):
    cp, cudf, *_ = gpu_libraries
    frame = trip_fixture().iloc[:4].copy()
    frame["source_token"] = np.array([8, 4, 2, 1], dtype=np.uint64)
    frame["source_row"] = np.array([8, 4, 2, 1], dtype=np.uint64)
    expected = PrioritySample(3, 42)
    expected.add(frame)
    actual = gpu.GpuPrioritySample(3, 42, cp, cudf)
    actual.add(cudf.from_pandas(frame.iloc[:2]))
    actual.add(cudf.from_pandas(frame.iloc[2:]))
    assert actual.metadata() == expected.metadata()
    features = np.tile([np.log1p(3.0), np.log1p(600.0), 0.0, 1.0], (1001, 1))
    mean, scale = gpu.gpu_fit_scaler(cp.asarray(features), cp)
    np.testing.assert_array_equal(cp.asnumpy(mean), features[0])
    np.testing.assert_array_equal(cp.asnumpy(scale), np.ones(4))
    np.testing.assert_array_equal(
        cp.asnumpy(gpu.gpu_scale(cp.asarray(features), mean, scale, cp)),
        np.zeros_like(features, dtype=np.float32),
    )


@pytest.mark.gpu
def test_native_gpu_fit_writes_filtered_artifacts_without_pandas_conversion(
    gpu_libraries, monkeypatch, tmp_path
):
    cp, cudf, *_ = gpu_libraries
    data = trip_fixture()
    bad = data.iloc[:1].copy()
    bad["record_id"] = "bad-distance"
    bad["source_row"] = np.uint64(999)
    bad["trip_distance_miles"] = 100125.14
    bad["duration_seconds"] = 780
    source = tmp_path / "canonical"
    source.mkdir()
    pd.concat([data, bad], ignore_index=True).to_parquet(source / "part.parquet", index=False)

    def forbid_rows(original):
        def guarded(frame, *args, **kwargs):
            if len(frame):
                raise AssertionError(
                    "Per-row pandas conversion attempted in device-native clustering"
                )
            # cuDF's Parquet writer builds pandas-compatible schema metadata from
            # an empty frame. No trip values cross to CPU through that operation.
            return original(frame, *args, **kwargs)

        return guarded

    with monkeypatch.context() as guard:
        guard.setattr(cudf.DataFrame, "to_pandas", forbid_rows(cudf.DataFrame.to_pandas))
        guard.setattr(cudf.Series, "to_pandas", forbid_rows(cudf.Series.to_pandas))
        first = gpu.fit_clusters_gpu(source, config(), tmp_path / "first")
    meta = first["metadata"]
    assert meta["backend"] == "gpu" and meta["execution_mode"] == "native"
    assert meta["prediction_rows"] == len(data)
    assert meta["eligibility"]["excluded_from_canonical_eligible_rows"] == 1
    cpu_reference = fit_clusters(source, config(), tmp_path / "cpu", backend="cpu")
    assert meta["eligibility"] == cpu_reference["metadata"]["eligibility"]
    assert meta["test_status"] == "evaluated"
    for name in [
        "features",
        "sampling",
        "scaler_fit",
        "scaler_transform",
        "kmeans_fit",
        "kmeans_predict",
        "evaluation",
    ]:
        assert meta["execution_evidence"][name]["backend"] == "gpu"
    assert meta["execution_evidence"]["kmeans_fit"]["output_type"] == "cupy"
    assert (tmp_path / "first" / first["tables"]["trip_cluster_assignment"]).is_dir()
    assignments = pd.read_parquet(tmp_path / "first" / first["tables"]["trip_cluster_assignment"])
    assert set(assignments["record_id"]) == set(data["record_id"])
    summary = pd.read_parquet(tmp_path / "first/cluster_summary.parquet").set_index("cluster")
    expected = (
        data.merge(assignments, on="record_id")
        .groupby("cluster")
        .agg(
            trip_count=("record_id", "size"),
            mean_distance_miles=("trip_distance_miles", "mean"),
            mean_duration_minutes=("duration_seconds", lambda values: values.mean() / 60),
        )
    )
    pd.testing.assert_frame_equal(
        summary[expected.columns],
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-10,
        atol=1e-10,
    )
    plotted = pd.read_parquet(tmp_path / "first/trip_plot_sample.parquet")
    assert len(plotted) == 10
    assert plotted["inclusion_fraction"].eq(10 / len(data)).all()
    sample = PrioritySample(40, 42)
    sample.add(data.loc[data["pickup_local"].lt(pd.Timestamp("2024-04-01"))])
    assert meta["samples"]["train"] == sample.metadata()
    first_params = json.loads((tmp_path / "first/cluster_parameters.json").read_text())
    identities = (
        sample.result()
        .iloc[initial_center_indices(sample.result(), meta["selected_k"], 42)]["record_id"]
        .tolist()
    )
    assert first_params["initial_center_record_ids"] == identities
    data.loc[data["pickup_local"].dt.month.eq(5), ["trip_distance_miles", "duration_seconds"]] = [
        80,
        7200,
    ]
    pd.concat([data, bad], ignore_index=True).to_parquet(source / "part.parquet", index=False)
    second = gpu.fit_clusters_gpu(source, config(), tmp_path / "second")
    second_params = json.loads((tmp_path / "second/cluster_parameters.json").read_text())
    assert first_params == second_params
    assert meta["selected_k"] == second["metadata"]["selected_k"]


@pytest.mark.gpu
def test_gpu_metrics_match_cpu_reference_on_device_inputs(gpu_libraries):
    cp, _, _, KMeans, silhouette, ari = gpu_libraries
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    matrix = np.array(
        [[0, 0], [0.1, 0.2], [-0.1, 0.1], [5, 5], [5.1, 5.2], [4.9, 5.1]], dtype=np.float32
    )
    device = cp.asarray(matrix)
    model = KMeans(n_clusters=2, init=cp.asarray(matrix[[0, 3]]), n_init=1, output_type="cupy")
    model.fit(device)
    assert isinstance(model.predict(device), cp.ndarray)
    assessment = gpu._evaluate(model, device, cp, silhouette)
    labels = cp.asnumpy(model.predict(device))
    assert assessment["silhouette"] == pytest.approx(
        silhouette_score(matrix, labels), rel=1e-5, abs=1e-5
    )
    other = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    assert float(ari(cp.asarray(labels), cp.asarray(other))) == pytest.approx(
        adjusted_rand_score(labels, other)
    )
