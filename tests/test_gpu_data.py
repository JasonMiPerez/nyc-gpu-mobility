"""GPU ingestion parity with the existing, independently implemented CPU rules."""

import os

import numpy as np
import pandas as pd
import pytest

from mobility.aggregate import HISTOGRAM_KEYS, daily_histograms
from mobility.backends import gpu_data
from mobility.data.quality import dst_unsafe_duration, normalize_yellow


SOURCE = {"sha256": "a" * 64, "source_id": "gpu-parity", "nominal_month": "2024-01"}
GPU = pytest.mark.skipif(
    os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1",
    reason="Set MOBILITY_RUN_GPU_TESTS=1 in the NVIDIA environment",
)


def raw_trips(count=1):
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.Series(
                [pd.Timestamp("2024-01-01 10:10")] * count, dtype="datetime64[ns]"
            ),
            "tpep_dropoff_datetime": pd.Series(
                [pd.Timestamp("2024-01-01 10:20")] * count, dtype="datetime64[ns]"
            ),
            "PULocationID": np.ones(count, dtype=float),
            "DOLocationID": np.full(count, 2.0),
            "trip_distance": np.full(count, 2.0),
        }
    )


def timezone_fixture():
    calendar = gpu_data.new_york_transition_calendar()
    starts, ends = calendar["unsafe_start_ns"], calendar["unsafe_end_ns"]
    # Every historical and future transition, including both exact boundaries.
    points = np.concatenate([starts - 1, starts, ends - 1, ends])
    pickup = np.concatenate([points, starts - 3600 * 10**9, [gpu_data.NAT]])
    dropoff = np.concatenate([points + 600 * 10**9, ends + 3600 * 10**9, [gpu_data.NAT]])
    return pickup, dropoff


def test_transition_calendar_matches_pandas_at_every_boundary(monkeypatch):
    # Exercise the exact array algorithm with NumPy, independently of CUDA.
    monkeypatch.setattr(gpu_data, "_gpu_modules", lambda: (None, np))
    pickup, dropoff = timezone_fixture()
    expected = dst_unsafe_duration(
        pd.Series(pickup.view("datetime64[ns]")),
        pd.Series(dropoff.view("datetime64[ns]")),
    )
    actual = gpu_data.dst_unsafe_duration_gpu(pickup, dropoff)
    np.testing.assert_array_equal(actual, expected.to_numpy())
    calendar = gpu_data.new_york_transition_calendar()
    assert np.all(np.diff(calendar["unsafe_start_ns"]) > 0)
    assert len(calendar["offset_ns"]) == len(calendar["unsafe_start_ns"]) + 1


def assert_normalization_parity(raw, source=SOURCE, row_offset=0):
    import cudf

    expected, expected_quality = normalize_yellow(raw, source, {1, 2}, row_offset)
    device, quality = gpu_data.normalize_yellow_gpu(
        cudf.from_pandas(raw), source, {1, 2}, row_offset
    )
    actual = device.to_pandas()
    # Pandas uses nullable Int32; cuDF uses int32 plus a validity bitmap.
    for name in ["pickup_zone_id", "dropoff_zone_id"]:
        assert device[name].dtype == np.dtype("int32")
        actual[name] = actual[name].astype("Int32")
    assert device.quality_flags.dtype == np.dtype("uint32")
    assert device.source_row.dtype == device.source_token.dtype == np.dtype("uint64")
    assert device.pickup_local.dtype == device.dropoff_local.dtype == np.dtype("datetime64[ns]")
    pd.testing.assert_frame_equal(
        actual[expected.columns], expected, check_dtype=False, rtol=1e-14, atol=1e-12
    )
    pd.testing.assert_frame_equal(actual[expected.columns].isna(), expected.isna())
    pd.testing.assert_frame_equal(quality, expected_quality)
    np.testing.assert_array_equal(
        actual.source_row, np.arange(len(raw), dtype="uint64") + row_offset
    )
    assert actual.source_token.eq(int(source["sha256"][:16], 16)).all()
    return device


@GPU
@pytest.mark.gpu
def test_gpu_original_quality_edge_cases():
    raw = raw_trips(24)
    raw.loc[0, "tpep_pickup_datetime"] = pd.NaT
    raw.loc[1, "tpep_dropoff_datetime"] = pd.NaT
    raw.loc[2, "DOLocationID"] = 999
    raw.loc[3, "tpep_pickup_datetime"] = pd.Timestamp("2023-12-31 23:50")
    raw.loc[3, "tpep_dropoff_datetime"] = pd.Timestamp("2024-01-01 00:10")
    raw.loc[4, "tpep_pickup_datetime"] = pd.Timestamp("2024-01-31 23:50")
    raw.loc[4, "tpep_dropoff_datetime"] = pd.Timestamp("2024-02-01 00:10")
    for index, value in enumerate([np.nan, np.inf, -np.inf, -2, 0, 100125.14], 5):
        raw.loc[index, "trip_distance"] = value
    for index, value in enumerate([1.2, np.nan, np.inf, 2**40, -(2**31), 2**31 - 1], 11):
        raw.loc[index, "DOLocationID"] = value
    raw.loc[17, "tpep_dropoff_datetime"] = raw.loc[17, "tpep_pickup_datetime"]
    raw.loc[18, "tpep_dropoff_datetime"] = pd.Timestamp("2024-01-01 10:00")
    raw.loc[19, "tpep_dropoff_datetime"] = pd.Timestamp("2024-01-03 10:20")
    raw.loc[20, "tpep_dropoff_datetime"] += pd.Timedelta(123456789, unit="ns")
    raw.loc[21, "PULocationID"] = np.nan
    raw.loc[22, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]] = pd.NaT
    raw.loc[23, "PULocationID"] = 1.2
    assert_normalization_parity(raw, row_offset=8192)


@GPU
@pytest.mark.gpu
def test_gpu_optional_columns_and_negative_fares():
    assert_normalization_parity(raw_trips(2))
    raw = raw_trips(4)
    raw["fare_amount"] = [-5.0, np.nan, 7.0, -np.inf]
    raw["total_amount"] = [5.0, -1.0, np.nan, np.inf]
    raw["passenger_count"] = [2.0, np.nan, 0.0, -1.0]
    assert_normalization_parity(raw)


@GPU
@pytest.mark.gpu
def test_gpu_dst_endpoint_rules_match_every_calendar_transition():
    import cupy as cp

    pickup, dropoff = timezone_fixture()
    actual = gpu_data.dst_unsafe_duration_gpu(cp.asarray(pickup), cp.asarray(dropoff))
    expected = dst_unsafe_duration(
        pd.Series(pickup.view("datetime64[ns]")),
        pd.Series(dropoff.view("datetime64[ns]")),
    )
    np.testing.assert_array_equal(cp.asnumpy(actual), expected.to_numpy())


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("month", ["2024-03", "2024-11"])
def test_gpu_dst_normalization_preserves_endpoints(month):
    pairs = {
        "2024-03": [
            ("2024-03-10 01:55", "2024-03-10 03:05"),
            ("2024-03-10 02:10", "2024-03-10 03:10"),
            ("2024-03-10 03:00", "2024-03-10 03:10"),
            (None, "2024-03-10 02:10"),
        ],
        "2024-11": [
            ("2024-11-03 00:55", "2024-11-03 02:05"),
            ("2024-11-03 01:15", "2024-11-03 01:45"),
            ("2024-11-03 02:00", "2024-11-03 02:10"),
            ("2024-11-03 01:15", None),
        ],
    }[month]
    raw = raw_trips(len(pairs))
    raw["tpep_pickup_datetime"] = pd.to_datetime([p for p, _ in pairs])
    raw["tpep_dropoff_datetime"] = pd.to_datetime([d for _, d in pairs])
    assert_normalization_parity(raw, {**SOURCE, "nominal_month": month})


@GPU
@pytest.mark.gpu
def test_gpu_row_identity_is_stable_across_batches():
    import cudf

    raw = raw_trips(3)
    whole = assert_normalization_parity(raw, row_offset=2**32)
    first, _ = gpu_data.normalize_yellow_gpu(cudf.from_pandas(raw.iloc[:1]), SOURCE, {1, 2}, 2**32)
    second, _ = gpu_data.normalize_yellow_gpu(
        cudf.from_pandas(raw.iloc[1:]), SOURCE, {1, 2}, 2**32 + 1
    )
    pd.testing.assert_frame_equal(
        cudf.concat([first, second], ignore_index=True).to_pandas(), whole.to_pandas()
    )


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("empty", [False, True])
def test_gpu_histograms_match_cpu_bins_and_overflow(empty):
    distances = [0, 0.99, 1, 2, 3, 5, 10, 20, 49.99, 50, 100125.14, np.inf, -1, np.nan]
    minutes = [0, 4.99, 5, 10, 15, 20, 30, 45, 60, 120, 240, 2000, -1, np.nan]
    raw = raw_trips(len(distances))
    raw["trip_distance"] = distances
    raw["tpep_dropoff_datetime"] = raw.tpep_pickup_datetime + pd.to_timedelta(minutes, unit="min")
    if empty:
        raw = raw.iloc[:0]
    device = assert_normalization_parity(raw)
    canonical, _ = normalize_yellow(raw, SOURCE, {1, 2})
    actual = gpu_data.daily_histograms_gpu(device).to_pandas()
    expected = daily_histograms(canonical)
    pd.testing.assert_frame_equal(
        actual.sort_values(HISTOGRAM_KEYS).reset_index(drop=True),
        expected.sort_values(HISTOGRAM_KEYS).reset_index(drop=True),
        check_dtype=False,
    )


@GPU
@pytest.mark.gpu
def test_gpu_rejects_timezone_aware_input():
    import cudf

    raw = raw_trips()
    raw["tpep_pickup_datetime"] = raw.tpep_pickup_datetime.dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-aware"):
        gpu_data.normalize_yellow_gpu(cudf.from_pandas(raw), SOURCE, {1, 2})


@GPU
@pytest.mark.gpu
def test_gpu_requires_device_dataframe():
    with pytest.raises(TypeError, match="cuDF DataFrame"):
        gpu_data.normalize_yellow_gpu(raw_trips(), SOURCE, {1, 2})
