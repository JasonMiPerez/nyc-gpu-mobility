"""Actual GPU detector parity, including holdouts and seeded synthetic windows."""

import os

import numpy as np
import pandas as pd
import pytest

from mobility.models.anomalies import (
    evaluate_injections,
    fit_anomalies,
    fit_seasonal_baseline,
    score_anomalies,
)
from mobility.models.duration_anomalies import aggregate_duration_hours, fit_duration_anomalies
from mobility.models.gpu_anomalies import (
    aggregate_duration_hours_gpu,
    evaluate_injections_gpu,
    fit_anomalies_gpu,
    fit_duration_anomalies_gpu,
    fit_seasonal_baseline_gpu,
    score_anomalies_gpu,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1",
        reason="Set MOBILITY_RUN_GPU_TESTS=1 in the NVIDIA environment",
    ),
]


def activity_fixture():
    times = (
        list(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"))
        + list(pd.date_range("2024-04-01", periods=48, freq="h"))
        + list(pd.date_range("2024-05-01", periods=24 * 14, freq="h"))
    )
    rows = []
    for zone in (1, 2):
        for hour in times:
            count = float(100 + zone * 10 + hour.hour + hour.day % 5)
            coverage = "observed"
            if hour.month == 5 and hour.hour == 12:
                count += 100
            if hour.month == 5 and hour.day == 3 and hour.hour == 10:
                count, coverage = np.nan, "missing_source"
            if hour.month == 5 and hour.day == 5 and hour.hour == 11:
                count = 0
            rows.append(
                {
                    "dataset_type": "yellow",
                    "zone_id": zone,
                    "hour": hour,
                    "pickup_count": count,
                    "pickup_coverage": coverage,
                    "time_policy_version": "nyc_local_wall_time_v1",
                }
            )
    return pd.DataFrame(rows)


def assert_table(actual, expected, keys):
    if hasattr(actual, "to_pandas"):
        actual = actual.to_pandas()
    # Float reductions may change a model's byte-level baseline hash. Compare
    # scientific results using keys and tolerances, not the resulting model ID.
    actual = actual.drop(columns="model_id", errors="ignore")
    expected = expected.drop(columns="model_id", errors="ignore")
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True)[expected.columns],
        expected.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-10,
    )


def test_gpu_baseline_scores_and_injections_match_cpu():
    import cudf

    frame = activity_fixture()
    device = cudf.from_pandas(frame)
    baseline = fit_seasonal_baseline(frame, {})
    gpu_baseline = fit_seasonal_baseline_gpu(device, {})
    assert_table(gpu_baseline, baseline, ["dataset_type", "zone_id", "hour_of_week"])
    test = frame.loc[frame.hour.dt.month.eq(5)]
    scores = score_anomalies(test, baseline, {})
    gpu_scores = score_anomalies_gpu(cudf.from_pandas(test), gpu_baseline, {})
    assert_table(gpu_scores, scores, ["dataset_type", "zone_id", "hour"])
    expected_metadata, expected_rows = evaluate_injections(scores)
    metadata, rows = evaluate_injections_gpu(gpu_scores)
    assert metadata == expected_metadata
    assert_table(rows, expected_rows, ["seed", "event_id", "hour"])
    # Evaluation must preserve the original score table.
    assert_table(gpu_scores, scores, ["dataset_type", "zone_id", "hour"])


def test_gpu_volume_artifacts_and_calibration_match_cpu(tmp_path):
    import cudf

    frame = activity_fixture()
    cpu_dir, gpu_dir = tmp_path / "cpu", tmp_path / "gpu"
    expected = fit_anomalies(frame, {}, cpu_dir)
    actual = fit_anomalies_gpu(cudf.from_pandas(frame), {}, gpu_dir)
    for name in (
        "status",
        "threshold",
        "baseline_groups",
        "supported_baseline_groups",
        "calibration",
        "calibration_scored_bins",
        "test_scored_bins",
        "test_alert_bins",
        "test_incidents",
        "test_status_counts",
        "synthetic_evaluation",
    ):
        assert actual["metadata"][name] == expected["metadata"][name]
    keys = {
        "anomaly_baseline": ["dataset_type", "zone_id", "hour_of_week"],
        "anomaly_scores": ["dataset_type", "zone_id", "hour"],
        "anomaly_incidents": ["dataset_type", "zone_id", "start_hour", "direction"],
        "anomaly_calibration": ["threshold"],
        "synthetic_injections": ["seed", "event_id", "hour"],
    }
    for name, sort in keys.items():
        assert_table(
            pd.read_parquet(gpu_dir / actual["tables"][name]),
            pd.read_parquet(cpu_dir / expected["tables"][name]),
            sort,
        )


def duration_fixture():
    rows = []
    training = list(pd.date_range("2024-01-01 10:00", periods=10, freq="7D"))
    times = training + [
        pd.Timestamp("2024-04-01 10:00"),
        pd.Timestamp("2024-05-06 10:00"),
        pd.Timestamp("2024-05-13 10:00"),
        pd.Timestamp("2024-03-10 02:00"),
        pd.Timestamp("2024-07-01 10:00"),
    ]
    for hour in times:
        count = 19 if hour.day == 13 and hour.month == 5 else 25
        minutes = 24.25 if hour.month == 5 else 10.25
        for position in range(count):
            rows.append(
                {
                    "record_id": f"{hour.isoformat()}:{position}",
                    "dataset_type": "yellow",
                    "pickup_local": hour + pd.Timedelta(seconds=position),
                    "pickup_zone_id": 1,
                    "dropoff_zone_id": 2,
                    "trip_distance_miles": 3.0,
                    "duration_seconds": minutes * 60,
                    "eligible_ml": True,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[0, "trip_distance_miles"] = 100125.14
    frame.loc[1, "duration_seconds"] = 181 * 60
    frame.loc[2, "eligible_ml"] = False
    frame.loc[3, "trip_distance_miles"] = 0
    frame["source_row"] = np.arange(len(frame), dtype="uint64")
    frame["source_token"] = np.uint64(0xFFFFFFFFFFFFFFFF)
    return frame


def test_gpu_duration_aggregation_and_artifacts_match_cpu(tmp_path):
    source = tmp_path / "canonical"
    source.mkdir()
    duration_fixture().to_parquet(source / "part.parquet", index=False, row_group_size=50)
    config = {"processing": {"batch_rows": 37}}
    cpu_hours, cpu_metadata = aggregate_duration_hours(source, config)
    gpu_hours, gpu_metadata = aggregate_duration_hours_gpu(source, config)
    assert gpu_metadata == cpu_metadata
    assert_table(gpu_hours, cpu_hours, ["dataset_type", "zone_id", "hour"])
    cpu_dir, gpu_dir = tmp_path / "cpu", tmp_path / "gpu"
    expected = fit_duration_anomalies(source, config, cpu_dir)
    actual = fit_duration_anomalies_gpu(source, config, gpu_dir)
    for name in (
        "status",
        "threshold",
        "aggregation",
        "calibration",
        "test_scored_bins",
        "test_alert_bins",
        "test_incidents",
        "test_status_counts",
    ):
        assert actual["metadata"][name] == expected["metadata"][name]
    for name, keys in {
        "duration_hour": ["dataset_type", "zone_id", "hour"],
        "duration_baseline": ["dataset_type", "zone_id", "hour_of_week"],
        "duration_scores": ["dataset_type", "zone_id", "hour"],
        "duration_incidents": ["dataset_type", "zone_id", "start_hour"],
        "duration_calibration": ["threshold"],
    }.items():
        assert_table(
            pd.read_parquet(gpu_dir / actual["tables"][name]),
            pd.read_parquet(cpu_dir / expected["tables"][name]),
            keys,
        )


def test_gpu_future_values_do_not_change_frozen_baseline_or_threshold(tmp_path):
    import cudf

    frame = activity_fixture()
    first = fit_anomalies_gpu(cudf.from_pandas(frame), {}, tmp_path / "first")
    frame.loc[frame.hour.dt.month.eq(5), "pickup_count"] = 1e8
    second = fit_anomalies_gpu(cudf.from_pandas(frame), {}, tmp_path / "second")
    for name in ("baseline_sha256", "threshold", "calibration", "model_id"):
        assert first["metadata"][name] == second["metadata"][name]


def test_gpu_no_holdout_or_duration_data_has_explicit_empty_artifacts(tmp_path):
    import cudf

    frame = activity_fixture()
    frame = frame.loc[frame.hour.dt.month.eq(1)]
    volume = fit_anomalies_gpu(cudf.from_pandas(frame), {}, tmp_path / "volume")
    assert volume["metadata"]["status"] == "insufficient_training_history"
    assert volume["metadata"]["synthetic_evaluation"]["status"] == "no_supported_test_bins"
    source = tmp_path / "canonical"
    source.mkdir()
    duration = fit_duration_anomalies_gpu(source, {}, tmp_path / "duration")
    assert duration["metadata"]["status"] == "insufficient_training_history"
    for name in duration["tables"].values():
        assert (tmp_path / "duration" / name).exists()


def test_gpu_rejects_duplicate_hour_keys_and_cpu_input():
    import cudf

    frame = activity_fixture().iloc[:1]
    with pytest.raises(TypeError, match="cuDF DataFrame"):
        fit_seasonal_baseline_gpu(frame, {})
    with pytest.raises(ValueError, match="unique"):
        fit_seasonal_baseline_gpu(cudf.from_pandas(pd.concat([frame, frame])), {})
