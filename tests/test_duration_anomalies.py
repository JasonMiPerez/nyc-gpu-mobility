"""Duration alerts preserve support, units, counts, and chronological holdouts."""

import numpy as np
import pandas as pd
import pytest

from mobility.models.duration_anomalies import (
    aggregate_duration_hours,
    duration_parameters,
    fit_duration_anomalies,
    fit_duration_baseline,
    score_duration_anomalies,
)


def trips_at(hours, minutes=10.25, count=20, zone=1):
    rows = []
    for hour in pd.to_datetime(hours):
        for position in range(count):
            rows.append(
                {
                    "record_id": f"{zone}:{hour.isoformat()}:{position}",
                    "dataset_type": "yellow",
                    "pickup_local": hour + pd.Timedelta(seconds=position),
                    "pickup_zone_id": zone,
                    "dropoff_zone_id": 2,
                    "trip_distance_miles": 3.0,
                    "duration_seconds": minutes * 60,
                    "eligible_ml": True,
                }
            )
    return pd.DataFrame(rows)


def training_trips():
    mondays = pd.date_range("2024-01-01 10:00", periods=8, freq="7D")
    return trips_at(list(mondays) + list(mondays + pd.Timedelta(hours=1)))


def write_canonical(tmp_path, frame):
    source = tmp_path / "canonical"
    source.mkdir(exist_ok=True)
    frame.to_parquet(source / "part.parquet", index=False)
    return source


def test_duration_hour_reconciles_and_uses_weighted_trip_mean_across_batches(tmp_path):
    first = trips_at(["2024-01-01 10:00"], minutes=10.25, count=20)
    more = trips_at(["2024-01-01 10:00"], minutes=20.25, count=5)
    more["record_id"] += ":additional"
    low_count = trips_at(["2024-01-01 11:00"], count=19)
    source = write_canonical(tmp_path, pd.concat([first, more, low_count], ignore_index=True))
    chunked, metadata = aggregate_duration_hours(source, {"processing": {"batch_rows": 7}})
    whole, _ = aggregate_duration_hours(source, {"processing": {"batch_rows": 1000}})
    pd.testing.assert_frame_equal(chunked, whole)
    assert chunked["eligible_trip_count"].tolist() == [25, 19]
    assert chunked["mean_duration_minutes"].tolist() == [12.25, 10.25]
    assert chunked["coverage"].tolist() == ["observed", "insufficient_trips"]
    assert metadata["counts"]["aggregated_eligible_trips"] == 44
    assert chunked["eligible_trip_count"].sum() == 44
    assert metadata["trip_count_reconciliation"] == "passed"
    assert len(chunked) == 2  # Absent hours are not fabricated as zero-minute trips.


def test_duration_trip_filters_periods_dst_and_canonical_rows_are_preserved(tmp_path):
    normal = trips_at(["2024-01-01 10:00"])
    bad = trips_at(["2024-01-01 10:00"], count=4)
    bad["record_id"] += ":bad"
    bad.loc[0, "trip_distance_miles"] = 100125.14
    bad.loc[0, "duration_seconds"] = 780
    bad.loc[1, "duration_seconds"] = 181 * 60
    bad.loc[2, "trip_distance_miles"] = 0  # Still an eligible duration observation.
    bad.loc[3, "eligible_ml"] = False
    outside = trips_at(["2024-07-01 10:00", "2025-01-06 10:00"], count=1)
    dst = trips_at(["2024-03-10 02:00"], count=1)
    data = pd.concat([normal, bad, outside, dst], ignore_index=True)
    source = write_canonical(tmp_path, data)
    hours, metadata = aggregate_duration_hours(source, {})
    counts = metadata["counts"]
    assert counts == {
        "canonical_rows_scanned": 27,
        "canonical_eligible_rows": 26,
        "study_filter_eligible_rows": 24,
        "excluded_outside_study_or_splits": 2,
        "excluded_invalid_group_keys": 0,
        "excluded_dst_unsafe_hour": 1,
        "aggregated_eligible_trips": 21,
    }
    assert hours["eligible_trip_count"].tolist() == [21]
    assert hours["mean_duration_minutes"].tolist() == [10.25]
    pd.testing.assert_frame_equal(pd.read_parquet(source / "part.parquet"), data)


def test_duration_known_shift_alert_and_low_trip_hours_are_separate(tmp_path):
    data = pd.concat(
        [
            training_trips(),
            trips_at(["2024-04-01 10:00", "2024-04-01 11:00"]),
            trips_at(["2024-05-06 10:00", "2024-05-06 11:00"], minutes=22.25),
            trips_at(["2024-05-06 12:00"], minutes=22.25),
            trips_at(["2024-05-13 10:00"], minutes=30.25, count=19),
        ],
        ignore_index=True,
    )
    source = write_canonical(tmp_path, data)
    output = tmp_path / "models"
    result = fit_duration_anomalies(source, {}, output)
    metadata = result["metadata"]
    assert metadata["metric"] == "mean_duration_minutes"
    assert metadata["unit"] == "minutes"
    assert metadata["status"] == "evaluated"
    assert metadata["threshold"] == 4
    assert metadata["calibration_scored_bins"] == 2
    assert metadata["test_scored_bins"] == 2
    assert metadata["test_alert_bins"] == 2
    assert metadata["test_incidents"] == 1
    scores = pd.read_parquet(output / "duration_scores.parquet")
    test = scores.loc[scores["split"].eq("test")].sort_values("hour")
    assert test["status"].tolist() == [
        "scored",
        "scored",
        "insufficient_history",
        "insufficient_trips",
    ]
    assert test["score"].iloc[:2].tolist() == [6.0, 6.0]
    assert test["baseline"].iloc[:2].tolist() == [10.25, 10.25]
    assert test["eligible_trip_count"].tolist() == [20, 20, 20, 19]
    assert test["is_alert"].tolist() == [True, True, False, False]
    assert test["score"].iloc[2:].isna().all()
    incidents = pd.read_parquet(output / "duration_incidents.parquet")
    assert incidents["affected_hours"].tolist() == [2]
    assert incidents["total_deviation"].tolist() == [24.0]
    assert incidents["label"].eq("unresolved").all()
    assert "not total passenger delay" in metadata["incident_total_deviation_meaning"]
    assert "does not establish congestion" in metadata["limitations"]


def test_duration_test_values_cannot_change_training_baseline_or_calibration(tmp_path):
    future = trips_at(["2024-05-06 10:00", "2024-05-06 11:00"])
    data = pd.concat(
        [training_trips(), trips_at(["2024-04-01 10:00", "2024-04-01 11:00"]), future],
        ignore_index=True,
    )
    source = write_canonical(tmp_path, data)
    first = fit_duration_anomalies(source, {}, tmp_path / "first")
    data.loc[data["pickup_local"].dt.month.eq(5), "duration_seconds"] = 120.5 * 60
    data.to_parquet(source / "part.parquet", index=False)
    second = fit_duration_anomalies(source, {}, tmp_path / "second")
    for name in ("baseline_sha256", "threshold", "model_id", "calibration"):
        assert first["metadata"][name] == second["metadata"][name]
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "first/duration_baseline.parquet"),
        pd.read_parquet(tmp_path / "second/duration_baseline.parquet"),
    )
    assert first["metadata"]["test_alert_bins"] == 0
    assert second["metadata"]["test_alert_bins"] == 2


def test_duration_baseline_requires_eight_populated_observations_not_zero_fill(tmp_path):
    mondays = pd.date_range("2024-01-01 10:00", periods=8, freq="7D")
    data = pd.concat(
        [trips_at(mondays[:7]), trips_at([mondays[7]], count=19), trips_at(["2024-05-06 10:00"])],
        ignore_index=True,
    )
    hours, _ = aggregate_duration_hours(write_canonical(tmp_path, data), {})
    baseline = fit_duration_baseline(hours, {})
    assert baseline["history_count"].tolist() == [7]
    assert baseline["baseline"].tolist() == [10.25]
    assert baseline["support_status"].tolist() == ["insufficient_history"]
    test = hours.loc[hours["hour"].dt.month.eq(5)]
    scores = score_duration_anomalies(test, baseline, {})
    assert scores["status"].tolist() == ["insufficient_history"]
    assert scores["score"].isna().all()


def test_duration_calibration_and_fractional_minute_thresholds(tmp_path):
    # Baseline 10.25 and floor 2 => a 9-minute calibration shift scores 4.5.
    data = pd.concat(
        [training_trips(), trips_at(["2024-04-01 10:00"], minutes=19.25)], ignore_index=True
    )
    source = write_canonical(tmp_path, data)
    result = fit_duration_anomalies(source, {}, tmp_path / "models")
    assert result["metadata"]["threshold"] == 5
    hours, _ = aggregate_duration_hours(source, {})
    baseline = fit_duration_baseline(hours, {})
    probe = hours.iloc[:1].copy()
    probe["mean_duration_minutes"] = 15.5  # Delta 5.25 must not be rounded to 5.
    config = {"duration_anomalies": {"score_threshold": 2, "absolute_threshold_minutes": 5.2}}
    score = score_duration_anomalies(probe, baseline, config)
    assert score["deviation"].iloc[0] == 5.25
    assert score["score"].iloc[0] == 2.625
    assert score["is_alert"].iloc[0]
    config["duration_anomalies"]["absolute_threshold_minutes"] = 5.3
    assert not score_duration_anomalies(probe, baseline, config)["is_alert"].iloc[0]


def test_duration_missing_source_has_explicit_empty_state(tmp_path):
    source = tmp_path / "canonical"
    source.mkdir()
    result = fit_duration_anomalies(source, {}, tmp_path / "models")
    assert result["metadata"]["status"] == "insufficient_training_history"
    assert result["metadata"]["calibration"]["status"] == "insufficient_calibration_history"
    for filename in result["tables"].values():
        assert (tmp_path / "models" / filename).is_file()
    assert pd.read_parquet(tmp_path / "models/duration_scores.parquet").empty


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_duration_rejects_invalid_minute_scale(value):
    with pytest.raises(ValueError, match="finite and positive"):
        duration_parameters({"duration_anomalies": {"scale_floor_minutes": value}})
