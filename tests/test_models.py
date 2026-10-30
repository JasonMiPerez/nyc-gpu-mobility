"""Scientific checks: leakage, temporal support, reproducibility, and honest states."""

import json

import numpy as np
import pandas as pd
import pytest

from mobility.models.anomalies import (
    calibrate_threshold,
    consolidate_incidents,
    evaluate_injections,
    fit_anomalies,
    fit_seasonal_baseline,
    score_anomalies,
)
from mobility.models.clustering import fit_clusters
from mobility.models.common import HashSample, feature_matrix, split_bounds


def hourly_frame(times, counts=100, coverage="observed"):
    return pd.DataFrame(
        {
            "dataset_type": "yellow",
            "zone_id": 1,
            "hour": pd.to_datetime(times),
            "pickup_count": counts,
            "pickup_coverage": coverage,
            "time_policy_version": "nyc_local_wall_time_v1",
        }
    )


def weekly_training():
    return hourly_frame(pd.date_range("2024-01-01 10:00", periods=10, freq="7D"))


def trip_fixture():
    rng = np.random.default_rng(42)
    count = 120
    pickups = (
        list(pd.date_range("2024-01-04", periods=60, freq="h"))
        + list(pd.date_range("2024-04-04", periods=30, freq="h"))
        + list(pd.date_range("2024-05-04", periods=30, freq="h"))
    )
    return pd.DataFrame(
        {
            "record_id": [f"source:{i}" for i in range(count)],
            "dataset_type": "yellow",
            "pickup_local": pickups,
            "pickup_zone_id": 1,
            "dropoff_zone_id": 2,
            "trip_distance_miles": rng.lognormal(1, 0.8, count),
            "duration_seconds": rng.lognormal(6, 0.7, count),
            "eligible_ml": True,
        }
    )


def small_config():
    return {
        "clustering": {
            "k_values": [2, 3],
            "training_max_rows": 40,
            "evaluation_max_rows": 20,
            "seed": 42,
        },
        "processing": {"batch_rows": 17, "cpu_threads": 1},
        "dashboard": {"max_points": 10},
    }


def test_hash_sample_does_not_depend_on_batch_order():
    data = trip_fixture()
    whole = HashSample(12, 42)
    whole.add(data)
    chunks = HashSample(12, 42)
    for group in [data.iloc[80:], data.iloc[:30], data.iloc[30:80]]:
        chunks.add(group)
    pd.testing.assert_frame_equal(whole.result(), chunks.result())
    assert whole.metadata() == chunks.metadata()
    assert whole.metadata()["sample_rows"] == 12


def test_temporal_splits_reject_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_bounds({"splits": {"train_end": "2024-04-02"}})


def test_future_observations_cannot_change_seasonal_baseline():
    training = weekly_training()
    future = hourly_frame(["2024-05-06 10:00"], 999999)
    baseline = fit_seasonal_baseline(pd.concat([training, future]), {})
    assert baseline["baseline"].iloc[0] == 100
    assert baseline["history_count"].iloc[0] == 10
    assert baseline["scale"].iloc[0] == 5


def test_missing_training_bins_do_not_count_toward_support():
    training = weekly_training()
    training.loc[:2, "pickup_coverage"] = "missing_source"
    baseline = fit_seasonal_baseline(training, {})
    assert baseline["history_count"].iloc[0] == 7
    scored = score_anomalies(hourly_frame(["2024-05-06 10:00"], 1000), baseline, {})
    assert scored["status"].iloc[0] == "insufficient_history"
    assert pd.isna(scored["score"].iloc[0])
    assert not scored["is_alert"].iloc[0]


def test_zero_observation_and_missing_source_are_different():
    baseline = fit_seasonal_baseline(weekly_training(), {})
    observed = hourly_frame(["2024-05-06 10:00"], 0)
    missing = hourly_frame(["2024-05-13 10:00"], np.nan, "missing_source")
    dst = hourly_frame(["2024-05-20 10:00"], 999, "dst_ambiguous")
    scores = score_anomalies(pd.concat([observed, missing, dst]), baseline, {})
    assert list(scores["status"]) == ["scored", "missing_source", "dst_ambiguous"]
    assert list(scores["is_alert"]) == [True, False, False]
    assert scores.iloc[0]["direction"] == "lower"


def test_alert_requires_both_relative_and_absolute_deviation():
    baseline = fit_seasonal_baseline(weekly_training(), {})
    config = {"anomalies": {"score_threshold": 1, "absolute_threshold": 20}}
    scores = score_anomalies(hourly_frame(["2024-05-06 10:00"], 110), baseline, config)
    assert scores["score"].iloc[0] == 2
    assert not scores["is_alert"].iloc[0]


def test_calibration_budget_and_missing_calibration_are_explicit():
    baseline = fit_seasonal_baseline(weekly_training(), {})
    calibration = hourly_frame(["2024-04-01 10:00"], 125)
    threshold, meta, table = calibrate_threshold(calibration, baseline, {})
    assert threshold == 6
    assert meta["status"] == "calibrated_alert_budget"
    assert table["alert_rows"].tolist() == [1, 1, 1, 0]
    _, empty_meta, _ = calibrate_threshold(calibration.iloc[:0], baseline, {})
    assert empty_meta["status"] == "insufficient_calibration_history"


def test_incidents_do_not_bridge_missing_hours_or_directions():
    scores = pd.DataFrame(
        {
            "model_id": "model",
            "dataset_type": "yellow",
            "zone_id": 1,
            "split": "test",
            "hour": pd.to_datetime(
                ["2024-05-01 10:00", "2024-05-01 11:00", "2024-05-01 13:00", "2024-05-01 12:00"]
            ),
            "direction": ["upper", "upper", "upper", "lower"],
            "is_alert": True,
            "score": [5, 6, 7, -8],
            "deviation": [30, 40, 50, -60],
            "history_count": 10,
        }
    )
    incidents = consolidate_incidents(scores)
    assert sorted(incidents["affected_hours"].tolist()) == [1, 1, 2]
    assert incidents["label"].eq("unresolved").all()


def test_one_month_has_no_claim_of_calibrated_anomaly_results(tmp_path):
    frame = hourly_frame(pd.date_range("2024-01-01", periods=24 * 31, freq="h"))
    result = fit_anomalies(frame, {}, tmp_path)
    assert result["metadata"]["status"] == "insufficient_training_history"
    assert result["metadata"]["calibration"]["status"] == "insufficient_calibration_history"
    assert result["metadata"]["synthetic_evaluation"]["status"] == "no_supported_test_bins"
    assert pd.read_parquet(tmp_path / "anomaly_scores.parquet").empty


def test_injection_evaluation_preserves_original_and_never_calls_background_false_positives():
    training = hourly_frame(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"))
    baseline = fit_seasonal_baseline(training, {})
    test = hourly_frame(pd.date_range("2024-05-01", periods=24 * 7, freq="h"))
    scores = score_anomalies(test, baseline, {})
    original = scores.copy(deep=True)
    metadata, injections = evaluate_injections(scores)
    pd.testing.assert_frame_equal(scores, original)
    assert metadata["status"] == "evaluated"
    assert len(metadata["trials"]) == 3
    assert injections["label"].eq("synthetic_positive").all()
    assert set(injections["requested_duration_hours"]) == {1, 3, 6}
    assert all("untouched_background_alerts" in trial for trial in metadata["trials"])
    assert "not verified false positives" in metadata["limitation"]
    again, again_rows = evaluate_injections(scores)
    assert again == metadata
    pd.testing.assert_frame_equal(injections, again_rows)


def test_clustering_scaler_and_selection_never_use_test_values(tmp_path):
    data = trip_fixture()
    source = tmp_path / "canonical"
    source.mkdir()
    data.to_parquet(source / "part.parquet", index=False)
    config = small_config()
    first = fit_clusters(source, config, tmp_path / "first")
    first_parameters = json.loads((tmp_path / "first" / "cluster_parameters.json").read_text())
    train_sample = HashSample(40, 42)
    train_sample.add(data.loc[data["pickup_local"].lt(pd.Timestamp("2024-04-01"))])
    np.testing.assert_allclose(
        first_parameters["mean"],
        feature_matrix(train_sample.result()).mean(axis=0),
        rtol=1e-5,
        atol=1e-7,
    )
    test_mask = data["pickup_local"].ge(pd.Timestamp("2024-05-01"))
    data.loc[test_mask, ["trip_distance_miles", "duration_seconds"]] = 1e9
    data.to_parquet(source / "part.parquet", index=False)
    second = fit_clusters(source, config, tmp_path / "second")
    second_parameters = json.loads((tmp_path / "second" / "cluster_parameters.json").read_text())
    assert first_parameters == second_parameters
    assert first["metadata"]["selected_k"] == second["metadata"]["selected_k"]
    assert first["metadata"]["prediction_rows"] == len(data)
    summary = pd.read_parquet(tmp_path / "first" / "cluster_summary.parquet")
    assert summary["trip_count"].sum() == len(data)
    plotted = pd.read_parquet(tmp_path / "first" / "trip_plot_sample.parquet")
    assert len(plotted) == 10
    assert plotted["inclusion_fraction"].eq(10 / len(data)).all()


def test_clustering_without_calibration_is_marked_exploratory(tmp_path):
    source = tmp_path / "canonical"
    source.mkdir()
    trip_fixture().iloc[:60].to_parquet(source / "part.parquet", index=False)
    result = fit_clusters(source, small_config(), tmp_path / "models")
    assert result["metadata"]["status"] == "exploratory_no_calibration"
    assert result["metadata"]["selected_k"] == 2
    assert result["metadata"]["test_status"] == "no_held_out_rows"


def test_empty_training_has_explicit_status(tmp_path):
    source = tmp_path / "canonical"
    source.mkdir()
    result = fit_clusters(source, small_config(), tmp_path / "models")
    assert result["metadata"]["status"] == "insufficient_training_rows"
    assert pd.read_parquet(tmp_path / "models" / "trip_plot_sample.parquet").empty
