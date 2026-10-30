"""Seasonal anomalies in mean trip duration, separate from trip-volume alerts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mobility.data.quality import hourly_dst_mask

from .anomalies import (
    calibrate_threshold,
    consolidate_incidents,
    fit_seasonal_baseline,
    score_anomalies,
)
from .common import (
    canonical_batches,
    in_period,
    ml_filter_parameters,
    split_bounds,
    stable_id,
    valid_ml_rows,
    write_json,
)


HOUR_KEYS = ["dataset_type", "zone_id", "hour"]
DEFAULT_PARAMETERS = {
    "min_history": 8,
    "min_trips_per_hour": 20,
    "scale_floor_minutes": 2.0,
    "absolute_threshold_minutes": 5.0,
    "score_threshold": 4.0,
    "threshold_candidates": [3.0, 4.0, 5.0, 6.0],
    "calibration_max_alert_fraction": 0.01,
}


def duration_parameters(config: dict) -> dict:
    supplied = config.get("duration_anomalies", {})
    params = {name: supplied.get(name, default) for name, default in DEFAULT_PARAMETERS.items()}
    for name in ("min_history", "min_trips_per_hour"):
        if int(params[name]) != params[name] or params[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
        params[name] = int(params[name])
    for name in ("scale_floor_minutes", "score_threshold", "absolute_threshold_minutes"):
        value = float(params[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        params[name] = value
    candidates = [float(value) for value in params["threshold_candidates"]]
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("Duration threshold candidates must be finite and positive.")
    params["threshold_candidates"] = sorted(set(candidates))
    budget = float(params["calibration_max_alert_fraction"])
    if not 0 < budget < 1:
        raise ValueError("Duration calibration alert fraction must be between zero and one.")
    params["calibration_max_alert_fraction"] = budget
    return params


def _seasonal_config(config: dict) -> dict:
    """Reuse the numerical detector with minute-valued parameters, without rounding."""
    params = duration_parameters(config)
    return {
        "splits": config.get("splits", {}),
        "anomalies": {
            "min_history": params["min_history"],
            "scale_floor": params["scale_floor_minutes"],
            "absolute_threshold": params["absolute_threshold_minutes"],
            "score_threshold": params["score_threshold"],
            "threshold_candidates": params["threshold_candidates"],
            "calibration_max_alert_fraction": params["calibration_max_alert_fraction"],
        },
    }


def _empty_hours() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_type": pd.Series(dtype="str"),
            "zone_id": pd.Series(dtype="int32"),
            "hour": pd.Series(dtype="datetime64[ns]"),
            "eligible_trip_count": pd.Series(dtype="int64"),
            "duration_sum_minutes": pd.Series(dtype="float64"),
            "mean_duration_minutes": pd.Series(dtype="float64"),
            "coverage": pd.Series(dtype="str"),
            "time_policy_version": pd.Series(dtype="str"),
        }
    )


def _merge_sums(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(HOUR_KEYS, observed=True, sort=False)[
            ["eligible_trip_count", "duration_sum_minutes"]
        ]
        .sum()
        .reset_index()
    )


def aggregate_duration_hours(canonical_dir: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Stream trips into sparse hourly means; absent hours never become zero minutes."""
    bounds = split_bounds(config)
    params = duration_parameters(config)
    filters = ml_filter_parameters(config.get("clustering", {}))
    study = config.get("data", {})
    year = int(study.get("year", 2024))
    months = study.get("months", list(range(1, 7)))
    counts = {
        "canonical_rows_scanned": 0,
        "canonical_eligible_rows": 0,
        "study_filter_eligible_rows": 0,
        "excluded_outside_study_or_splits": 0,
        "excluded_invalid_group_keys": 0,
        "excluded_dst_unsafe_hour": 0,
        "aggregated_eligible_trips": 0,
    }
    parts = []
    for batch in canonical_batches(
        Path(canonical_dir), int(config.get("processing", {}).get("batch_rows", 250_000))
    ):
        counts["canonical_rows_scanned"] += len(batch)
        counts["canonical_eligible_rows"] += int(batch["eligible_ml"].fillna(False).sum())
        selected = valid_ml_rows(batch, filters)
        counts["study_filter_eligible_rows"] += len(selected)
        times = selected["pickup_local"]
        in_splits = pd.Series(False, index=selected.index)
        for period in ("train", "calibration", "test"):
            in_splits |= in_period(times, bounds, period)
        in_study = times.dt.year.eq(year) & times.dt.month.isin(months) & in_splits
        counts["excluded_outside_study_or_splits"] += int((~in_study).sum())
        selected = selected.loc[in_study].copy()
        zone = pd.to_numeric(selected["pickup_zone_id"], errors="coerce")
        valid_keys = (
            selected["dataset_type"].notna() & np.isfinite(zone) & zone.ge(1) & zone.mod(1).eq(0)
        )
        counts["excluded_invalid_group_keys"] += int((~valid_keys).sum())
        selected = selected.loc[valid_keys].copy()
        if selected.empty:
            continue
        selected["zone_id"] = zone.loc[valid_keys].astype("int32")
        selected["hour"] = selected["pickup_local"].dt.floor("h")
        unsafe = hourly_dst_mask(pd.DatetimeIndex(selected["hour"]))
        counts["excluded_dst_unsafe_hour"] += int(unsafe.sum())
        selected = selected.loc[~unsafe].copy()
        if selected.empty:
            continue
        counts["aggregated_eligible_trips"] += len(selected)
        selected["duration_minutes"] = selected["duration_seconds"] / 60.0
        parts.append(
            selected.groupby(HOUR_KEYS, observed=True, sort=False)
            .agg(
                eligible_trip_count=("record_id", "size"),
                duration_sum_minutes=("duration_minutes", "sum"),
            )
            .reset_index()
        )
        # Bound intermediate storage by hourly groups, rather than trip rows.
        if len(parts) >= 8:
            parts = [_merge_sums(parts)]
    hours = _merge_sums(parts) if parts else _empty_hours()
    hours["mean_duration_minutes"] = hours["duration_sum_minutes"] / hours["eligible_trip_count"]
    hours["coverage"] = np.where(
        hours["eligible_trip_count"].ge(params["min_trips_per_hour"]),
        "observed",
        "insufficient_trips",
    )
    hours["time_policy_version"] = config.get("time_policy", "nyc_local_wall_time_v1")
    hours = hours.sort_values(HOUR_KEYS).reset_index(drop=True)
    if int(hours["eligible_trip_count"].sum()) != counts["aggregated_eligible_trips"]:
        raise ValueError("Duration-hour trip counts failed reconciliation.")
    metadata = {
        "counts": counts,
        "hourly_rows": len(hours),
        "sufficient_trip_hours": int(hours["coverage"].eq("observed").sum()),
        "insufficient_trip_hours": int(hours["coverage"].eq("insufficient_trips").sum()),
        "trip_count_reconciliation": "passed",
        "trip_filter_thresholds": filters,
        "study_year": year,
        "study_months": months,
        "coverage_meaning": (
            "Observed means at least the configured number of eligible trip records in that "
            "pickup-zone hour; it does not prove complete demand or source coverage. "
            "Hours with fewer eligible trips are retained as insufficient_trips. Hours with "
            "no eligible observations have no row, are not zero-filled, and cannot contribute "
            "baseline history or receive scores. DST-unsafe local hours are excluded."
        ),
    }
    return hours, metadata


def _detector_hours(hours: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Adapt named duration values to the existing generic seasonal calculations."""
    required = HOUR_KEYS + ["mean_duration_minutes", "eligible_trip_count", "coverage"]
    missing = set(required) - set(hours.columns)
    if missing:
        raise ValueError(f"Duration input lacks {sorted(missing)}")
    frame = hours[required].copy()
    supported = frame["eligible_trip_count"].ge(duration_parameters(config)["min_trips_per_hour"])
    frame.loc[frame["coverage"].eq("observed") & ~supported, "coverage"] = "insufficient_trips"
    return frame.rename(
        columns={"mean_duration_minutes": "pickup_count", "coverage": "pickup_coverage"}
    )


def fit_duration_baseline(hours: pd.DataFrame, config: dict) -> pd.DataFrame:
    return fit_seasonal_baseline(_detector_hours(hours, config), _seasonal_config(config))


def score_duration_anomalies(
    hours: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict,
    model_id: str = "duration-seasonal",
    split: str = "test",
    threshold: float | None = None,
) -> pd.DataFrame:
    scores = score_anomalies(
        _detector_hours(hours, config),
        baseline,
        _seasonal_config(config),
        model_id,
        split,
        threshold,
    )
    scores = scores.merge(
        hours[HOUR_KEYS + ["eligible_trip_count"]], on=HOUR_KEYS, validate="one_to_one"
    )
    scores["metric"] = "mean_duration_minutes"
    scores["unit"] = "minutes"
    return scores


def fit_duration_anomalies(canonical_dir: Path, config: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = split_bounds(config)
    params = duration_parameters(config)
    hours, aggregation = aggregate_duration_hours(canonical_dir, config)
    baseline = fit_duration_baseline(hours, config)
    calibration = hours.loc[in_period(hours["hour"], bounds, "calibration")]
    test = hours.loc[in_period(hours["hour"], bounds, "test")]
    threshold, calibration_metadata, sensitivity = calibrate_threshold(
        _detector_hours(calibration, config), baseline, _seasonal_config(config)
    )
    baseline_hash = hashlib.sha256(
        pd.util.hash_pandas_object(baseline, index=False).values.tobytes()
    ).hexdigest()
    model_parameters = {
        "method": "frozen_seasonal_hourly_mean_duration_median_mad_v1",
        "metric": "mean_duration_minutes",
        "unit": "minutes",
        "baseline_sha256": baseline_hash,
        "threshold": threshold,
        "duration_anomalies_config": params,
        "trip_filter_thresholds": aggregation["trip_filter_thresholds"],
        "splits": {name: value.isoformat() for name, value in bounds.items()},
    }
    model_id = stable_id("duration-seasonal", model_parameters)
    calibration_scores = score_duration_anomalies(
        calibration, baseline, config, model_id, "calibration", threshold
    )
    test_scores = score_duration_anomalies(test, baseline, config, model_id, "test", threshold)
    scores = pd.concat([calibration_scores, test_scores], ignore_index=True)
    incidents = consolidate_incidents(scores)
    incidents["metric"] = "mean_duration_minutes"
    incidents["unit"] = "minutes"
    tables = {}
    for name, table in [
        ("duration_hour", hours),
        ("duration_baseline", baseline),
        ("duration_scores", scores),
        ("duration_incidents", incidents),
        ("duration_calibration", sensitivity),
    ]:
        tables[name] = f"{name}.parquet"
        table.to_parquet(output_dir / tables[name], index=False)
    supported = int(baseline["support_status"].eq("supported").sum())
    scored_test = int(test_scores["status"].eq("scored").sum())
    if not supported:
        status = "insufficient_training_history"
    elif test_scores.empty:
        status = "no_held_out_rows"
    elif not scored_test:
        status = "no_supported_test_bins"
    else:
        status = "evaluated"
    metadata = model_parameters | {
        "model_id": model_id,
        "status": status,
        "computation_backend": "CPU pandas/numpy; bounded canonical Parquet batches",
        "aggregation": aggregation,
        "baseline_groups": len(baseline),
        "supported_baseline_groups": supported,
        "calibration": calibration_metadata,
        "calibration_scored_bins": int(calibration_scores["status"].eq("scored").sum()),
        "test_scored_bins": scored_test,
        "test_alert_bins": int(test_scores["is_alert"].sum()),
        "test_incidents": int(incidents["split"].eq("test").sum()),
        "test_status_counts": {
            str(key): int(value) for key, value in test_scores["status"].value_counts().items()
        },
        "baseline_meaning": (
            "Training-period median of hourly mean eligible-trip durations for the same "
            "pickup zone, weekday, and hour; each sufficiently populated hour has equal weight."
        ),
        "score_meaning": (
            "(Observed hourly mean minutes - baseline minutes) / max(1.4826 * training MAD, "
            "configured minute floor). Alerts also require the configured absolute minute change. "
            "This is a robust deviation score, not an accuracy percentage or probability."
        ),
        "incident_rule": (
            "Same pickup zone/direction/split in strictly adjacent one-hour bins. Missing, "
            "low-trip, or unsupported hours do not bridge incidents. End hour is the last bin's start."
        ),
        "incident_total_deviation_meaning": (
            "Sum of deviations in hourly mean minutes, not total passenger delay or total trip time."
        ),
        "manual_review_status": "not_performed",
        "review_note": "Labels remain unresolved until a person reviews the incident.",
        "limitations": (
            "Changes in route, distance, destination, passenger demand, or trip composition can "
            "change mean duration. This model does not hold routes or distance constant and "
            "does not establish congestion, an event cause, or real-world detection accuracy. "
            "Only trips passing the configured fixed study bounds contribute."
        ),
    }
    write_json(output_dir / "duration_anomalies.json", metadata)
    return {"tables": tables, "metadata": metadata}
