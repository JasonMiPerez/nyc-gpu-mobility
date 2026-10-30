"""Device-native seasonal detectors with bounded transfers for reporting.

cuDF/CuPy perform the hourly aggregation, training medians and MADs, calibration,
scoring, and synthetic perturbations. The host handles small metadata, incident
rows already selected as alerts, and seeded choices of tiny injection windows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mobility.backends.gpu import require_cudf
from mobility.backends.cuda_features import require_cupy
from mobility.backends.gpu_data import NAT, dst_unsafe_duration_gpu

from .anomalies import KEYS, consolidate_incidents
from .common import ml_filter_parameters, split_bounds, stable_id, write_json
from .duration_anomalies import HOUR_KEYS, _empty_hours, _seasonal_config, duration_parameters
from .gpu_clustering import canonical_gpu_batches, gpu_eligibility_masks


def _period(times, bounds, name):
    return (times >= bounds[f"{name}_start"]) & (times < bounds[f"{name}_end"])


def prepare_hours_gpu(zone_hour):
    cudf = require_cudf()
    if not isinstance(zone_hour, cudf.DataFrame):
        raise TypeError("Native GPU anomalies require a cuDF DataFrame")
    required = ["dataset_type", "zone_id", "hour", "pickup_count", "pickup_coverage"]
    missing = set(required) - set(zone_hour.columns)
    if missing:
        raise ValueError(f"Anomaly input lacks {sorted(missing)}")
    selected = required + (["time_policy_version"] if "time_policy_version" in zone_hour else [])
    frame = zone_hour[selected].copy()
    frame["hour"] = cudf.to_datetime(frame["hour"]).astype("datetime64[ns]")
    if bool(frame.hour.isna().any()):
        raise ValueError("Hourly activity keys cannot contain a missing timestamp.")
    if bool(frame.duplicated(HOUR_KEYS).any()):
        raise ValueError("Hourly activity keys must be unique before anomaly fitting.")
    frame["hour_of_week"] = (frame.hour.dt.weekday * 24 + frame.hour.dt.hour).astype("int32")
    frame["pickup_count"] = cudf.to_numeric(frame.pickup_count, errors="coerce")
    return frame


def _finite_nonnegative(series):
    cp, cudf = require_cupy(), require_cudf()
    values = series.astype("float64").to_cupy(na_value=np.nan)
    return cudf.Series(cp.isfinite(values) & (values >= 0), index=series.index)


def fit_seasonal_baseline_gpu(zone_hour, config):
    cudf, cp = require_cudf(), require_cupy()
    frame = prepare_hours_gpu(zone_hour)
    bounds = split_bounds(config)
    params = config.get("anomalies", {})
    floor, minimum = float(params.get("scale_floor", 5)), int(params.get("min_history", 8))
    if floor <= 0 or minimum < 1:
        raise ValueError("Scale floor and minimum history must be positive.")
    valid = (
        _period(frame.hour, bounds, "train")
        & frame.pickup_coverage.eq("observed")
        & _finite_nonnegative(frame.pickup_count)
    ).fillna(False)
    training = frame.loc[valid]
    grouped = training.groupby(KEYS)
    baseline = (
        grouped[["pickup_count"]]
        .median()
        .reset_index()
        .rename(columns={"pickup_count": "baseline"})
    )
    counts = grouped.size().rename("history_count").reset_index()
    baseline = baseline.merge(counts, on=KEYS)
    baseline["history_count"] = baseline.history_count.astype("int64")
    deviations = training.merge(baseline[KEYS + ["baseline"]], on=KEYS)
    deviations["absolute_deviation"] = (deviations.pickup_count - deviations.baseline).abs()
    mad = (
        deviations.groupby(KEYS)[["absolute_deviation"]]
        .median()
        .reset_index()
        .rename(columns={"absolute_deviation": "mad"})
    )
    baseline = baseline.merge(mad, on=KEYS).sort_values(KEYS).reset_index(drop=True)
    baseline["scale"] = cudf.Series(
        cp.maximum(1.4826 * baseline["mad"].to_cupy(na_value=np.nan), floor), index=baseline.index
    )
    baseline["support_status"] = "insufficient_history"
    baseline.loc[baseline.history_count >= minimum, "support_status"] = "supported"
    return baseline[KEYS + ["baseline", "history_count", "mad", "scale", "support_status"]]


def score_anomalies_gpu(
    zone_hour, baseline, config, model_id="seasonal", split="test", threshold=None
):
    frame = prepare_hours_gpu(zone_hour)
    params = config.get("anomalies", {})
    minimum = int(params.get("min_history", 8))
    threshold = float(threshold if threshold is not None else params.get("score_threshold", 4))
    absolute = float(params.get("absolute_threshold", 20))
    if threshold <= 0 or absolute < 0:
        raise ValueError("Anomaly thresholds must be positive (absolute threshold may be zero).")
    scores = frame.merge(baseline, on=KEYS, how="left").rename(
        columns={"pickup_count": "observed", "pickup_coverage": "coverage"}
    )
    scores["history_count"] = scores.history_count.fillna(0).astype("int32")
    scores["status"] = scores.coverage.fillna("missing_coverage")
    observed = scores.coverage.eq("observed").fillna(False)
    has_value = _finite_nonnegative(scores.observed)
    supported = (scores.history_count >= minimum) & scores.baseline.notna()
    scores.loc[observed & ~has_value, "status"] = "missing_observation"
    scores.loc[observed & has_value & ~supported, "status"] = "insufficient_history"
    valid = observed & has_value & supported
    scores.loc[valid, "status"] = "scored"
    scores["deviation"] = scores.observed - scores.baseline
    scores["score"] = (scores.deviation / scores["scale"]).where(valid)
    scores["threshold"], scores["absolute_threshold"] = threshold, absolute
    scores["is_alert"] = (
        valid & (scores.score.abs() >= threshold) & (scores.deviation.abs() >= absolute)
    ).fillna(False)
    scores["direction"] = "none"
    scores.loc[scores.is_alert & (scores.deviation >= 0), "direction"] = "upper"
    scores.loc[scores.is_alert & (scores.deviation < 0), "direction"] = "lower"
    scores["model_id"], scores["metric"], scores["split"] = model_id, "pickup_count", split
    return scores.drop(columns="support_status")


def calibrate_threshold_gpu(calibration, baseline, config):
    params = config.get("anomalies", {})
    default = float(params.get("score_threshold", 4))
    candidates = sorted(
        set(map(float, params.get("threshold_candidates", [3, 4, 5, 6]))) | {default}
    )
    budget = float(params.get("calibration_max_alert_fraction", 0.01))
    if not 0 < budget < 1:
        raise ValueError("Calibration alert-fraction budget must be between zero and one.")
    scores = score_anomalies_gpu(
        calibration, baseline, config, split="calibration", threshold=default
    )
    valid = scores.status.eq("scored")
    count = int(valid.sum())
    rows = []
    for candidate in candidates:
        flags = (
            valid
            & (scores.score.abs() >= candidate)
            & (scores.deviation.abs() >= scores.absolute_threshold)
        )
        alerts = int(flags.fillna(False).sum())
        rows.append(
            {
                "threshold": candidate,
                "scored_rows": count,
                "alert_rows": alerts,
                "alert_fraction": alerts / count if count else None,
            }
        )
    table = pd.DataFrame(rows)  # Only the four/few reduced candidate totals.
    if not count:
        return (
            default,
            {
                "status": "insufficient_calibration_history",
                "selected_threshold": default,
                "selection_rule": "Configured threshold retained; no calibration claim.",
                "target_alert_fraction": budget,
            },
            table,
        )
    acceptable = [
        row for row in rows if row["threshold"] >= default and row["alert_fraction"] <= budget
    ]
    chosen = acceptable[0]["threshold"] if acceptable else max(default, max(candidates))
    return (
        chosen,
        {
            "status": "calibrated_alert_budget" if acceptable else "calibration_budget_not_met",
            "selected_threshold": chosen,
            "target_alert_fraction": budget,
            "selection_rule": "Smallest candidate at least the configured threshold meeting the calibration alert-fraction budget; highest candidate if none meets it.",
            "meaning": "Controls calibration alert burden, not a false-positive probability or detection accuracy.",
        },
        table,
    )


def evaluate_injections_gpu(test_scores, seeds=(42, 43, 44)):
    """Use the CPU oracle's seeded windows, with all count changes/scoring on device."""
    cudf, cp = require_cudf(), require_cupy()
    eligible = (
        test_scores.loc[test_scores.status.eq("scored")]
        .sort_values(["dataset_type", "zone_id", "hour"])
        .reset_index(drop=True)
    )
    if not len(eligible):
        return {
            "status": "no_supported_test_bins",
            "trials": [],
            "limitation": "No held-out synthetic sensitivity evaluation was possible.",
        }, pd.DataFrame(columns=["seed", "event_id", "dataset_type", "zone_id", "hour", "label"])
    # At most one count per dataset/zone, not the full held-out hourly table.
    groups = eligible.groupby(["dataset_type", "zone_id"]).size().to_pandas().sort_index()
    sizes = groups.to_numpy(dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    background_alerts = int(eligible.is_alert.sum())
    scenarios = [
        (direction, magnitude, duration)
        for direction in ("spike", "drop")
        for magnitude in (0.5, 1.0, 2.0)
        for duration in (1, 3, 6)
    ]
    injections, trials = [], []
    for seed in seeds:
        rng, used, trial_rows = np.random.default_rng(seed), set(), []
        for event_index, (direction, magnitude, duration) in enumerate(scenarios):
            chosen = None
            for _ in range(30):
                group = int(rng.integers(len(sizes)))
                if sizes[group] < duration:
                    continue
                start = int(offsets[group] + rng.integers(sizes[group] - duration + 1))
                positions = np.arange(start, start + duration)
                hours = eligible.iloc[start : start + duration].hour.to_pandas()
                if hours.diff().dropna().eq(pd.Timedelta(hours=1)).all() and not used.intersection(
                    positions.tolist()
                ):
                    chosen = positions
                    break
            if chosen is None:
                continue
            altered = eligible.iloc[int(chosen[0]) : int(chosen[-1]) + 1].copy()
            values = altered.observed.astype("float64").to_cupy()
            shift = cp.maximum(1, cp.rint(altered.baseline.to_cupy() * magnitude))
            changed = values + shift if direction == "spike" else cp.maximum(0, values - shift)
            mask = changed != values
            # Only up to six chosen positions cross to the host for overlap control.
            used.update(chosen[cp.asnumpy(mask)].tolist())
            altered = altered.loc[cudf.Series(mask, index=altered.index)].copy()
            if not len(altered):
                continue
            altered["original_count"] = altered.observed
            altered["perturbed_count"] = cudf.Series(changed[mask], index=altered.index)
            altered["original_score"], altered["original_alert"] = altered.score, altered.is_alert
            deviation = altered.perturbed_count - altered.baseline
            altered["perturbed_score"] = deviation / altered["scale"]
            altered["perturbed_alert"] = (altered.perturbed_score.abs() >= altered.threshold) & (
                deviation.abs() >= altered.absolute_threshold
            )
            altered["seed"], altered["event_id"] = seed, f"injection-{seed}-{event_index}"
            altered["injection_direction"], altered["magnitude_baseline_multiple"] = (
                direction,
                magnitude,
            )
            altered["requested_duration_hours"], altered["label"] = duration, "synthetic_positive"
            trial_rows.append(altered)
        if not trial_rows:
            trials.append(
                {
                    "seed": seed,
                    "status": "no_changeable_windows",
                    "events_attempted": len(scenarios),
                }
            )
            continue
        combined = cudf.concat(trial_rows, ignore_index=True)
        before, after = int(combined.original_alert.sum()), int(combined.perturbed_alert.sum())
        unmatched = background_alerts - before
        post_total = unmatched + after
        # Boolean max is equivalent to any here; cuDF 25.10 lacks groupby.any.
        detections = combined.groupby("event_id")["perturbed_alert"].max()
        trials.append(
            {
                "seed": seed,
                "status": "evaluated",
                "events_attempted": len(scenarios),
                "injected_events": len(detections),
                "injected_bins": len(combined),
                "detected_injected_bins": after,
                "injection_relative_recall": after / len(combined),
                "injection_relative_precision": after / post_total if post_total else None,
                "event_recall": float(detections.mean()),
                "event_matching_rule": "Any alerted changed bin within the injected event window.",
                "newly_alerted_injected_bins": int(
                    (combined.perturbed_alert & ~combined.original_alert).sum()
                ),
                "already_alerted_injected_bins": before,
                "untouched_background_alerts": unmatched,
                "post_injection_total_alerts": post_total,
            }
        )
        injections.append(combined)
    columns = [
        "seed",
        "event_id",
        "dataset_type",
        "zone_id",
        "hour",
        "injection_direction",
        "magnitude_baseline_multiple",
        "requested_duration_hours",
        "original_count",
        "perturbed_count",
        "baseline",
        "scale",
        "threshold",
        "absolute_threshold",
        "original_score",
        "perturbed_score",
        "original_alert",
        "perturbed_alert",
        "label",
    ]
    result = (
        cudf.concat(injections, ignore_index=True)[columns]
        if injections
        else pd.DataFrame(columns=columns)
    )
    return {
        "status": "evaluated" if len(result) else "no_changeable_windows",
        "trials": trials,
        "scope": "Detector-level perturbations of held-out aggregate counts, not raw-trip perturbations.",
        "scenario_design": "Both spike/drop; 0.5, 1, 2 times baseline; 1, 3, 6 consecutive hours; independent fixed seeds; drop counts clipped at zero.",
        "limitation": "Injected labels are synthetic. Untouched real alerts are unmatched background alerts, not verified false positives. These values do not measure real-world precision or recall.",
    }, result


def _baseline_hash(baseline):
    # One seasonal value per zone/week-hour (about 44k rows), never raw trips.
    return hashlib.sha256(
        pd.util.hash_pandas_object(baseline.to_pandas(), index=False).values.tobytes()
    ).hexdigest()


def _write_tables(output_dir, entries):
    tables = {}
    for name, table in entries:
        tables[name] = f"{name}.parquet"
        table.to_parquet(output_dir / tables[name], index=False)
    return tables


def _summary(baseline, calibration, test, incidents):
    supported = int(baseline.support_status.eq("supported").sum())
    scored = int(test.status.eq("scored").sum())
    status = (
        "insufficient_training_history"
        if not supported
        else "no_held_out_rows"
        if not len(test)
        else "no_supported_test_bins"
        if not scored
        else "evaluated"
    )
    return {
        "status": status,
        "computation_backend": "GPU cuDF/CuPy; bounded host reporting",
        "baseline_groups": len(baseline),
        "supported_baseline_groups": supported,
        "calibration_scored_bins": int(calibration.status.eq("scored").sum()),
        "test_scored_bins": scored,
        "test_alert_bins": int(test.is_alert.sum()),
        "test_incidents": int(incidents["split"].eq("test").sum()),
        "test_status_counts": {
            str(k): int(v) for k, v in test.status.value_counts().to_pandas().items()
        },
    }


def fit_anomalies_gpu(zone_hour, config, output_dir):
    cudf = require_cudf()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = split_bounds(config)
    frame = prepare_hours_gpu(zone_hour)
    baseline = fit_seasonal_baseline_gpu(zone_hour, config)
    calibration = frame.loc[_period(frame.hour, bounds, "calibration")]
    test = frame.loc[_period(frame.hour, bounds, "test")]
    threshold, calibration_metadata, sensitivity = calibrate_threshold_gpu(
        calibration, baseline, config
    )
    parameters = {
        "method": "frozen_seasonal_median_mad_v1",
        "metric": "pickup_count",
        "baseline_sha256": _baseline_hash(baseline),
        "threshold": threshold,
        "anomalies_config": config.get("anomalies", {}),
        "splits": {name: value.isoformat() for name, value in bounds.items()},
    }
    model_id = stable_id("seasonal", parameters)
    calibration_scores = score_anomalies_gpu(
        calibration, baseline, config, model_id, "calibration", threshold
    )
    test_scores = score_anomalies_gpu(test, baseline, config, model_id, "test", threshold)
    scores = cudf.concat([calibration_scores, test_scores], ignore_index=True)
    incidents = consolidate_incidents(scores.loc[scores.is_alert].to_pandas())
    injection_metadata, injection_rows = evaluate_injections_gpu(test_scores)
    tables = _write_tables(
        output_dir,
        [
            ("anomaly_baseline", baseline),
            ("anomaly_scores", scores),
            ("anomaly_incidents", incidents),
            ("anomaly_calibration", sensitivity),
            ("synthetic_injections", injection_rows),
        ],
    )
    metadata = (
        parameters
        | _summary(baseline, calibration_scores, test_scores, incidents)
        | {
            "model_id": model_id,
            "calibration": calibration_metadata,
            "score_meaning": "Robust deviation score; not a calibrated z-test or p-value.",
            "incident_rule": "Same dataset/zone/direction/split in strictly adjacent one-hour bins; no missing-hour gaps bridged. end_hour is the last affected bin's start.",
            "manual_review_status": "not_performed",
            "review_note": "Incident labels remain unresolved until a person records a review; no real-world accuracy claim.",
            "synthetic_evaluation": injection_metadata,
        }
    )
    write_json(output_dir / "anomalies.json", metadata)
    return {"tables": tables, "metadata": metadata}


def aggregate_duration_hours_gpu(canonical_dir, config):
    cudf, cp = require_cudf(), require_cupy()
    bounds, params = split_bounds(config), duration_parameters(config)
    filters = ml_filter_parameters(config.get("clustering", {}))
    study = config.get("data", {})
    year, months = int(study.get("year", 2024)), study.get("months", list(range(1, 7)))
    counts = dict.fromkeys(
        [
            "canonical_rows_scanned",
            "canonical_eligible_rows",
            "study_filter_eligible_rows",
            "excluded_outside_study_or_splits",
            "excluded_invalid_group_keys",
            "excluded_dst_unsafe_hour",
            "aggregated_eligible_trips",
        ],
        0,
    )
    parts = []

    def merge(parts):
        return (
            cudf.concat(parts, ignore_index=True)
            .groupby(HOUR_KEYS)[["eligible_trip_count", "duration_sum_minutes"]]
            .sum()
            .reset_index()
        )

    for batch in canonical_gpu_batches(
        Path(canonical_dir), int(config.get("processing", {}).get("batch_rows", 250_000)), cudf
    ):
        counts["canonical_rows_scanned"] += len(batch)
        masks = gpu_eligibility_masks(batch, filters, cp)
        counts["canonical_eligible_rows"] += int(masks["canonical_eligible"].sum())
        selected = batch.loc[cudf.Series(masks["model_eligible"], index=batch.index)].copy()
        counts["study_filter_eligible_rows"] += len(selected)
        times = selected.pickup_local
        in_splits = (
            _period(times, bounds, "train")
            | _period(times, bounds, "calibration")
            | _period(times, bounds, "test")
        )
        in_study = ((times.dt.year == year) & times.dt.month.isin(months) & in_splits).fillna(False)
        counts["excluded_outside_study_or_splits"] += int((~in_study).sum())
        selected = selected.loc[in_study].copy()
        zone = cudf.to_numeric(selected.pickup_zone_id, errors="coerce").astype("float64")
        values = zone.to_cupy(na_value=np.nan)
        valid = (
            cudf.Series(
                cp.isfinite(values) & (values >= 1) & (values % 1 == 0), index=selected.index
            )
            & selected.dataset_type.notna()
        )
        counts["excluded_invalid_group_keys"] += int((~valid).sum())
        selected = selected.loc[valid].copy()
        if not len(selected):
            continue
        selected["zone_id"] = zone.loc[valid].astype("int32")
        selected["hour"] = selected.pickup_local.dt.floor("h").astype("datetime64[ns]")
        hour_ns = selected.hour.astype("int64").to_cupy(na_value=NAT)
        unsafe = dst_unsafe_duration_gpu(hour_ns, hour_ns)
        counts["excluded_dst_unsafe_hour"] += int(unsafe.sum())
        selected = selected.loc[cudf.Series(~unsafe, index=selected.index)].copy()
        if not len(selected):
            continue
        counts["aggregated_eligible_trips"] += len(selected)
        selected["duration_sum_minutes"] = selected.duration_seconds / 60.0
        selected["eligible_trip_count"] = np.int64(1)
        parts.append(
            selected.groupby(HOUR_KEYS)[["eligible_trip_count", "duration_sum_minutes"]]
            .sum()
            .reset_index()
        )
        if len(parts) >= 8:
            parts = [merge(parts)]
    hours = merge(parts) if parts else cudf.from_pandas(_empty_hours())
    hours["mean_duration_minutes"] = hours.duration_sum_minutes / hours.eligible_trip_count
    hours["coverage"] = "insufficient_trips"
    hours.loc[hours.eligible_trip_count >= params["min_trips_per_hour"], "coverage"] = "observed"
    hours["time_policy_version"] = config.get("time_policy", "nyc_local_wall_time_v1")
    hours = hours.sort_values(HOUR_KEYS).reset_index(drop=True)
    if int(hours.eligible_trip_count.sum()) != counts["aggregated_eligible_trips"]:
        raise ValueError("Duration-hour trip counts failed reconciliation.")
    return hours, {
        "counts": counts,
        "hourly_rows": len(hours),
        "sufficient_trip_hours": int(hours.coverage.eq("observed").sum()),
        "insufficient_trip_hours": int(hours.coverage.eq("insufficient_trips").sum()),
        "trip_count_reconciliation": "passed",
        "trip_filter_thresholds": filters,
        "study_year": year,
        "study_months": months,
        "coverage_meaning": "Observed means at least the configured number of eligible trip records in that pickup-zone hour; it does not prove complete demand or source coverage. Hours with fewer eligible trips are retained as insufficient_trips. Hours with no eligible observations have no row, are not zero-filled, and cannot contribute baseline history or receive scores. DST-unsafe local hours are excluded.",
    }


def _duration_detector(hours, config):
    required = HOUR_KEYS + ["mean_duration_minutes", "eligible_trip_count", "coverage"]
    missing = set(required) - set(hours.columns)
    if missing:
        raise ValueError(f"Duration input lacks {sorted(missing)}")
    frame = hours[required].copy()
    supported = frame.eligible_trip_count >= duration_parameters(config)["min_trips_per_hour"]
    frame.loc[frame.coverage.eq("observed") & ~supported, "coverage"] = "insufficient_trips"
    return frame.rename(
        columns={"mean_duration_minutes": "pickup_count", "coverage": "pickup_coverage"}
    )


def score_duration_anomalies_gpu(
    hours, baseline, config, model_id="duration-seasonal", split="test", threshold=None
):
    scores = score_anomalies_gpu(
        _duration_detector(hours, config),
        baseline,
        _seasonal_config(config),
        model_id,
        split,
        threshold,
    )
    scores = scores.merge(hours[HOUR_KEYS + ["eligible_trip_count"]], on=HOUR_KEYS)
    scores["metric"], scores["unit"] = "mean_duration_minutes", "minutes"
    return scores


def fit_duration_anomalies_gpu(canonical_dir, config, output_dir):
    cudf = require_cudf()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds, params = split_bounds(config), duration_parameters(config)
    hours, aggregation = aggregate_duration_hours_gpu(canonical_dir, config)
    baseline = fit_seasonal_baseline_gpu(
        _duration_detector(hours, config), _seasonal_config(config)
    )
    calibration = hours.loc[_period(hours.hour, bounds, "calibration")]
    test = hours.loc[_period(hours.hour, bounds, "test")]
    threshold, calibration_metadata, sensitivity = calibrate_threshold_gpu(
        _duration_detector(calibration, config), baseline, _seasonal_config(config)
    )
    parameters = {
        "method": "frozen_seasonal_hourly_mean_duration_median_mad_v1",
        "metric": "mean_duration_minutes",
        "unit": "minutes",
        "baseline_sha256": _baseline_hash(baseline),
        "threshold": threshold,
        "duration_anomalies_config": params,
        "trip_filter_thresholds": aggregation["trip_filter_thresholds"],
        "splits": {name: value.isoformat() for name, value in bounds.items()},
    }
    model_id = stable_id("duration-seasonal", parameters)
    calibration_scores = score_duration_anomalies_gpu(
        calibration, baseline, config, model_id, "calibration", threshold
    )
    test_scores = score_duration_anomalies_gpu(test, baseline, config, model_id, "test", threshold)
    scores = cudf.concat([calibration_scores, test_scores], ignore_index=True)
    incidents = consolidate_incidents(scores.loc[scores.is_alert].to_pandas())
    incidents["metric"], incidents["unit"] = "mean_duration_minutes", "minutes"
    tables = _write_tables(
        output_dir,
        [
            ("duration_hour", hours),
            ("duration_baseline", baseline),
            ("duration_scores", scores),
            ("duration_incidents", incidents),
            ("duration_calibration", sensitivity),
        ],
    )
    metadata = (
        parameters
        | _summary(baseline, calibration_scores, test_scores, incidents)
        | {
            "model_id": model_id,
            "aggregation": aggregation,
            "calibration": calibration_metadata,
            "baseline_meaning": "Training-period median of hourly mean eligible-trip durations for the same pickup zone, weekday, and hour; each sufficiently populated hour has equal weight.",
            "score_meaning": "(Observed hourly mean minutes - baseline minutes) / max(1.4826 * training MAD, configured minute floor). Alerts also require the configured absolute minute change. This is a robust deviation score, not an accuracy percentage or probability.",
            "incident_rule": "Same pickup zone/direction/split in strictly adjacent one-hour bins. Missing, low-trip, or unsupported hours do not bridge incidents. End hour is the last bin's start.",
            "incident_total_deviation_meaning": "Sum of deviations in hourly mean minutes, not total passenger delay or total trip time.",
            "manual_review_status": "not_performed",
            "review_note": "Labels remain unresolved until a person reviews the incident.",
            "limitations": "Changes in route, distance, destination, passenger demand, or trip composition can change mean duration. This model does not hold routes or distance constant and does not establish congestion, an event cause, or real-world detection accuracy. Only trips passing the configured fixed study bounds contribute.",
        }
    )
    write_json(output_dir / "duration_anomalies.json", metadata)
    return {"tables": tables, "metadata": metadata}
