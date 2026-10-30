"""Frozen seasonal baselines and explicitly limited anomaly evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import in_period, split_bounds, stable_id, write_json


KEYS = ["dataset_type", "zone_id", "hour_of_week"]
INCIDENT_COLUMNS = [
    "model_id",
    "dataset_type",
    "zone_id",
    "direction",
    "split",
    "start_hour",
    "end_hour",
    "affected_hours",
    "peak_score",
    "total_deviation",
    "baseline_support_min",
    "label",
]


def prepare_hours(zone_hour: pd.DataFrame) -> pd.DataFrame:
    required = ["dataset_type", "zone_id", "hour", "pickup_count", "pickup_coverage"]
    missing = set(required) - set(zone_hour.columns)
    if missing:
        raise ValueError(f"Anomaly input lacks {sorted(missing)}")
    selected = required + (["time_policy_version"] if "time_policy_version" in zone_hour else [])
    frame = zone_hour[selected].copy()
    frame["hour"] = pd.to_datetime(frame["hour"])
    if frame["hour"].isna().any():
        raise ValueError("Hourly activity keys cannot contain a missing timestamp.")
    if frame.duplicated(["dataset_type", "zone_id", "hour"]).any():
        raise ValueError("Hourly activity keys must be unique before anomaly fitting.")
    frame["hour_of_week"] = frame["hour"].dt.day_of_week * 24 + frame["hour"].dt.hour
    frame["pickup_count"] = pd.to_numeric(frame["pickup_count"], errors="coerce")
    return frame


def fit_seasonal_baseline(zone_hour: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Use training observations only, without treating missing bins as zeros."""
    frame = prepare_hours(zone_hour)
    bounds = split_bounds(config)
    params = config.get("anomalies", {})
    valid = (
        in_period(frame["hour"], bounds, "train")
        & frame["pickup_coverage"].eq("observed")
        & np.isfinite(frame["pickup_count"])
        & frame["pickup_count"].ge(0)
    )
    training = frame.loc[valid]
    grouped = training.groupby(KEYS, observed=True)["pickup_count"]
    baseline = grouped.agg(baseline="median", history_count="size").reset_index()
    deviations = training.merge(baseline[KEYS + ["baseline"]], on=KEYS, validate="many_to_one")
    deviations["absolute_deviation"] = (deviations["pickup_count"] - deviations["baseline"]).abs()
    mad = (
        deviations.groupby(KEYS, observed=True)["absolute_deviation"]
        .median()
        .rename("mad")
        .reset_index()
    )
    baseline = baseline.merge(mad, on=KEYS, validate="one_to_one")
    floor = float(params.get("scale_floor", 5.0))
    minimum = int(params.get("min_history", 8))
    if floor <= 0 or minimum < 1:
        raise ValueError("Scale floor and minimum history must be positive.")
    baseline["scale"] = np.maximum(1.4826 * baseline["mad"], floor)
    baseline["support_status"] = np.where(
        baseline["history_count"].ge(minimum), "supported", "insufficient_history"
    )
    return baseline.sort_values(KEYS).reset_index(drop=True)


def score_anomalies(
    zone_hour: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict,
    model_id: str = "seasonal",
    split: str = "test",
    threshold: float | None = None,
) -> pd.DataFrame:
    frame = prepare_hours(zone_hour)
    params = config.get("anomalies", {})
    minimum = int(params.get("min_history", 8))
    score_threshold = float(
        threshold if threshold is not None else params.get("score_threshold", 4.0)
    )
    absolute_threshold = float(params.get("absolute_threshold", 20))
    if score_threshold <= 0 or absolute_threshold < 0:
        raise ValueError("Anomaly thresholds must be positive (absolute threshold may be zero).")
    scores = frame.merge(baseline, on=KEYS, how="left", validate="many_to_one")
    scores = scores.rename(columns={"pickup_count": "observed", "pickup_coverage": "coverage"})
    scores["history_count"] = scores["history_count"].fillna(0).astype(np.int32)
    scores["status"] = scores["coverage"].fillna("missing_coverage").astype(str)
    observed = scores["coverage"].eq("observed")
    has_value = np.isfinite(scores["observed"]) & scores["observed"].ge(0)
    supported = scores["history_count"].ge(minimum) & scores["baseline"].notna()
    scores.loc[observed & ~has_value, "status"] = "missing_observation"
    scores.loc[observed & has_value & ~supported, "status"] = "insufficient_history"
    valid = observed & has_value & supported
    scores.loc[valid, "status"] = "scored"
    scores["deviation"] = scores["observed"] - scores["baseline"]
    scores["score"] = np.nan
    scores.loc[valid, "score"] = scores.loc[valid, "deviation"] / scores.loc[valid, "scale"]
    scores["threshold"] = score_threshold
    scores["absolute_threshold"] = absolute_threshold
    scores["is_alert"] = (
        valid
        & scores["score"].abs().ge(score_threshold)
        & scores["deviation"].abs().ge(absolute_threshold)
    )
    scores["direction"] = np.where(
        scores["is_alert"], np.where(scores["deviation"].ge(0), "upper", "lower"), "none"
    )
    scores["model_id"] = model_id
    scores["metric"] = "pickup_count"
    scores["split"] = split
    return scores.drop(columns="support_status", errors="ignore")


def consolidate_incidents(scores: pd.DataFrame) -> pd.DataFrame:
    """Merge strictly adjacent alerts; never bridge gaps or opposite directions."""
    if scores.empty or not scores["is_alert"].any():
        return pd.DataFrame(columns=INCIDENT_COLUMNS)
    keys = ["model_id", "dataset_type", "zone_id", "direction", "split"]
    alerts = scores.loc[scores["is_alert"]].sort_values(keys + ["hour"]).copy()
    gap = alerts.groupby(keys, observed=True)["hour"].diff().ne(pd.Timedelta(hours=1))
    alerts["incident_sequence"] = gap.cumsum()
    alerts["absolute_score"] = alerts["score"].abs()
    incidents = (
        alerts.groupby(keys + ["incident_sequence"], observed=True)
        .agg(
            start_hour=("hour", "min"),
            end_hour=("hour", "max"),
            affected_hours=("hour", "size"),
            peak_score=("absolute_score", "max"),
            total_deviation=("deviation", "sum"),
            baseline_support_min=("history_count", "min"),
        )
        .reset_index()
    )
    # end_hour denotes the final affected bin's start, not an exclusive end time.
    incidents["label"] = "unresolved"
    return (
        incidents[INCIDENT_COLUMNS]
        .sort_values("peak_score", ascending=False)
        .reset_index(drop=True)
    )


def calibrate_threshold(
    calibration: pd.DataFrame, baseline: pd.DataFrame, config: dict
) -> tuple[float, dict, pd.DataFrame]:
    params = config.get("anomalies", {})
    default = float(params.get("score_threshold", 4.0))
    candidates = sorted(
        set(float(x) for x in params.get("threshold_candidates", [3, 4, 5, 6])) | {default}
    )
    budget = float(params.get("calibration_max_alert_fraction", 0.01))
    if not 0 < budget < 1:
        raise ValueError("Calibration alert-fraction budget must be between zero and one.")
    base_scores = score_anomalies(
        calibration, baseline, config, split="calibration", threshold=default
    )
    valid = base_scores["status"].eq("scored")
    scored_rows = int(valid.sum())
    sensitivity = []
    for threshold in candidates:
        flags = (
            valid
            & base_scores["score"].abs().ge(threshold)
            & base_scores["deviation"].abs().ge(base_scores["absolute_threshold"])
        )
        sensitivity.append(
            {
                "threshold": threshold,
                "scored_rows": scored_rows,
                "alert_rows": int(flags.sum()),
                "alert_fraction": float(flags.sum() / scored_rows) if scored_rows else None,
            }
        )
    table = pd.DataFrame(sensitivity)
    if not scored_rows:
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
        row
        for row in sensitivity
        if row["threshold"] >= default and row["alert_fraction"] <= budget
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


def evaluate_injections(
    test_scores: pd.DataFrame, seeds: tuple = (42, 43, 44)
) -> tuple[dict, pd.DataFrame]:
    """Perturb a separate copy of held-out bins; untouched real bins aren't negatives.

    The original frozen baseline and thresholds stay fixed. Evaluating only changed
    bins is exact for this pointwise detector; all other scores remain unchanged.
    """
    eligible = test_scores.loc[test_scores["status"].eq("scored")].copy()
    if eligible.empty:
        return {
            "status": "no_supported_test_bins",
            "trials": [],
            "limitation": "No held-out synthetic sensitivity evaluation was possible.",
        }, pd.DataFrame(columns=["seed", "event_id", "dataset_type", "zone_id", "hour", "label"])
    eligible = eligible.sort_values(["dataset_type", "zone_id", "hour"]).reset_index(drop=True)
    groups = eligible.groupby(["dataset_type", "zone_id"], observed=True).indices
    group_keys = list(groups)
    injections = []
    trials = []
    background_alerts = int(eligible["is_alert"].sum())
    # Scenarios are defined before looking at the detector response.
    scenarios = [
        (direction, magnitude, duration)
        for direction in ("spike", "drop")
        for magnitude in (0.5, 1.0, 2.0)
        for duration in (1, 3, 6)
    ]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        used = set()
        trial_rows = []
        events_attempted = 0
        for event_index, (direction, magnitude, duration) in enumerate(scenarios):
            events_attempted += 1
            chosen = None
            for _ in range(30):
                group_key = group_keys[int(rng.integers(len(group_keys)))]
                positions = groups[group_key]
                if len(positions) < duration:
                    continue
                start = int(rng.integers(len(positions) - duration + 1))
                selection = positions[start : start + duration]
                hours = eligible.iloc[selection]["hour"]
                if hours.diff().dropna().eq(pd.Timedelta(hours=1)).all() and not used.intersection(
                    selection.tolist()
                ):
                    chosen = selection
                    break
            if chosen is None:
                continue
            altered = eligible.iloc[chosen].copy()
            shift = np.maximum(1, np.rint(altered["baseline"].to_numpy() * magnitude))
            if direction == "spike":
                changed = altered["observed"].to_numpy() + shift
            else:
                changed = np.maximum(0, altered["observed"].to_numpy() - shift)
            changed_mask = changed != altered["observed"].to_numpy()
            altered = altered.loc[changed_mask].copy()
            if altered.empty:
                continue
            used.update(chosen[changed_mask].tolist())
            altered["original_count"] = altered["observed"]
            altered["perturbed_count"] = changed[changed_mask]
            altered["original_score"] = altered["score"]
            altered["original_alert"] = altered["is_alert"]
            deviation = altered["perturbed_count"] - altered["baseline"]
            altered["perturbed_score"] = deviation / altered["scale"]
            altered["perturbed_alert"] = altered["perturbed_score"].abs().ge(
                altered["threshold"]
            ) & deviation.abs().ge(altered["absolute_threshold"])
            altered["seed"] = seed
            altered["event_id"] = f"injection-{seed}-{event_index}"
            altered["injection_direction"] = direction
            altered["magnitude_baseline_multiple"] = magnitude
            altered["requested_duration_hours"] = duration
            altered["label"] = "synthetic_positive"
            trial_rows.append(altered)
        if not trial_rows:
            trials.append(
                {
                    "seed": seed,
                    "status": "no_changeable_windows",
                    "events_attempted": events_attempted,
                }
            )
            continue
        combined = pd.concat(trial_rows, ignore_index=True)
        before = int(combined["original_alert"].sum())
        after = int(combined["perturbed_alert"].sum())
        unmatched = background_alerts - before
        post_total = unmatched + after
        event_detections = combined.groupby("event_id")["perturbed_alert"].any()
        trials.append(
            {
                "seed": seed,
                "status": "evaluated",
                "events_attempted": events_attempted,
                "injected_events": len(event_detections),
                "injected_bins": len(combined),
                "detected_injected_bins": after,
                "injection_relative_recall": after / len(combined),
                "injection_relative_precision": after / post_total if post_total else None,
                "event_recall": float(event_detections.mean()),
                "event_matching_rule": "Any alerted changed bin within the injected event window.",
                "newly_alerted_injected_bins": int(
                    (combined["perturbed_alert"] & ~combined["original_alert"]).sum()
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
        pd.concat(injections, ignore_index=True)[columns]
        if injections
        else pd.DataFrame(columns=columns)
    )
    metadata = {
        "status": "evaluated" if len(result) else "no_changeable_windows",
        "trials": trials,
        "scope": "Detector-level perturbations of held-out aggregate counts, not raw-trip perturbations.",
        "scenario_design": "Both spike/drop; 0.5, 1, 2 times baseline; 1, 3, 6 consecutive hours; independent fixed seeds; drop counts clipped at zero.",
        "limitation": "Injected labels are synthetic. Untouched real alerts are unmatched background alerts, not verified false positives. These values do not measure real-world precision or recall.",
    }
    return metadata, result


def fit_anomalies(zone_hour: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = split_bounds(config)
    frame = prepare_hours(zone_hour)
    baseline = fit_seasonal_baseline(zone_hour, config)
    calibration = zone_hour.loc[in_period(frame["hour"], bounds, "calibration")]
    test = zone_hour.loc[in_period(frame["hour"], bounds, "test")]
    threshold, calibration_metadata, sensitivity = calibrate_threshold(
        calibration, baseline, config
    )
    baseline_hash = hashlib.sha256(
        pd.util.hash_pandas_object(baseline, index=False).values.tobytes()
    ).hexdigest()
    model_parameters = {
        "method": "frozen_seasonal_median_mad_v1",
        "metric": "pickup_count",
        "baseline_sha256": baseline_hash,
        "threshold": threshold,
        "anomalies_config": config.get("anomalies", {}),
        "splits": {name: value.isoformat() for name, value in bounds.items()},
    }
    model_id = stable_id("seasonal", model_parameters)
    calibration_scores = score_anomalies(
        calibration, baseline, config, model_id, "calibration", threshold
    )
    test_scores = score_anomalies(test, baseline, config, model_id, "test", threshold)
    scores = pd.concat([calibration_scores, test_scores], ignore_index=True)
    incidents = consolidate_incidents(scores)
    injection_metadata, injection_rows = evaluate_injections(test_scores)
    tables = {}
    for name, table in [
        ("anomaly_baseline", baseline),
        ("anomaly_scores", scores),
        ("anomaly_incidents", incidents),
        ("anomaly_calibration", sensitivity),
        ("synthetic_injections", injection_rows),
    ]:
        tables[name] = f"{name}.parquet"
        table.to_parquet(output_dir / tables[name], index=False)
    supported_baselines = int(baseline["support_status"].eq("supported").sum())
    if not supported_baselines:
        status = "insufficient_training_history"
    elif test_scores.empty:
        status = "no_held_out_rows"
    elif not test_scores["status"].eq("scored").any():
        status = "no_supported_test_bins"
    else:
        status = "evaluated"
    metadata = model_parameters | {
        "model_id": model_id,
        "status": status,
        "computation_backend": "CPU pandas/numpy",
        "baseline_groups": len(baseline),
        "supported_baseline_groups": supported_baselines,
        "calibration": calibration_metadata,
        "calibration_scored_bins": int(calibration_scores["status"].eq("scored").sum()),
        "test_scored_bins": int(test_scores["status"].eq("scored").sum()),
        "test_alert_bins": int(test_scores["is_alert"].sum()),
        "test_incidents": int(incidents["split"].eq("test").sum()),
        "test_status_counts": {
            str(key): int(value) for key, value in test_scores["status"].value_counts().items()
        },
        "score_meaning": "Robust deviation score; not a calibrated z-test or p-value.",
        "incident_rule": "Same dataset/zone/direction/split in strictly adjacent one-hour bins; no missing-hour gaps bridged. end_hour is the last affected bin's start.",
        "manual_review_status": "not_performed",
        "review_note": "Incident labels remain unresolved until a person records a review; no real-world accuracy claim.",
        "synthetic_evaluation": injection_metadata,
    }
    write_json(output_dir / "anomalies.json", metadata)
    return {"tables": tables, "metadata": metadata}
