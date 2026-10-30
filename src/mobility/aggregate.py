"""Additive endpoint and origin–destination aggregates shared by all runs."""

import numpy as np
import pandas as pd

ZONE_KEYS = ["dataset_type", "zone_id", "hour"]
OD_KEYS = ["dataset_type", "pickup_zone_id", "dropoff_zone_id", "hour"]
STATS = ["duration_sum_seconds", "duration_count", "distance_sum_miles", "distance_count"]


def _statistics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    valid_duration = out.eligible_duration & out.duration_seconds.notna()
    distance = out.trip_distance_miles
    valid_distance = distance.notna() & np.isfinite(distance) & distance.ge(0)
    out["duration_sum_seconds"] = out.duration_seconds.where(valid_duration, 0).fillna(0)
    out["duration_count"] = valid_duration.astype("int64")
    out["distance_sum_miles"] = distance.where(valid_distance, 0).fillna(0)
    out["distance_count"] = valid_distance.astype("int64")
    return out


def aggregate_cpu(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _statistics(frame)
    pickups = frame.loc[frame.eligible_pickup].copy()
    pickups["hour"] = pickups.pickup_local.dt.floor("h")
    pickups["zone_id"] = pickups.pickup_zone_id
    pickups["pickup_count"] = np.int64(1)
    pickup_groups = pickups.groupby(ZONE_KEYS, observed=True)[["pickup_count"] + STATS].sum()
    dropoffs = frame.loc[frame.eligible_dropoff].copy()
    dropoffs["hour"] = dropoffs.dropoff_local.dt.floor("h")
    dropoffs["zone_id"] = dropoffs.dropoff_zone_id
    dropoffs["dropoff_count"] = np.int64(1)
    dropoff_groups = dropoffs.groupby(ZONE_KEYS, observed=True)[["dropoff_count"]].sum()
    zone = pickup_groups.join(dropoff_groups, how="outer").fillna(0).reset_index()
    trips = frame.loc[frame.eligible_od].copy()
    trips["hour"] = trips.pickup_local.dt.floor("h")
    trips["trip_count"] = np.int64(1)
    od = trips.groupby(OD_KEYS, observed=True)[["trip_count"] + STATS].sum().reset_index()
    for table in [zone, od]:
        for column in table.columns:
            if column.endswith("_count"):
                table[column] = table[column].astype("int64")
    return zone, od


def merge_aggregates(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        raise ValueError("No aggregate parts were supplied")
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(keys, as_index=False, observed=True)
        .sum(numeric_only=True)
    )


HISTOGRAM_BINS = {
    "distance_miles": np.array([0, 1, 2, 3, 5, 10, 20, 50, np.inf]),
    "duration_minutes": np.array([0, 5, 10, 15, 20, 30, 45, 60, 120, 240, np.inf]),
}
HISTOGRAM_KEYS = [
    "dataset_type",
    "pickup_date",
    "feature",
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_label",
    "bin_version",
]


def daily_histograms(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact daily bins with an explicit final overflow bin; no zone filter claim."""
    parts = []
    for feature, edges in HISTOGRAM_BINS.items():
        values = (
            frame.trip_distance_miles
            if feature == "distance_miles"
            else frame.duration_seconds / 60
        )
        valid = frame.eligible_pickup & np.isfinite(values) & values.ge(0)
        if feature == "duration_minutes":
            valid &= frame.eligible_duration
        data = frame.loc[valid, ["dataset_type", "pickup_local"]].copy()
        data["pickup_date"] = data.pickup_local.dt.floor("D")
        data["bin_index"] = np.searchsorted(edges, values[valid], side="right") - 1
        grouped = (
            data.groupby(["dataset_type", "pickup_date", "bin_index"])
            .size()
            .rename("count")
            .reset_index()
        )
        grouped["feature"] = feature
        grouped["bin_left"] = edges[grouped.bin_index.to_numpy(dtype=int)]
        grouped["bin_right"] = edges[grouped.bin_index.to_numpy(dtype=int) + 1]
        grouped["bin_label"] = [
            f"{left:g}+" if np.isinf(right) else f"{left:g}–{right:g}"
            for left, right in zip(grouped.bin_left, grouped.bin_right)
        ]
        grouped["bin_version"] = "daily_fixed_v1"
        parts.append(grouped[HISTOGRAM_KEYS + ["count"]])
    return pd.concat(parts, ignore_index=True)
