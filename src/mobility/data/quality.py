"""Normalize yellow trips without discarding independently valid endpoints."""

from enum import IntFlag

import numpy as np
import pandas as pd


class QualityFlag(IntFlag):
    MISSING_PICKUP = 1
    MISSING_DROPOFF = 2
    UNMAPPED_PICKUP = 4
    UNMAPPED_DROPOFF = 8
    INVALID_DURATION = 16
    INVALID_DISTANCE = 32
    ZERO_DISTANCE = 64
    DST_UNSAFE_DURATION = 128
    OUTSIDE_SOURCE_MONTH = 256
    EXTREME_DURATION = 512
    EXTREME_SPEED = 1024
    NEGATIVE_FARE = 2048


def localize_for_validation(times: pd.Series) -> pd.Series:
    return times.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")


def dst_unsafe_duration(pickup: pd.Series, dropoff: pd.Series) -> pd.Series:
    """Exclude uncertain wall-clock intervals, rather than guessing their UTC offset."""
    p = localize_for_validation(pickup)
    d = localize_for_validation(dropoff)
    p_offset = p.dt.tz_localize(None) - p.dt.tz_convert("UTC").dt.tz_localize(None)
    d_offset = d.dt.tz_localize(None) - d.dt.tz_convert("UTC").dt.tz_localize(None)
    uncertain_endpoint = (pickup.notna() & p.isna()) | (dropoff.notna() & d.isna())
    changed_offset = p.notna() & d.notna() & (p_offset != d_offset)
    return uncertain_endpoint | changed_offset


def normalize_yellow(raw: pd.DataFrame, source: dict, mapped_ids: set[int],
                     row_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.reset_index(drop=True)
    out = pd.DataFrame(index=raw.index)
    out["record_id"] = [f"{source['sha256']}:{i}" for i in range(row_offset, row_offset + len(raw))]
    out["source_id"] = source["source_id"]
    out["dataset_type"] = "yellow"
    out["source_month"] = source["nominal_month"]
    for original, name in [("tpep_pickup_datetime", "pickup_local"),
                           ("tpep_dropoff_datetime", "dropoff_local")]:
        times = pd.to_datetime(raw[original], errors="coerce")
        if times.dt.tz is not None:
            raise ValueError("Unexpected timezone-aware TLC source; a new adapter is required")
        out[name] = times.astype("datetime64[ns]")
    for original, name in [("PULocationID", "pickup_zone_id"), ("DOLocationID", "dropoff_zone_id")]:
        values = pd.to_numeric(raw[original], errors="coerce")
        valid_int = values.notna() & np.isfinite(values) & (values % 1 == 0)
        out[name] = values.where(valid_int & values.between(-2**31, 2**31 - 1)).astype("Int32")
    for original, name in [("trip_distance", "trip_distance_miles"),
                           ("fare_amount", "fare_amount"), ("total_amount", "total_amount"),
                           ("passenger_count", "passenger_count")]:
        values = raw[original] if original in raw else pd.Series(np.nan, index=raw.index)
        out[name] = pd.to_numeric(values, errors="coerce").astype("float64")

    pu, do = out.pickup_local, out.dropoff_local
    duration = (do - pu).dt.total_seconds()
    unsafe = dst_unsafe_duration(pu, do)
    distance = out.trip_distance_miles
    valid_duration = duration.notna() & np.isfinite(duration) & duration.gt(0) & ~unsafe
    valid_distance = distance.notna() & np.isfinite(distance) & distance.ge(0)
    out["duration_seconds"] = duration.where(valid_duration)
    out["mean_speed_mph"] = (distance * 3600 / duration.where(valid_duration)).where(valid_distance)
    outside = pu.notna() & (pu.dt.strftime("%Y-%m") != source["nominal_month"])
    mapped_pu = out.pickup_zone_id.isin(mapped_ids)
    mapped_do = out.dropoff_zone_id.isin(mapped_ids)
    out["eligible_pickup"] = pu.notna() & mapped_pu & ~outside
    out["eligible_dropoff"] = do.notna() & mapped_do
    out["eligible_od"] = out.eligible_pickup & mapped_do
    out["eligible_duration"] = valid_duration
    out["eligible_ml"] = out.eligible_pickup & valid_duration & valid_distance
    out["eligible_distance"] = valid_distance
    out["dst_unsafe_duration"] = unsafe
    rules = {
        QualityFlag.MISSING_PICKUP: pu.isna(),
        QualityFlag.MISSING_DROPOFF: do.isna(),
        QualityFlag.UNMAPPED_PICKUP: ~mapped_pu,
        QualityFlag.UNMAPPED_DROPOFF: ~mapped_do,
        QualityFlag.INVALID_DURATION: ~valid_duration,
        QualityFlag.INVALID_DISTANCE: ~valid_distance,
        QualityFlag.ZERO_DISTANCE: distance.eq(0),
        QualityFlag.DST_UNSAFE_DURATION: unsafe,
        QualityFlag.OUTSIDE_SOURCE_MONTH: outside,
        QualityFlag.EXTREME_DURATION: duration.gt(24 * 3600),
        QualityFlag.EXTREME_SPEED: out.mean_speed_mph.gt(100),
        QualityFlag.NEGATIVE_FARE: out.fare_amount.lt(0) | out.total_amount.lt(0),
    }
    flags = np.zeros(len(out), dtype=np.uint32)
    quality = []
    for flag, mask in rules.items():
        flags[mask.fillna(False).to_numpy(dtype=bool)] |= int(flag)
        quality.append({"source_id": source["source_id"], "rule": flag.name.lower(),
                        "affected_rows": int(mask.sum()), "total_rows": len(out),
                        "action": "flag; exclude only affected analytical views"})
    out["quality_flags"] = flags
    return out, pd.DataFrame(quality)


def hourly_dst_mask(hours: pd.DatetimeIndex) -> np.ndarray:
    return hours.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").isna()
