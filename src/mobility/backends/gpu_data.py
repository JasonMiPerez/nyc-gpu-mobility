"""Device-side TLC normalization and exact daily histogram reductions.

Only timezone transition metadata and reduced quality counts use the CPU.
Individual trip columns stay in cuDF/CuPy throughout these operations.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from mobility.aggregate import HISTOGRAM_BINS, HISTOGRAM_KEYS
from mobility.data.quality import QualityFlag

NAT = np.iinfo(np.int64).min
SECOND = 1_000_000_000


@lru_cache(maxsize=1)
def new_york_transition_calendar():
    """Build a small transition table from the CPU oracle's timezone database.

    The fixed calendar covers the datetime64[ns] domain. Daily UTC probes find
    offset changes; a binary search resolves each change to its exact second.
    These are calendar probes, never source trip rows. New York transitions
    are separated by more than a day, including its historical LMT change.
    """
    utc = pd.date_range("1677-09-23 12:00", "2262-04-10 12:00", freq="D", tz="UTC")
    local = utc.tz_convert("America/New_York")
    offsets = (
        (local.tz_localize(None) - utc.tz_localize(None))
        .to_numpy(dtype="timedelta64[ns]")
        .view("int64")
    )
    utc_seconds = utc.as_unit("ns").asi8 // SECOND
    starts, ends, after = [], [], [int(offsets[0])]
    for index in np.flatnonzero(offsets[1:] != offsets[:-1]) + 1:
        old, new = int(offsets[index - 1]), int(offsets[index])
        low, high = int(utc_seconds[index - 1]), int(utc_seconds[index])
        while high - low > 1:
            middle = (low + high) // 2
            value = pd.Timestamp(middle, unit="s", tz="UTC").tz_convert("America/New_York")
            middle_offset = int(value.utcoffset().total_seconds()) * SECOND
            if middle_offset == old:
                low = middle
            else:
                high = middle
        boundary = high * SECOND
        starts.append(boundary + min(old, new))
        ends.append(boundary + max(old, new))
        after.append(new)
    return {
        "unsafe_start_ns": np.asarray(starts, dtype="int64"),
        "unsafe_end_ns": np.asarray(ends, dtype="int64"),
        "offset_ns": np.asarray(after, dtype="int64"),
    }


def _gpu_modules():
    from .gpu import require_cudf
    from .cuda_features import require_cupy

    return require_cudf(), require_cupy()


def dst_unsafe_duration_gpu(pickup_ns, dropoff_ns):
    """Match CPU ambiguous/nonexistent endpoint and offset-change rules on GPU."""
    _, cp = _gpu_modules()
    calendar = new_york_transition_calendar()
    starts = cp.asarray(calendar["unsafe_start_ns"])
    ends = cp.asarray(calendar["unsafe_end_ns"])
    offsets = cp.asarray(calendar["offset_ns"])

    def classify(values):
        valid = values != NAT
        index = cp.searchsorted(starts, values, side="right") - 1
        unsafe = valid & (index >= 0) & (values < ends[cp.maximum(index, 0)])
        offset = offsets[cp.searchsorted(ends, values, side="right")]
        return valid, unsafe, offset

    pv, pu, po = classify(pickup_ns)
    dv, du, do = classify(dropoff_ns)
    return pu | du | (pv & dv & ~pu & ~du & (po != do))


def normalize_yellow_gpu(raw, source, mapped_ids, row_offset=0):
    """Return (canonical cuDF, small pandas quality summary), without CPU row work."""
    cudf, cp = _gpu_modules()
    if not isinstance(raw, cudf.DataFrame):
        raise TypeError("GPU normalization requires a cuDF DataFrame; no CPU conversion is used")
    if row_offset < 0 or row_offset + len(raw) > 2**64:
        raise ValueError("Source row identities must fit uint64")
    raw = raw.reset_index(drop=True)
    out = cudf.DataFrame(index=raw.index)
    row_numbers = cudf.Series(cp.arange(len(raw), dtype=cp.uint64) + cp.uint64(row_offset))
    out["record_id"] = str(source["sha256"]) + ":" + row_numbers.astype("str")
    out["source_id"] = source["source_id"]
    out["dataset_type"] = "yellow"
    out["source_month"] = source["nominal_month"]
    for original, name in [
        ("tpep_pickup_datetime", "pickup_local"),
        ("tpep_dropoff_datetime", "dropoff_local"),
    ]:
        values = raw[original]
        if getattr(values.dtype, "tz", None) is not None:
            raise ValueError("Unexpected timezone-aware TLC source; a new adapter is required")
        if not pd.api.types.is_datetime64_dtype(values.dtype):
            raise ValueError("GPU TLC adapter requires typed Parquet timestamp columns")
        # cuDF 25.10 does not implement errors='coerce' for a non-scalar input.
        # Official TLC Parquet columns already have a timestamp dtype and mask.
        times = values
        result = times.astype("datetime64[ns]")
        # Timestamp-unit conversion must not silently wrap out-of-range values.
        if bool((times.notna() & result.astype(times.dtype).ne(times)).any()):
            raise OverflowError("Timestamp cannot be represented as datetime64[ns]")
        out[name] = result
    for original, name in [("PULocationID", "pickup_zone_id"), ("DOLocationID", "dropoff_zone_id")]:
        values = cudf.to_numeric(raw[original], errors="coerce").astype("float64")
        device = values.to_cupy(na_value=np.nan)
        valid = (
            cp.isfinite(device) & (device % 1 == 0) & (device >= -(2**31)) & (device <= 2**31 - 1)
        )
        integers = cudf.Series(cp.where(valid, device, 0).astype(cp.int32), index=out.index)
        out[name] = integers.where(cudf.Series(valid, index=out.index))
    for original, name in [
        ("trip_distance", "trip_distance_miles"),
        ("fare_amount", "fare_amount"),
        ("total_amount", "total_amount"),
        ("passenger_count", "passenger_count"),
    ]:
        out[name] = (
            cudf.to_numeric(raw[original], errors="coerce").astype("float64")
            if original in raw
            else cudf.Series(cp.full(len(raw), cp.nan, dtype=cp.float64))
        )
    pickup = out.pickup_local.astype("int64").to_cupy(na_value=NAT)
    dropoff = out.dropoff_local.astype("int64").to_cupy(na_value=NAT)
    pickup_valid, dropoff_valid = pickup != NAT, dropoff != NAT
    delta = dropoff - pickup
    overflow = pickup_valid & dropoff_valid & (((dropoff ^ pickup) & (dropoff ^ delta)) < 0)
    if bool(cp.any(overflow)):
        raise OverflowError("Overflow in int64 timestamp subtraction")
    duration = cp.where(pickup_valid & dropoff_valid, delta.astype(cp.float64) / SECOND, cp.nan)
    unsafe = dst_unsafe_duration_gpu(pickup, dropoff)
    distance = out.trip_distance_miles.to_cupy(na_value=np.nan)
    valid_duration = cp.isfinite(duration) & (duration > 0) & ~unsafe
    valid_distance = cp.isfinite(distance) & (distance >= 0)
    stored_duration = cp.where(valid_duration, duration, cp.nan)
    speed = cp.where(valid_distance, distance * 3600 / stored_duration, cp.nan)
    out["duration_seconds"] = cudf.Series(stored_duration, index=out.index)
    out["mean_speed_mph"] = cudf.Series(speed, index=out.index)
    outside = (
        (
            out.pickup_local.notna()
            & out.pickup_local.dt.strftime("%Y-%m").ne(source["nominal_month"])
        )
        .fillna(False)
        .to_cupy()
    )
    mapped = list(sorted(mapped_ids))  # Only the small zone lookup crosses to the device.
    mapped_pickup = out.pickup_zone_id.isin(mapped).fillna(False).to_cupy()
    mapped_dropoff = out.dropoff_zone_id.isin(mapped).fillna(False).to_cupy()
    eligible_pickup = pickup_valid & mapped_pickup & ~outside
    eligible = {
        "eligible_pickup": eligible_pickup,
        "eligible_dropoff": dropoff_valid & mapped_dropoff,
        "eligible_od": eligible_pickup & mapped_dropoff,
        "eligible_duration": valid_duration,
        "eligible_ml": eligible_pickup & valid_duration & valid_distance,
        "eligible_distance": valid_distance,
        "dst_unsafe_duration": unsafe,
    }
    for name, mask in eligible.items():
        out[name] = cudf.Series(mask, index=out.index)
    fare = out.fare_amount.to_cupy(na_value=np.nan)
    total = out.total_amount.to_cupy(na_value=np.nan)
    rules = {
        QualityFlag.MISSING_PICKUP: ~pickup_valid,
        QualityFlag.MISSING_DROPOFF: ~dropoff_valid,
        QualityFlag.UNMAPPED_PICKUP: ~mapped_pickup,
        QualityFlag.UNMAPPED_DROPOFF: ~mapped_dropoff,
        QualityFlag.INVALID_DURATION: ~valid_duration,
        QualityFlag.INVALID_DISTANCE: ~valid_distance,
        QualityFlag.ZERO_DISTANCE: distance == 0,
        QualityFlag.DST_UNSAFE_DURATION: unsafe,
        QualityFlag.OUTSIDE_SOURCE_MONTH: outside,
        QualityFlag.EXTREME_DURATION: duration > 24 * 3600,
        QualityFlag.EXTREME_SPEED: speed > 100,
        QualityFlag.NEGATIVE_FARE: (fare < 0) | (total < 0),
    }
    flags = cp.zeros(len(out), dtype=cp.uint32)
    for flag, mask in rules.items():
        flags |= mask.astype(cp.uint32) * cp.uint32(flag)
    out["quality_flags"] = cudf.Series(flags, index=out.index)
    out["source_row"] = row_numbers
    out["source_token"] = cp.uint64(int(source["sha256"][:16], 16))
    # The quality summary copies only twelve integer totals to the host.
    counts = cp.asnumpy(cp.stack([mask.sum(dtype=cp.int64) for mask in rules.values()]))
    quality = pd.DataFrame(
        [
            {
                "source_id": source["source_id"],
                "rule": flag.name.lower(),
                "affected_rows": int(count),
                "total_rows": len(out),
                "action": "flag; exclude only affected analytical views",
            }
            for flag, count in zip(rules, counts)
        ]
    )
    return out, quality


def daily_histograms_gpu(frame):
    """Return the same exact daily bins as aggregate.daily_histograms, on device."""
    cudf, cp = _gpu_modules()
    parts = []
    for feature, edges in HISTOGRAM_BINS.items():
        values = (
            frame.trip_distance_miles
            if feature == "distance_miles"
            else frame.duration_seconds / 60
        ).to_cupy(na_value=np.nan)
        valid = frame.eligible_pickup.fillna(False).to_cupy() & cp.isfinite(values) & (values >= 0)
        if feature == "duration_minutes":
            valid &= frame.eligible_duration.fillna(False).to_cupy()
        data = frame.loc[
            cudf.Series(valid, index=frame.index), ["dataset_type", "pickup_local"]
        ].copy()
        data["pickup_date"] = data.pickup_local.dt.floor("D")
        bins = cp.searchsorted(cp.asarray(edges), values[valid], side="right") - 1
        data["bin_index"] = cudf.Series(bins.astype(cp.int64), index=data.index)
        data["count"] = np.int64(1)
        grouped = (
            data.groupby(["dataset_type", "pickup_date", "bin_index"])[["count"]]
            .sum()
            .reset_index()
        )
        bin_lookup = cudf.DataFrame(
            {
                "bin_index": np.arange(len(edges) - 1, dtype=np.int64),
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "bin_label": [
                    f"{left:g}+" if np.isinf(right) else f"{left:g}–{right:g}"
                    for left, right in zip(edges[:-1], edges[1:])
                ],
            }
        )
        grouped = grouped.merge(bin_lookup, on="bin_index", how="left")
        grouped["feature"] = feature
        grouped["bin_version"] = "daily_fixed_v1"
        parts.append(grouped[HISTOGRAM_KEYS + ["count"]])
    return cudf.concat(parts, ignore_index=True)
