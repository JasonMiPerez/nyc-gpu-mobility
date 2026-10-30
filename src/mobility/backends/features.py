"""The CPU reference for the small fused CUDA experiment.

These flags describe this kernel's inputs, not the canonical quality bitmask.
Timezone decisions are made upstream and passed as an explicit mask.
"""

from enum import IntFlag

import numpy as np


class FeatureFlag(IntFlag):
    MISSING_PICKUP = 1
    MISSING_DROPOFF = 2
    TIME_EXCLUDED = 4
    NONPOSITIVE_DURATION = 8
    INVALID_DISTANCE = 16
    ZERO_DISTANCE = 32
    INVALID_ZONE = 64


INPUT_DTYPES = {
    "pickup_ns": np.dtype("int64"),
    "dropoff_ns": np.dtype("int64"),
    "distance": np.dtype("float64"),
    "pickup_valid": np.dtype("uint8"),
    "dropoff_valid": np.dtype("uint8"),
    "time_allowed": np.dtype("uint8"),
    "pickup_zone_valid": np.dtype("uint8"),
    "dropoff_zone_valid": np.dtype("uint8"),
}
OUTPUT_DTYPES = {
    "duration_seconds": np.dtype("float64"),
    "mean_speed_mph": np.dtype("float64"),
    "feature_eligible": np.dtype("uint8"),
    "feature_flags": np.dtype("uint32"),
}
NAT = np.iinfo(np.int64).min


def validate_inputs(inputs, array_type=np.ndarray):
    """Reject implicit casts or strided views before handing pointers to CUDA."""
    if set(inputs) != set(INPUT_DTYPES):
        raise ValueError(f"Expected input names: {', '.join(INPUT_DTYPES)}")
    size = None
    device = None
    for name, dtype in INPUT_DTYPES.items():
        value = inputs[name]
        if not isinstance(value, array_type):
            raise TypeError(f"{name} must be a {array_type.__name__}")
        if value.ndim != 1 or value.dtype != dtype or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be a contiguous 1-D {dtype} array")
        if size is not None and len(value) != size:
            raise ValueError("All feature buffers must have the same length")
        size = len(value)
        if hasattr(value, "device") and hasattr(value.device, "id"):
            if device is not None and device != value.device.id:
                raise ValueError("All feature buffers must use the same CUDA device")
            device = value.device.id
    return size or 0


def reference_features(inputs):
    """Vectorized NumPy baseline; invalid outputs remain NaN.

    Divide timestamps into seconds and remainders before subtraction. This
    avoids int64 overflow and preserves subsecond differences near large epochs.
    """
    n = validate_inputs(inputs)
    pickup = inputs["pickup_ns"]
    dropoff = inputs["dropoff_ns"]
    distance = inputs["distance"]
    pickup_ok = (inputs["pickup_valid"] != 0) & (pickup != NAT)
    dropoff_ok = (inputs["dropoff_valid"] != 0) & (dropoff != NAT)
    time_ok = inputs["time_allowed"] != 0
    zones_ok = (inputs["pickup_zone_valid"] != 0) & (inputs["dropoff_zone_valid"] != 0)
    distance_ok = np.isfinite(distance) & (distance >= 0)
    valid_times = pickup_ok & dropoff_ok & time_ok
    duration = np.full(n, np.nan, dtype=np.float64)
    # Only valid timestamp positions participate; NaT is never subtracted.
    p = pickup[valid_times]
    d = dropoff[valid_times]
    duration[valid_times] = (d // 1_000_000_000 - p // 1_000_000_000).astype(float)
    duration[valid_times] += (d % 1_000_000_000 - p % 1_000_000_000) * 1e-9
    positive = valid_times & (duration > 0)
    nonpositive = valid_times & ~positive
    duration[~positive] = np.nan
    speed = np.full(n, np.nan, dtype=np.float64)
    usable_speed = positive & distance_ok
    speed[usable_speed] = distance[usable_speed] / duration[usable_speed] * 3600.0
    flags = np.zeros(n, dtype=np.uint32)
    for mask, flag in [
        (~pickup_ok, FeatureFlag.MISSING_PICKUP),
        (~dropoff_ok, FeatureFlag.MISSING_DROPOFF),
        (~time_ok, FeatureFlag.TIME_EXCLUDED),
        (nonpositive, FeatureFlag.NONPOSITIVE_DURATION),
        (~distance_ok, FeatureFlag.INVALID_DISTANCE),
        (distance_ok & (distance == 0), FeatureFlag.ZERO_DISTANCE),
        (~zones_ok, FeatureFlag.INVALID_ZONE),
    ]:
        flags[mask] |= np.uint32(flag)
    return {
        "duration_seconds": duration,
        "mean_speed_mph": speed,
        "feature_eligible": (positive & distance_ok & zones_ok).astype(np.uint8),
        "feature_flags": flags,
    }


def inputs_from_pandas(frame):
    """Explicit conversion stage; include it in full feature-stage timings."""
    import pandas as pd

    def timestamps(column):
        values = pd.to_datetime(frame[column]).astype("datetime64[ns]")
        return np.ascontiguousarray(values.to_numpy().view("int64"))

    # Prefer the calendar-only mask so this experiment checks chronology itself.
    # Older canonical schemas without it conservatively use duration eligibility.
    if "dst_unsafe_duration" in frame:
        allowed = (~frame["dst_unsafe_duration"].fillna(True)).to_numpy(dtype=np.uint8)
    else:
        allowed = frame["eligible_duration"].fillna(False).to_numpy(dtype=np.uint8)
    return {
        "pickup_ns": timestamps("pickup_local"),
        "dropoff_ns": timestamps("dropoff_local"),
        "distance": np.ascontiguousarray(
            frame["trip_distance_miles"].to_numpy(dtype="float64", na_value=np.nan)
        ),
        "pickup_valid": np.ascontiguousarray(
            frame["pickup_local"].notna().to_numpy(dtype=np.uint8)
        ),
        "dropoff_valid": np.ascontiguousarray(
            frame["dropoff_local"].notna().to_numpy(dtype=np.uint8)
        ),
        "time_allowed": np.ascontiguousarray(allowed),
        "pickup_zone_valid": np.ascontiguousarray(
            frame["eligible_pickup"].fillna(False).to_numpy(dtype=np.uint8)
        ),
        "dropoff_zone_valid": np.ascontiguousarray(
            frame["eligible_dropoff"].fillna(False).to_numpy(dtype=np.uint8)
        ),
    }
