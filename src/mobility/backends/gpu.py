"""Explicit cuDF/CuPy analytics; there is deliberately no CPU fallback."""

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .cuda_features import require_cupy

_memory_resources = {}


def configure_memory(fraction=0.65):
    """Share a bounded RMM pool between cuDF and CuPy before processing batches.

    Hold resources for the process lifetime; never reinitialize an allocator
    underneath live arrays. The caller must choose smaller batches after OOM.
    """
    if not 0 < fraction < 1:
        raise ValueError("GPU memory fraction must be strictly between 0 and 1")
    cp = require_cupy()
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    device = cp.cuda.Device().id
    if device in _memory_resources:
        previous, pool = _memory_resources[device]
        if previous != fraction:
            raise RuntimeError("Start a fresh process to change the GPU pool limit")
        return {"allocator": "rmm_pool", "fraction": fraction}
    free, total = cp.cuda.runtime.memGetInfo()
    limit = int(min(total * fraction, free * 0.9)) // 256 * 256
    initial = min(64 * 1024**2, limit)
    upstream = rmm.mr.CudaMemoryResource()
    pool = rmm.mr.PoolMemoryResource(upstream, initial_pool_size=initial, maximum_pool_size=limit)
    rmm.mr.set_current_device_resource(pool)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    _memory_resources[device] = (fraction, pool)
    return {
        "allocator": "rmm_pool",
        "fraction": fraction,
        "maximum_pool_bytes": limit,
        "initial_pool_bytes": initial,
    }


def require_cudf():
    require_cupy()
    try:
        import cudf

        return cudf
    except ImportError as exc:
        raise RuntimeError("cuDF is required for the GPU dataframe backend") from exc


def synchronize():
    # cuDF can use streams other than the current CuPy stream.
    require_cupy().cuda.runtime.deviceSynchronize()


def to_device(frame):
    cudf = require_cudf()
    return frame if isinstance(frame, cudf.DataFrame) else cudf.from_pandas(frame)


def derive_features(frame, config=None):
    """Derive device features while preserving canonical quality/eligibility.

    Native production supplies GPU-normalized columns; the explicit legacy
    adapter can also transfer CPU-normalized columns with the same schema.
    """
    cp = require_cupy()
    cudf = require_cudf()
    frame = to_device(frame).copy(deep=False)
    distance = frame["trip_distance_miles"].to_cupy(na_value=np.nan)
    duration = frame["duration_seconds"].to_cupy(na_value=np.nan)
    valid_duration = (
        frame["eligible_duration"].fillna(False).to_cupy() & cp.isfinite(duration) & (duration > 0)
    )
    valid_distance = cp.isfinite(distance) & (distance >= 0)
    if (config or {}).get("processing", {}).get("custom_cuda", False):
        from .cuda_features import fused_features
        from .features import NAT

        def mask(name):
            return frame[name].fillna(False).astype("uint8").to_cupy()

        inputs = {
            "pickup_ns": frame["pickup_local"]
            .astype("datetime64[ns]")
            .astype("int64")
            .to_cupy(na_value=NAT),
            "dropoff_ns": frame["dropoff_local"]
            .astype("datetime64[ns]")
            .astype("int64")
            .to_cupy(na_value=NAT),
            "distance": cp.ascontiguousarray(distance, dtype=cp.float64),
            "pickup_valid": frame["pickup_local"].notna().astype("uint8").to_cupy(),
            "dropoff_valid": frame["dropoff_local"].notna().astype("uint8").to_cupy(),
            "time_allowed": (
                (~frame["dst_unsafe_duration"].fillna(True)).astype("uint8").to_cupy()
                if "dst_unsafe_duration" in frame
                else mask("eligible_duration")
            ),
            "pickup_zone_valid": mask("eligible_pickup"),
            "dropoff_zone_valid": mask("eligible_dropoff"),
        }
        result = fused_features(inputs)
        # Preserve source-specific canonical duration (for example reported FHV
        # duration). This first kernel integration is explicitly yellow-only.
        if not bool((frame["dataset_type"] == "yellow").all()):
            raise ValueError(
                "The fused timestamp-duration pipeline currently supports yellow trips only"
            )
        duration = result["duration_seconds"]
        frame["duration_seconds"] = cudf.Series(duration, index=frame.index)
        frame["kernel_feature_flags"] = cudf.Series(result["feature_flags"], index=frame.index)
        speed = result["mean_speed_mph"]
    else:
        safe_duration = cp.where(valid_duration, duration, 1.0)
        speed = cp.where(valid_duration & valid_distance, distance / safe_duration * 3600.0, cp.nan)
    frame["mean_speed_mph"] = cudf.Series(speed, index=frame.index)
    frame["log_distance"] = cudf.Series(
        cp.log1p(cp.where(valid_distance, distance, cp.nan)), index=frame.index
    )
    frame["log_duration"] = cudf.Series(
        cp.log1p(cp.where(valid_duration, duration, cp.nan)), index=frame.index
    )
    frame["pickup_hour"] = frame["pickup_local"].dt.hour
    frame["pickup_weekday"] = frame["pickup_local"].dt.weekday
    hour = frame["pickup_hour"].astype("float64").to_cupy(na_value=np.nan)
    frame["hour_sin"] = cudf.Series(cp.sin(2 * cp.pi * hour / 24), index=frame.index)
    frame["hour_cos"] = cudf.Series(cp.cos(2 * cp.pi * hour / 24), index=frame.index)
    return frame


def aggregate(frame):
    """Return GPU partial sums/counts for continued device aggregation.

    Pickup and drop-off time bins are independent. Duration and distance
    statistics are pickup-anchored. Coverage grids are built after merging.
    """
    cudf = require_cudf()
    frame = to_device(frame)
    stats = ["duration_sum_seconds", "duration_count", "distance_sum_miles", "distance_count"]
    source = frame.copy(deep=False)
    duration_ok = (
        source["eligible_duration"].fillna(False) & source["duration_seconds"].notna()
    ).fillna(False)
    distance = source["trip_distance_miles"]
    distance_ok = (distance.notna() & (distance >= 0) & (distance < np.inf)).fillna(False)
    source["duration_sum_seconds"] = (
        source["duration_seconds"].where(duration_ok, 0).fillna(0).astype("float64")
    )
    source["duration_count"] = duration_ok.astype("int64")
    source["distance_sum_miles"] = distance.where(distance_ok, 0).fillna(0).astype("float64")
    source["distance_count"] = distance_ok.astype("int64")
    pu = source.loc[
        source["eligible_pickup"].fillna(False),
        ["dataset_type", "pickup_zone_id", "pickup_local", *stats],
    ].copy()
    pu = pu.rename(columns={"pickup_zone_id": "zone_id", "pickup_local": "hour"})
    pu["hour"] = pu["hour"].dt.floor("h")
    pu["pickup_count"], pu["dropoff_count"] = np.int64(1), np.int64(0)
    do = source.loc[
        source["eligible_dropoff"].fillna(False),
        ["dataset_type", "dropoff_zone_id", "dropoff_local"],
    ].copy()
    do = do.rename(columns={"dropoff_zone_id": "zone_id", "dropoff_local": "hour"})
    do["hour"] = do["hour"].dt.floor("h")
    do["pickup_count"], do["dropoff_count"] = np.int64(0), np.int64(1)
    for name in stats:
        do[name] = np.float64(0) if "sum" in name else np.int64(0)
    zone = cudf.concat([pu, do], ignore_index=True)
    zone = (
        zone.groupby(["dataset_type", "zone_id", "hour"], sort=True)[
            ["pickup_count", "dropoff_count", *stats]
        ]
        .sum()
        .reset_index()
    )
    od = source.loc[
        source["eligible_od"].fillna(False),
        ["dataset_type", "pickup_zone_id", "dropoff_zone_id", "pickup_local", *stats],
    ].copy()
    od = od.rename(columns={"pickup_local": "hour"})
    od["hour"] = od["hour"].dt.floor("h")
    od["trip_count"] = np.int64(1)
    od = (
        od.groupby(["dataset_type", "pickup_zone_id", "dropoff_zone_id", "hour"], sort=True)[
            ["trip_count", *stats]
        ]
        .sum()
        .reset_index()
    )
    return zone, od


def doctor():
    """Perform real device work, not just an import or driver-version check."""
    cp = require_cupy()
    cudf = require_cudf()
    import cuml
    import rmm
    from cuml.cluster import KMeans

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    total = int(cudf.Series([1, 2, 3], dtype="int64").sum())
    fitted = KMeans(n_clusters=2, random_state=42, n_init=1).fit(
        cp.asarray([[0.0, 0.0], [0.0, 1.0], [9.0, 9.0], [9.0, 10.0]], dtype=cp.float32)
    )
    predictions = fitted.predict(cp.asarray([[0.0, 0.0], [9.0, 9.0]], dtype=cp.float32))
    synchronize()
    if total != 6 or len(predictions) != 2:
        raise RuntimeError("GPU smoke computations returned unexpected results")
    packages = {}
    for name in [
        "cupy-cuda12x",
        "cupy-cuda13x",
        "cupy",
        "cudf-cu12",
        "cudf-cu13",
        "cuml-cu12",
        "cuml-cu13",
        "rmm-cu12",
        "rmm-cu13",
    ]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "status": "passed",
        "device_id": device.id,
        "gpu": properties["name"].decode()
        if isinstance(properties["name"], bytes)
        else str(properties["name"]),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "total_device_memory_bytes": int(properties["totalGlobalMem"]),
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "packages": packages,
        "cudf_version": cudf.__version__,
        "cuml_version": cuml.__version__,
        "rmm_version": rmm.__version__,
    }
