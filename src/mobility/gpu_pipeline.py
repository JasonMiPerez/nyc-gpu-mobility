"""Native GPU production path: device columns from decoding through analysis.

The CPU handles file metadata, checksums, a small geographic/time calendar and
bounded result packaging. No pandas normalization or CPU model implementation is
called for trip rows. Cross-backend comparisons happen outside this function.
"""

from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid

import pandas as pd
import pyarrow.parquet as pq

from mobility.aggregate import HISTOGRAM_KEYS, OD_KEYS, ZONE_KEYS
from mobility.data.quality import hourly_dst_mask
from mobility.execution import execution_plan
from mobility.io import atomic_json, utc_now


def _merge(parts, keys):
    import cudf

    if not parts:
        raise ValueError("No device aggregate parts")
    columns = [name for name in parts[0].columns if name not in keys]
    return (
        cudf.concat(parts, ignore_index=True).groupby(keys, sort=True)[columns].sum().reset_index()
    )


def _coverage(zone, mapped_ids, sources, config):
    import cudf
    import cupy as cp

    year, months = config["data"]["year"], config["data"]["months"]
    start = pd.Timestamp(year=year, month=min(months), day=1)
    end = pd.Timestamp(year=year, month=max(months), day=1) + pd.offsets.MonthBegin(1)
    hours = pd.date_range(start, end, inclusive="left", freq="h")
    identifiers = sorted(mapped_ids)
    grid = cudf.DataFrame(
        {
            "zone_id": cp.tile(cp.asarray(identifiers, dtype=cp.int64), len(hours)),
            "hour": cp.repeat(cp.asarray(hours.as_unit("ns").asi8), len(identifiers)),
        }
    )
    grid["hour"] = grid.hour.astype("datetime64[ns]")
    grid["dataset_type"] = "yellow"
    result = grid.merge(zone, on=ZONE_KEYS, how="left")
    acquired = [int(source["nominal_month"][5:7]) for source in sources]
    observed = result.hour.dt.month.isin(acquired)
    counts = ["pickup_count", "dropoff_count", "duration_count", "distance_count"]
    for column in counts + ["duration_sum_seconds", "distance_sum_miles"]:
        result[column] = result[column].fillna(0).where(observed)
        result[column] = result[column].astype("int64" if column in counts else "float64")
    result["pickup_coverage"] = "observed"
    result["dropoff_coverage"] = "partial_source_context"
    result.loc[~observed, ["pickup_coverage", "dropoff_coverage"]] = "missing_source"
    unsafe = cudf.Series(hours[hourly_dst_mask(hours)])
    dst = observed & result.hour.isin(unsafe)
    result.loc[dst, ["pickup_coverage", "dropoff_coverage"]] = "dst_ambiguous"
    result["time_policy_version"] = config["time_policy"]
    return result.sort_values(["hour", "zone_id"]).reset_index(drop=True)


def _validate_gpu(zone, od_paths, expected):
    import cudf

    if bool(zone.duplicated(ZONE_KEYS).any()):
        raise ValueError("Duplicate GPU zone/hour keys")
    if bool((zone.pickup_count < 0).any()) or bool((zone.dropoff_count < 0).any()):
        raise ValueError("Negative GPU endpoint counts")
    if bool(zone.loc[zone.pickup_coverage == "missing_source", "pickup_count"].notna().any()):
        raise ValueError("Missing GPU source coverage became zero")
    total, seen = 0, set()
    for path in od_paths:
        od = cudf.read_parquet(path)
        if bool(od.duplicated(OD_KEYS).any()) or bool((od.trip_count < 0).any()):
            raise ValueError("Invalid GPU OD keys/counts")
        months = set(od.hour.dt.strftime("%Y-%m").unique().to_arrow().to_pylist())
        if len(months) > 1 or months & seen:
            raise ValueError("Overlapping GPU OD partitions")
        seen |= months
        total += int(od.trip_count.sum())
    counts = {
        "pickup": int(zone.pickup_count.sum()),
        "dropoff": int(zone.dropoff_count.sum()),
        "od": total,
    }
    if counts != expected:
        raise ValueError(f"GPU output counts differ: {counts} != {expected}")
    return {
        "status": "passed",
        "counts": counts,
        "unique_zone_hour_keys": True,
        "missing_sources_are_null": True,
        "backend": "gpu",
    }


def run_gpu_pipeline(root: Path, config: dict, *, fit_models=True):
    import cudf
    from mobility.backends import gpu
    from mobility.backends.gpu_data import normalize_yellow_gpu, daily_histograms_gpu
    from mobility.pipeline import environment_record, source_list

    started = time.perf_counter()
    root = Path(root).resolve()
    sources = source_list(root / "data", config)
    zone_dir = root / "data/zones"
    zones = pd.read_parquet(zone_dir / "zones.parquet")  # Only the 265-row lookup.
    mapped_ids = set(zones.loc[zones.longitude.notna(), "zone_id"].astype(int))
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run = root / "artifacts/runs" / run_id
    canonical, output = run / "canonical", run / "tables"
    canonical.mkdir(parents=True)
    (output / "od_hour").mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "processing",
        "config": config,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "sources": sources,
        "environment": environment_record(),
        "backend": "gpu",
        "execution_mode": "native",
        "data_kind": "synthetic"
        if any(s.get("data_kind") == "synthetic" for s in sources)
        else "real",
    }
    atomic_json(run / "manifest.json", manifest)
    expected = {"pickup": 0, "dropoff": 0, "od": 0}
    all_zone, quality_parts, histogram_parts, timings, graph_parts = [], [], [], [], []
    rows_total = part_index = 0
    year, months = config["data"]["year"], config["data"]["months"]
    start = pd.Timestamp(year=year, month=min(months), day=1)
    end = pd.Timestamp(year=year, month=max(months), day=1) + pd.offsets.MonthBegin(1)
    acquired = [int(source["nominal_month"][5:7]) for source in sources]
    batch_rows = config["processing"]["batch_rows"]
    try:
        memory = gpu.configure_memory(config["processing"]["gpu_memory_fraction"])
        for source in sources:
            source_started = time.perf_counter()
            parquet = pq.ParquetFile(source["local_path"])  # Footer metadata only.
            columns = [
                name
                for name in [
                    "tpep_pickup_datetime",
                    "tpep_dropoff_datetime",
                    "PULocationID",
                    "DOLocationID",
                    "trip_distance",
                    "passenger_count",
                    "fare_amount",
                    "total_amount",
                ]
                if name in parquet.schema_arrow.names
            ]
            zone_parts, od_parts, row_offset = [], [], 0
            for row_group in range(parquet.num_row_groups):
                raw = cudf.read_parquet(
                    source["local_path"], columns=columns, row_groups=[row_group]
                )
                for offset in range(0, len(raw), batch_rows):
                    frame, quality = normalize_yellow_gpu(
                        raw.iloc[offset : offset + batch_rows], source, mapped_ids, row_offset
                    )
                    frame.to_parquet(
                        canonical / f"part-{part_index:06d}.parquet",
                        index=False,
                        compression="zstd",
                    )
                    row_offset += len(frame)
                    rows_total += len(frame)
                    part_index += 1
                    quality_parts.append(quality)
                    histogram_parts.append(daily_histograms_gpu(frame))
                    p_in = (frame.pickup_local >= start) & (frame.pickup_local < end)
                    d_in = (frame.dropoff_local >= start) & (frame.dropoff_local < end)
                    d_covered = (
                        frame.dropoff_local.dt.year == year
                    ) & frame.dropoff_local.dt.month.isin(acquired)
                    expected["pickup"] += int((frame.eligible_pickup & p_in).sum())
                    expected["dropoff"] += int((frame.eligible_dropoff & d_in & d_covered).sum())
                    expected["od"] += int((frame.eligible_od & p_in).sum())
                    device = gpu.derive_features(frame, config)
                    zone_part, od_part = gpu.aggregate(device)
                    zone_parts.append(zone_part)
                    od_parts.append(od_part)
                    if len(od_parts) >= 4:
                        od_parts = [_merge(od_parts, OD_KEYS)]
                    if len(zone_parts) >= 4:
                        zone_parts = [_merge(zone_parts, ZONE_KEYS)]
                    del frame, device, zone_part, od_part
                del raw
            if row_offset != source["parquet_row_count"]:
                raise ValueError("GPU ingestion source row count mismatch")
            if zone_parts:
                all_zone.append(_merge(zone_parts, ZONE_KEYS))
                od = _merge(od_parts, OD_KEYS)
                od = od.loc[(od.hour >= start) & (od.hour < end)]
                od.sort_values("hour").to_parquet(
                    output / "od_hour" / f"{source['nominal_month']}.parquet",
                    index=False,
                    compression="zstd",
                )
                graph_parts.append(
                    od.groupby(["pickup_zone_id", "dropoff_zone_id"])["trip_count"]
                    .sum()
                    .reset_index()
                )
            gpu.synchronize()
            timings.append(
                {
                    "source_id": source["source_id"],
                    "rows": row_offset,
                    "elapsed_s": time.perf_counter() - source_started,
                    "scope": "GPU_decode_normalize_write_aggregate_with_device_reconciliation",
                }
            )
        if not all_zone:
            raise ValueError("Native GPU path requires at least one input row")
        zone_hour = _coverage(_merge(all_zone, ZONE_KEYS), mapped_ids, sources, config)
        zone_hour.to_parquet(output / "zone_hour.parquet", index=False, compression="zstd")
        zones.to_parquet(output / "zones.parquet", index=False)
        shutil.copy2(zone_dir / "zones.geojson", output / "zones.geojson")
        quality = (
            pd.concat(quality_parts, ignore_index=True)
            .groupby(["source_id", "rule", "action"], as_index=False)[
                ["affected_rows", "total_rows"]
            ]
            .sum()
        )
        quality.to_parquet(output / "quality_summary.parquet", index=False)
        histograms = _merge(histogram_parts, HISTOGRAM_KEYS)
        histograms.to_parquet(output / "trip_histogram.parquet", index=False)
        profile = zone_hour.loc[
            zone_hour.pickup_coverage == "observed",
            ["dataset_type", "zone_id", "hour", "pickup_count"],
        ].copy()
        profile["hour_of_week"] = profile.hour.dt.weekday * 24 + profile.hour.dt.hour
        profile = (
            profile.groupby(["dataset_type", "zone_id", "hour_of_week"])["pickup_count"]
            .sum()
            .reset_index()
        )
        totals = (
            profile.groupby(["dataset_type", "zone_id"])["pickup_count"]
            .sum()
            .reset_index()
            .rename(columns={"pickup_count": "total"})
        )
        profile = profile.merge(totals, on=["dataset_type", "zone_id"])
        profile["activity_share"] = profile.pickup_count / profile.total.where(profile.total > 0)
        profile.drop(columns="total").to_parquet(output / "zone_profile.parquet", index=False)
        del all_zone, zone_parts, od_parts, od, histogram_parts, histograms, profile, totals
        result = {"tables": {}, "models": {}, "status": "not_requested"}
        if fit_models:
            from mobility.models.gpu_clustering import fit_clusters_gpu
            from mobility.models.gpu_anomalies import fit_anomalies_gpu, fit_duration_anomalies_gpu
            from mobility.models.graph_analysis import analyze_graph

            models = {
                "clustering": fit_clusters_gpu(canonical, config, output),
                "anomalies": fit_anomalies_gpu(zone_hour, config, output),
            }
            if config.get("duration_anomalies", {}).get("enabled", True):
                models["duration_anomalies"] = fit_duration_anomalies_gpu(canonical, config, output)
            if config.get("graph", {}).get("enabled", True):
                edges = _merge(graph_parts, ["pickup_zone_id", "dropoff_zone_id"])
                # At most 263² pairs: bounded graph labels/metadata are host finalization.
                models["graph"] = analyze_graph(
                    edges.to_pandas(), zones, config, output, backend="gpu", validate_cpu=False
                )
            result = {
                "tables": {
                    name: path
                    for model in models.values()
                    for name, path in model["tables"].items()
                },
                "models": {name: model["metadata"] for name, model in models.items()},
                "status": {name: model["metadata"]["status"] for name, model in models.items()},
            }
        validation = _validate_gpu(
            zone_hour, sorted((output / "od_hour").glob("*.parquet")), expected
        )
        tables = {
            "zone_hour": "zone_hour.parquet",
            "od_hour": "od_hour",
            "zones": "zones.parquet",
            "quality_summary": "quality_summary.parquet",
            "trip_histogram": "trip_histogram.parquet",
            "zone_profile": "zone_profile.parquet",
            **result["tables"],
        }
        gpu.synchronize()
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            raw_rows=rows_total,
            canonical_rows=rows_total,
            structurally_quarantined_rows=0,
            study_start=start.isoformat(),
            study_end=end.isoformat(),
            tables=tables,
            models=result["models"],
            model_status=result["status"],
            gpu_batch_reference_checks=False,
            validation=validation,
            row_counts=validation["counts"],
            pipeline_timings=timings,
            elapsed_s=time.perf_counter() - started,
            time_policy=config["time_policy"],
            execution_plan=execution_plan(rows_total, result["models"], "gpu", len(zone_hour)),
            gpu_memory_policy=memory,
            limitations=[
                "Recorded taxi trips are not total NYC demand.",
                "Drop-off coverage has partial source context.",
                "Small calendar/zone metadata, cryptographic identity hashes and reduced result packaging use the host.",
                "CPU reference comparisons are external to the native GPU production run.",
            ],
        )
        atomic_json(run / "manifest.json", manifest)
        return run
    except BaseException as error:
        manifest.update(
            status="canceled" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_at=utc_now(),
        )
        atomic_json(run / "manifest.json", manifest)
        raise
