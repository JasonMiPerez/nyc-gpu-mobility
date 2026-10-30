"""Bounded ingestion, additive aggregation, model fitting, and atomic publication."""

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mobility.aggregate import (
    HISTOGRAM_KEYS,
    OD_KEYS,
    ZONE_KEYS,
    aggregate_cpu,
    daily_histograms,
    merge_aggregates,
)
from mobility.data.acquire import inspect_parquet
from mobility.data.quality import hourly_dst_mask, normalize_yellow
from mobility.io import atomic_json, read_json, sha256_file, utc_now

LOGGER = logging.getLogger(__name__)


def environment_record() -> dict:
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "scikit-learn", "dash", "plotly", "geopandas"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "git_commit": commit.stdout.strip() or None,
        "dirty_tree": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def source_list(data_root: Path, config: dict) -> list[dict]:
    sources = []
    for month in config["data"]["months"]:
        path = (
            data_root
            / "raw"
            / "yellow"
            / f"yellow_tripdata_{config['data']['year']}-{month:02d}.parquet"
        )
        metadata_path = path.with_suffix(".parquet.json")
        if not path.exists() or not metadata_path.exists():
            if config["processing"]["allow_missing_sources"]:
                continue
            raise FileNotFoundError(f"Missing source {path.name}; run mobility data fetch first")
        metadata = read_json(metadata_path)
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Source checksum changed: {path}")
        metadata.update(inspect_parquet(path), local_path=str(path.resolve()))
        sources.append(metadata)
    if not sources:
        raise ValueError("No validated source files are available")
    return sources


def add_coverage(
    zone: pd.DataFrame, zones: list[int], sources: list[dict], config: dict
) -> pd.DataFrame:
    """Keep an explicit requested-period calendar, including unacquired months."""
    year, months = config["data"]["year"], config["data"]["months"]
    start = pd.Timestamp(year=year, month=min(months), day=1)
    end = pd.Timestamp(year=year, month=max(months), day=1) + pd.offsets.MonthBegin(1)
    hours = pd.date_range(start, end, inclusive="left", freq="h")
    grid = pd.MultiIndex.from_product([["yellow"], zones, hours], names=ZONE_KEYS).to_frame(
        index=False
    )
    result = grid.merge(zone, how="left", on=ZONE_KEYS, validate="one_to_one")
    acquired = {source["nominal_month"] for source in sources}
    observed = result.hour.dt.strftime("%Y-%m").isin(acquired)
    counts = ["pickup_count", "dropoff_count", "duration_count", "distance_count"]
    for column in counts + ["duration_sum_seconds", "distance_sum_miles"]:
        result[column] = result[column].fillna(0).where(observed)
        if column in counts:
            result[column] = result[column].astype("Int64")
    coverage = np.where(observed, "observed", "missing_source")
    result["pickup_coverage"] = coverage
    # Without a validated upper trip-duration bound, drop-off completeness cannot
    # be proved from pickup-month sources. Preserve observed counts but mark partial.
    result["dropoff_coverage"] = np.where(observed, "partial_source_context", "missing_source")
    discontinuous = set(hours[hourly_dst_mask(hours)])
    dst = result.hour.isin(discontinuous) & observed
    result.loc[dst, ["pickup_coverage", "dropoff_coverage"]] = "dst_ambiguous"
    result["time_policy_version"] = config["time_policy"]
    return result.sort_values(["hour", "zone_id"]).reset_index(drop=True)


def validate_outputs(output_dir: Path, expected: dict | None = None) -> dict:
    zone = pd.read_parquet(output_dir / "zone_hour.parquet")
    if zone.duplicated(ZONE_KEYS).any():
        raise ValueError("Duplicate zone-hour keys")
    if (zone[["pickup_count", "dropoff_count"]] < 0).any().any():
        raise ValueError("Negative endpoint counts")
    missing = zone.pickup_coverage.eq("missing_source")
    if zone.loc[missing, "pickup_count"].notna().any():
        raise ValueError("Missing source coverage was converted to observed counts")
    od_total = 0
    seen_od_months = set()
    for part in sorted((output_dir / "od_hour").glob("*.parquet")):
        od = pd.read_parquet(part)
        if od.duplicated(OD_KEYS).any() or od.trip_count.lt(0).any():
            raise ValueError("Invalid OD keys/counts")
        part_months = set(pd.to_datetime(od.hour).dt.strftime("%Y-%m"))
        if len(part_months) > 1 or seen_od_months.intersection(part_months):
            raise ValueError("OD month partitions overlap; duplicate keys could cross files")
        seen_od_months.update(part_months)
        od_total += int(od.trip_count.sum())
    counts = {
        "pickup": int(zone.pickup_count.sum()),
        "dropoff": int(zone.dropoff_count.sum()),
        "od": od_total,
    }
    if expected is not None:
        for key, value in counts.items():
            if value != expected[key]:
                raise ValueError(f"{key} reconciliation failed: {value} != {expected[key]}")
    return {
        "status": "passed",
        "counts": counts,
        "unique_zone_hour_keys": True,
        "missing_sources_are_null": True,
    }


def run_pipeline(root: Path, config: dict, *, fit_models: bool = True) -> Path:
    started = time.perf_counter()
    root = root.resolve()
    data_root, artifacts = root / "data", root / "artifacts"
    sources = source_list(data_root, config)
    zone_dir = data_root / "zones"
    if not (zone_dir / "zones.parquet").exists():
        raise FileNotFoundError("Run mobility zones prepare before the pipeline")
    zones = pd.read_parquet(zone_dir / "zones.parquet")
    mapped_ids = set(zones.loc[zones.longitude.notna(), "zone_id"].astype(int))
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = artifacts / "runs" / run_id
    canonical = run_dir / "canonical"
    output = run_dir / "tables"
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
        "backend": config["processing"]["backend"],
        "data_kind": "synthetic"
        if any(s.get("data_kind") == "synthetic" for s in sources)
        else "real",
    }
    atomic_json(run_dir / "manifest.json", manifest)
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    expected = {"pickup": 0, "dropoff": 0, "od": 0}
    all_zone, quality_parts, histogram_parts, timings = [], [], [], []
    rows_total, part_index = 0, 0
    year, months = config["data"]["year"], config["data"]["months"]
    start = pd.Timestamp(year=year, month=min(months), day=1)
    end = pd.Timestamp(year=year, month=max(months), day=1) + pd.offsets.MonthBegin(1)
    acquired = {s["nominal_month"] for s in sources}
    try:
        if config["processing"]["backend"] == "gpu":
            from mobility.backends import gpu

            gpu.configure_memory(config["processing"]["gpu_memory_fraction"])
        for source in sources:
            source_started = time.perf_counter()
            parquet = pq.ParquetFile(source["local_path"])
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
            for batch in parquet.iter_batches(
                batch_size=config["processing"]["batch_rows"], columns=columns
            ):
                frame, quality = normalize_yellow(batch.to_pandas(), source, mapped_ids, row_offset)
                frame.to_parquet(
                    canonical / f"part-{part_index:06d}.parquet", index=False, compression="zstd"
                )
                row_offset += len(frame)
                rows_total += len(frame)
                part_index += 1
                quality_parts.append(quality)
                histogram_parts.append(daily_histograms(frame))
                p_in = frame.pickup_local.ge(start) & frame.pickup_local.lt(end)
                d_in = frame.dropoff_local.ge(start) & frame.dropoff_local.lt(end)
                d_covered = frame.dropoff_local.dt.strftime("%Y-%m").isin(acquired)
                expected["pickup"] += int((frame.eligible_pickup & p_in).sum())
                expected["dropoff"] += int((frame.eligible_dropoff & d_in & d_covered).sum())
                expected["od"] += int((frame.eligible_od & p_in).sum())
                if config["processing"]["backend"] == "gpu":
                    from mobility.backends import gpu

                    device = gpu.derive_features(gpu.to_device(frame), config)
                    zone_part, od_part = gpu.aggregate(device)
                    gpu.synchronize()
                    zone_part, od_part = zone_part.to_pandas(), od_part.to_pandas()
                    if config["processing"].get("verify_gpu_batches", True):
                        # Production validation checks each batch. Matched benchmarks
                        # compare complete CPU/GPU outputs after their timed windows.
                        cpu_zone, cpu_od = aggregate_cpu(frame)
                        for actual, reference, keys in [
                            (zone_part, cpu_zone, ZONE_KEYS),
                            (od_part, cpu_od, OD_KEYS),
                        ]:
                            a = actual.sort_values(keys).reset_index(drop=True)
                            b = reference.sort_values(keys).reset_index(drop=True)
                            pd.testing.assert_frame_equal(
                                a[b.columns],
                                b,
                                check_dtype=False,
                                check_exact=False,
                                rtol=1e-9,
                                atol=1e-7,
                            )
                else:
                    zone_part, od_part = aggregate_cpu(frame)
                zone_parts.append(zone_part)
                od_parts.append(od_part)
                if len(od_parts) >= 4:
                    od_parts = [merge_aggregates(od_parts, OD_KEYS)]
                LOGGER.info(
                    "%s: %s / %s rows",
                    source["nominal_month"],
                    f"{row_offset:,}",
                    f"{source['parquet_row_count']:,}",
                )
            if row_offset != source["parquet_row_count"]:
                raise ValueError("Canonical ingestion did not reconcile with source row count")
            if zone_parts:
                all_zone.append(merge_aggregates(zone_parts, ZONE_KEYS))
                od = merge_aggregates(od_parts, OD_KEYS)
                od = od.loc[od.hour.ge(start) & od.hour.lt(end)]
                od.sort_values("hour").to_parquet(
                    output / "od_hour" / f"{source['nominal_month']}.parquet",
                    index=False,
                    compression="zstd",
                )
            timings.append(
                {
                    "source_id": source["source_id"],
                    "rows": row_offset,
                    "elapsed_s": time.perf_counter() - source_started,
                    "scope": "read_normalize_write_aggregate_with_validation",
                }
            )
        if not all_zone:
            # An empty input still has a valid schema and observed zero-count grid.
            base = pd.DataFrame(
                columns=ZONE_KEYS
                + [
                    "pickup_count",
                    "dropoff_count",
                    "duration_sum_seconds",
                    "duration_count",
                    "distance_sum_miles",
                    "distance_count",
                ]
            )
        else:
            base = merge_aggregates(all_zone, ZONE_KEYS)
        zone_hour = add_coverage(base, sorted(mapped_ids), sources, config)
        zone_hour.to_parquet(output / "zone_hour.parquet", index=False, compression="zstd")
        zones.to_parquet(output / "zones.parquet", index=False)
        shutil.copy2(zone_dir / "zones.geojson", output / "zones.geojson")
        if quality_parts:
            quality = (
                pd.concat(quality_parts, ignore_index=True)
                .groupby(["source_id", "rule", "action"], as_index=False)[
                    ["affected_rows", "total_rows"]
                ]
                .sum()
            )
        else:
            quality = pd.DataFrame(
                columns=["source_id", "rule", "action", "affected_rows", "total_rows"]
            )
        quality.to_parquet(output / "quality_summary.parquet", index=False)
        histograms = (
            pd.concat(histogram_parts, ignore_index=True)
            .groupby(HISTOGRAM_KEYS, as_index=False)["count"]
            .sum()
            if histogram_parts
            else pd.DataFrame(columns=HISTOGRAM_KEYS + ["count"])
        )
        histograms.to_parquet(output / "trip_histogram.parquet", index=False)
        profile = zone_hour.loc[
            zone_hour.pickup_coverage.eq("observed"),
            ["dataset_type", "zone_id", "hour", "pickup_count"],
        ].copy()
        profile["hour_of_week"] = profile.hour.dt.day_of_week * 24 + profile.hour.dt.hour
        profile = profile.groupby(["dataset_type", "zone_id", "hour_of_week"], as_index=False)[
            "pickup_count"
        ].sum()
        denominator = profile.groupby(["dataset_type", "zone_id"])["pickup_count"].transform("sum")
        profile["activity_share"] = profile.pickup_count / denominator.where(denominator.gt(0))
        profile.to_parquet(output / "zone_profile.parquet", index=False)
        model_result = {"tables": {}, "models": {}, "status": "not_requested"}
        if fit_models:
            from mobility.models.analysis import run_models

            LOGGER.info("Fitting chronological models and evaluating available holdouts")
            model_result = run_models(
                canonical, zone_hour, config, output, backend=config["processing"]["backend"]
            )
        validation = validate_outputs(output, expected)
        tables = {
            "zone_hour": "zone_hour.parquet",
            "od_hour": "od_hour",
            "zones": "zones.parquet",
            "quality_summary": "quality_summary.parquet",
            "trip_histogram": "trip_histogram.parquet",
            "zone_profile": "zone_profile.parquet",
            **model_result.get("tables", {}),
        }
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            raw_rows=rows_total,
            canonical_rows=rows_total,
            structurally_quarantined_rows=0,
            study_start=start.isoformat(),
            study_end=end.isoformat(),
            tables=tables,
            models=model_result.get("models", {}),
            model_status=model_result.get("status"),
            gpu_batch_reference_checks=(
                config["processing"]["backend"] == "gpu"
                and config["processing"].get("verify_gpu_batches", True)
            ),
            validation=validation,
            row_counts=validation["counts"],
            pipeline_timings=timings,
            elapsed_s=time.perf_counter() - started,
            time_policy=config["time_policy"],
            limitations=[
                "Recorded taxi trips are not total NYC travel demand.",
                "Drop-off coverage is partial without a bound on earlier pickup context.",
                "TLC local timestamps do not resolve repeated DST hours.",
                "Pipeline durations include correctness checks; use the benchmark harness for controlled comparisons.",
            ],
        )
        atomic_json(run_dir / "manifest.json", manifest)
        return run_dir
    except KeyboardInterrupt:
        manifest.update(status="canceled", canceled_at=utc_now())
        atomic_json(run_dir / "manifest.json", manifest)
        LOGGER.info("Run %s canceled; the published snapshot is unchanged", run_id)
        raise
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error), failed_at=utc_now()
        )
        atomic_json(run_dir / "manifest.json", manifest)
        LOGGER.exception("Run %s failed; the active published snapshot is unchanged", run_id)
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def publish_run(run_dir: Path, artifacts_root: Path) -> Path:
    run_dir, artifacts_root = run_dir.resolve(), artifacts_root.resolve()
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Only successfully completed runs can be published")
    validation = validate_outputs(run_dir / "tables", manifest["validation"]["counts"])
    snapshot_id = manifest["run_id"] + "-" + uuid.uuid4().hex[:6]
    snapshots = artifacts_root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    staging = snapshots / (".staging-" + snapshot_id)
    final = snapshots / snapshot_id
    staging.mkdir()
    tables, hashes = {}, {}
    # Per-trip predictions stay in research artifacts; the public site gets bounded summaries.
    excluded = {"trip_cluster_assignment", "trip_features"}
    try:
        for name, relative in manifest["tables"].items():
            if name in excluded:
                continue
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe artifact path in run manifest")
            source = (run_dir / "tables" / relative).resolve()
            if not source.is_relative_to((run_dir / "tables").resolve()):
                raise ValueError("Table path escapes the run")
            target = staging / relative
            if not target.resolve().is_relative_to(staging.resolve()):
                raise ValueError("Artifact destination escapes staging")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                if any(item.is_symlink() for item in source.rglob("*")):
                    raise ValueError("Published artifact directories must not contain symlinks")
                shutil.copytree(source, target)
                for part in target.rglob("*.parquet"):
                    hashes[str(part.relative_to(staging))] = sha256_file(part)
            else:
                shutil.copy2(source, target)
                hashes[relative] = sha256_file(target)
            tables[name] = relative
        shutil.copy2(run_dir / "tables" / "zones.geojson", staging / "zones.geojson")
        public = {
            **manifest,
            "snapshot_id": snapshot_id,
            "tables": tables,
            "validation": validation,
            "file_hashes": hashes,
            "published_at": utc_now(),
        }
        public["sources"] = [
            {k: v for k, v in source.items() if k != "local_path"} for source in manifest["sources"]
        ]
        atomic_json(staging / "manifest.json", public)
        os.replace(staging, final)
        atomic_json(
            artifacts_root / "current.json",
            {"snapshot_id": snapshot_id, "path": str(final.relative_to(artifacts_root))},
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def current_snapshot(artifacts_root: Path) -> Path:
    pointer = read_json(artifacts_root / "current.json")
    snapshot = (artifacts_root / pointer["path"]).resolve()
    if (
        not snapshot.is_relative_to(artifacts_root.resolve())
        or not (snapshot / "manifest.json").is_file()
    ):
        raise ValueError("Invalid active snapshot pointer")
    return snapshot
