"""Export a bounded, read-only website from a validated analytical snapshot.

The full Dash application remains available for richer hourly queries. This
portable website performs supported aggregate filtering in the browser and never
starts GPU jobs or exposes the raw source files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import warnings

import pandas as pd
from plotly.offline import get_plotlyjs

from mobility.serving.repository import SnapshotRepository

MAX_JSON_BYTES = 50_000_000
MAX_ZONE_DAYS = 400_000
MAX_ALERTS = 100
MAX_INCIDENTS = 200


def _records(frame: pd.DataFrame, columns=None, *, precision=6) -> list[dict]:
    if columns is not None:
        frame = frame[[column for column in columns if column in frame]]
    return json.loads(
        frame.to_json(orient="records", date_format="iso", double_precision=precision)
    )


def _model_summary(models: dict) -> dict:
    """Only publish explanatory values, never model paths or training identities."""
    allowed = {
        "status",
        "model_id",
        "k",
        "selected_k",
        "features",
        "feature_names",
        "training_rows",
        "training_sample_rows",
        "prediction_rows",
        "eligible_rows",
        "train_start",
        "train_end",
        "calibration_start",
        "calibration_end",
        "test_start",
        "test_end",
        "score_threshold",
        "absolute_threshold",
        "min_history",
        "scale_floor",
        "calibration_status",
        "evaluation_status",
        "backend",
        "preprocessing_backend",
        "eligible_fraction",
        "canonical_eligible_rows",
        "threshold",
        "metric",
        "computation_backend",
        "score_meaning",
        "manual_review_status",
        "unit",
        "baseline_meaning",
        "limitations",
    }
    # These structures are generated explanations and numeric policy values.
    # Training samples, record identities, paths and software environments stay private.
    explanatory = {
        "eligibility",
        "feature_explanation",
        "splits",
        "anomalies_config",
        "calibration",
        "duration_anomalies_config",
        "trip_filter_thresholds",
    }
    return {
        name: {
            key: value
            for key, value in model.items()
            if (key in allowed and isinstance(value, (str, int, float, bool, list, type(None))))
            or (key in explanatory and isinstance(value, dict))
        }
        for name, model in models.items()
        if isinstance(model, dict)
    }


def _daily(hours: pd.DataFrame) -> pd.DataFrame:
    data = hours.copy()
    data["date"] = pd.to_datetime(data.hour).dt.strftime("%Y-%m-%d")
    data["observed_hours"] = data.pickup_coverage.eq("observed").astype(int)
    data["dropoff_observed_hours"] = data.dropoff_coverage.eq("observed").astype(int)
    data["total_hours"] = 1
    keys = ["dataset_type", "zone_id", "date"]
    counts = [
        "pickup_count",
        "dropoff_count",
        "duration_sum_seconds",
        "duration_count",
        "observed_hours",
        "dropoff_observed_hours",
        "total_hours",
    ]
    result = data.groupby(keys, as_index=False)[counts].sum(min_count=1)
    return result.rename(
        columns={
            "pickup_count": "pickups",
            "dropoff_count": "dropoffs",
            "duration_sum_seconds": "duration_sum",
            "duration_count": "duration_n",
        }
    )


def _hour_profile(hours: pd.DataFrame) -> list[dict]:
    observed = hours[hours.pickup_coverage.eq("observed")]
    totals = observed.groupby("hour").pickup_count.sum(min_count=1).reset_index()
    totals["weekday"] = pd.to_datetime(totals.hour).dt.day_of_week
    totals["hour_of_day"] = pd.to_datetime(totals.hour).dt.hour
    grouped = (
        totals.groupby(["weekday", "hour_of_day"]).pickup_count.agg(["mean", "count"]).reset_index()
    )
    return _records(grouped.rename(columns={"mean": "mean_pickups", "count": "observed_hours"}))


def _anomalies(repo: SnapshotRepository, prefix="anomaly") -> dict:
    scores = (
        repo.read(f"{prefix}_scores", repo.default_filters())
        if repo.has(f"{prefix}_scores")
        else pd.DataFrame()
    )
    incidents = (
        repo.read(f"{prefix}_incidents") if repo.has(f"{prefix}_incidents") else pd.DataFrame()
    )
    result = {
        "alerts": [],
        "incidents": [],
        "contexts": {},
        "status_counts": [],
        "total_alert_hours": 0,
        "total_incidents": len(incidents),
        "alert_limit": MAX_ALERTS,
        "incident_limit": MAX_INCIDENTS,
    }
    if not incidents.empty:
        incidents = (
            incidents.assign(rank=incidents.peak_score.abs())
            .sort_values("rank", ascending=False)
            .head(MAX_INCIDENTS)
        )
        result["incidents"] = _records(
            incidents,
            [
                "zone_id",
                "direction",
                "start_hour",
                "end_hour",
                "affected_hours",
                "peak_score",
                "total_deviation",
                "baseline_support_min",
                "label",
                "split",
            ],
        )
    if scores.empty:
        return result
    result["status_counts"] = _records(
        scores.status.value_counts().rename_axis("status").reset_index(name="zone_hours")
    )
    alerts = scores[scores.is_alert.fillna(False)].copy()
    result["total_alert_hours"] = len(alerts)
    alerts["rank"] = alerts.score.abs()
    alerts = alerts.sort_values(["rank", "hour"], ascending=[False, True]).head(MAX_ALERTS)
    result["alerts"] = _records(
        alerts,
        [
            "zone_id",
            "hour",
            "observed",
            "baseline",
            "score",
            "threshold",
            "absolute_threshold",
            "history_count",
            "direction",
            "status",
            "split",
            "eligible_trip_count",
        ],
    )
    for index, row in enumerate(alerts.itertuples()):
        hour = pd.Timestamp(row.hour)
        context = scores[
            (scores.zone_id == row.zone_id)
            & (pd.to_datetime(scores.hour) >= hour - pd.Timedelta(hours=48))
            & (pd.to_datetime(scores.hour) <= hour + pd.Timedelta(hours=48))
        ].sort_values("hour")
        if prefix == "duration":
            # Missing duration observations are gaps, not zero-minute trips or
            # a line interpolated across unsupported/missing hours.
            context = context.set_index("hour").reindex(
                pd.date_range(
                    hour - pd.Timedelta(hours=48), hour + pd.Timedelta(hours=48), freq="h"
                )
            )
            context.index.name = "hour"
            context = context.reset_index()
            context.loc[context.status.ne("scored"), ["observed", "baseline"]] = float("nan")
        result["contexts"][str(index)] = _records(
            context, ["hour", "observed", "baseline", "is_alert", "status"]
        )
        result["alerts"][index]["context_id"] = str(index)
    return result


def _benchmark_payload(repo: SnapshotRepository, trials: pd.DataFrame, *, pipeline=False) -> dict:
    comparisons = []
    if not trials.empty:
        valid = trials[pd.to_numeric(trials.elapsed_s, errors="coerce").gt(0)]
        valid = valid[
            valid.validation.astype(str).str.lower().isin(["passed", "pass", "true", "ok"])
        ]
        if "phase" in valid:
            valid = valid[valid.phase.eq("measured")]
        protocol = "protocol_hash" if "protocol_hash" in valid else "config_hash"
        keys = [
            key
            for key in ["experiment_id", "stage", "timing_scope", "input_rows", protocol]
            if key in valid
        ]
        if keys and not valid.empty:
            paired = (
                valid.groupby(keys + ["backend"], dropna=False)
                .elapsed_s.median()
                .unstack("backend")
            )
            if "cpu" in paired:
                for backend in ["gpu", "gpu_library", "gpu_custom", "cuda", "custom_cuda"]:
                    if backend not in paired:
                        continue
                    frame = (
                        paired[["cpu", backend]]
                        .dropna()
                        .reset_index()
                        .rename(columns={"cpu": "cpu_seconds", backend: "gpu_seconds"})
                    )
                    frame["gpu_backend"] = backend
                    frame["speedup"] = frame.cpu_seconds / frame.gpu_seconds
                    comparisons.extend(_records(frame, precision=12))
    return {
        "trials": _records(
            trials.head(2000),
            [
                "experiment_id",
                "repeat",
                "backend",
                "stage",
                "timing_scope",
                "input_rows",
                "elapsed_s",
                "validation",
                "config_hash",
                "peak_rss_bytes",
                "peak_device_bytes",
                "threads",
                "cpu_threads",
                "peak_host_rss_sampled_bytes",
                "rss_sample_interval_s",
                "device_pool_used_after_bytes",
                "device_pool_reserved_after_bytes",
                "host_device_transfer_bytes",
                "rows_per_second",
                "scope_detail",
                "cache_condition",
                "memory_measurement",
                "protocol_hash",
                "phase",
                "months",
                "month_count",
                "peak_gpu_memory_sampled_bytes",
                "gpu_sample_interval_s",
                "rss_samples",
                "gpu_samples",
                "rss_process_tree_complete",
                "rss_max_observed_gap_s",
                "gpu_max_observed_gap_s",
                "rss_sampling_errors",
                "gpu_sampling_errors",
                "resource_status",
                "gpu_background_processes",
                "background_gpu_processes",
                "selected_k",
                "cluster_eligible_rows",
                "validation_elapsed_s",
            ],
            precision=12,
        ),
        "total_trials": len(trials),
        "display_limit": 2000,
        "comparisons": comparisons,
        "environment": _public_environment(
            repo.manifest.get(
                "pipeline_benchmark_environment" if pipeline else "benchmark_environment", {}
            )
        ),
    }


def _public_environment(environment: dict) -> dict:
    """Publish hardware/method evidence without server names or library file paths."""
    allowed = {
        "recorded_at",
        "platform",
        "machine",
        "python",
        "cpu_logical_cores",
        "cpu_physical_cores",
        "host_memory_bytes",
        "packages",
        "git_commit",
        "git_dirty",
        "experiment_id",
        "requested_gpu",
        "input_selection",
        "precision",
        "cache_state",
        "scope_notes",
        "memory_notes",
        "numeric_tolerance",
        "logical_bytes_per_row",
        "actual_sizes",
        "status",
        "warm_initialization_s",
        "gpu",
        "requested_month_counts",
        "repeats",
        "warmups",
        "cpu_threads",
        "timing_scope",
        "resource_notes",
        "validation_notes",
        "model_notes",
    }
    result = {key: value for key, value in environment.items() if key in allowed}
    settings = environment.get("config", {})
    if "benchmark" in settings:
        settings = settings["benchmark"]
    # The full-pipeline CLI can override the project defaults. Publish its actual
    # protocol, rather than showing the older stage benchmark's repetition count.
    settings = {
        **settings,
        **{
            key: environment[key]
            for key in ["warmups", "repeats", "cpu_threads", "seed"]
            if key in environment
        },
    }
    result["config"] = {
        key: value
        for key, value in settings.items()
        if key
        in {
            "warmups",
            "repeats",
            "cpu_threads",
            "sizes",
            "seed",
            "randomize_backend_order",
            "synchronize_gpu",
        }
    }
    for key in ["threadpools", "effective_threadpools"]:
        if key in environment:
            result[key] = [
                {field: value for field, value in pool.items() if field != "filepath"}
                for pool in environment[key]
            ]
    return result


def _histogram_payload(repo: SnapshotRepository) -> list[dict]:
    if not repo.has("trip_histogram"):
        return []
    data = repo.read("trip_histogram")
    if data.empty:
        return []
    bins = (
        data.groupby(["feature", "bin_index", "bin_label"], as_index=False, dropna=False)["count"]
        .sum()
        .sort_values(["feature", "bin_index"])
    )
    return _records(bins)


def export_public(snapshot_dir: Path, destination_dist: Path) -> Path:
    """Write a self-contained website, failing before an oversized data publication."""
    repo = SnapshotRepository(snapshot_dir)
    validation = repo.manifest.get("validation", {})
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("Refusing to publish a snapshot without passed validation.")
    destination = Path(destination_dist).resolve()
    if (
        destination == repo.root
        or destination.is_relative_to(repo.root)
        or repo.root.is_relative_to(destination)
    ):
        raise ValueError("The public destination must be separate from the analytical snapshot.")
    hours = repo.read("zone_hour", repo.default_filters())
    days = _daily(hours)
    if len(days) > MAX_ZONE_DAYS:
        raise ValueError(
            f"Public export has {len(days):,} zone-days; the supported limit is {MAX_ZONE_DAYS:,}. Publish a narrower study."
        )
    zones = repo.zones()
    summary = repo.read("cluster_summary")
    samples = (
        repo.read("trip_plot_sample", repo.default_filters()).head(repo.max_points)
        if repo.has("trip_plot_sample")
        else pd.DataFrame()
    )
    connections = repo.od_totals(repo.default_filters())
    if len(connections) > 100_000:
        raise ValueError("The public OD matrix exceeds 100,000 zone pairs.")
    trials = repo.read("benchmark_runs")
    quality = repo.read("quality_summary")
    sources = repo.manifest.get("sources", [])
    if isinstance(sources, dict):
        sources = list(sources.values())
    public_sources = [
        {
            key: value
            for key, value in source.items()
            if key
            in {
                "source_id",
                "dataset_type",
                "nominal_month",
                "source_month",
                "sha256",
                "byte_count",
                "parquet_row_count",
                "url",
                "downloaded_at_utc",
            }
        }
        for source in sources
        if isinstance(source, dict)
    ]
    meta = {
        "snapshot_id": repo.snapshot_id,
        "created_at": repo.manifest.get("created_at"),
        "data_kind": repo.manifest.get("data_kind", "unknown"),
        "study_start": repo.manifest["study_start"],
        "study_end": repo.manifest["study_end"],
        "dataset": repo.default_filters().dataset,
        "raw_rows": repo.manifest.get("raw_rows"),
        "models": _model_summary(repo.manifest.get("models", {})),
        "splits": repo.manifest.get("config", {}).get("splits", {}),
        "anomaly_parameters": repo.manifest.get("config", {}).get("anomalies", {}),
        "processing": {
            "aggregation_backend": repo.manifest.get("config", {})
            .get("processing", {})
            .get("backend", "unknown"),
            "clustering_backend": repo.manifest.get("models", {})
            .get("clustering", {})
            .get("backend", "unknown"),
            "anomaly_backend": "cpu",
            "duration_anomaly_backend": "cpu" if repo.has("duration_scores") else None,
            "graph_backend": repo.manifest.get("models", {}).get("graph", {}).get("backend"),
            "browser_execution": "Precomputed snapshot; no GPU jobs run in the browser.",
        },
        "time_convention": "America/New_York — source local wall time",
        "export_kind": "static_published_snapshot",
        "max_flows": repo.max_flows,
        "max_points": repo.max_points,
        "max_export_rows": repo.max_export,
        "capabilities": {
            "overview": ["date", "zone"],
            "od": ["origin", "destination"],
            "patterns": ["sample_cluster"],
            "anomalies": ["exported_alert_zone"],
            "performance": ["timing_scope", "backend"],
        },
    }
    payloads = {
        "summary.json": {
            "meta": meta,
            "zones": _records(zones, ["zone_id", "zone", "borough", "longitude", "latitude"]),
            "zone_days": _records(days),
        },
        "patterns.json": {
            "hour_profile": _hour_profile(hours),
            "clusters": _records(summary),
            "histograms": _histogram_payload(repo),
            "samples": _records(
                samples,
                [
                    "trip_distance_miles",
                    "duration_seconds",
                    "cluster",
                    "inclusion_fraction",
                    "sample_design",
                ],
            ),
        },
        "od.json": {
            "connections": _records(
                connections,
                ["pickup_zone_id", "dropoff_zone_id", "trip_count", "mean_duration_minutes"],
            ),
            "total_pairs": len(connections),
            "scope": "fixed_full_snapshot_period",
        },
        "anomalies.json": _anomalies(repo),
        "duration-anomalies.json": _anomalies(repo, "duration"),
        "graph.json": {
            "nodes": _records(
                repo.read("graph_nodes"),
                [
                    "zone_id",
                    "zone",
                    "borough",
                    "pagerank",
                    "community",
                    "in_degree",
                    "out_degree",
                    "degree",
                    "in_trip_strength",
                    "out_trip_strength",
                    "self_trip_count",
                    "is_isolated",
                ],
                precision=12,
            )
            if repo.has("graph_nodes")
            else [],
            "metadata": repo.manifest.get("models", {}).get("graph", {}),
            "scope": "fixed_full_snapshot_period",
        },
        "performance.json": _benchmark_payload(repo, trials),
        "pipeline-performance.json": _benchmark_payload(
            repo,
            repo.read("pipeline_benchmark_runs")
            if repo.has("pipeline_benchmark_runs")
            else pd.DataFrame(),
            pipeline=True,
        ),
        "methods.json": {
            "sources": public_sources,
            "quality": _records(quality.head(2000)),
            "quality_total_rows": len(quality),
            "models": meta["models"],
            "validation": validation,
        },
    }
    geometry = repo.geojson()
    if geometry:
        payloads["zones.geojson"] = geometry
    encoded = {
        name: json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False
        ).encode()
        for name, payload in payloads.items()
    }
    total = sum(len(value) for value in encoded.values())
    if total > MAX_JSON_BYTES:
        raise ValueError(
            f"Public JSON totals {total / 1_000_000:.1f} MB, exceeding the 50 MB budget. Narrow the published study or revise the aggregate contract."
        )
    if total > 20_000_000:
        warnings.warn(
            f"Public snapshot data is {total / 1_000_000:.1f} MB before compression; consider a narrower public study for slow connections.",
            stacklevel=2,
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(exist_ok=True)
    if not geometry:
        (destination / "data" / "zones.geojson").unlink(missing_ok=True)
    for name, content in encoded.items():
        (destination / "data" / name).write_bytes(content)
    template = Path(__file__).parent / "static"
    for name in ["index.html", "app.js", "style.css"]:
        shutil.copyfile(template / name, destination / name)
    (destination / "plotly.min.js").write_text(get_plotlyjs())
    export_manifest = {
        "snapshot_id": repo.snapshot_id,
        "source_manifest_sha256": hashlib.sha256(
            (repo.root / "manifest.json").read_bytes()
        ).hexdigest(),
        "data_kind": meta["data_kind"],
        "json_bytes": total,
        "files": {
            f"data/{name}": {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in encoded.items()
        },
        "limits": {
            "zone_days": MAX_ZONE_DAYS,
            "json_bytes": MAX_JSON_BYTES,
            "sample_points": repo.max_points,
            "alert_hours": MAX_ALERTS,
            "incidents": MAX_INCIDENTS,
        },
    }
    (destination / "export-manifest.json").write_text(json.dumps(export_manifest, indent=2))
    return destination
