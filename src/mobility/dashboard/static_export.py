"""Export a bounded, read-only website from a validated analytical snapshot.

The full Dash application remains available for richer hourly queries. This
portable website performs supported aggregate filtering in the browser and never
starts GPU jobs or exposes the raw source files.
"""

from __future__ import annotations

import hashlib
import json
import math
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
        "initialization_policy",
        "sampling_policy",
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
    result = {
        name: {
            key: value
            for key, value in model.items()
            if (key in allowed and isinstance(value, (str, int, float, bool, list, type(None))))
            or (key in explanatory and isinstance(value, dict))
        }
        for name, model in models.items()
        if isinstance(model, dict)
    }
    for name, model in models.items():
        if name in result:
            # Publish the sampling method, never the individual sample identities.
            method = model.get("samples", {}).get("train", {}).get("method")
            if isinstance(method, str):
                result[name].setdefault("sampling_policy", method)
    return result


def processing_metadata(manifest: dict) -> dict:
    """Describe the saved run, including old snapshots without execution evidence."""
    settings = manifest.get("config", {}).get("processing", {})
    mode = manifest.get("execution_mode", settings.get("execution_mode", "legacy"))
    plan = {
        stage: {key: value[key] for key in ("backend", "implementation", "rows") if key in value}
        for stage, value in manifest.get("execution_plan", {}).items()
        if isinstance(value, dict)
    }
    models = manifest.get("models", {})

    def backend(stage, model_name=None, fallback=None):
        model = models.get(model_name, {})
        value = plan.get(stage, {}).get("backend") or model.get("backend")
        value = value or model.get("computation_backend") or fallback
        if isinstance(value, str):
            for label in ("cpu", "gpu"):
                if value.lower().startswith(label):
                    return label
        return value or "unknown"

    # Older volume/duration implementations were CPU-only. Do not infer that
    # default for a native model whose execution evidence is missing.
    def old_detector(name):
        return "cpu" if mode == "legacy" and name in models else None

    return {
        "execution_mode": mode,
        "execution_plan": plan,
        "aggregation_backend": backend(
            "aggregation", fallback=settings.get("backend", manifest.get("backend"))
        ),
        "clustering_backend": backend("kmeans_fit", "clustering"),
        "anomaly_backend": backend("volume_scoring", "anomalies", old_detector("anomalies")),
        "duration_anomaly_backend": backend(
            "duration_scoring", "duration_anomalies", old_detector("duration_anomalies")
        ),
        "graph_backend": backend("graph", "graph"),
        "browser_execution": "Precomputed snapshot; no GPU jobs run in the browser.",
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


def _json_value(value):
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _finite_number(value):
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def clustering_quality_rows(trials: pd.DataFrame) -> list[dict]:
    """Project scoped model diagnostics; never export whole model manifests."""
    rows = trials.loc[trials.phase.eq("measured")] if "phase" in trials else trials
    result = []
    for row in rows.head(2000).to_dict("records"):
        models = _json_value(row.get("model_quality_json"))
        model = models.get("clustering", {}) if isinstance(models, dict) else {}
        model = model if isinstance(model, dict) else {}
        selected_k = _finite_number(model.get("selected_k", row.get("selected_k")))
        candidates = model.get("candidates", [])
        chosen = (
            next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, dict) and candidate.get("k") == selected_k
                ),
                {},
            )
            if isinstance(candidates, list)
            else {}
        )
        calibration = chosen.get("calibration", {})
        test = model.get("test", {})
        calibration = calibration if isinstance(calibration, dict) else {}
        test = test if isinstance(test, dict) else {}
        selection_status = model.get("status")
        test_status = model.get("test_status")
        result.append(
            {
                key: row.get(key)
                for key in (
                    "experiment_id",
                    "protocol_hash",
                    "backend",
                    "repeat",
                    "validation",
                )
            }
            | {
                "months": row.get("months", row.get("month_count")),
                "selected_k": selected_k,
                "diagnostics_status": "recorded" if model else "not_recorded",
                "selection_status": selection_status
                if selection_status
                in {
                    "calibration_selected",
                    "exploratory_no_calibration",
                    "insufficient_training_rows",
                }
                else "not_recorded",
                "test_status": test_status
                if test_status
                in {
                    "evaluated",
                    "no_held_out_rows",
                }
                else "not_recorded",
                "training_rows": _finite_number(chosen.get("training_rows")),
                "training_inertia_per_row": _finite_number(chosen.get("training_inertia_per_row")),
                "calibration_rows": _finite_number(calibration.get("rows")),
                "calibration_silhouette": _finite_number(calibration.get("silhouette")),
                "test_rows": _finite_number(test.get("rows")),
                "test_silhouette": _finite_number(test.get("silhouette")),
                "test_mean_squared_center_distance": _finite_number(
                    test.get("mean_squared_center_distance")
                ),
            }
        )
    return result


def cpu_execution_evidence(trials: pd.DataFrame) -> dict:
    """Summarize recorded native CPU isolation; missing NVML is never zero usage."""
    rows = [
        row
        for row in trials.to_dict("records")
        if row.get("backend") == "cpu"
        and row.get("phase") == "measured"
        and (
            row.get("execution_mode") == "native"
            or row.get("stage") == "native_production_pipeline"
        )
    ]
    guards = sum(
        row.get("cpu_import_guard_enabled") is True
        and _json_value(row.get("loaded_gpu_modules_json")) == []
        for row in rows
    )
    observed = [
        row
        for row in rows
        if (_finite_number(row.get("gpu_samples")) or 0) > 0
        and _finite_number(row.get("peak_gpu_memory_sampled_bytes")) is not None
    ]
    return {
        "measured_native_cpu_trials": len(rows),
        "guard_and_no_cuda_imports_confirmed": guards,
        "zero_sampled_gpu_memory_trials": sum(
            row["peak_gpu_memory_sampled_bytes"] == 0 for row in observed
        ),
        "nonzero_sampled_gpu_memory_trials": sum(
            row["peak_gpu_memory_sampled_bytes"] > 0 for row in observed
        ),
        "gpu_memory_unavailable_trials": len(rows) - len(observed),
    }


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
                "execution_mode",
                "execution_plan_json",
                "cpu_import_guard_enabled",
                "cpu_cuda_context_check",
                "loaded_gpu_modules_json",
            ],
            precision=12,
        ),
        "total_trials": len(trials),
        "display_limit": 2000,
        "comparisons": comparisons,
        "clustering_quality": clustering_quality_rows(trials) if pipeline else [],
        "cpu_execution_evidence": cpu_execution_evidence(trials) if pipeline else {},
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
        "execution_mode",
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
    processing = processing_metadata(repo.manifest)
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
        "processing": processing,
        "execution_mode": processing["execution_mode"],
        "execution_plan": processing["execution_plan"],
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
            "processing": processing,
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
