"""Sampled KMeans training, chronological selection, and batched full prediction."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import (
    FEATURE_NAMES,
    HashSample,
    canonical_batches,
    feature_matrix,
    in_period,
    split_bounds,
    stable_id,
    valid_ml_rows,
    write_json,
)


SUMMARY_COLUMNS = [
    "model_id",
    "dataset_type",
    "cluster",
    "trip_count",
    "share",
    "mean_distance_miles",
    "mean_duration_minutes",
    "scoring_start",
    "scoring_end",
    "sampled_median_distance_miles",
    "sampled_median_duration_minutes",
]
PLOT_COLUMNS = [
    "record_id",
    "dataset_type",
    "pickup_local",
    "pickup_zone_id",
    "dropoff_zone_id",
    "trip_distance_miles",
    "duration_seconds",
    "cluster",
    "model_id",
    "inclusion_fraction",
    "sample_design",
]


def _sync(backend: str) -> None:
    if backend == "gpu":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def _fit_model(matrix: np.ndarray, k: int, seed: int, backend: str):
    # Identical explicit starting centers make subsequent backend comparisons useful.
    centers, _ = kmeans_plusplus(matrix, n_clusters=k, random_state=seed)
    parameters = dict(
        n_clusters=k, init=centers, n_init=1, max_iter=300, tol=1e-4, random_state=seed
    )
    if backend == "gpu":
        try:
            from cuml.cluster import KMeans as GPUKMeans
        except ImportError as error:
            raise RuntimeError("GPU clustering requires cuML; no CPU fallback was used.") from error
        model = GPUKMeans(**parameters, output_type="numpy")
    elif backend == "cpu":
        model = KMeans(**parameters, algorithm="lloyd")
    else:
        raise ValueError("Clustering backend must be cpu or gpu.")
    _sync(backend)
    start = perf_counter()
    model.fit(matrix)
    _sync(backend)
    return model, centers, perf_counter() - start


def _evaluate(model, matrix: np.ndarray, backend: str) -> dict:
    if len(matrix) == 0:
        return {"rows": 0, "silhouette": None, "mean_squared_center_distance": None}
    labels = np.asarray(model.predict(matrix)).astype(np.int32)
    _sync(backend)
    centers = np.asarray(model.cluster_centers_)
    squared = np.square(matrix - centers[labels]).sum(axis=1)
    n_labels = len(np.unique(labels))
    silhouette = float(silhouette_score(matrix, labels)) if 1 < n_labels < len(matrix) else None
    counts = np.bincount(labels, minlength=len(centers))
    return {
        "rows": len(matrix),
        "silhouette": silhouette,
        "mean_squared_center_distance": float(squared.mean()),
        "cluster_counts": counts.tolist(),
    }


def fit_clusters(canonical_dir: Path, config: dict, output_dir: Path, backend: str = "cpu") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = config.get("clustering", {})
    seed = int(parameters.get("seed", 42))
    bounds = split_bounds(config)
    batch_rows = int(config.get("processing", {}).get("batch_rows", 250_000))
    evaluation_limit = min(5000, int(parameters.get("evaluation_max_rows", 2000)))
    samples = {
        "train": HashSample(int(parameters.get("training_max_rows", 200_000)), seed),
        "calibration": HashSample(evaluation_limit, seed),
        "test": HashSample(evaluation_limit, seed),
    }
    total_rows = 0
    eligible_rows = 0
    for batch in canonical_batches(canonical_dir, batch_rows):
        total_rows += len(batch)
        eligible = valid_ml_rows(batch)
        eligible_rows += len(eligible)
        for period, sample in samples.items():
            sample.add(eligible.loc[in_period(eligible["pickup_local"], bounds, period)])
    training = samples["train"].result()
    k_values = sorted(set(int(k) for k in parameters.get("k_values", [4, 6, 8])))
    k_values = [k for k in k_values if 2 <= k < len(training)]
    base_metadata = {
        "backend": backend,
        "feature_names": FEATURE_NAMES,
        "feature_dtype": "float32",
        "feature_version": "log_distance_duration_local_hour_v1",
        "preprocessing_backend": "CPU: numpy and sklearn StandardScaler",
        "splits": {key: value.isoformat() for key, value in bounds.items()},
        "samples": {key: sample.metadata() for key, sample in samples.items()},
        "canonical_rows": total_rows,
        "eligible_rows": eligible_rows,
        "eligible_fraction": eligible_rows / total_rows if total_rows else 0.0,
        "software": {name: version(name) for name in ["numpy", "pandas", "scikit-learn"]},
    }
    tables = {
        "cluster_summary": "cluster_summary.parquet",
        "trip_plot_sample": "trip_plot_sample.parquet",
    }
    if not k_values:
        for name, columns in [
            ("cluster_summary", SUMMARY_COLUMNS),
            ("trip_plot_sample", PLOT_COLUMNS),
        ]:
            pd.DataFrame(columns=columns).to_parquet(output_dir / tables[name], index=False)
        metadata = base_metadata | {"status": "insufficient_training_rows", "model_id": None}
        write_json(output_dir / "clustering.json", metadata)
        return {"tables": tables, "metadata": metadata}

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(feature_matrix(training)).astype(np.float32)
    calibration = samples["calibration"].result()
    calibration_matrix = (
        scaler.transform(feature_matrix(calibration)).astype(np.float32)
        if len(calibration)
        else np.empty((0, 4), dtype=np.float32)
    )
    threads = int(config.get("processing", {}).get("cpu_threads", 4))
    candidates = []
    fitted = {}
    with threadpool_limits(limits=threads):
        for k in k_values:
            model, centers, elapsed = _fit_model(train_matrix, k, seed, backend)
            assessment = _evaluate(model, calibration_matrix, backend)
            candidates.append(
                {
                    "k": k,
                    "training_rows": len(training),
                    "fit_seconds": elapsed,
                    "timing_scope": "fit including selected library input transfer; warm state unspecified",
                    "training_inertia_per_row": float(model.inertia_) / len(training),
                    "calibration": assessment,
                }
            )
            fitted[k] = (model, centers)
        selectable = [item for item in candidates if item["calibration"]["silhouette"] is not None]
        if selectable:
            chosen = max(
                selectable, key=lambda item: (item["calibration"]["silhouette"], -item["k"])
            )
            selection = "highest silhouette on one fixed calibration sample; smaller k breaks ties"
            status = "calibration_selected"
        else:
            chosen = candidates[0]
            selection = (
                "smallest feasible configured k; exploratory, not calibrated model selection"
            )
            status = "exploratory_no_calibration"
        model, initial_centers = fitted[chosen["k"]]
        del fitted
        held_out = samples["test"].result()
        test_matrix = (
            scaler.transform(feature_matrix(held_out)).astype(np.float32)
            if len(held_out)
            else np.empty((0, 4), dtype=np.float32)
        )
        test_assessment = _evaluate(model, test_matrix, backend)
        stability_matrix = (
            calibration_matrix if len(calibration_matrix) else train_matrix[:evaluation_limit]
        )
        stability_scope = (
            "calibration" if len(calibration_matrix) else "training exploratory sample"
        )
        original_labels = np.asarray(model.predict(stability_matrix)).astype(np.int32)
        stability = []
        for alternate_seed in (seed + 1, seed + 2):
            alternate, _, _ = _fit_model(train_matrix, chosen["k"], alternate_seed, backend)
            other_labels = np.asarray(alternate.predict(stability_matrix)).astype(np.int32)
            _sync(backend)
            stability.append(
                {
                    "seed": alternate_seed,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(original_labels, other_labels)
                    ),
                }
            )

    model_parameters = {
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "centers": np.asarray(model.cluster_centers_).tolist(),
        "initial_centers": initial_centers.tolist(),
        "k": chosen["k"],
        "seed": seed,
        "n_init": 1,
        "max_iter": 300,
        "tol": 1e-4,
        "training_sample_hash": samples["train"].metadata()["sample_ids_sha256"],
    }
    model_id = stable_id("kmeans", model_parameters)
    model_parameters["model_id"] = model_id
    write_json(output_dir / "cluster_parameters.json", model_parameters)
    # Preserve actual identities as well as the hash rule, without unsafe pickle loading.
    for period, sample in samples.items():
        selected = sample.result()
        if len(selected):
            selected[["record_id", "pickup_local"]].to_parquet(
                output_dir / f"cluster_{period}_sample_ids.parquet", index=False
            )

    plot_sample = HashSample(int(config.get("dashboard", {}).get("max_points", 5000)), seed)
    summaries = []
    prediction_rows = 0
    writer = None
    assignment_path = output_dir / "trip_cluster_assignment.parquet"
    prediction_start = perf_counter()
    with threadpool_limits(limits=threads):
        try:
            for batch in canonical_batches(canonical_dir, batch_rows):
                eligible = valid_ml_rows(batch)
                if eligible.empty:
                    continue
                matrix = scaler.transform(feature_matrix(eligible)).astype(np.float32)
                labels = np.asarray(model.predict(matrix)).astype(np.int32)
                _sync(backend)
                eligible["cluster"] = labels
                eligible["model_id"] = model_id
                assignments = eligible[["record_id", "model_id", "cluster"]].copy()
                centers = np.asarray(model.cluster_centers_)
                assignments["distance_to_center"] = np.linalg.norm(matrix - centers[labels], axis=1)
                arrow = pa.Table.from_pandas(assignments, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(assignment_path, arrow.schema, compression="zstd")
                writer.write_table(arrow)
                prediction_rows += len(eligible)
                plot_sample.add(eligible.drop(columns="eligible_ml"))
                summaries.append(
                    eligible.groupby(["dataset_type", "cluster"], observed=True)
                    .agg(
                        trip_count=("record_id", "size"),
                        distance_sum=("trip_distance_miles", "sum"),
                        duration_sum=("duration_seconds", "sum"),
                        scoring_start=("pickup_local", "min"),
                        scoring_end=("pickup_local", "max"),
                    )
                    .reset_index()
                )
        finally:
            if writer is not None:
                writer.close()
    prediction_seconds = perf_counter() - prediction_start
    plotted = plot_sample.result()
    plotted["inclusion_fraction"] = plot_sample.metadata()["inclusion_fraction"]
    plotted["sample_design"] = "uniform deterministic hash sample of eligible trips"
    plotted[PLOT_COLUMNS].to_parquet(output_dir / tables["trip_plot_sample"], index=False)
    summary = (
        pd.concat(summaries)
        .groupby(["dataset_type", "cluster"], observed=True)
        .agg(
            trip_count=("trip_count", "sum"),
            distance_sum=("distance_sum", "sum"),
            duration_sum=("duration_sum", "sum"),
            scoring_start=("scoring_start", "min"),
            scoring_end=("scoring_end", "max"),
        )
        .reset_index()
    )
    summary["model_id"] = model_id
    summary["share"] = summary["trip_count"] / summary.groupby("dataset_type")[
        "trip_count"
    ].transform("sum")
    summary["mean_distance_miles"] = summary["distance_sum"] / summary["trip_count"]
    summary["mean_duration_minutes"] = summary["duration_sum"] / summary["trip_count"] / 60
    medians = (
        plotted.groupby(["dataset_type", "cluster"], observed=True)
        .agg(
            sampled_median_distance_miles=("trip_distance_miles", "median"),
            sampled_median_duration_minutes=("duration_seconds", "median"),
        )
        .reset_index()
    )
    medians["sampled_median_duration_minutes"] /= 60
    summary = summary.merge(
        medians, on=["dataset_type", "cluster"], how="left", validate="one_to_one"
    )
    summary[SUMMARY_COLUMNS].to_parquet(output_dir / tables["cluster_summary"], index=False)
    tables["trip_cluster_assignment"] = assignment_path.name
    metadata = base_metadata | {
        "model_id": model_id,
        "status": status,
        "selected_k": chosen["k"],
        "selection_rule": selection,
        "candidates": candidates,
        "test": test_assessment,
        "test_status": "evaluated" if len(held_out) else "no_held_out_rows",
        "stability": {"scope": stability_scope, "trials": stability},
        "prediction_rows": prediction_rows,
        "prediction_seconds": prediction_seconds,
        "prediction_timing_scope": "canonical read, feature/scaler transforms, prediction, assignment write",
        "plot_sample": plot_sample.metadata(),
        "parameters_file": "cluster_parameters.json",
        "summary_scope": "All feature-eligible canonical rows; static scoring-period summaries.",
        "median_scope": "Medians use the bounded plot sample; counts and means use all eligible rows.",
        "limitations": "Clusters are geometric groupings, not ground-truth passenger categories. Fit timings are diagnostics, not controlled benchmark repetitions.",
    }
    write_json(output_dir / "clustering.json", metadata)
    return {"tables": tables, "metadata": metadata}
