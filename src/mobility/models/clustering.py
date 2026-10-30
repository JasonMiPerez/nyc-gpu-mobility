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
    ML_EXCLUSION_RULES,
    HashSample,
    PopulationScaler,
    PrioritySample,
    canonical_batches,
    feature_matrix,
    in_period,
    initial_center_indices,
    ml_filter_parameters,
    ml_row_masks,
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


def _fit_model(matrix: np.ndarray, k: int, seed: int, backend: str, initial_centers=None):
    # Identical explicit starting centers make subsequent backend comparisons useful.
    centers = initial_centers
    if centers is None:
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
    native = config.get("processing", {}).get("execution_mode", "legacy") == "native"
    if native and backend == "gpu":
        from .gpu_clustering import fit_clusters_gpu

        return fit_clusters_gpu(canonical_dir, config, output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = config.get("clustering", {})
    filter_parameters = ml_filter_parameters(parameters)
    eligibility_policy = {
        "version": "fixed_trip_plausibility_v1",
        "thresholds": filter_parameters,
    }
    seed = int(parameters.get("seed", 42))
    bounds = split_bounds(config)
    batch_rows = int(config.get("processing", {}).get("batch_rows", 250_000))
    evaluation_limit = min(5000, int(parameters.get("evaluation_max_rows", 2000)))
    sampler = PrioritySample if native else HashSample

    def matrix_for(frame):
        return feature_matrix(frame, np.float64 if native else np.float32)

    samples = {
        "train": sampler(int(parameters.get("training_max_rows", 200_000)), seed),
        "calibration": sampler(evaluation_limit, seed),
        "test": sampler(evaluation_limit, seed),
    }
    total_rows = 0
    canonical_eligible_rows = 0
    eligible_rows = 0
    excluded_by_rule = dict.fromkeys(ML_EXCLUSION_RULES, 0)
    for batch in canonical_batches(canonical_dir, batch_rows):
        total_rows += len(batch)
        masks = ml_row_masks(batch, filter_parameters)
        canonical_eligible_rows += int(masks["canonical_eligible"].sum())
        for name in ML_EXCLUSION_RULES:
            excluded_by_rule[name] += int(masks[name].sum())
        eligible = batch.loc[masks["model_eligible"]].copy()
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
        "feature_version": "log_distance_duration_local_hour_float64_v2"
        if native
        else "log_distance_duration_local_hour_v1",
        "preprocessing_backend": "CPU NumPy float64 population scaling"
        if native
        else "CPU: numpy and sklearn StandardScaler",
        "initialization_policy": "seeded_source_row_priority_first_k_v1"
        if native
        else "sklearn_kmeans_plusplus",
        "execution_evidence": {
            name: {
                "backend": "cpu",
                "implementation": "NumPy/pandas/scikit-learn",
                "rows": eligible_rows,
            }
            for name in (
                "features",
                "sampling",
                "scaler_fit",
                "scaler_transform",
                "kmeans_fit",
                "kmeans_predict",
                "evaluation",
            )
        },
        "splits": {key: value.isoformat() for key, value in bounds.items()},
        "samples": {key: sample.metadata() for key, sample in samples.items()},
        "canonical_rows": total_rows,
        "canonical_eligible_rows": canonical_eligible_rows,
        "eligible_rows": eligible_rows,
        "eligible_fraction": eligible_rows / total_rows if total_rows else 0.0,
        "eligibility": {
            "policy_version": eligibility_policy["version"],
            "thresholds": filter_parameters,
            "canonical_rows": total_rows,
            "canonical_eligible_rows": canonical_eligible_rows,
            "model_eligible_rows": eligible_rows,
            "excluded_from_canonical_eligible_rows": canonical_eligible_rows - eligible_rows,
            "excluded_by_rule": excluded_by_rule,
            "model_eligible_fraction_of_canonical_rows": (
                eligible_rows / total_rows if total_rows else 0.0
            ),
            "model_eligible_fraction_of_canonical_eligible_rows": (
                eligible_rows / canonical_eligible_rows if canonical_eligible_rows else 0.0
            ),
            "exclusion_counting": (
                "Rule counts apply to canonical-eligible rows and can overlap; "
                "the total excluded count counts each trip once."
            ),
            "scope": (
                "Clustering only: training, calibration, test, predictions, and plotted trips. "
                "Raw records, canonical rows, and endpoint/OD aggregates are unchanged."
            ),
            "interpretation": (
                "Fixed study limits, inclusive at the boundary, applied before every model pass. "
                "These bounds are not learned from the data and are not universal validity rules. "
                "Mean speed is reported miles divided by elapsed hours, including stops. "
                "Zero-distance trips remain eligible when their other features pass."
            ),
        },
        "feature_explanation": {
            "method": "KMeans groups trips by similarity in four standardized features.",
            "distance": "log(1 + reported trip distance in miles)",
            "duration": "log(1 + elapsed trip duration in seconds)",
            "pickup_time": (
                "Sine and cosine of the local pickup hour; midnight and 11 p.m. are adjacent."
            ),
            "standardization": "Feature means and scales are fitted on the training sample only.",
            "not_features": ["pickup zone", "drop-off zone", "fare", "passenger identity"],
            "cluster_numbers": "Cluster numbers are arbitrary identifiers, not rankings.",
        },
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

    scaler = PopulationScaler() if native else StandardScaler()
    train_matrix = scaler.fit_transform(matrix_for(training)).astype(np.float32)
    calibration = samples["calibration"].result()
    calibration_matrix = (
        scaler.transform(matrix_for(calibration)).astype(np.float32)
        if len(calibration)
        else np.empty((0, 4), dtype=np.float32)
    )
    threads = int(config.get("processing", {}).get("cpu_threads", 4))
    candidates = []
    fitted = {}
    with threadpool_limits(limits=threads):
        for k in k_values:
            centers = (
                train_matrix[initial_center_indices(training, k, seed)].copy() if native else None
            )
            model, centers, elapsed = _fit_model(train_matrix, k, seed, backend, centers)
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
            scaler.transform(matrix_for(held_out)).astype(np.float32)
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
            centers = (
                train_matrix[initial_center_indices(training, chosen["k"], alternate_seed)].copy()
                if native
                else None
            )
            alternate, _, _ = _fit_model(
                train_matrix, chosen["k"], alternate_seed, backend, centers
            )
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
        "eligibility_policy": eligibility_policy,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "centers": np.asarray(model.cluster_centers_).tolist(),
        "initial_centers": initial_centers.tolist(),
        "initial_center_record_ids": training.iloc[
            initial_center_indices(training, chosen["k"], seed)
        ].record_id.tolist()
        if native
        else None,
        "k": chosen["k"],
        "seed": seed,
        "n_init": 1,
        "max_iter": 300,
        "tol": 1e-4,
        "training_sample_hash": samples["train"].metadata()["sample_ids_sha256"],
        "sampling_policy": samples["train"].metadata()["method"],
        "initialization_policy": base_metadata["initialization_policy"],
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

    plot_sample = sampler(min(5000, int(config.get("dashboard", {}).get("max_points", 5000))), seed)
    summaries = []
    prediction_rows = 0
    writer = None
    assignment_path = output_dir / (
        "trip_cluster_assignment" if native else "trip_cluster_assignment.parquet"
    )
    if native:
        assignment_path.mkdir()
    prediction_start = perf_counter()
    with threadpool_limits(limits=threads):
        try:
            for batch_index, batch in enumerate(canonical_batches(canonical_dir, batch_rows)):
                eligible = valid_ml_rows(batch, filter_parameters)
                if eligible.empty:
                    continue
                matrix = scaler.transform(matrix_for(eligible)).astype(np.float32)
                labels = np.asarray(model.predict(matrix)).astype(np.int32)
                _sync(backend)
                eligible["cluster"] = labels
                eligible["model_id"] = model_id
                assignments = eligible[["record_id", "model_id", "cluster"]].copy()
                centers = np.asarray(model.cluster_centers_)
                assignments["distance_to_center"] = np.linalg.norm(matrix - centers[labels], axis=1)
                arrow = pa.Table.from_pandas(assignments, preserve_index=False)
                if native:
                    pq.write_table(
                        arrow,
                        assignment_path / f"part-{batch_index:06d}.parquet",
                        compression="zstd",
                    )
                else:
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
    plotted["sample_design"] = "uniform deterministic hash sample of model-eligible trips"
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
        "summary_scope": (
            "All canonical-eligible rows that pass the fixed clustering study filters; "
            "static scoring-period summaries in original miles and minutes."
        ),
        "median_scope": (
            "Medians use the bounded model-eligible plot sample; "
            "counts and raw means use all model-eligible rows."
        ),
        "limitations": "Clusters are geometric groupings, not ground-truth passenger categories. Fit timings are diagnostics, not controlled benchmark repetitions.",
    }
    write_json(output_dir / "clustering.json", metadata)
    return {"tables": tables, "metadata": metadata}
