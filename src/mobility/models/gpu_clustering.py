"""Device-native clustering with bounded host metadata and explicit GPU evidence.

cuDF reads and filters canonical rows; CuPy computes priorities, features and
training-only scaling; cuML fits, predicts and evaluates device arrays. Only
bounded sample identities and small final model statistics cross to the host.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

from .common import (
    FEATURE_NAMES,
    ML_EXCLUSION_RULES,
    TRIP_COLUMNS,
    ml_filter_parameters,
    split_bounds,
    stable_id,
    write_json,
)


SAMPLE_METHOD = "smallest_splitmix64_source_row_v1"
INITIALIZATION_METHOD = "seeded_source_row_priority_first_k_v1"
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
REQUIRED_COLUMNS = list(dict.fromkeys([*TRIP_COLUMNS, "source_token", "source_row"]))


def priority_values(source_token, source_row, seed: int, xp):
    """Identical unsigned-64 SplitMix priority arithmetic for NumPy and CuPy."""
    value = (
        xp.asarray(source_token, dtype=xp.uint64)
        ^ xp.asarray(source_row, dtype=xp.uint64)
        ^ xp.uint64(int(seed) & ((1 << 64) - 1))
    )
    value = value + xp.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)
    return value ^ (value >> xp.uint64(31))


def _libraries():
    try:
        import cupy as cp
        import cudf
        import cuml
        from cuml.cluster import KMeans
        from cuml.metrics.cluster import adjusted_rand_score, silhouette_score
    except ImportError as error:
        raise RuntimeError(
            "Native GPU clustering requires CuPy, cuDF and cuML; no CPU fallback was used."
        ) from error
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("Native GPU clustering requires an available CUDA device.")
    return cp, cudf, cuml, KMeans, silhouette_score, adjusted_rand_score


def _sync(cp):
    cp.cuda.get_current_stream().synchronize()


def _device_array(value, cp, name):
    """Reject accidental host outputs instead of silently copying them back to GPU."""
    if hasattr(value, "to_output"):
        value = value.to_output("cupy")
    if not isinstance(value, cp.ndarray):
        raise RuntimeError(
            f"{name} returned a host array; native GPU clustering requires device outputs."
        )
    return value


def canonical_gpu_batches(directory: Path, batch_rows: int, cudf):
    """Read bounded device tables; PyArrow is used only for file/schema metadata."""
    if batch_rows < 1:
        raise ValueError("Clustering batch_rows must be positive.")
    for path in sorted(Path(directory).rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(
                f"Native GPU clustering needs canonical provenance columns; {path.name} lacks {sorted(missing)}. Rebuild canonical data in native mode."
            )
        for offset in range(0, parquet.metadata.num_rows, batch_rows):
            yield cudf.read_parquet(
                str(path),
                columns=REQUIRED_COLUMNS,
                skip_rows=offset,
                nrows=min(batch_rows, parquet.metadata.num_rows - offset),
            )


def gpu_eligibility_masks(frame, parameters, cp):
    """Mirror the fixed CPU study policy using device arrays, including overlaps."""
    distance = frame["trip_distance_miles"].astype("float64").to_cupy(na_value=np.nan)
    duration = frame["duration_seconds"].astype("float64").to_cupy(na_value=np.nan)
    canonical = frame["eligible_ml"].fillna(False).astype("bool").to_cupy()
    has_time = frame["pickup_local"].notna().to_cupy()
    distance_finite, duration_finite = cp.isfinite(distance), cp.isfinite(duration)
    features_valid = has_time & distance_finite & (distance >= 0) & duration_finite & (duration > 0)
    mean_speed = distance / (cp.where(duration > 0, duration, 1.0) / 3600.0)
    masks = {
        "canonical_eligible": canonical,
        "invalid_features": canonical & ~features_valid,
        "distance_exceeds_limit": canonical
        & distance_finite
        & (distance > parameters["max_distance_miles"]),
        "duration_exceeds_limit": canonical
        & duration_finite
        & (duration > parameters["max_duration_minutes"] * 60),
        "mean_speed_exceeds_limit": canonical
        & distance_finite
        & duration_finite
        & (duration > 0)
        & (mean_speed > parameters["max_mean_speed_mph"]),
    }
    excluded = masks[ML_EXCLUSION_RULES[0]].copy()
    for name in ML_EXCLUSION_RULES[1:]:
        excluded |= masks[name]
    masks["model_eligible"] = canonical & ~excluded
    return masks


def gpu_feature_matrix(frame, cp):
    """Float64 numerical features stay on device until small model metadata is saved."""
    distance = frame["trip_distance_miles"].astype("float64").to_cupy()
    duration = frame["duration_seconds"].astype("float64").to_cupy()
    hour = frame["pickup_local"].dt.hour.astype("float64").to_cupy()
    return cp.ascontiguousarray(
        cp.column_stack(
            [
                cp.log1p(distance),
                cp.log1p(duration),
                cp.sin(2 * cp.pi * hour / 24),
                cp.cos(2 * cp.pi * hour / 24),
            ]
        ),
        dtype=cp.float64,
    )


def gpu_fit_scaler(features, cp):
    mean = features.mean(axis=0, dtype=cp.float64)
    variance = cp.square(features - mean).mean(axis=0, dtype=cp.float64)
    scale = cp.sqrt(variance)
    constant = (features.max(axis=0) == features.min(axis=0)) | (scale == 0)
    mean = cp.where(constant, features[0], mean)
    scale = cp.where(constant, 1.0, scale)
    return mean, scale


def gpu_scale(features, mean, scale, cp):
    return cp.ascontiguousarray((features - mean) / scale, dtype=cp.float32)


def _priority_order(frame, seed, cp):
    tokens = frame["source_token"].astype("uint64").to_cupy()
    rows = frame["source_row"].astype("uint64").to_cupy()
    priorities = priority_values(tokens, rows, seed, cp)
    return cp.lexsort(cp.stack((rows, tokens, priorities)))


class GpuPrioritySample:
    """Keep at most limit device rows, independent of file order or batch boundaries."""

    def __init__(self, limit, seed, cp, cudf):
        if limit < 1:
            raise ValueError("Sample limit must be positive.")
        self.limit, self.seed = int(limit), int(seed)
        self.cp, self.cudf = cp, cudf
        self.frame = None
        self.rows_seen = 0

    def add(self, frame):
        if len(frame) == 0:
            return
        self.rows_seen += len(frame)
        candidate = frame.copy(deep=False)
        if self.frame is not None:
            candidate = self.cudf.concat([self.frame, candidate], ignore_index=True)
        order = _priority_order(candidate, self.seed, self.cp)[: self.limit]
        self.frame = candidate.take(order).reset_index(drop=True)

    def result(self):
        return self.frame

    def metadata(self):
        # Only bounded identifiers cross to CPU for the existing audit hash contract.
        identities = (
            self.frame["record_id"].to_arrow().to_pylist() if self.frame is not None else []
        )
        encoded = "\n".join(map(str, identities)).encode()
        return {
            "method": SAMPLE_METHOD,
            "seed": self.seed,
            "eligible_rows": self.rows_seen,
            "sample_rows": len(identities),
            "sample_ids_sha256": hashlib.sha256(encoded).hexdigest(),
            "inclusion_fraction": len(identities) / self.rows_seen if self.rows_seen else 0.0,
        }


def gpu_initial_centers(training, matrix, k, seed, cp):
    order = _priority_order(training, seed, cp)[:k]
    return cp.ascontiguousarray(matrix[order], dtype=cp.float32)


def _fit_model(training, matrix, k, seed, cp, KMeans):
    centers = gpu_initial_centers(training, matrix, k, seed, cp)
    model = KMeans(
        n_clusters=k,
        init=centers.copy(),
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=seed,
        output_type="cupy",
    )
    _sync(cp)
    start = perf_counter()
    model.fit(matrix)
    _device_array(model.cluster_centers_, cp, "cuML cluster centers")
    _sync(cp)
    return model, centers, perf_counter() - start


def _evaluate(model, matrix, cp, silhouette_score):
    if len(matrix) == 0:
        return {"rows": 0, "silhouette": None, "mean_squared_center_distance": None}
    labels = _device_array(model.predict(matrix), cp, "cuML prediction").astype(cp.int32)
    centers = _device_array(model.cluster_centers_, cp, "cuML cluster centers")
    squared = cp.square(matrix - centers[labels]).sum(axis=1)
    label_count = len(cp.unique(labels))
    silhouette = (
        float(silhouette_score(matrix, labels, metric="euclidean"))
        if 1 < label_count < len(matrix)
        else None
    )
    counts = cp.bincount(labels, minlength=len(centers))
    _sync(cp)
    return {
        "rows": len(matrix),
        "silhouette": silhouette,
        "mean_squared_center_distance": float(squared.mean()),
        "cluster_counts": cp.asnumpy(counts).tolist(),
    }


def _empty_artifacts(output_dir, tables, cudf):
    strings = {"record_id", "model_id", "dataset_type", "sample_design"}
    timestamps = {"pickup_local", "scoring_start", "scoring_end"}
    integers = {"pickup_zone_id", "dropoff_zone_id", "cluster", "trip_count"}
    for table, columns in [
        ("cluster_summary", SUMMARY_COLUMNS),
        ("trip_plot_sample", PLOT_COLUMNS),
    ]:
        frame = cudf.DataFrame(
            {
                name: cudf.Series(
                    [],
                    dtype="str"
                    if name in strings
                    else "datetime64[ns]"
                    if name in timestamps
                    else "int64"
                    if name in integers
                    else "float64",
                )
                for name in columns
            }
        )
        frame.to_parquet(str(output_dir / tables[table]), index=False)


def _batch_summary(eligible):
    grouped = eligible.groupby(["dataset_type", "cluster"])
    summary = grouped.agg(
        {"record_id": "count", "trip_distance_miles": "sum", "duration_seconds": "sum"}
    )
    summary = summary.rename(
        columns={
            "record_id": "trip_count",
            "trip_distance_miles": "distance_sum",
            "duration_seconds": "duration_sum",
        }
    )
    summary["scoring_start"] = grouped["pickup_local"].min()
    summary["scoring_end"] = grouped["pickup_local"].max()
    return summary.reset_index()


def fit_clusters_gpu(canonical_dir: Path, config: dict, output_dir: Path) -> dict:
    cp, cudf, cuml, KMeans, silhouette_score, adjusted_rand_score = _libraries()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = config.get("clustering", {})
    filters = ml_filter_parameters(parameters)
    seed = int(parameters.get("seed", 42))
    bounds = split_bounds(config)
    batch_rows = int(config.get("processing", {}).get("batch_rows", 250_000))
    evaluation_limit = min(5000, int(parameters.get("evaluation_max_rows", 2000)))
    samples = {
        "train": GpuPrioritySample(
            int(parameters.get("training_max_rows", 200_000)), seed, cp, cudf
        ),
        "calibration": GpuPrioritySample(evaluation_limit, seed, cp, cudf),
        "test": GpuPrioritySample(evaluation_limit, seed, cp, cudf),
    }
    total_rows = canonical_eligible = eligible_rows = 0
    excluded_by_rule = dict.fromkeys(ML_EXCLUSION_RULES, 0)
    for batch in canonical_gpu_batches(canonical_dir, batch_rows, cudf):
        total_rows += len(batch)
        masks = gpu_eligibility_masks(batch, filters, cp)
        canonical_eligible += int(masks["canonical_eligible"].sum())
        for name in ML_EXCLUSION_RULES:
            excluded_by_rule[name] += int(masks[name].sum())
        eligible = batch.loc[cudf.Series(masks["model_eligible"], index=batch.index)]
        eligible_rows += len(eligible)
        for period, sample in samples.items():
            times = eligible["pickup_local"]
            mask = (times >= bounds[f"{period}_start"].to_datetime64()) & (
                times < bounds[f"{period}_end"].to_datetime64()
            )
            sample.add(eligible.loc[mask])
    sample_metadata = {name: sample.metadata() for name, sample in samples.items()}
    training = samples["train"].result()
    train_rows = 0 if training is None else len(training)
    k_values = sorted(
        {int(k) for k in parameters.get("k_values", [4, 6, 8]) if 2 <= int(k) < train_rows}
    )
    policy = {"version": "fixed_trip_plausibility_v1", "thresholds": filters}
    evidence = {
        "canonical_read": {
            "backend": "gpu",
            "implementation": "cuDF",
            "array_type": "cudf.DataFrame",
            "rows": total_rows,
            "batch_rows": batch_rows,
        },
        "sampling": {
            "backend": "gpu",
            "implementation": "CuPy/cuDF",
            "array_type": "cupy.ndarray uint64 / cudf.DataFrame",
            "rows": eligible_rows,
            "method": SAMPLE_METHOD,
        },
        "features": {
            "backend": "gpu",
            "implementation": "CuPy",
            "array_type": "cupy.ndarray",
            "dtype": "float64",
            "training_rows": train_rows,
        },
        "scaler_fit": {
            "backend": "gpu",
            "implementation": "CuPy",
            "array_type": "cupy.ndarray",
            "rows": train_rows,
            "dtype": "float64",
            "ddof": 0,
        },
        "scaler_transform": {
            "backend": "gpu",
            "implementation": "CuPy",
            "array_type": "cupy.ndarray",
            "output_dtype": "float32",
        },
        "kmeans_fit": {
            "backend": "gpu",
            "implementation": "cuML",
            "array_type": "cupy.ndarray",
            "output_type": "cupy",
            "training_rows": train_rows,
        },
        "kmeans_predict": {
            "backend": "gpu",
            "implementation": "cuML",
            "array_type": "cupy.ndarray",
            "output_type": "cupy",
            "rows": 0,
        },
        "evaluation": {
            "backend": "gpu",
            "implementation": "cuML/CuPy",
            "array_type": "cupy.ndarray",
            "metrics": ["silhouette", "adjusted_rand_index", "squared_center_distance"],
        },
    }
    base_metadata = {
        "backend": "gpu",
        "execution_mode": "native",
        "feature_names": FEATURE_NAMES,
        "feature_dtype": "float32",
        "feature_version": "log_distance_duration_local_hour_float64_v2",
        "preprocessing_backend": "GPU: cuDF/CuPy float64 features and training-only population scaling; float32 matrices",
        "initialization_policy": INITIALIZATION_METHOD,
        "host_transfer_scope": "Bounded sample record IDs for SHA256, scalar diagnostics, and small scaler/center arrays only; no per-trip pandas conversion.",
        "splits": {name: value.isoformat() for name, value in bounds.items()},
        "samples": sample_metadata,
        "canonical_rows": total_rows,
        "canonical_eligible_rows": canonical_eligible,
        "eligible_rows": eligible_rows,
        "eligible_fraction": eligible_rows / total_rows if total_rows else 0.0,
        "eligibility": {
            "policy_version": policy["version"],
            "thresholds": filters,
            "canonical_rows": total_rows,
            "canonical_eligible_rows": canonical_eligible,
            "model_eligible_rows": eligible_rows,
            "excluded_from_canonical_eligible_rows": canonical_eligible - eligible_rows,
            "excluded_by_rule": excluded_by_rule,
            "model_eligible_fraction_of_canonical_rows": eligible_rows / total_rows
            if total_rows
            else 0.0,
            "model_eligible_fraction_of_canonical_eligible_rows": eligible_rows / canonical_eligible
            if canonical_eligible
            else 0.0,
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
            "method": "KMeans groups trips using standardized log distance, log duration and cyclic pickup hour.",
            "distance": "log(1 + miles)",
            "duration": "log(1 + seconds)",
            "pickup_time": "Sine and cosine of local pickup hour; midnight and 11 p.m. are adjacent.",
            "standardization": "Population mean/std fitted on training sample only, in GPU float64; standardized values cast to float32.",
            "not_features": ["pickup zone", "drop-off zone", "fare", "passenger identity"],
            "cluster_numbers": "Arbitrary labels, not rankings.",
        },
        "software": {"cupy": cp.__version__, "cudf": cudf.__version__, "cuml": cuml.__version__},
        "execution_evidence": evidence,
    }
    tables = {
        "cluster_summary": "cluster_summary.parquet",
        "trip_plot_sample": "trip_plot_sample.parquet",
    }
    if not k_values:
        for stage in (
            "features",
            "scaler_fit",
            "scaler_transform",
            "kmeans_fit",
            "kmeans_predict",
            "evaluation",
        ):
            evidence[stage]["executed"] = False
        _empty_artifacts(output_dir, tables, cudf)
        metadata = base_metadata | {"status": "insufficient_training_rows", "model_id": None}
        write_json(output_dir / "clustering.json", metadata)
        return {"tables": tables, "metadata": metadata}

    train_features = gpu_feature_matrix(training, cp)
    mean, scale = gpu_fit_scaler(train_features, cp)
    train_matrix = gpu_scale(train_features, mean, scale, cp)
    del train_features

    def sample_matrix(period):
        frame = samples[period].result()
        return (
            gpu_scale(gpu_feature_matrix(frame, cp), mean, scale, cp)
            if frame is not None
            else cp.empty((0, len(FEATURE_NAMES)), dtype=cp.float32)
        )

    calibration_matrix = sample_matrix("calibration")
    candidates, fitted = [], {}
    for k in k_values:
        model, initial, elapsed = _fit_model(training, train_matrix, k, seed, cp, KMeans)
        candidates.append(
            {
                "k": k,
                "training_rows": train_rows,
                "fit_seconds": elapsed,
                "timing_scope": "cuML fit on resident device matrix, including completion; initialization centers prepared before timer",
                "training_inertia_per_row": float(model.inertia_) / train_rows,
                "calibration": _evaluate(model, calibration_matrix, cp, silhouette_score),
            }
        )
        fitted[k] = (model, initial)
    selectable = [
        candidate for candidate in candidates if candidate["calibration"]["silhouette"] is not None
    ]
    if selectable:
        chosen = max(
            selectable,
            key=lambda candidate: (candidate["calibration"]["silhouette"], -candidate["k"]),
        )
        status = "calibration_selected"
        selection = "Highest silhouette on one fixed calibration sample; smaller k breaks ties."
    else:
        chosen = candidates[0]
        status = "exploratory_no_calibration"
        selection = "Smallest feasible configured k; exploratory because calibration selection is unavailable."
    model, initial_centers = fitted[chosen["k"]]
    del fitted
    test_matrix = sample_matrix("test")
    test_assessment = _evaluate(model, test_matrix, cp, silhouette_score)
    stability_matrix = (
        calibration_matrix if len(calibration_matrix) else train_matrix[:evaluation_limit]
    )
    original_labels = _device_array(
        model.predict(stability_matrix), cp, "cuML stability prediction"
    ).astype(cp.int32)
    stability = []
    for alternate_seed in (seed + 1, seed + 2):
        alternate, _, _ = _fit_model(
            training, train_matrix, chosen["k"], alternate_seed, cp, KMeans
        )
        other_labels = _device_array(
            alternate.predict(stability_matrix), cp, "cuML stability prediction"
        ).astype(cp.int32)
        stability.append(
            {
                "seed": alternate_seed,
                "adjusted_rand_index": float(adjusted_rand_score(original_labels, other_labels)),
            }
        )
    _sync(cp)
    evidence["kmeans_fit"]["fit_count"] = len(k_values) + 2
    evidence["evaluation"].update(
        calibration_rows=len(calibration_matrix), test_rows=len(test_matrix)
    )
    model_parameters = {
        "feature_names": FEATURE_NAMES,
        "eligibility_policy": policy,
        "initialization_policy": INITIALIZATION_METHOD,
        "sampling_policy": SAMPLE_METHOD,
        "mean": cp.asnumpy(mean).tolist(),
        "scale": cp.asnumpy(scale).tolist(),
        "centers": cp.asnumpy(
            _device_array(model.cluster_centers_, cp, "cuML cluster centers")
        ).tolist(),
        "initial_centers": cp.asnumpy(initial_centers).tolist(),
        "initial_center_record_ids": training.take(
            _priority_order(training, seed, cp)[: chosen["k"]]
        )["record_id"]
        .to_arrow()
        .to_pylist(),
        "k": chosen["k"],
        "seed": seed,
        "n_init": 1,
        "max_iter": 300,
        "tol": 1e-4,
        "training_sample_hash": sample_metadata["train"]["sample_ids_sha256"],
    }
    model_id = stable_id("kmeans", model_parameters)
    model_parameters["model_id"] = model_id
    write_json(output_dir / "cluster_parameters.json", model_parameters)
    for period, sample in samples.items():
        if sample.result() is not None:
            sample.result()[["record_id", "pickup_local"]].to_parquet(
                str(output_dir / f"cluster_{period}_sample_ids.parquet"), index=False
            )

    plot_sample = GpuPrioritySample(
        min(5000, int(config.get("dashboard", {}).get("max_points", 5000))), seed, cp, cudf
    )
    assignment_dir = output_dir / "trip_cluster_assignment"
    assignment_dir.mkdir(exist_ok=False)
    summaries, prediction_rows = [], 0
    _sync(cp)
    prediction_start = perf_counter()
    for part, batch in enumerate(canonical_gpu_batches(canonical_dir, batch_rows, cudf)):
        valid = gpu_eligibility_masks(batch, filters, cp)["model_eligible"]
        eligible = batch.loc[cudf.Series(valid, index=batch.index)].reset_index(drop=True)
        if len(eligible) == 0:
            continue
        matrix = gpu_scale(gpu_feature_matrix(eligible, cp), mean, scale, cp)
        labels = _device_array(model.predict(matrix), cp, "cuML prediction").astype(cp.int32)
        eligible["cluster"] = cudf.Series(labels)
        eligible["model_id"] = model_id
        assignments = eligible[["record_id", "model_id", "cluster"]].copy(deep=False)
        assignments["distance_to_center"] = cudf.Series(
            cp.linalg.norm(
                matrix - _device_array(model.cluster_centers_, cp, "cuML cluster centers")[labels],
                axis=1,
            )
        )
        assignments.to_parquet(
            str(assignment_dir / f"part-{part:06d}.parquet"), index=False, compression="ZSTD"
        )
        prediction_rows += len(eligible)
        plot_sample.add(eligible)
        summaries.append(_batch_summary(eligible))
    _sync(cp)
    prediction_seconds = perf_counter() - prediction_start
    if prediction_rows != eligible_rows:
        raise ValueError("GPU prediction rows do not reconcile to the eligibility scan.")
    plot_metadata = plot_sample.metadata()
    plotted = plot_sample.result().copy(deep=False)
    plotted["inclusion_fraction"] = plot_metadata["inclusion_fraction"]
    plotted["sample_design"] = "deterministic SplitMix64 priority sample of model-eligible trips"
    plotted[PLOT_COLUMNS].to_parquet(str(output_dir / tables["trip_plot_sample"]), index=False)
    keys = ["dataset_type", "cluster"]
    summary = (
        cudf.concat(summaries, ignore_index=True)
        .groupby(keys)
        .agg(
            {
                "trip_count": "sum",
                "distance_sum": "sum",
                "duration_sum": "sum",
                "scoring_start": "min",
                "scoring_end": "max",
            }
        )
        .reset_index()
    )
    totals = (
        summary.groupby("dataset_type")["trip_count"]
        .sum()
        .reset_index()
        .rename(columns={"trip_count": "dataset_trip_count"})
    )
    summary = summary.merge(totals, on="dataset_type", how="left")
    summary["model_id"] = model_id
    summary["share"] = summary["trip_count"] / summary["dataset_trip_count"]
    summary["mean_distance_miles"] = summary["distance_sum"] / summary["trip_count"]
    summary["mean_duration_minutes"] = summary["duration_sum"] / summary["trip_count"] / 60
    medians = (
        plotted.groupby(keys)[["trip_distance_miles", "duration_seconds"]]
        .median()
        .reset_index()
        .rename(
            columns={
                "trip_distance_miles": "sampled_median_distance_miles",
                "duration_seconds": "sampled_median_duration_minutes",
            }
        )
    )
    medians["sampled_median_duration_minutes"] /= 60
    summary = summary.merge(medians, on=keys, how="left").sort_values(keys)
    summary[SUMMARY_COLUMNS].to_parquet(str(output_dir / tables["cluster_summary"]), index=False)
    tables["trip_cluster_assignment"] = assignment_dir.name
    evidence["features"]["prediction_rows"] = prediction_rows
    evidence["scaler_transform"].update(
        training_rows=train_rows,
        calibration_rows=len(calibration_matrix),
        test_rows=len(test_matrix),
        prediction_rows=prediction_rows,
    )
    evidence["kmeans_predict"]["rows"] = prediction_rows
    for stage in evidence.values():
        stage["executed"] = True
    metadata = base_metadata | {
        "model_id": model_id,
        "status": status,
        "selected_k": chosen["k"],
        "selection_rule": selection,
        "candidates": candidates,
        "test": test_assessment,
        "test_status": "evaluated" if len(test_matrix) else "no_held_out_rows",
        "stability": {
            "scope": "calibration" if len(calibration_matrix) else "training exploratory sample",
            "trials": stability,
        },
        "prediction_rows": prediction_rows,
        "prediction_seconds": prediction_seconds,
        "prediction_timing_scope": "cuDF canonical read, CuPy features/scaling, cuML prediction, device summaries and Parquet assignment writes, including GPU completion",
        "plot_sample": plot_metadata,
        "parameters_file": "cluster_parameters.json",
        "summary_scope": "All model-eligible canonical trips; counts and means computed on GPU in original units.",
        "median_scope": "GPU medians of the bounded plot sample; full-population means/counts use every eligible trip.",
        "limitations": "Clusters are geometric groupings, not ground-truth passenger categories. GPU/CPU fits can differ numerically. Fit diagnostics are not repeated performance experiments.",
    }
    write_json(output_dir / "clustering.json", metadata)
    return {"tables": tables, "metadata": metadata}
