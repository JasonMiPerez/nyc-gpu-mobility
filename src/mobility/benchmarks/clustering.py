"""Matched KMeans fit and frozen-center prediction benchmarks.

Preparation, scaling, sample selection, and GPU transfers happen before loaded
compute timing. Fit and prediction are distinct experiments. Numerical checks and
host conversion for validation happen after completion has been timed.
"""

from __future__ import annotations

import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mobility.models.common import (
    FEATURE_NAMES,
    HashSample,
    feature_matrix,
    in_period,
    split_bounds,
    valid_ml_rows,
)


# Declare tolerances before experiments, rather than changing them to pass a run.
FIT_OBJECTIVE_RTOL = 0.05
SELF_INERTIA_RTOL = 5e-4
PREDICTION_DISTANCE_RTOL = 1e-5
PREDICTION_DISTANCE_ATOL = 1e-5


def _prepare_inputs(frame: pd.DataFrame, config: dict):
    frame = frame.copy()
    frame["pickup_local"] = pd.to_datetime(frame["pickup_local"])
    eligible = valid_ml_rows(frame)
    bounds = split_bounds(config)
    parameters = config.get("clustering", {})
    options = config.get("benchmarks", config.get("benchmark", {}))
    seed = int(options.get("seed", 42))
    training = eligible.loc[in_period(eligible.pickup_local, bounds, "train")]
    sampler = HashSample(int(parameters.get("training_max_rows", 200_000)), seed)
    sampler.add(training)
    selected = sampler.result()
    k = min(int(value) for value in parameters.get("k_values", [4, 6, 8]))
    if len(selected) <= k:
        raise ValueError(
            "The benchmark prefix needs more eligible training-period rows than clusters; no future rows were used as a substitute."
        )
    scaler = StandardScaler()
    train = np.ascontiguousarray(scaler.fit_transform(feature_matrix(selected)), dtype=np.float32)
    prediction = np.ascontiguousarray(scaler.transform(feature_matrix(eligible)), dtype=np.float32)
    initial, _ = kmeans_plusplus(train, n_clusters=k, random_state=seed)
    initial = np.ascontiguousarray(initial, dtype=np.float32)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_dtype": "float32",
        "training_rows_available": len(training),
        "training_sample": sampler.metadata(),
        "prediction_rows": len(prediction),
        "prefix_rows": len(frame),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_matrix_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "prediction_matrix_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "initial_centers_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "k": k,
        "seed": seed,
        "n_init": 1,
        "max_iter": 300,
        "tol": 1e-4,
        "k_choice": "smallest configured candidate, fixed for the benchmark, not selected on the test period",
        "convergence_note": "The tolerance value is matched, but library stopping rules can differ. Report iteration counts, objective differences, and label-invariant agreement; do not assume identical convergence.",
        "fit_quality_relative_objective_tolerance": FIT_OBJECTIVE_RTOL,
        "self_inertia_relative_tolerance": SELF_INERTIA_RTOL,
        "prediction_distance_rtol": PREDICTION_DISTANCE_RTOL,
        "prediction_distance_atol": PREDICTION_DISTANCE_ATOL,
        "scope": "Already transformed host/device float32 matrices; estimator creation, output allocation, and completed library operations included. File IO, sampling, scaling, initial-center generation, explicit transfers, and correctness checks excluded.",
    }
    return train, prediction, initial, metadata


def _numpy(value, cp=None):
    # cuML 25.10 stores some fitted attributes as CumlArray internally, while its
    # output descriptor can expose NumPy/CuPy arrays outside estimator methods.
    if hasattr(value, "to_output"):
        value = value.to_output("numpy")
    return (
        cp.asnumpy(value) if cp is not None and isinstance(value, cp.ndarray) else np.asarray(value)
    )


def _prediction_quality(labels, matrix, centers, expected_labels) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != (len(matrix),) or np.any(labels < 0) or np.any(labels >= len(centers)):
        raise AssertionError("KMeans returned an invalid assignment shape or cluster index.")
    different = labels != expected_labels
    mismatch_count = int(different.sum())
    if mismatch_count:
        points = matrix[different].astype(np.float64)
        chosen_distance = np.square(points - centers[labels[different]].astype(np.float64)).sum(
            axis=1
        )
        reference_distance = np.square(
            points - centers[expected_labels[different]].astype(np.float64)
        ).sum(axis=1)
        allowance = PREDICTION_DISTANCE_ATOL + PREDICTION_DISTANCE_RTOL * reference_distance
        if np.any(chosen_distance > reference_distance + allowance):
            raise AssertionError(
                "Frozen-center prediction differs beyond the declared near-boundary tolerance."
            )
    return {
        "prediction_label_mismatches": mismatch_count,
        "prediction_near_boundary_mismatches": mismatch_count,
    }


def run_clustering_benchmarks(
    frame: pd.DataFrame, config: dict, include_gpu: bool, experiment_id: str
) -> list[dict]:
    """Return raw trials ready to append to the pipeline benchmark table."""
    options = config.get("benchmarks", config.get("benchmark", {}))
    repeats = int(options.get("repeats", 7))
    warmups = int(options.get("warmups", 2))
    threads = int(options.get("cpu_threads", 4))
    if repeats < 1 or warmups < 0 or threads < 1:
        raise ValueError("Benchmark repeats/threads must be positive and warmups nonnegative.")
    train, prediction, initial, metadata = _prepare_inputs(frame, config)
    k, seed = metadata["k"], metadata["seed"]
    parameters = dict(n_clusters=k, n_init=1, max_iter=300, tol=1e-4, random_state=seed)
    cp = None
    gpu_error = None
    gpu_class = None
    device_train = device_prediction = device_initial = None
    records = []

    def synchronize():
        if cp is not None and gpu_error is None:
            cp.cuda.runtime.deviceSynchronize()

    def cpu_fit():
        return KMeans(**parameters, init=initial.copy(), algorithm="lloyd").fit(train)

    def gpu_fit():
        return gpu_class(**parameters, init=device_initial.copy(), output_type="cupy").fit(
            device_train
        )

    with threadpool_limits(limits=threads):
        cpu_reference = cpu_fit()
        frozen_centers = np.ascontiguousarray(cpu_reference.cluster_centers_, dtype=np.float32)
        expected_prediction = cpu_reference.predict(prediction)
        reference_labels = cpu_reference.predict(train)
        reference_inertia = float(
            np.square(
                train.astype(np.float64) - frozen_centers[reference_labels].astype(np.float64)
            ).sum()
        )
        evaluation_count = min(5000, len(train))
        evaluation_indices = np.random.default_rng(seed).choice(
            len(train), evaluation_count, replace=False
        )
        metadata["frozen_centers_sha256"] = hashlib.sha256(frozen_centers.tobytes()).hexdigest()
        metadata["evaluation_rows"] = evaluation_count
        metadata["reference_inertia"] = reference_inertia
        metadata["prediction_center_rule"] = (
            "Both libraries predict with the identical frozen CPU-reference float32 centers, asserted byte-for-byte before timing. Fit comparisons train independent models from identical initial centers."
        )
        # A fitted estimator supplies all version-specific bookkeeping. Public
        # center arrays are then frozen explicitly; this setup is not timed.
        cpu_predictor = KMeans(**parameters, init=frozen_centers.copy(), algorithm="lloyd").fit(
            frozen_centers
        )
        cpu_predictor.cluster_centers_ = frozen_centers.copy()
        np.testing.assert_array_equal(cpu_predictor.cluster_centers_, frozen_centers)

        if include_gpu:
            try:
                import cupy as cp_module
                from cuml.cluster import KMeans as GPUKMeans
                from cuml.internals.array import CumlArray

                cp = cp_module
                gpu_class = GPUKMeans
                device_train = cp.asarray(train)
                device_prediction = cp.asarray(prediction)
                device_initial = cp.asarray(initial)
                device_frozen = cp.asarray(frozen_centers)
                gpu_predictor = gpu_class(
                    **parameters, init=device_frozen.copy(), output_type="cupy"
                ).fit(device_frozen)
                # The installed cuML implementation expects its internal array
                # wrapper (including .ptr), not a bare CuPy ndarray, when using
                # fitted centers inside predict. Verify this capability below.
                gpu_predictor.cluster_centers_ = CumlArray(device_frozen.copy())
                synchronize()
                np.testing.assert_array_equal(
                    _numpy(gpu_predictor.cluster_centers_, cp), frozen_centers
                )
                predicted = _numpy(gpu_predictor.predict(device_prediction), cp)
                _prediction_quality(predicted, prediction, frozen_centers, expected_prediction)
            except Exception as error:
                gpu_error = f"{type(error).__name__}: {error}"
                records.append(
                    {
                        "experiment_id": experiment_id,
                        "repeat": -1,
                        "backend": "gpu_library",
                        "stage": "kmeans_initialization",
                        "timing_scope": "loaded_compute",
                        "input_rows": len(train),
                        "elapsed_s": np.nan,
                        "validation": "failed",
                        "error": gpu_error,
                        "cpu_threads": threads,
                    }
                )

        operations = {
            ("cpu", "kmeans_fit"): cpu_fit,
            ("cpu", "kmeans_prediction"): lambda: cpu_predictor.predict(prediction),
        }
        if include_gpu and gpu_error is None:
            operations[("gpu_library", "kmeans_fit")] = gpu_fit
            operations[("gpu_library", "kmeans_prediction")] = lambda: gpu_predictor.predict(
                device_prediction
            )

        def assess_fit(model, backend):
            centers = _numpy(model.cluster_centers_, cp).astype(np.float32)
            labels = _numpy(model.labels_, cp).astype(np.int32)
            if centers.shape != (k, train.shape[1]) or not np.isfinite(centers).all():
                raise AssertionError("Invalid fitted KMeans centers.")
            if labels.shape != (len(train),) or np.any(labels < 0) or np.any(labels >= k):
                raise AssertionError("Invalid fitted KMeans assignments.")
            actual_inertia = float(
                np.square(train.astype(np.float64) - centers[labels].astype(np.float64)).sum()
            )
            np.testing.assert_allclose(
                float(model.inertia_), actual_inertia, rtol=SELF_INERTIA_RTOL, atol=1e-5
            )
            sampled = train[evaluation_indices].astype(np.float64)
            distances = np.square(sampled[:, None, :] - centers.astype(np.float64)[None, :, :]).sum(
                axis=2
            )
            assigned = distances[np.arange(evaluation_count), labels[evaluation_indices]]
            minimum = distances.min(axis=1)
            if np.any(
                assigned > minimum + PREDICTION_DISTANCE_ATOL + PREDICTION_DISTANCE_RTOL * minimum
            ):
                raise AssertionError("Fit assignments are not nearest to their returned centers.")
            relative_difference = abs(actual_inertia - reference_inertia) / max(
                reference_inertia, 1e-12
            )
            details = {
                "fit_objective": actual_inertia,
                "fit_reference_objective": reference_inertia,
                "fit_relative_objective_difference": relative_difference,
                "fit_adjusted_rand_index": float(
                    adjusted_rand_score(
                        reference_labels[evaluation_indices], labels[evaluation_indices]
                    )
                ),
                "fit_iterations": int(model.n_iter_),
                "fit_cluster_counts_json": json.dumps(np.bincount(labels, minlength=k).tolist()),
                "fit_quality_comparable": relative_difference <= FIT_OBJECTIVE_RTOL,
            }
            return details

        failed_operations = []
        for (backend, stage), function in operations.items():
            try:
                for _ in range(warmups):
                    result = function()
                    synchronize()
                    if stage == "kmeans_fit":
                        assess_fit(result, backend)
                    else:
                        _prediction_quality(
                            _numpy(result, cp), prediction, frozen_centers, expected_prediction
                        )
            except Exception as error:
                failed_operations.append((backend, stage))
                records.append(
                    {
                        "experiment_id": experiment_id,
                        "repeat": -1,
                        "backend": backend,
                        "stage": stage,
                        "timing_scope": "loaded_compute",
                        "input_rows": len(train) if stage == "kmeans_fit" else len(prediction),
                        "elapsed_s": np.nan,
                        "validation": "failed",
                        "error": f"Warmup failed: {type(error).__name__}: {error}",
                        "cpu_threads": threads,
                    }
                )
        for key in failed_operations:
            del operations[key]

        rng = random.Random(seed)
        metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
        for repeat in range(repeats):
            order = list(operations)
            rng.shuffle(order)
            for backend, stage in order:
                record = {
                    "experiment_id": experiment_id,
                    "repeat": repeat,
                    "backend": backend,
                    "stage": stage,
                    "timing_scope": "loaded_compute",
                    "input_rows": len(train) if stage == "kmeans_fit" else len(prediction),
                    "elapsed_s": np.nan,
                    "validation": "failed",
                    "error": None,
                    "cpu_threads": threads,
                    "feature_dimensions": train.shape[1],
                    "clusters": k,
                    "precision": "float32",
                    "model_parameters_json": metadata_json,
                    "host_device_transfer_bytes": 0,
                }
                try:
                    synchronize()
                    started = time.perf_counter()
                    result = operations[(backend, stage)]()
                    synchronize()
                    elapsed = time.perf_counter() - started
                    record["elapsed_s"] = elapsed
                    if stage == "kmeans_fit":
                        details = assess_fit(result, backend)
                        record.update(details)
                        if not details["fit_quality_comparable"]:
                            raise AssertionError(
                                "Fit objective differs from CPU reference beyond the predeclared 5% tolerance; timing is not a comparable speedup."
                            )
                    else:
                        record.update(
                            _prediction_quality(
                                _numpy(result, cp), prediction, frozen_centers, expected_prediction
                            )
                        )
                    record.update(
                        validation="passed",
                        rows_per_second=record["input_rows"] / elapsed if elapsed else None,
                    )
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                records.append(record)
    return records
