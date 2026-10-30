"""Same-host feature experiments with synchronized timing and saved raw trials.

This harness measures one declared stage, not an invented full-pipeline speedup.
Pipeline acquisition, model fitting and publication remain separate measurements.
"""

from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import threading
import time
import uuid

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import psutil
from threadpoolctl import threadpool_limits, threadpool_info

from mobility.backends.features import inputs_from_pandas, reference_features


FEATURE_COLUMNS = [
    "pickup_local",
    "dropoff_local",
    "trip_distance_miles",
    "eligible_duration",
    "eligible_pickup",
    "eligible_dropoff",
]


def environment_record(config):
    packages = {}
    for package in ["numpy", "pandas", "pyarrow", "scikit-learn", "psutil", "threadpoolctl"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "host_memory_bytes": psutil.virtual_memory().total,
        "packages": packages,
        "threadpools": threadpool_info(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config,
    }


class MemorySampler:
    """Sample process RSS; the interval and short-peak limitation are reported."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()

    def _sample(self):
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def _bounded_rows(canonical_dir, limit):
    files = sorted(Path(canonical_dir).rglob("*.parquet"))
    if not files:
        raise ValueError("No canonical Parquet files are available to benchmark")
    # Sorted-file prefix is intentional and recorded, not a representative sample.
    batches, remaining = [], limit
    for path in files:
        dataset = ds.dataset(path, format="parquet")
        columns = FEATURE_COLUMNS + (
            ["dst_unsafe_duration"] if "dst_unsafe_duration" in dataset.schema.names else []
        )
        scanner = dataset.scanner(columns=columns, batch_size=min(250_000, limit))
        for batch in scanner.to_batches():
            used = min(len(batch), remaining)
            batches.append(batch.slice(0, used).to_pandas())
            remaining -= used
            if remaining == 0:
                return pd.concat(batches, ignore_index=True)
    return (
        pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=FEATURE_COLUMNS)
    )


def _validate(actual, expected, cp=None):
    for name in expected:
        result = cp.asnumpy(actual[name]) if cp is not None else actual[name]
        if expected[name].dtype.kind in "ui":
            np.testing.assert_array_equal(result, expected[name], err_msg=name)
        else:
            np.testing.assert_allclose(
                result, expected[name], rtol=1e-10, atol=1e-9, equal_nan=True, err_msg=name
            )


def summarize_trials(trials):
    """Never calculate speedup from failed or mismatched-scope measurements."""
    if trials.empty:
        return pd.DataFrame(
            columns=[
                "backend",
                "stage",
                "timing_scope",
                "input_rows",
                "median_s",
                "q25_s",
                "q75_s",
                "repetitions",
                "speedup_vs_cpu",
            ]
        )
    passed = trials.loc[(trials.validation == "passed") & trials.elapsed_s.notna()]
    keys = ["backend", "stage", "timing_scope", "input_rows"]
    if passed.empty:
        return pd.DataFrame(
            columns=[*keys, "median_s", "q25_s", "q75_s", "repetitions", "speedup_vs_cpu"]
        )
    summary = passed.groupby(keys, as_index=False).agg(
        median_s=("elapsed_s", "median"),
        q25_s=("elapsed_s", lambda values: values.quantile(0.25)),
        q75_s=("elapsed_s", lambda values: values.quantile(0.75)),
        repetitions=("elapsed_s", "count"),
    )
    cpu = summary.loc[summary.backend == "cpu", ["stage", "timing_scope", "input_rows", "median_s"]]
    cpu = cpu.rename(columns={"median_s": "cpu_median_s"})
    summary = summary.merge(
        cpu, on=["stage", "timing_scope", "input_rows"], how="left", validate="many_to_one"
    )
    summary["speedup_vs_cpu"] = summary.cpu_median_s / summary.median_s.where(summary.median_s > 0)
    return summary


def run_benchmarks(canonical_dir, output_dir, config, include_gpu=False):
    """Write raw Parquet trials, summary and environment JSON for real inputs.

    Default sizes are prefixes of up to 100k and 1m *available* rows. A request
    larger than the available data uses the actual size, without duplicating
    records. GPU absence/failure is saved and raised, never relabeled as CPU.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = config.get("benchmarks", config.get("benchmark", {}))
    maximum = int(options.get("max_rows", 1_000_000))
    sizes = sorted(set(int(n) for n in options.get("sizes", [min(100_000, maximum), maximum])))
    repeats, warmups = int(options.get("repeats", 7)), int(options.get("warmups", 2))
    threads, seed = int(options.get("cpu_threads", 4)), int(options.get("seed", 42))
    if not sizes or sizes[0] <= 0 or repeats < 1 or warmups < 0 or threads < 1:
        raise ValueError(
            "Benchmark sizes/threads/repeats must be positive; warmups must be nonnegative"
        )
    experiment = "features-" + uuid.uuid4().hex[:12]
    environment = environment_record(options)
    kernel_source = Path(__file__).parents[1] / "backends" / "kernels" / "fused_trip_features.cu"
    environment["custom_kernel_sha256"] = hashlib.sha256(kernel_source.read_bytes()).hexdigest()
    environment["custom_kernel_compile_options"] = ["--std=c++17"]
    environment.update(
        {
            "experiment_id": experiment,
            "requested_gpu": include_gpu,
            "input_selection": "prefix of lexicographically sorted canonical files; not a random sample",
            "precision": "float64",
            "cache_state": "not controlled; warm in-process trials",
            "scope_notes": {
                "loaded_compute": "Inputs already resident in host/device memory; output allocation included; GPU completion awaited.",
                "kernel_only": "CUDA events around the custom kernel on its current stream; inputs and outputs preallocated.",
                "host_to_host_features": "Host input arrays through feature operations to host output arrays; includes GPU transfers; excludes Parquet IO and schema conversion.",
            },
            "memory_notes": "Host RSS sampled every 10ms; short peaks can be missed. Device pool observations after trials are not peak VRAM.",
            "numeric_tolerance": {"rtol": 1e-10, "atol": 1e-9},
            "logical_bytes_per_row": 50,
        }
    )
    rows = _bounded_rows(canonical_dir, sizes[-1])
    sizes = sorted({min(size, len(rows)) for size in sizes})
    if not len(rows):
        raise ValueError("The canonical dataset has no records to benchmark")
    records, cp, gpu_error = [], None, None
    if include_gpu:
        try:
            from mobility.backends.cuda_features import (
                require_cupy,
                to_device,
                library_features,
                fused_features,
                allocate_outputs,
                prepare_fused_launch,
            )
            from mobility.backends.gpu import doctor

            cp = require_cupy()
            environment["gpu"] = doctor()
        except Exception as exc:
            gpu_error = f"{type(exc).__name__}: {exc}"
            environment["gpu_initialization_error"] = gpu_error
            records.append(
                {
                    "experiment_id": experiment,
                    "repeat": -1,
                    "backend": "gpu",
                    "stage": "initialization",
                    "timing_scope": "cold_start",
                    "input_rows": 0,
                    "elapsed_s": np.nan,
                    "validation": "unavailable",
                    "error": gpu_error,
                }
            )
            cp = None
    rng = random.Random(seed)
    with threadpool_limits(limits=threads):
        environment["effective_threadpools"] = threadpool_info()
        for size in sizes:
            inputs = inputs_from_pandas(rows.iloc[:size])
            digest = hashlib.sha256()
            for name, values in inputs.items():
                digest.update(name.encode())
                digest.update(values.tobytes())
            environment.setdefault("input_sha256_by_size", {})[str(size)] = digest.hexdigest()
            expected = reference_features(inputs)
            operations = {
                ("cpu", "loaded_compute"): lambda: reference_features(inputs),
                ("cpu", "host_to_host_features"): lambda: reference_features(inputs),
            }
            if cp is not None:
                # Compile and validate before steady-state timing. Initialization
                # is recorded separately, not hidden inside the first GPU repeat.
                initialization_start = time.perf_counter()
                try:
                    device_inputs = to_device(inputs)
                    outputs = allocate_outputs(size)
                    _validate(library_features(device_inputs), expected, cp)
                    _validate(fused_features(device_inputs, outputs), expected, cp)
                    kernel_launch = prepare_fused_launch(device_inputs, outputs)
                    cp.cuda.runtime.deviceSynchronize()
                    environment.setdefault("warm_initialization_s", {})[str(size)] = (
                        time.perf_counter() - initialization_start
                    )
                except Exception as exc:
                    gpu_error = f"{type(exc).__name__}: {exc}"
                    records.append(
                        {
                            "experiment_id": experiment,
                            "repeat": -1,
                            "backend": "gpu",
                            "stage": "validation",
                            "timing_scope": "loaded_compute",
                            "input_rows": size,
                            "elapsed_s": np.nan,
                            "validation": "failed",
                            "error": gpu_error,
                        }
                    )
                    cp = None
            if cp is not None:
                operations[("gpu_library", "loaded_compute")] = lambda: library_features(
                    device_inputs
                )
                operations[("gpu_custom", "loaded_compute")] = lambda: fused_features(device_inputs)
                operations[("gpu_custom", "kernel_only")] = kernel_launch

                def roundtrip(function):
                    result = function(to_device(inputs))
                    return {name: cp.asnumpy(values) for name, values in result.items()}

                operations[("gpu_library", "host_to_host_features")] = lambda: roundtrip(
                    library_features
                )
                operations[("gpu_custom", "host_to_host_features")] = lambda: roundtrip(
                    fused_features
                )
            for function in operations.values():
                for _ in range(warmups):
                    function()
                    if cp is not None:
                        cp.cuda.runtime.deviceSynchronize()
            for repeat in range(repeats):
                order = list(operations)
                rng.shuffle(order)
                for backend, scope in order:
                    function = operations[(backend, scope)]
                    record = {
                        "experiment_id": experiment,
                        "repeat": repeat,
                        "backend": backend,
                        "stage": "fused_trip_features",
                        "timing_scope": scope,
                        "input_rows": size,
                        "elapsed_s": np.nan,
                        "validation": "failed",
                        "error": None,
                        "cpu_threads": threads,
                        "rss_sample_interval_s": 0.01,
                        "host_device_transfer_bytes": size * 50
                        if backend.startswith("gpu") and scope == "host_to_host_features"
                        else 0,
                        "logical_io_bytes": size * 50,
                        "memory_measurement": "sampled RSS; post-trial GPU pool",
                    }
                    try:
                        if cp is not None:
                            cp.cuda.runtime.deviceSynchronize()
                        with MemorySampler() as memory:
                            if scope == "kernel_only":
                                start, end = cp.cuda.Event(), cp.cuda.Event()
                                start.record()
                                result = function()
                                end.record()
                                end.synchronize()
                                elapsed = cp.cuda.get_elapsed_time(start, end) / 1000
                            else:
                                started = time.perf_counter()
                                result = function()
                                if backend.startswith("gpu"):
                                    cp.cuda.runtime.deviceSynchronize()
                                elapsed = time.perf_counter() - started
                        is_device_result = (
                            backend.startswith("gpu") and scope != "host_to_host_features"
                        )
                        _validate(result, expected, cp if is_device_result else None)
                        record.update(
                            elapsed_s=elapsed,
                            validation="passed",
                            peak_host_rss_sampled_bytes=memory.peak,
                            rows_per_second=size / elapsed if elapsed else None,
                        )
                        if cp is not None and backend.startswith("gpu"):
                            record["device_pool_used_after_bytes"] = (
                                cp.get_default_memory_pool().used_bytes()
                            )
                            record["device_pool_reserved_after_bytes"] = (
                                cp.get_default_memory_pool().total_bytes()
                            )
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    records.append(record)
    if options.get("include_extended", True):
        # Retain completed raw trials if a later stage fails or is interrupted.
        pd.DataFrame(records).to_parquet(output_dir / "benchmark_runs.parquet", index=False)
        try:
            from mobility.benchmarks.aggregation import canonical_prefix, run_aggregation_benchmarks

            canonical = canonical_prefix(canonical_dir, max(sizes))
            records.extend(
                run_aggregation_benchmarks(
                    canonical, output_dir, config, cp is not None, experiment
                )
            )
            pd.DataFrame(records).to_parquet(output_dir / "benchmark_runs.parquet", index=False)
            from mobility.benchmarks.clustering import run_clustering_benchmarks

            records.extend(run_clustering_benchmarks(canonical, config, cp is not None, experiment))
        except Exception as exc:
            records.append(
                {
                    "experiment_id": experiment,
                    "repeat": -1,
                    "backend": "gpu" if include_gpu else "cpu",
                    "stage": "extended_stage_setup",
                    "timing_scope": "setup",
                    "input_rows": max(sizes),
                    "elapsed_s": np.nan,
                    "validation": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        environment["scope_notes"]["end_to_end_aggregation"] = (
            "Bounded canonical Parquet read through endpoint/OD aggregation and Parquet writes; "
            "required GPU transfers included; excludes raw normalization, models and publication."
        )
    trials = pd.DataFrame(records)
    trials.to_parquet(output_dir / "benchmark_runs.parquet", index=False)
    summary = summarize_trials(trials)
    summary.to_parquet(output_dir / "benchmark_summary.parquet", index=False)
    environment["actual_sizes"] = sizes
    environment["status"] = "passed" if (trials.validation == "passed").all() else "failed"
    (output_dir / "benchmark_environment.json").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )
    if gpu_error:
        raise RuntimeError(
            f"GPU benchmark unavailable; CPU trials and diagnostic saved to {output_dir}: {gpu_error}"
        )
    if environment["status"] != "passed":
        raise RuntimeError(
            f"Benchmark trial failed; inspect {output_dir / 'benchmark_runs.parquet'}"
        )
    return {
        "status": "passed",
        "tables": {
            "benchmark_runs": "benchmark_runs.parquet",
            "benchmark_summary": "benchmark_summary.parquet",
        },
        "experiment_id": experiment,
        "environment": "benchmark_environment.json",
        "actual_sizes": sizes,
        "gpu_measured": bool(include_gpu),
    }
