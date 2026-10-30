"""Controlled experiments using the same aggregation functions as the pipeline."""

from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from threadpoolctl import threadpool_limits

from mobility.aggregate import aggregate_cpu, ZONE_KEYS, OD_KEYS


def canonical_prefix(directory, limit):
    """Bound the input without repeating records to reach an advertised size."""
    batches, remaining = [], limit
    for path in sorted(Path(directory).rglob("*.parquet")):
        for batch in (
            ds.dataset(path, format="parquet").scanner(batch_size=min(limit, 250_000)).to_batches()
        ):
            used = min(remaining, len(batch))
            batches.append(batch.slice(0, used).to_pandas())
            remaining -= used
            if remaining == 0:
                return pd.concat(batches, ignore_index=True)
    if not batches:
        raise ValueError("No canonical records are available")
    return pd.concat(batches, ignore_index=True)


def validate_aggregates(actual, expected):
    """Counts are exact; only floating sums receive a numeric tolerance."""
    for result, reference, keys in zip(actual, expected, [ZONE_KEYS, OD_KEYS]):
        if hasattr(result, "to_pandas"):
            result = result.to_pandas()
        result = result.sort_values(keys).reset_index(drop=True)
        reference = reference.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(result[keys], reference[keys], check_dtype=False)
        for column in reference.columns.difference(keys):
            if column.endswith("_count"):
                np.testing.assert_array_equal(
                    result[column].to_numpy(), reference[column].to_numpy()
                )
            else:
                np.testing.assert_allclose(
                    result[column].to_numpy(),
                    reference[column].to_numpy(),
                    rtol=1e-9,
                    atol=1e-7,
                    equal_nan=True,
                )


def run_aggregation_benchmarks(frame, output_dir, config, include_gpu, experiment_id):
    from mobility.benchmarks.runner import MemorySampler

    options = config.get("benchmarks", config.get("benchmark", {}))
    maximum = int(options.get("max_rows", 1_000_000))
    requested = options.get("sizes", [min(100_000, maximum), maximum])
    sizes = sorted({min(int(size), len(frame)) for size in requested})
    warmups, repeats = int(options.get("warmups", 2)), int(options.get("repeats", 7))
    threads = int(options.get("cpu_threads", 4))
    rng = random.Random(int(options.get("seed", 42)))
    records = []
    if include_gpu:
        from mobility.backends import gpu

        cudf = gpu.require_cudf()
    with (
        tempfile.TemporaryDirectory(prefix="aggregation-experiment-", dir=output_dir) as scratch,
        threadpool_limits(limits=threads),
    ):
        scratch = Path(scratch)
        input_file = scratch / "canonical.parquet"
        for size in sizes:
            part = frame.iloc[:size].copy()
            reference = aggregate_cpu(part)
            part.to_parquet(input_file, index=False, compression="zstd")
            operations = {("cpu", "loaded_compute"): lambda: aggregate_cpu(part)}
            if include_gpu:
                device = gpu.to_device(part)
                validate_aggregates(gpu.aggregate(device), reference)
                operations[("gpu_library", "loaded_compute")] = lambda: gpu.aggregate(device)

            def file_pipeline(backend):
                if backend == "cpu":
                    result = aggregate_cpu(pd.read_parquet(input_file))
                else:
                    result = gpu.aggregate(cudf.read_parquet(input_file))
                    gpu.synchronize()
                    result = tuple(table.to_pandas() for table in result)
                paths = [scratch / f"{backend}-zone.parquet", scratch / f"{backend}-od.parquet"]
                for table, path in zip(result, paths):
                    table.to_parquet(path, index=False, compression="zstd")
                return paths

            operations[("cpu", "end_to_end_aggregation")] = lambda: file_pipeline("cpu")
            if include_gpu:
                operations[("gpu_library", "end_to_end_aggregation")] = lambda: file_pipeline(
                    "gpu_library"
                )
            for (backend, _), operation in operations.items():
                for _ in range(warmups):
                    operation()
                    if backend != "cpu":
                        gpu.synchronize()
            for repeat in range(repeats):
                order = list(operations)
                rng.shuffle(order)
                for backend, scope in order:
                    record = {
                        "experiment_id": experiment_id,
                        "repeat": repeat,
                        "backend": backend,
                        "stage": "zone_hour_and_od_groupby",
                        "timing_scope": scope,
                        "input_rows": size,
                        "elapsed_s": np.nan,
                        "validation": "failed",
                        "error": None,
                        "cpu_threads": threads,
                        "input_columns": len(part.columns),
                        "input_parquet_bytes": input_file.stat().st_size,
                        "cache_condition": "warm in-process; filesystem cache not controlled",
                        "scope_detail": (
                            "preloaded canonical rows through both endpoint and OD aggregates"
                            if scope == "loaded_compute"
                            else "bounded canonical Parquet read, both aggregates, required GPU transfers and Parquet writes; excludes models, raw normalization and publication"
                        ),
                    }
                    try:
                        if include_gpu:
                            gpu.synchronize()
                        with MemorySampler() as memory:
                            started = time.perf_counter()
                            result = operations[(backend, scope)]()
                            if backend != "cpu":
                                gpu.synchronize()
                            elapsed = time.perf_counter() - started
                        if scope == "end_to_end_aggregation":
                            result = tuple(pd.read_parquet(path) for path in result)
                        validate_aggregates(result, reference)
                        record.update(
                            validation="passed",
                            elapsed_s=elapsed,
                            zone_groups=len(result[0]),
                            od_groups=len(result[1]),
                            peak_host_rss_sampled_bytes=memory.peak,
                            rss_sample_interval_s=0.01,
                            rows_per_second=size / elapsed if elapsed else None,
                        )
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    records.append(record)
    return records
