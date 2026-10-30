"""Isolated production-pipeline trials with sampled resources and external checks.

Acquisition and publication are deliberately absent. Internal reconciliation and
source checksum checks remain inside the measured production call. Comparisons
of output aggregates, model inputs, and artifact hashes happen after timing.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid


SCOPE = "raw_files_to_native_models_and_artifacts_external_reference_validation"
STAGE = "native_production_pipeline"
LEGACY_SCOPE = "raw_files_to_models_and_artifacts_with_reconciliation"
GPU_MODULES = {
    "cupy",
    "cupyx",
    "cudf",
    "pylibcudf",
    "cuml",
    "cugraph",
    "pylibcugraph",
    "rmm",
    "pylibraft",
}
BULK_STAGES = {
    "normalization",
    "eligibility",
    "features",
    "sampling",
    "scaler_fit",
    "scaler_transform",
    "aggregation",
    "kmeans_fit",
    "kmeans_predict",
    "duration_aggregation",
    "duration_scoring",
    "volume_scoring",
    "graph",
}
MARKER = ".pipeline-benchmark-owned"


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def protocol_hash(config):
    """Pair backends by their common inputs/settings, not their execution flags."""
    common = deepcopy(config)
    for key in ("backend", "custom_cuda"):
        common["processing"].pop(key, None)
    return hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()


def source_fingerprint(sources):
    identity = [
        {key: source[key] for key in ("source_id", "nominal_month", "sha256", "parquet_row_count")}
        for source in sources
    ]
    identity.sort(key=lambda source: source["nominal_month"])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def timing_identity(config):
    native = config["processing"].get("execution_mode", "native") == "native"
    return (STAGE, SCOPE) if native else ("production_pipeline", LEGACY_SCOPE)


def loaded_gpu_modules():
    return sorted(name for name in sys.modules if name.split(".", 1)[0] in GPU_MODULES)


class CPUOnlyImports:
    """Fail instead of allowing an optional GPU package to initialize in a CPU worker."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in GPU_MODULES:
            raise RuntimeError(f"CPU-only worker forbids CUDA package import: {fullname}")
        return None


def validate_execution_plan(manifest, backend):
    if manifest.get("execution_mode") != "native":
        raise ValueError("Native benchmark did not produce a native execution manifest")
    plan = manifest.get("execution_plan", {})
    for name in BULK_STAGES:
        stage = plan.get(name, {})
        if stage.get("backend") != backend or not stage.get("implementation"):
            raise ValueError(
                f"Native {backend} stage {name} has missing or incorrect backend evidence"
            )
        if not isinstance(stage.get("rows"), int) or stage["rows"] < 0:
            raise ValueError(f"Native stage {name} must declare its nonnegative row count")
    allowed = {backend, "host_io", "host_finalize"}
    if any(stage.get("backend") not in allowed for stage in plan.values()):
        raise ValueError("Execution plan contains an undeclared compute fallback")
    if backend == "gpu":
        graph = manifest.get("models", {}).get("graph", {})
        if graph.get("cpu_reference_elapsed_s") not in (None, 0):
            raise ValueError("Native GPU graph ran a CPU reference inside the timed pipeline")
        if manifest.get("gpu_batch_reference_checks"):
            raise ValueError("Native GPU benchmark ran timed per-batch CPU references")
    return plan


def trial_config(config, months, backend, threads):
    if backend not in {"cpu", "gpu"}:
        raise ValueError("Backend must be cpu or gpu; no automatic fallback is allowed")
    result = deepcopy(config)
    result["data"]["months"] = list(months)
    result["processing"].update(
        backend=backend,
        execution_mode=result["processing"].get("execution_mode", "native"),
        cpu_threads=threads,
        custom_cuda=backend == "gpu",
        verify_gpu_batches=False,
        allow_missing_sources=False,
    )
    result.setdefault("benchmark", {})["cpu_threads"] = threads
    return result


def _owned_directory(parent, name, token):
    path = Path(parent) / name
    path.mkdir(parents=True, exist_ok=False)
    (path / MARKER).write_text(token)
    return path


def remove_owned_directory(path, parent, token):
    """Never delete a run supplied by the user or follow a scratch-root symlink."""
    path, parent = Path(path), Path(parent).resolve()
    if path.is_symlink() or path.parent.resolve() != parent:
        raise ValueError("Refusing to remove a directory outside the owned scratch parent")
    marker = path / MARKER
    if not marker.is_file() or marker.read_text() != token:
        raise ValueError("Refusing to remove an unowned benchmark directory")
    shutil.rmtree(path)


class ResourceSampler:
    """Sample process-tree RSS and NVML memory; these are not exact high-water marks."""

    def __init__(self, pid, rss_interval=0.01, gpu_interval=0.02):
        import psutil

        self.psutil = psutil
        self.pid = pid
        self.rss_interval = rss_interval
        self.gpu_interval = gpu_interval
        self.stop = threading.Event()
        self.pids = {pid}
        self.rss_peak = self.rss_samples = self.rss_errors = 0
        self.rss_child_errors = 0
        self.rss_error_detail = None
        self.gpu_peak = self.gpu_samples = self.gpu_errors = 0
        self.gpu_memory_queries = self.gpu_unknown_memory = 0
        self.background = {}
        self.devices = {}
        self.nvml_error = None
        self.nvml = None
        self.last_rss = self.last_gpu = None
        self.rss_gap = self.gpu_gap = 0.0

    def _rss(self):
        while not self.stop.is_set():
            now = time.monotonic()
            try:
                root = self.psutil.Process(self.pid)
                processes = [root]
                try:
                    processes.extend(root.children(recursive=True))
                except (self.psutil.Error, OSError) as error:
                    self.rss_child_errors += 1
                    self.rss_error_detail = f"Descendant enumeration: {error}"
                self.pids = {process.pid for process in processes}
                rss = sum(process.memory_info().rss for process in processes)
                self.rss_peak = max(self.rss_peak, rss)
                self.rss_samples += 1
                if self.last_rss is not None:
                    self.rss_gap = max(self.rss_gap, now - self.last_rss)
                self.last_rss = now
            except (self.psutil.Error, OSError) as error:
                self.rss_errors += 1
                self.rss_error_detail = str(error)
            self.stop.wait(self.rss_interval)

    def _gpu(self):
        nvml = self.nvml
        try:
            handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]
        except nvml.NVMLError as error:
            self.nvml_error = str(error)
            return
        while not self.stop.is_set():
            now, total_owned = time.monotonic(), 0
            for index, handle in enumerate(handles):
                try:
                    identifier = str(nvml.nvmlDeviceGetUUID(handle))
                    device = self.devices.setdefault(
                        identifier,
                        {
                            "index": index,
                            "name": str(nvml.nvmlDeviceGetName(handle)),
                            "utilization_samples": 0,
                            "utilization_sum_percent": 0,
                            "utilization_peak_percent": 0,
                            "sampled_owned_peak_bytes": 0,
                        },
                    )
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    device["utilization_samples"] += 1
                    device["utilization_sum_percent"] += utilization
                    device["utilization_peak_percent"] = max(
                        device["utilization_peak_percent"], utilization
                    )
                    processes = {}
                    for getter in (
                        nvml.nvmlDeviceGetComputeRunningProcesses,
                        nvml.nvmlDeviceGetGraphicsRunningProcesses,
                    ):
                        try:
                            items = getter(handle)
                            self.gpu_memory_queries += 1
                            for item in items:
                                memory = item.usedGpuMemory
                                if not isinstance(memory, int) or memory < 0 or memory >= 2**60:
                                    memory = None
                                    self.gpu_unknown_memory += 1
                                previous = processes.get(item.pid)
                                processes[item.pid] = max(previous or 0, memory or 0)
                        except nvml.NVMLError:
                            self.gpu_errors += 1
                    owned = sum(memory for pid, memory in processes.items() if pid in self.pids)
                    device["sampled_owned_peak_bytes"] = max(
                        device["sampled_owned_peak_bytes"], owned
                    )
                    total_owned += owned
                    for pid, memory in processes.items():
                        if pid not in self.pids:
                            key = f"{identifier}:{pid}"
                            entry = self.background.setdefault(
                                key,
                                {
                                    "device_uuid": identifier,
                                    "pid": pid,
                                    "sample_count": 0,
                                    "sampled_peak_memory_bytes": 0,
                                },
                            )
                            entry["sample_count"] += 1
                            entry["sampled_peak_memory_bytes"] = max(
                                entry["sampled_peak_memory_bytes"], memory
                            )
                except nvml.NVMLError:
                    self.gpu_errors += 1
            self.gpu_peak = max(self.gpu_peak, total_owned)
            self.gpu_samples += 1
            if self.last_gpu is not None:
                self.gpu_gap = max(self.gpu_gap, now - self.last_gpu)
            self.last_gpu = now
            self.stop.wait(self.gpu_interval)

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
        except Exception as error:
            self.nvml_error = f"{type(error).__name__}: {error}"
        self.threads = [threading.Thread(target=self._rss, daemon=True)]
        if self.nvml is not None:
            self.threads.append(threading.Thread(target=self._gpu, daemon=True))
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        for thread in self.threads:
            thread.join()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()

    def result(self):
        for device in self.devices.values():
            n = device["utilization_samples"]
            device["system_gpu_utilization_mean_percent"] = (
                device.pop("utilization_sum_percent") / n if n else None
            )
        return {
            "scope": "Worker and descendants during the marked pipeline window; bootstrap excluded",
            "rss": {
                "sampled_peak_bytes": self.rss_peak if self.rss_samples else None,
                "samples": self.rss_samples,
                "requested_interval_s": self.rss_interval,
                "max_observed_gap_s": self.rss_gap,
                "sampling_errors": self.rss_errors,
                "descendant_enumeration_errors": self.rss_child_errors,
                "process_tree_complete": self.rss_child_errors == 0,
                "last_error": self.rss_error_detail,
            },
            "gpu": {
                "sampled_process_peak_bytes": self.gpu_peak if self.gpu_memory_queries else None,
                "samples": self.gpu_samples,
                "requested_interval_s": self.gpu_interval,
                "max_observed_gap_s": self.gpu_gap,
                "sampling_errors": self.gpu_errors,
                "successful_process_queries": self.gpu_memory_queries,
                "unavailable_process_memory_values": self.gpu_unknown_memory,
                "unavailable_reason": self.nvml_error,
                "devices": self.devices,
                "other_processes_observed": list(self.background.values()),
                "background_activity_observed": bool(self.background),
            },
            "limitations": [
                "Sampling can miss short peaks; no exact allocator or process high-water claim.",
                "Summed process RSS can double-count shared pages between related processes.",
                "GPU utilization is device-wide, not attributed to the worker.",
                "Observed other GPU processes flag possible contention, not its exact effect.",
                "Missing NVML process-memory values are omitted; partial coverage is a lower bound.",
                "OS scheduling and NVML overhead can make actual sample gaps exceed requests.",
            ],
        }


def _worker(request_path):
    request = _read(request_path)
    config = request["config"]
    backend = config["processing"]["backend"]
    if backend == "cpu":
        if loaded_gpu_modules():
            raise RuntimeError("CPU worker already has a CUDA package loaded")
        sys.meta_path.insert(0, CPUOnlyImports())
    threads = config["benchmark"]["cpu_threads"]
    # Parent also sets these before interpreter startup; Arrow has separate pools.
    import pyarrow as pa
    from threadpoolctl import threadpool_limits
    from mobility.pipeline import run_pipeline

    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    gpu = None
    if backend == "gpu":
        from mobility.backends import gpu

        gpu.require_cudf()
        gpu.synchronize()  # Strict preflight. Initialization is outside the stopwatch.
    print(json.dumps({"event": "ready", "pid": os.getpid()}), flush=True)
    if sys.stdin.readline().strip() != "go":
        raise RuntimeError("Missing benchmark start handshake")
    result = {"event": "done", "status": "failed"}
    with threadpool_limits(limits=threads):
        started = time.perf_counter()
        try:
            run = run_pipeline(Path(request["root"]), config, fit_models=True)
            if gpu is not None:
                gpu.synchronize()
            result.update(status="complete", run_dir=str(run))
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_s"] = time.perf_counter() - started
    # Auditing is after the stopwatch. No CPU reference algorithm runs in GPU timing.
    result["loaded_gpu_modules"] = loaded_gpu_modules()
    result["cpu_import_guard_enabled"] = backend == "cpu"
    if result["status"] == "complete":
        try:
            if backend == "cpu" and result["loaded_gpu_modules"]:
                raise ValueError("CPU worker imported CUDA packages")
            manifest = _read(Path(result["run_dir"]) / "manifest.json")
            if config["processing"].get("execution_mode", "native") == "native":
                result["execution_plan"] = validate_execution_plan(manifest, backend)
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


def _launch(root, config, trial_dir, project_root):
    request = trial_dir / "request.json"
    _write(request, {"root": str(root), "config": config})
    env = os.environ.copy()
    if config["processing"]["backend"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    threads = str(config["benchmark"]["cpu_threads"])
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = threads
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    with (trial_dir / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mobility.benchmarks.pipeline_scaling",
                "--worker",
                str(request),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=env,
            cwd=project_root,
        )
        try:
            ready = process.stdout.readline()
            if not ready or json.loads(ready).get("event") != "ready":
                process.wait()
                raise RuntimeError(f"Worker preflight failed; see {trial_dir / 'worker.log'}")
            with ResourceSampler(process.pid) as resources:
                process.stdin.write("go\n")
                process.stdin.flush()
                line = process.stdout.readline()
            result = json.loads(line) if line else {"status": "failed", "error": "Worker exited"}
            result["returncode"] = process.wait()
            _write(trial_dir / "resources.json", resources.result())
            _write(trial_dir / "worker_result.json", result)
            return result, _read(trial_dir / "resources.json")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _reference_files(run):
    tables = Path(run) / "tables"
    files = [
        tables / name
        for name in (
            "zone_hour.parquet",
            "quality_summary.parquet",
            "trip_histogram.parquet",
            "clustering.json",
            "cluster_parameters.json",
            "cluster_train_sample_ids.parquet",
            "cluster_calibration_sample_ids.parquet",
            "cluster_test_sample_ids.parquet",
            "anomaly_baseline.parquet",
            "anomaly_scores.parquet",
            "anomaly_calibration.parquet",
            "duration_hour.parquet",
            "duration_baseline.parquet",
            "duration_scores.parquet",
            "duration_calibration.parquet",
            "duration_incidents.parquet",
            "anomaly_incidents.parquet",
            "graph_nodes.parquet",
            "graph_edges.parquet",
            "graph_undirected_edges.parquet",
            "graph_analysis.json",
            "duration_anomalies.json",
            "anomalies.json",
        )
    ]
    return [path for path in files if path.exists()] + sorted(
        (tables / "od_hour").glob("*.parquet")
    )


def _save_reference(run, reference):
    for path in _reference_files(run):
        target = reference / path.relative_to(Path(run) / "tables")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def compare_numeric_table(
    actual, reference, keys, *, floating=(), ignored=(), rtol=1e-9, atol=1e-7
):
    """Keep counts, coverage, decisions and identifiers exact; tolerate only named numerics."""
    import numpy as np
    import pandas as pd

    left, right = pd.read_parquet(actual), pd.read_parquet(reference)
    if set(left.columns) != set(right.columns):
        raise ValueError(f"Table columns differ: {Path(actual).name}")
    left, right = [frame.sort_values(keys).reset_index(drop=True) for frame in (left, right)]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Duplicate comparison keys: {Path(actual).name}")
    exact = right.columns.difference([*floating, *ignored])
    pd.testing.assert_frame_equal(left[exact], right[exact], check_dtype=False, check_exact=True)
    for name in set(floating) & set(right):
        np.testing.assert_allclose(
            left[name].to_numpy(dtype=float, na_value=np.nan),
            right[name].to_numpy(dtype=float, na_value=np.nan),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"{Path(actual).name}:{name}",
        )


def compare_graph(actual, reference):
    """Independent CPU graph checks after timing, including each returned partition's quality."""
    import networkx as nx
    import numpy as np
    import pandas as pd

    required = [
        "graph_nodes.parquet",
        "graph_edges.parquet",
        "graph_undirected_edges.parquet",
        "graph_analysis.json",
    ]
    if not any((actual / name).exists() or (reference / name).exists() for name in required):
        return {"status": "not_requested"}
    if not all(
        (directory / name).exists() for directory in (actual, reference) for name in required
    ):
        raise ValueError("Graph comparison artifacts are incomplete")
    edges = pd.read_parquet(reference / "graph_edges.parquet")
    undirected = pd.read_parquet(reference / "graph_undirected_edges.parquet")
    nodes = pd.read_parquet(reference / "graph_nodes.parquet").sort_values("zone_id")
    compare_numeric_table(
        actual / required[1], reference / required[1], ["pickup_zone_id", "dropoff_zone_id"]
    )
    compare_numeric_table(actual / required[2], reference / required[2], ["zone_a", "zone_b"])
    compare_numeric_table(
        actual / required[0],
        reference / required[0],
        ["zone_id"],
        ignored=["pagerank", "community"],
    )
    options = _read(reference / "graph_analysis.json")["parameters"]
    directed, projection = nx.DiGraph(), nx.Graph()
    vertices = nodes.zone_id.tolist()
    directed.add_nodes_from(vertices)
    projection.add_nodes_from(vertices)
    directed.add_weighted_edges_from(
        edges[["pickup_zone_id", "dropoff_zone_id", "trip_count"]].itertuples(
            index=False, name=None
        )
    )
    projection.add_weighted_edges_from(
        undirected[["zone_a", "zone_b", "trip_count"]].itertuples(index=False, name=None)
    )
    expected = (
        nx.pagerank(
            directed,
            alpha=options["pagerank_alpha"],
            tol=options["pagerank_tol"] / max(1, len(vertices)),
            max_iter=options["pagerank_max_iter"],
            weight="weight",
            backend="networkx",
        )
        if vertices
        else {}
    )
    diagnostics = {}
    for name, directory in (("actual", actual), ("reference", reference)):
        table = pd.read_parquet(directory / "graph_nodes.parquet").sort_values("zone_id")
        metadata = _read(directory / "graph_analysis.json")
        if metadata["parameters"] != options or table.zone_id.tolist() != vertices:
            raise ValueError("Graph vertex set or algorithm parameters differ")
        values = table.pagerank.to_numpy(dtype=float)
        target = np.array([expected[vertex] for vertex in vertices])
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (len(values) and not np.isclose(values.sum(), 1, atol=1e-6, rtol=0))
        ):
            raise ValueError("Graph PageRank mass or values are invalid")
        np.testing.assert_allclose(
            values, target, rtol=options["pagerank_rtol"], atol=options["pagerank_atol"]
        )
        if table.community.isna().any():
            raise ValueError("Every graph vertex requires a community label")
        groups = [set(group.zone_id) for _, group in table.groupby("community")]
        for vertex in table.loc[table.is_isolated, "zone_id"]:
            if next(group for group in groups if vertex in group) != {vertex}:
                raise ValueError("Isolated graph vertices require singleton community labels")
        modularity = (
            nx.community.modularity(
                projection,
                groups,
                weight="weight",
                resolution=options["louvain_resolution"],
                backend="networkx",
            )
            if len(undirected)
            else None
        )
        reported = metadata.get("modularity")
        if (modularity is None) != (reported is None) or (
            modularity is not None and not np.isclose(modularity, reported, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError("Graph modularity differs from an independent CPU recomputation")
        diagnostics[name] = {
            "pagerank_max_abs_error": float(np.max(np.abs(values - target)))
            if len(values)
            else 0.0,
            "modularity": modularity,
            "community_count": len(groups),
        }
    return {
        "status": "passed",
        "reference": "Independent post-timing NetworkX computation",
        **diagnostics,
    }


def compare_outputs(run, reference):
    """Exact keys/counts; floating reductions have a declared numeric tolerance."""
    import numpy as np
    import pandas as pd
    from mobility.aggregate import OD_KEYS, ZONE_KEYS

    actual = Path(run) / "tables"
    files = [
        Path("zone_hour.parquet"),
        Path("quality_summary.parquet"),
        Path("trip_histogram.parquet"),
    ]
    files += [
        path.relative_to(reference) for path in sorted((reference / "od_hour").glob("*.parquet"))
    ]
    if sorted(path.name for path in (actual / "od_hour").glob("*.parquet")) != sorted(
        path.name for path in (reference / "od_hour").glob("*.parquet")
    ):
        raise ValueError("OD partitions differ across trials")
    compared = []
    for relative in files:
        left, right = pd.read_parquet(actual / relative), pd.read_parquet(reference / relative)
        if relative.name == "zone_hour.parquet":
            keys = ZONE_KEYS
        elif relative.parent.name == "od_hour":
            keys = OD_KEYS
        else:
            keys = sorted(set(right.columns) - {"count", "affected_rows", "total_rows"})
        left = left.sort_values(keys).reset_index(drop=True)
        right = right.sort_values(keys).reset_index(drop=True)
        if set(left.columns) != set(right.columns):
            raise ValueError(f"Aggregate columns differ: {relative}")
        left = left[right.columns]
        floating = [name for name in right if "sum_" in name or name.endswith("_sum")]
        exact = right.columns.difference(floating)
        pd.testing.assert_frame_equal(
            left[exact], right[exact], check_dtype=False, check_exact=True
        )
        for name in floating:
            np.testing.assert_allclose(
                left[name].to_numpy(dtype=float, na_value=np.nan),
                right[name].to_numpy(dtype=float, na_value=np.nan),
                rtol=1e-9,
                atol=1e-7,
                equal_nan=True,
            )
        compared.append(str(relative))
    first, second = _read(actual / "clustering.json"), _read(reference / "clustering.json")
    for name in ("eligible_rows", "canonical_rows", "prediction_rows", "samples", "eligibility"):
        if first.get(name) != second.get(name):
            raise ValueError(f"Clustering inputs differ: {name}")
    parameter_paths = [actual / "cluster_parameters.json", reference / "cluster_parameters.json"]
    if all(path.exists() for path in parameter_paths):
        a, b = map(_read, parameter_paths)
        for name in (
            "training_sample_hash",
            "feature_names",
            "initial_center_policy",
            "initialization_policy",
            "sampling_policy",
        ):
            if a.get(name) != b.get(name):
                raise ValueError(f"Clustering preprocessing differs: {name}")
        for name in ("mean", "scale"):
            np.testing.assert_allclose(a[name], b[name], rtol=1e-12, atol=1e-12, err_msg=name)
        ids_a, ids_b = a.get("initial_center_record_ids"), b.get("initial_center_record_ids")
        if (ids_a is None) != (ids_b is None):
            raise ValueError("Initial-center identity evidence differs")
        if ids_a is not None:
            common = min(len(ids_a), len(ids_b))
            if len(ids_a) != a["k"] or len(ids_b) != b["k"] or ids_a[:common] != ids_b[:common]:
                raise ValueError("Initial-center source identities differ")
        if (
            "initial_centers" in a
            and "initial_centers" in b
            and (a.get("k") == b.get("k") or ids_a is not None)
        ):
            common = min(len(a["initial_centers"]), len(b["initial_centers"]))
            np.testing.assert_allclose(
                a["initial_centers"][:common],
                b["initial_centers"][:common],
                rtol=1e-6,
                atol=1e-6,
                err_msg="float32 initial centers",
            )
    hour_keys = ["dataset_type", "zone_id", "hour"]
    baseline_keys = ["dataset_type", "zone_id", "hour_of_week"]
    detector_floats = [
        "observed",
        "baseline",
        "mad",
        "scale",
        "score",
        "deviation",
        "threshold",
        "absolute_threshold",
    ]
    specifications = {
        "anomaly_baseline.parquet": (baseline_keys, ["baseline", "mad", "scale"]),
        "duration_baseline.parquet": (baseline_keys, ["baseline", "mad", "scale"]),
        "duration_hour.parquet": (hour_keys, ["duration_sum_minutes", "mean_duration_minutes"]),
        "anomaly_scores.parquet": (hour_keys, detector_floats),
        "duration_scores.parquet": (hour_keys, detector_floats),
        "anomaly_calibration.parquet": (["threshold"], ["alert_fraction"]),
        "duration_calibration.parquet": (["threshold"], ["alert_fraction"]),
        "anomaly_incidents.parquet": (
            ["dataset_type", "zone_id", "start_hour", "direction", "split"],
            ["peak_score", "total_deviation"],
        ),
        "duration_incidents.parquet": (
            ["dataset_type", "zone_id", "start_hour", "direction", "split"],
            ["peak_score", "total_deviation"],
        ),
    }
    for period in ("train", "calibration", "test"):
        specifications[f"cluster_{period}_sample_ids.parquet"] = (["record_id"], [])
    for filename, (keys, floating) in specifications.items():
        present = [(directory / filename).exists() for directory in (actual, reference)]
        if present[0] != present[1]:
            raise ValueError(f"Comparison artifact availability differs: {filename}")
        if all(present):
            compare_numeric_table(
                actual / filename,
                reference / filename,
                keys,
                floating=floating,
                ignored=["model_id"],
            )
            compared.append(filename)
    graph_validation = compare_graph(actual, reference)
    return {
        "status": "passed",
        "aggregate_files": compared,
        "exact": "keys, integer counts, coverage, quality rules, histograms, model input samples",
        "floating_sum_tolerance": {"rtol": 1e-9, "atol": 1e-7},
        "scaler_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "graph_validation": graph_validation,
        "same_selected_k": first.get("selected_k") == second.get("selected_k"),
        "model_comparison": "Report quality separately; cluster labels and fitted centers need not match",
    }


def _run_details(run, trial_dir):
    from mobility.io import sha256_file

    manifest = _read(run / "manifest.json")
    _write(trial_dir / "run_manifest.json", manifest)
    hashes = {str(path.relative_to(run)): sha256_file(path) for path in _reference_files(run)}
    hashes["manifest.json"] = sha256_file(run / "manifest.json")
    _write(trial_dir / "artifact_checksums.json", hashes)
    cluster = manifest.get("models", {}).get("clustering", {})
    if cluster.get("model_id") and cluster.get("prediction_rows") != cluster.get("eligible_rows"):
        raise ValueError("Cluster prediction rows did not reconcile with eligible rows")
    for assessment in [cluster.get("test", {}), *cluster.get("candidates", [])]:
        silhouette = assessment.get("silhouette")
        if silhouette is not None and not -1 <= silhouette <= 1:
            raise ValueError("Invalid clustering quality metric")
    return manifest


def run_pipeline_scaling(
    root,
    output_dir,
    config,
    *,
    month_counts=(1, 3, 6),
    backends=("cpu", "gpu"),
    repeats=3,
    warmups=1,
    cpu_threads=4,
    keep_runs=False,
    seed=42,
):
    """Benchmark full chronological prefixes; retain failed states without substitution."""
    import pandas as pd
    from mobility.benchmarks.runner import environment_record, summarize_trials
    from mobility.pipeline import source_list

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    stage, scope = timing_identity(config)
    native = stage == STAGE
    counts = sorted(set(month_counts))
    available = sorted(config["data"]["months"])
    if (
        not counts
        or counts[0] < 1
        or counts[-1] > len(available)
        or repeats < 1
        or warmups < 0
        or cpu_threads < 1
        or not backends
        or len(set(backends)) != len(backends)
        or set(backends) - {"cpu", "gpu"}
    ):
        raise ValueError("Invalid month prefixes, backends, repetition counts, or thread count")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty output directory to preserve existing experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = ("native-pipeline-" if native else "pipeline-") + uuid.uuid4().hex[:12]
    scratch = _owned_directory(output_dir, "scratch", experiment)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir()
    preflight = trial_config(config, available[: counts[-1]], "cpu", cpu_threads)
    try:
        sources = source_list(root / "data", preflight)
    except Exception as error:
        _write(
            output_dir / "benchmark_environment.json",
            {
                "experiment_id": experiment,
                "status": "preflight_failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        remove_owned_directory(scratch, output_dir, experiment)
        raise
    environment = environment_record(config)
    environment.update(
        {
            "experiment_id": experiment,
            "status": "running",
            "requested_month_counts": counts,
            "backends": list(backends),
            "repeats": repeats,
            "warmups": warmups,
            "cpu_threads": cpu_threads,
            "seed": seed,
            "keep_runs": keep_runs,
            "sources": [{k: v for k, v in item.items() if k != "local_path"} for item in sources],
            "stage": stage,
            "execution_mode": "native" if native else "legacy",
            "timing_scope": scope,
            "scope_notes": "Raw-file checksums/read/normalization, features, aggregates, model fitting/evaluation/prediction, artifact writes and built-in reconciliation. No acquisition or publication. Worker startup/bootstrap imports, GPU dependency preflight, cross-trial comparisons and artifact hashing excluded. Lazy imports, initialization and compilation invoked inside the production pipeline remain timed.",
            "gpu_validation_mode": "CPU references for aggregates, detector outputs, model inputs and graphs run after timing"
            if native
            else "Legacy hybrid pipeline; per-batch aggregation reference disabled, internal graph reference retained",
            "cpu_execution_contract": "CUDA packages blocked before pipeline imports; CUDA_VISIBLE_DEVICES empty; end-of-run module audit and sampled process GPU memory recorded",
            "cache_state": "Fresh process each trial; filesystem and CUDA compilation caches uncontrolled. Warmups are full discarded runs, not persistent-process warming.",
            "model_notes": "Identical configuration, chronological samples and preprocessing; selected cluster counts and fitted model quality reported, not assumed identical.",
        }
    )
    _write(output_dir / "benchmark_environment.json", environment)
    rng, records, failures = random.Random(seed), [], []

    def persist():
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame.to_parquet(output_dir / "benchmark_runs.parquet", index=False)
            summarize_trials(frame).to_parquet(
                output_dir / "benchmark_summary.parquet", index=False
            )
        _write(
            output_dir / "resource_measurements.json",
            [
                {"trial_id": row["trial_id"], "resources": row.get("resource_file")}
                for row in records
            ],
        )

    try:
        for count in counts:
            months = available[:count]
            selected_sources = [
                source for source in sources if int(source["nominal_month"][-2:]) in months
            ]
            reference = _owned_directory(scratch, f"reference-{count}", experiment)
            reference_backend = None
            size_records = []
            for phase, n in (("warmup", warmups), ("measured", repeats)):
                for repeat in range(n):
                    order = list(backends)
                    rng.shuffle(order)
                    for backend in order:
                        trial_id = f"months-{count}-{phase}-{repeat}-{backend}"
                        trial_dir = trials_dir / trial_id
                        trial_dir.mkdir()
                        trial_root = _owned_directory(scratch, trial_id, experiment)
                        (trial_root / "data").symlink_to(root / "data", target_is_directory=True)
                        options = trial_config(config, months, backend, cpu_threads)
                        row = {
                            "experiment_id": experiment,
                            "trial_id": trial_id,
                            "repeat": repeat,
                            "phase": phase,
                            "backend": backend,
                            "stage": stage,
                            "timing_scope": scope,
                            "execution_mode": "native" if native else "legacy",
                            "months": count,
                            "input_rows": sum(s["parquet_row_count"] for s in selected_sources),
                            "source_set_sha256": source_fingerprint(selected_sources),
                            "elapsed_s": None,
                            "validation": "failed",
                            "error": None,
                            "protocol_hash": protocol_hash(options),
                            "cpu_threads": cpu_threads,
                            "rss_sample_interval_s": 0.01,
                            "gpu_sample_interval_s": 0.02,
                            "resource_file": str(
                                (trial_dir / "resources.json").relative_to(output_dir)
                            ),
                        }
                        try:
                            result, resources = _launch(trial_root, options, trial_dir, root)
                            row["elapsed_s"] = result.get("elapsed_s")
                            if result.get("status") != "complete" or result.get("returncode"):
                                raise RuntimeError(result.get("error", "Worker did not complete"))
                            gpu_bytes = resources["gpu"]["sampled_process_peak_bytes"]
                            if backend == "cpu" and gpu_bytes is not None and gpu_bytes > 0:
                                raise ValueError(
                                    "CPU-only worker acquired sampled GPU process memory"
                                )
                            run = Path(result["run_dir"])
                            validation_start = time.perf_counter()
                            manifest = _run_details(run, trial_dir)
                            if manifest["raw_rows"] != row["input_rows"]:
                                raise ValueError(
                                    "Raw source counts differ from benchmark preflight"
                                )
                            if source_fingerprint(manifest["sources"]) != row["source_set_sha256"]:
                                raise ValueError(
                                    "Source identities/checksums differ from preflight"
                                )
                            if reference_backend is None:
                                _save_reference(run, reference)
                                reference_backend = backend
                            comparison = compare_outputs(run, reference)
                            _write(trial_dir / "comparison.json", comparison)
                            cluster = manifest.get("models", {}).get("clustering", {})
                            row.update(
                                validation="warmup" if phase == "warmup" else "pending_match",
                                validation_elapsed_s=time.perf_counter() - validation_start,
                                loaded_gpu_modules_json=json.dumps(
                                    result.get("loaded_gpu_modules", [])
                                ),
                                cpu_import_guard_enabled=result.get(
                                    "cpu_import_guard_enabled", False
                                ),
                                execution_plan_json=json.dumps(result.get("execution_plan", {})),
                                cpu_cuda_context_check=(
                                    "zero sampled process GPU memory"
                                    if gpu_bytes == 0
                                    else "NVML unavailable; import/device guards enforced"
                                )
                                if backend == "cpu"
                                else "not_applicable",
                                peak_host_rss_sampled_bytes=resources["rss"]["sampled_peak_bytes"],
                                peak_gpu_memory_sampled_bytes=resources["gpu"][
                                    "sampled_process_peak_bytes"
                                ],
                                rss_samples=resources["rss"]["samples"],
                                rss_process_tree_complete=resources["rss"]["process_tree_complete"],
                                gpu_samples=resources["gpu"]["samples"],
                                background_gpu_processes=resources["gpu"][
                                    "background_activity_observed"
                                ],
                                selected_k=cluster.get("selected_k"),
                                cluster_eligible_rows=cluster.get("eligible_rows"),
                                model_quality_json=json.dumps(
                                    manifest.get("models", {}), default=str
                                ),
                                rows_per_second=row["input_rows"] / row["elapsed_s"],
                                retained_run=str(run) if keep_runs else None,
                            )
                            if phase == "measured":
                                size_records.append(row)
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                            row["validation"] = "failed"
                            if not (trial_dir / "resources.json").exists():
                                row["resource_file"] = None
                            failures.append(trial_id)
                        finally:
                            records.append(row)
                            _write(trial_dir / "trial.json", row)
                            persist()
                            if not keep_runs:
                                remove_owned_directory(trial_root, scratch, experiment)
                        print(f"{trial_id}: {row['validation']} ({row['elapsed_s']} s)", flush=True)
            # A failed or absent backend must never become a claimed speedup.
            successful = {row["backend"] for row in size_records}
            complete = all(
                sum(row["backend"] == backend for row in size_records) == repeats
                for backend in backends
            )
            for row in size_records:
                row["validation"] = (
                    "passed" if complete and successful == set(backends) else "incomplete_pair"
                )
                _write(trials_dir / row["trial_id"] / "trial.json", row)
            persist()
            remove_owned_directory(reference, scratch, experiment)
        environment["status"] = "failed" if failures else "passed"
    except BaseException:
        environment["status"] = "interrupted"
        raise
    finally:
        environment["failed_trials"] = failures
        _write(output_dir / "benchmark_environment.json", environment)
        persist()
        if not keep_runs:
            remove_owned_directory(scratch, output_dir, experiment)
    if failures:
        raise RuntimeError(
            f"{len(failures)} pipeline trials failed; diagnostics saved in {output_dir}"
        )
    return {
        "experiment_id": experiment,
        "status": environment["status"],
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(_worker(args.worker))
