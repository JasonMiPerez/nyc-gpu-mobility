# GPU implementation and experiments

The server is the main benchmark machine. Compare its CPU and GPU on the same
frozen inputs; laptop CPU timings must not be paired with server GPU timings.
The web dashboard needs only the CPU environment and published artifacts.

## What runs on the GPU

`mobility.backends.gpu` explicitly uses cuDF for endpoint/OD aggregation and CuPy
for numerical features. Source adaptation, calendar/DST decisions and quality
eligibility are prepared on the CPU. Their time and the transfer belong in a
pipeline measurement. There is no automatic CPU fallback for a requested GPU run.
Clustering has its own cuML implementation in `mobility.models`. Zone-network
analysis uses cuGraph for weighted directed PageRank and weighted undirected
Louvain communities, with NetworkX correctness checks. Both anomaly detectors
currently use CPU calculations. ML feature/scaler preparation also remains CPU
work; the published metadata distinguishes these stages.

`processing.custom_cuda: true` selects the fused custom feature kernel during a
yellow-taxi GPU pipeline. With `false`, ordinary GPU library operations are used.
Both paths preserve canonical quality/eligibility; the custom kernel's diagnostic
flags live in `kernel_feature_flags`, a separately documented bitmask.

## Custom CUDA contract

Source: `src/mobility/backends/kernels/fused_trip_features.cu`. Python compiles this
CUDA C++ file with CuPy RawModule/NVRTC. CMake compiles the same source in the
standalone executable under `cuda/`, so there are not two independent kernels.

Each thread handles a trip in a grid-stride loop. Input arrays are contiguous
int64 pickup/drop-off nanoseconds, float64 miles and explicit uint8 masks for
timestamp validity, allowed time policy, and endpoint-zone eligibility. Masks
are expanded bytes, not Arrow bitmaps. These conversions occur outside a
kernel-only timing, but must be included in full-stage timings.

Outputs are float64 duration/speed, a uint8 feature-eligibility mask and uint32
flags. Flag bits: missing pickup 1, missing drop-off 2, time excluded 4,
nonpositive duration 8, invalid distance 16, zero distance 32, invalid zone 64.
Feature eligibility here requires valid duration/distance and both eligible
zones; it is not an instruction to replace the canonical ML mask.

Invalid numerical outputs are NaN. Null sentinels never participate in timestamp
subtraction. Splitting into seconds/remainders avoids int64 subtraction overflow
and preserves useful timestamp precision. Division/logarithms are guarded.
Zero distance remains a valid distance, with a diagnostic flag. Zone validity
does not erase otherwise valid duration/speed.

The wrapper rejects wrong shapes, dtypes, devices, strided arrays and overlapping
output buffers. Launches use the current CuPy stream; callers retain arrays and
synchronize before inspecting results. Benchmarks synchronize full device work
at wall-clock boundaries and use CUDA events for kernel-only work.

## Environment

Observed server: Ubuntu 20.04, glibc 2.31, NVIDIA TITAN RTX (Turing 7.5, 24 GB),
driver 535.183.01. The project environment is isolated under `.gpu-env`; it does
not change drivers or system Python. The selected compatibility target is
RAPIDS 25.10 (including cuGraph 25.10.01), Python 3.12, CUDA/NVRTC/nvcc package selection 12.2, CuPy 13.6 and scikit-learn
1.7.2. The older sklearn pin is necessary for cuML 25.10's internal API usage;
the initially resolved sklearn 1.9 failed the import smoke test. Direct CUDA
driver API returned version 12020; `CONDA_OVERRIDE_CUDA=12.2` was used because
micromamba initially misidentified the virtual CUDA package as 11.6. The actual
CuPy process reports runtime API **12090 (12.9)**, while the driver API reports
**12020 (12.2)** and the standalone compiler identifies **nvcc 12.2.140**.
Do not describe these as one uniform CUDA version. Package selection and actual
loaded/linked runtime observations are both retained in the lock and run report.
[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
allows certain CUDA 12.x applications on this driver family, with feature/PTX
limitations. The passing tests establish compatibility for the operations used
here; they do not establish that every CUDA 12.9 feature will work on this server.

Actual server validation passed all 14 CPU/GPU contract tests, cuDF aggregation,
cuML KMeans, the custom NVRTC kernel (empty, single, 513, and 100,003 rows), and
the standalone nvcc/CMake executable. All four Compute Sanitizer tools reported
zero errors (racecheck also reported zero hazards). A separate Nsight Systems
trace captured `fused_trip_features` on actual canonical records. Nsight Compute
could not access hardware counters (`ERR_NVGPUCTRPERM`); the failed attempt and
permission error are preserved, and no occupancy/counter claims are made.
Save `artifacts/gpu-validation/smoke.json`
and the environment lock alongside final experiment artifacts. This validates
the tested implementation/environment, not arbitrary future package versions.

Exact packages are frozen in `environments/gpu-linux-64.lock`; durable constraints
are in `environments/gpu-pins.txt` and the server environment's `conda-meta/pinned`.
To reproduce on a compatible Linux NVIDIA machine:

```sh
micromamba create -p "$PWD/.gpu-env" -f environments/gpu-linux-64.lock
cp environments/gpu-pins.txt .gpu-env/conda-meta/pinned
micromamba run -p "$PWD/.gpu-env" python -m pip install --no-deps -e .
micromamba run -p "$PWD/.gpu-env" python scripts/gpu_smoke.py
```

Compatibility sources: [NVIDIA RAPIDS platform matrix](https://docs.nvidia.com/datascience/platform-support/index.html),
[CuPy RawModule](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawModule.html),
[CuPy performance timing](https://docs.cupy.dev/en/stable/user_guide/performance.html).

## Validation and profiling

From the project root on the server, with the GPU environment active:

```sh
python scripts/gpu_smoke.py
MOBILITY_RUN_GPU_TESTS=1 python -m pytest tests/test_gpu_contract.py -q
cmake -S cuda -B cuda/build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build cuda/build
ctest --test-dir cuda/build --output-on-failure
compute-sanitizer --tool memcheck cuda/build/cuda_smoke
compute-sanitizer --tool racecheck cuda/build/cuda_smoke
compute-sanitizer --tool initcheck cuda/build/cuda_smoke
compute-sanitizer --tool synccheck cuda/build/cuda_smoke
nsys profile -o artifacts/profiles/features python scripts/gpu_smoke.py
ncu --set full cuda/build/cuda_smoke
```

Sanitizer and profiler tools must be installed in the GPU environment or supplied
by the administrator. Save their output, tool version and exit code. Do not use
heavily instrumented profiling runs as final timing trials.

## Benchmark API and interpretation

The full study can be run with all implemented analyses:

```sh
python -m mobility --config configs/gpu-six-month.yaml pipeline run --publish
```

The broader benchmark runs complete production pipelines on the same host:

```sh
python scripts/benchmark_pipeline_scaling.py \
  --root "$PWD" --config configs/default.yaml \
  --output artifacts/new-pipeline-experiment \
  --month-counts 1 3 6 --repeats 3 --warmups 1 --cpu-threads 4 --keep-runs
```

Use a new output directory. These trials include raw-source normalization,
artifact writes, cuML/CPU clustering, both anomaly detectors and graph analysis.
The expensive per-batch CPU aggregate reference is disabled during timing;
complete outputs and model inputs are compared afterward. Built-in source and
count reconciliation and the graph's CPU reference remain timed. Graph checks
do not assume the same Louvain partition on both backends. One/three-month runs
lack the later calibration/test data available to the six-month run.

Each trial is a fresh process. Requested sampling intervals are 10 ms for host
process-tree RSS and 20 ms for GPU process memory; actual gaps and missing readings
are retained. These are sampled peaks, not guaranteed high-water marks. Desktop
GPU services remain running and are disclosed as background activity. See
[extended methods](extended-methods.md) for the complete measurement contract.

The smaller stage-level benchmark remains available separately:

`run_benchmarks(canonical_dir, output_dir, config, include_gpu=False)` runs actual
available canonical records. Set `include_gpu=True` explicitly on the NVIDIA
machine. It saves `benchmark_runs.parquet`, `benchmark_summary.parquet`, and
`benchmark_environment.json`, including all measured repetitions and failures.

The default uses up to 100,000 and 1,000,000 actual rows from sorted canonical
files. This is a deterministic file prefix, not a representative random sample
or a claim that every record was processed. Input-buffer hashes are saved. Set
`benchmark.max_rows`, `warmups`, `repeats`, `cpu_threads` and `seed` explicitly for
final experiments. Default repetition counts are two warmups and seven trials.

- `loaded_compute`: identical resident inputs through feature outputs, including
  output allocation. GPU transfers and file reading are excluded.
- `kernel_only`: CUDA events around the custom kernel with preallocated outputs.
  Validation/compilation are performed before recording events. The event window
  can still include host-launch scheduling gaps; it is not identical to a
  profiler's device-only kernel duration. There is no CPU speedup attached to
  this unmatched scope.
- `host_to_host_features`: identical host arrays through computation and back to
  host arrays. GPU transfer is included; Parquet decoding, source adaptation,
  model work and publication are excluded. This is **not full-pipeline time**.

CPU uses vectorized NumPy, GPU library uses separate CuPy operations, and GPU
custom uses the fused kernel. Orders are randomized, inputs/precision match,
all integer flags agree exactly, and float64 outputs use declared tolerances.
The summary computes speedup only for passed trials with matching stage, scope
and row count. CPU-only results are explicitly CPU-only.

Memory fields distinguish sampled host RSS from post-trial device-pool usage.
The latter is not peak VRAM; short RSS peaks can also be missed at 10ms sampling.
Logical read/write traffic is estimated at 50 bytes per row, excluding caches,
temporary arrays and transactions; it is not measured DRAM bandwidth.

The harness also measures production endpoint+OD aggregation as one explicitly
combined stage, matched KMeans fit, and prediction using identical saved centers.
`end_to_end_aggregation` reads a bounded canonical Parquet input, aggregates and
writes both output tables. It includes required GPU transfers, but excludes raw
schema normalization, model fitting and publication. Filesystem caches are not
flushed; results are labeled warm rather than cold-disk measurements. Temporary
bounded input/output files are removed after validation.

Full raw-source pipeline timings remain in pipeline stage logs and include
correctness checks. They are not interchangeable with these repeated isolated
experiments. Graph algorithms are optional and have no invented benchmark row.

The first accepted server experiment (`features-018231779281`) contains 182
passed measured trials: two data sizes (100,000 and 1,000,000 real canonical
rows), seven repeats per applicable operation, a 200,000-row training sample,
and prediction for 995,597 eligible rows. Its tables and exact environment are
under `artifacts/server-benchmarks-final/`. An earlier experiment with a failed
cuML center-assignment check remains archived separately; it was not silently
replaced with invented measurements.
