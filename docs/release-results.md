# First release: measured results

These observations were recorded on October 30, 2026. They are measured results,
not proposed targets from the original specification.

## Chart and clustering correction

The current clustering run is `20261030T193520-94b28e21`, model
`kmeans-10a1a53e43121915`. A plotted source value of 100,125.14 miles in 13 minutes
exposed an overly permissive clustering policy and an unreadable linear axis.
Clustering now restricts reported distance to 100 miles, duration to 180 minutes
and mean speed to 100 mph. These are inclusive study bounds, not universal
validity claims. They exclude 18,290 of the 20,248,369 previously eligible rows
(0.09033%), leaving 20,230,079 model-eligible trips. The original records and all
pickup/drop-off/OD counts are unchanged.

The refit still selects six clusters. Its 2,000-trip May–June evaluation has a
silhouette of 0.3041. Because the policy revision followed inspection of the
published full-period sample, this is a retrospective chronological holdout
result, not fresh unseen confirmation. A new untouched period is needed for
that claim. Cluster numbers were reassigned and should not be compared by ID
with the earlier model. Anomaly results and all historical benchmark timings
were preserved without rerunning them.

The chart now defaults to 0–50 miles and 0–120 minutes, identifies the three
sampled trips outside that view, and offers full-range inspection. Public charts
also offer zero-preserving logarithmic spacing. All 5,000 new sampled distances
are present, including 75 reported zero distances. Sample distance bands include
an overflow category. OD lines have top-N selection and scope explanations;
anomaly definitions and CPU/GPU winner labels are explicit.

Current verification: 105 CPU tests passed, six GPU-only tests skipped on the Mac.
The original release measurements below remain a historical record; GPU trials
were not rerun for this interface/model correction. See
[proposal-coverage.md](proposal-coverage.md) for the remaining proposal gaps.

## Data and analysis

CPU study run: `20261030T141931-b2c5ac43`.

| Quantity | Result |
|---|---:|
| Official source records, January–June 2024 | 20,332,093 |
| Mapped eligible pickups | 20,254,960 |
| Mapped eligible drop-offs in available source context | 20,149,129 |
| Eligible pickup-anchored origin–destination records | 20,123,078 |
| Selected KMeans clusters | 6 |
| Held-out clustering evaluation sample | 2,000 trips |
| Held-out silhouette on that sample | 0.300 |
| Scored held-out zone-hours | 385,032 |
| Held-out alert bins | 3,224 |
| Merged held-out incidents | 2,463 |

Training uses January–March, calibration April, and held-out evaluation May–June.
KMeans fits a deterministic 200,000-row training sample and predicts eligible
trips in batches. Cluster numbers are algorithmic groupings, not ground-truth
trip purposes. Alerts are statistical deviations with unresolved causes.

The complete custom-CUDA/cuDF/cuML pipeline also processed January's 2,964,624
source records successfully on the TITAN RTX server. Run:
`20261030T143820-f27dd8bf`. Each GPU aggregate batch was compared with the CPU
reference. January-only model output is explicitly exploratory because later
calibration/test data is absent from that run.

## CPU/GPU experiments

Accepted experiment: `features-018231779281`. All **182 measured trials** passed
their applicable correctness gates. Each configuration uses two warmups and
seven repetitions. CPU and GPU trials ran on the same server; source rows,
precision, initial centers, and output semantics were matched where applicable.

| Workload | Rows | CPU median | GPU median | CPU/GPU ratio |
|---|---:|---:|---:|---:|
| Loaded endpoint + OD aggregation | 1,000,000 | 0.9683 s | 0.06335 s | 15.28× |
| Canonical Parquet read → aggregate → write | 1,000,000 | 1.5456 s | 0.2912 s | 5.31× |
| Custom features, including host/device round trip | 1,000,000 | 0.05846 s | 0.01037 s | 5.64× |
| GPU-library features, including host/device round trip | 100,000 | 0.004186 s | 0.004763 s | 0.88× |
| Matched KMeans fitting | 200,000 | 0.05950 s | 0.01435 s | 4.15× |
| Prediction with identical frozen centers | 995,597 | 0.005516 s | 0.002991 s | 1.84× |

The smaller GPU-library feature job lost to the CPU once transfer overhead was
included. Keep that result: GPU acceleration is workload-dependent.

The read/aggregate/write measurement starts from canonical Parquet. It does not
include network acquisition, raw-source normalization, clustering, anomaly
modeling, or dashboard rendering. The loaded custom-kernel speedup is not a
whole-project speedup. Raw repetitions, interquartile ranges, input hashes,
thread limits, environment information, and failures remain archived under
`artifacts/server-benchmarks-final/` and in the published snapshot.

## Correctness and interface checks

- Local CPU test suite: **93 passed, 6 GPU-only tests skipped** on the Mac.
- Dedicated GPU contract suite: **14 passed** on the TITAN RTX.
- Standalone CUDA CMake test passed.
- Compute Sanitizer memcheck, racecheck, initcheck, and synccheck: zero errors.
- Nsight Systems captured the actual custom kernel and CUDA timeline.
- Ruff and JavaScript syntax checks passed.
- All six full Dash pages and public website pages were checked against actual
  data. Public date/zone filtering and a downloaded CSV matched independent
  Parquet queries. The 390-pixel mobile view had no page-width overflow, and the
  final public browser run reported no JavaScript errors.

The browser measurements are representative smoke checks, not a statistically
established p95 latency claim. The first uncached full-period Dash OD view took
about 3.45 seconds in this check; cached/public snapshot views were quicker.

## Remaining limitations

Nsight Compute hardware counters require server-administrator permission
(`ERR_NVGPUCTRPERM`). No driver or administrator configuration was changed.
The four sanitizer checks and Nsight Systems trace succeeded independently.

Synthetic anomaly metrics are not real-world precision or recall; the tested
perturbations also reveal missed small/sparse changes under the absolute-count
threshold. Three assistant-prepared case studies are available for review,
but human review and causal corroboration have not been completed. Incident
labels remain unresolved. Full-year scaling, feature/scale-floor ablations,
ride-hailing, weather, graph algorithms, and distributed execution remain
extensions rather than completed findings.

The public site uses compact published data with an explicit filter-capability
contract. The full Dash app supports finer hourly/date-specific queries. Both
retain missing coverage, sampling limits, source attribution, and model scope.
