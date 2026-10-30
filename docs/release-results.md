# Release results and measurement history

These observations were recorded on October 30, 2026. They are measured results,
not proposed targets from the original specification.

The latest complete experiment is summarized in
[pipeline-scaling-results.md](pipeline-scaling-results.md). The expanded release
appears first; subsequent sections preserve earlier measurement scopes.

## Expanded six-month GPU analysis

Run `20261030T164900-a9f2d0b4` processes all **20,332,093** January–June 2024
source records using the GPU pipeline. It retains the same verified totals:
20,254,960 mapped pickups, 20,149,129 mapped drop-offs and 20,123,078
pickup-anchored origin–destination trips.

The cuML model `kmeans-f87033b3680d81ad` fits the deterministic 200,000-trip
January–March training sample, selects **six clusters** using April, and predicts
all **20,230,079** study-eligible trips. Its 2,000-trip May–June evaluation has
silhouette **0.301884**. This is a retrospective chronological evaluation under
the revised study bounds, not a new untouched-data validation. The two alternate
initialization checks have adjusted Rand indices 0.838 and 0.458, so the groups
should not be treated as uniquely determined passenger categories.

The separate duration detector scores hourly mean eligible-trip minutes against
the frozen zone/weekday/hour baseline. The selected threshold is four, with at
least 20 trips per hour, eight historical hours and a five-minute absolute
change. May–June contains **550 alert hours**, consolidated into **374 candidate
incidents**. These are unreviewed statistical deviations, not confirmed congestion
or measured passenger delay. An independent calculation from canonical timestamps
reproduced the May 24, 11:00 JFK result exactly: 170 trips, mean 70.291176 minutes,
baseline 40.821756 minutes, 13 historical hours and score 14.734710.

Actual cuGraph PageRank and Louvain run on **263 mapped zones**, **43,501 directed
cross-zone pairs** and 19,131,355 cross-zone trips. The 991,723 within-zone trips
are counted separately. Three isolated zones remain included. This selected run
has seven communities including their singleton labels, with weighted modularity
0.199421. Every PageRank value passes the NetworkX comparison; the maximum
absolute difference is **2.43 × 10⁻¹⁷**. Matching communities in this run do not
imply that CPU and GPU Louvain partitions must always match.

The full-pipeline experiment retains one warmup and three measured repetitions
per backend at one, three and six months. Its fixed publication rule selects the
first measured six-month GPU repetition, independently of runtime. See the
[scaling report](pipeline-scaling-results.md) for the completed raw-trial summary,
runtime comparisons and sampled resource measurements, and
[extended methods](extended-methods.md) for the timing boundaries.

All 18 measured trials passed. Six-month medians are **342.734 seconds CPU** and
**334.459 seconds GPU**, a **1.025×** CPU/GPU ratio. This is a modest measured
whole-pipeline advantage, not the much larger acceleration of some isolated
stages. The three repetitions do not establish a universal or statistically
significant winner. Maximum sampled six-month host RSS is 2.231 GiB for CPU runs
and 3.009 GiB for GPU runs; GPU process memory reaches 7.891 GiB, including context
and allocator reservations. Sampling reported no errors, but actual gaps reached
about 32 ms for host memory and 45 ms for GPU monitoring. The server's desktop
GPU services remained active.

The original experiment, trial manifests, resource records, source checksums and
exact code archive are retained under `artifacts/pipeline-scaling-20261030-complete/`.
The SHA-256 of `computation-source.tar.gz` is
`a91af44bfd79c75bc18a0dc13b0acf5468ae353c9423650fb71c338fed003fe3`.
The later configuration fix propagates nondefault CPU thread counts into model
fitting; the recorded experiment used four threads, matching its inner model
limit, and was completed with its source frozen.

Local verification: **159 passed, 7 GPU-only tests skipped**. Dedicated server
validation: **36 passed**, including GPU contracts, duration checks and actual
cuGraph agreement tests. The public browser check is recorded in
`artifacts/qa/browser-public.json` against the selected published data.
All five duration-result tables also match the independent local CPU reference
(`artifacts/qa/final-gpu-data-check.json`).

## Earlier chart and clustering correction

The CPU clustering correction used run `20261030T193520-94b28e21`, model
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

Verification for that correction: 105 CPU tests passed, six GPU-only tests skipped on the Mac.
The original release measurements below remain a historical record; GPU trials
were not rerun for this interface/model correction. See
[proposal-coverage.md](proposal-coverage.md) for current proposal coverage.

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
ride-hailing, weather, and distributed execution remain
extensions rather than completed findings.

The public site uses compact published data with an explicit filter-capability
contract. The full Dash app supports finer hourly/date-specific queries. Both
retain missing coverage, sampling limits, source attribution, and model scope.
