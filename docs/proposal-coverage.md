# Proposal coverage and interpretation

Audit date: October 30, 2026. Native experiment `native-pipeline-484be339070f`
completed all 24 trials: 18 measured runs and six discarded warmups. Retained GPU
run `20261030T191130-202b1964` supplies the new analysis. Its six-month CPU/GPU
median times were **309.137/36.499 seconds**, a measured **8.470×** ratio for local
raw-file processing through saved analytical outputs. Independent CPU reference
checks happen after timing. The proposal's analysis components are implemented
and exercised, subject to the limitations below. Completed computation does not
by itself establish deployment of a new public snapshot.

The native implementation moves bulk normalization, eligibility, sampling/scaling
and both anomaly detectors to the selected CPU or GPU backend. The historical
hybrid experiment's **1.025×** result remains separate evidence; it describes
different execution and sampling policies and cannot be pooled with native trials.

| Proposal component | Implementation and evidence | Interpretation or verification boundary |
|---|---|---|
| Collect TLC records | January–June 2024 yellow taxis: 20,332,093 source rows with source checksums. | Further months and taxi types are extensions. |
| Clean and prepare | Schema and timestamp validation; independent endpoint eligibility; distance, duration, time and quality flags. Clustering and duration analysis apply explicit study limits while preserving raw rows and endpoint counts. | Limits define the analysis population; they cannot establish every retained record's accuracy. |
| CUDA and cuDF | Native trials ran bulk normalization, eligibility, numerical features, sampling/scaling and aggregation on the TITAN RTX using CUDA/CuPy/cuDF. Every bulk stage records its backend and rows. | Host file services, metadata, job control and reduced output finalization remain explicit. Earlier kernel-only experiments have a narrower scope. |
| cuML clustering | The retained native model fits a deterministic 200,000-trip training sample, selects eight groups using April, and predicts all 20,230,079 study-eligible trips. Its 2,000-trip May–June silhouette is 0.283492. CPU/sklearn and GPU/cuML use the same numeric sampling and initial-center policy. | Matching inputs do not guarantee equal fitted quality. Per-trial diagnostics remain visible. May–June evaluation is retrospective under study bounds revised after historical inspection; labels are not known passenger categories. |
| Anomaly detection | Native GPU detectors perform aggregation, historical baselines, calibration and scoring on device. The six-month duration model produces 550 May–June alert hours and 374 candidate incidents; the count model produces 3,224 hours and 2,463 incidents. | Recorded pickups are observed activity, not total or unmet demand. Duration changes can reflect route/distance composition. All incidents remain unreviewed retrospective findings. |
| cuGraph | Actual cuGraph PageRank and Louvain executed on 263 mapped zones. The selected run has seven communities, including isolated-zone labels; maximum PageRank absolute error versus NetworkX is 2.43 × 10⁻¹⁷. | A graph of taxi zones is small; no graph-specific GPU speed advantage is claimed. Community IDs are arbitrary and valid CPU/GPU partitions can differ. |
| Visualization | Six-section public and local dashboards support endpoint maps, sampled patterns, OD flows, separate count/duration alerts, PageRank/community views, stage/full-pipeline performance and methods. | A selected immutable snapshot shows only the artifacts it contains. Public exports are bounded; graph/model summaries keep their fixed period while supported hourly views honor filters. |
| CPU/GPU evaluation | Native experiment: three measured trials and one discarded full warmup per backend at 1/3/6 months, all passed. Six-month CPU/GPU medians are 309.137/36.499 s, ratio 8.470×. CPU workers block CUDA imports and recorded zero sampled process GPU memory. | Three repetitions on this server do not establish a universal winner or statistical significance. Sampled process RSS/NVML memory are not exact peaks. Historical hybrid and stage trials retain separate scopes. |

Verification: 187 local tests passed, 28 GPU-only tests were skipped on the Mac,
27 actual server native checks passed, and all 24 pipeline trials passed. The
[native scaling report](native-pipeline-results.md) records the experiment;
[release results](release-results.md) retain model IDs and historical measurements.

## What runs where

Native GPU execution uses cuDF/CuPy for bulk normalization, timestamp/DST and
eligibility decisions, numerical features, numeric priority sampling, training-only
scaling, aggregation and both anomaly detectors. cuML supplies KMeans; cuGraph
supplies PageRank/Louvain. The CPU counterpart uses CPU libraries throughout.
Both use the same sampling and initial-center policies. Manifests record each
stage's backend and row count, and the website derives its descriptions from
that saved evidence.

Host services handle file discovery/checksums, metadata, small calendar/zone
inputs, orchestration and reduced output finalization. Independent CPU reference
checks run after native GPU benchmark timing. Historical legacy runs instead
used CPU source preparation, model sampling/scaling and anomaly work, and included
the graph's CPU reference inside pipeline time. Missing GPU dependencies fail
explicitly; they do not authorize an unreported CPU fallback.

## Meaning of the displayed counts

A pickup is the recorded start of one taxi trip; a drop-off is its recorded end.
They count trip endpoints, not passengers or unique people. Mapped means the
endpoint has an eligible timestamp and a location ID that can be joined to a
taxi-zone polygon. It does not mean the precise street address is known.

The full-period totals are 20,254,960 mapped pickups and 20,149,129 mapped
drop-offs. Each uses its own endpoint time and validity checks. One endpoint
can qualify while the other does not, and a trip can cross a date boundary.
The available monthly files are organized around pickups, so drop-off coverage
also has incomplete earlier-pickup context. These totals need not be equal.

## Clustering and chart correction

KMeans uses four standardized features: log(1 + miles), log(1 + duration in
seconds), and sine/cosine of pickup hour. The circular time encoding keeps
11 p.m. close to midnight. Scaling and model centers are fitted on a deterministic
January–March sample; April selects among k = 4, 6 and 8; May–June evaluates the
frozen result. Zone IDs, fares and passenger identities are not clustering inputs.

The retained native model `kmeans-336273a6b0eac929` selected eight groups, with
test mean squared center distance 1.133352 in standardized feature space.
Native sampling uses numeric SplitMix64 source-row priorities, and both backends
start from the same priority-selected training centers. This differs from the
historical six-group fit's BLAKE2 sampling/initialization policy; the changed
cluster count is a new fitted result, not a renaming of the old groups.

The initial chart sample contained a reported 100,125.14-mile trip lasting
13 minutes. Its full-range axis obscured ordinary trips, and the same permissive
eligibility could distort cluster summaries. The correction preserves original
records and endpoint counts but refits clustering with documented study bounds:
at most 100 miles, 180 minutes and 100 mph mean speed. Zero distance remains
eligible when the other features qualify. Exclusions are counted in model
metadata; individual reasons can overlap.

The chart defaults to 0–50 miles and 0–120 minutes, labels trips outside that
view, and provides a full-range option. Public charts also offer a log(1 + x)
view that retains zero values. The sample histogram uses visible fixed bands
and an overflow category. These chart ranges are separate from model inclusion.

The completed CPU refit `20261030T193520-94b28e21` excluded 18,290 otherwise eligible
trips (0.09033%), retaining 20,230,079 for clustering. This revision followed inspection of the published
full-period sample. Revised May–June evaluation is therefore retrospective;
another untouched period is needed for fresh confirmation. Cluster IDs can
change after refitting and do not identify persistent categories across models.

Historical stage benchmark timings retain their original configuration and inputs.
They were not retroactively rewritten under the revised clustering policy. New
full-pipeline measurements use a separate experiment and recorded configuration.

## Separate pickup-count and duration alerts

The baseline is the January–March median pickup count for a zone and the same
weekday/hour. History is the number of comparable observed training hours.
The default count score is `(observed - baseline) / max(1.4826 * MAD, 5)`, where
MAD is the median absolute difference from that training baseline. Defaults
require at least eight history observations, a calibrated absolute score threshold
of at least four, and an absolute change of twenty pickups. The saved model records
the selected threshold. Scores are neither percentages nor probabilities.

The duration detector instead observes the mean eligible-trip duration, in minutes,
for each pickup-zone hour. An hour needs at least twenty trips, and the seasonal
baseline needs eight comparable populated training hours. Its default score is
`(mean minutes - baseline minutes) / max(1.4826 * MAD, 2 minutes)`. An alert must
also change by at least five minutes. Missing hours are absent and low-trip hours
remain explicitly unsupported; neither becomes a zero-minute observation.

April calibration selects the threshold before evaluation on May–June. The
calibration alert budget limits review volume, not false-positive probability.
"Unreviewed" displays the stored `unresolved` label: no person has confirmed a
cause or classified the alert. Seasonal changes can also trigger a winter-trained
baseline. A shift in trip distance, routes or destinations can change mean duration
without demonstrating congestion. Adding deviations in hourly mean minutes does
not measure total passenger delay. See [extended methods](extended-methods.md)
for both support rules and the exact score definitions.

## Zone-network analysis

The graph keeps every mapped zone and weights directed origin–destination edges by
their recorded trip counts. Within-zone trips are preserved separately and excluded
from cross-zone degree, PageRank and Louvain calculations. PageRank describes
weighted directed connectivity; it is neither a trip count nor the empirical
probability of a passenger's next destination. Louvain uses an undirected projection
whose weights sum the retained directions. Community numbers are arbitrary labels,
and valid CPU/GPU partitions can differ.

Graph maps and rankings summarize the saved whole-period network. Date, origin and
destination filters on the OD views do not recompute that network. The manifest
records graph backend, edge threshold, isolate handling and numerical checks.
The retained native GPU run includes 43,501 directed cross-zone pairs and three isolated
zones; 991,723 within-zone trips remain separately accounted for. Graph validation
checks counts, PageRank values and modularity rather than assuming that CPU and GPU
community labels must be identical.

## Timing scopes and resource measurements

The performance ratio is CPU seconds divided by GPU seconds. It is positive
because both durations are positive; above one favors GPU, below one favors CPU.
For example, 0.004186 CPU seconds versus 0.004763 GPU seconds gives about 0.879,
meaning the CPU is about 1.14 times faster for that measured scope. Six decimal
places and an explicit winner prevent the previous rounding from hiding this.

The native full stopwatch includes raw-source checksums/read/normalization,
features, aggregation, model fitting/evaluation/prediction, duration and graph
analysis, artifact writes and built-in reconciliation. It excludes acquisition,
publication, worker bootstrap, GPU preflight and later cross-trial comparisons.
Lazy initialization reached inside the pipeline remains timed. The independent
CPU graph reference occurs afterward, alongside checks of aggregates, both
detectors and source/model-input identities. Historical hybrid trials instead
included their graph reference in GPU timing. Stage benchmarks measure narrower
operations and do not describe complete-pipeline speedup.

The 1/3-month prefixes do not contain April calibration or May–June test data.
Six-month trials add that evaluation work, so their increase measures more than
just row-count growth. Host process-tree RSS and NVML memory are sampled at requested
10 ms and 20 ms intervals respectively. Short spikes can be missed, shared pages
can appear more than once in RSS, and unavailable GPU readings are not zero usage.

The completed native measurements are:

| Source prefix | Rows | CPU median seconds | GPU median seconds | CPU/GPU ratio |
|---|---:|---:|---:|---:|
| January | 2,964,624 | 47.829 | 7.287 | 6.564× |
| January–March | 9,554,778 | 148.470 | 17.564 | 8.453× |
| January–June | 20,332,093 | 309.137 | 36.499 | 8.470× |

At six months, the largest sampled CPU host RSS was 2.231 GiB. GPU trials reached
1.635 GiB sampled host RSS and 7.926 GiB sampled process GPU memory. These are maxima
across the measured repetitions, not exact high-water marks. Other GPU processes
were observed, so possible contention remains part of the experiment conditions.
See the [native report](native-pipeline-results.md) for raw-trial summaries,
interquartile ranges, sample gaps and model diagnostics. All measured CPU trials
recorded active CUDA import guards, no imported CUDA libraries and zero sampled
process GPU memory. The earlier hybrid experiment measured 342.734/334.459 s at
six months (1.025×); its [original report](pipeline-scaling-results.md) preserves
that result and its different scope. Comparing those releases does not isolate
which individual optimization produced the change.

The public website serves precomputed results; visiting it does not execute
CUDA. The completed native GPU analysis processed all six months; its retained
candidate was the first measured six-month GPU repetition, selected independently
of runtime. The earlier January-only GPU model remains exploratory because later
calibration/test months were absent. See [release-results.md](release-results.md)
for the expanded analysis and original experiment scopes, and
[case-studies.md](case-studies.md) for alerts prepared for human review.

Further months, other taxi types, weather enrichment, prospective alert evaluation,
distributed execution and multiple GPUs remain extensions. The source adapter
continues to reject unsupported families explicitly.
