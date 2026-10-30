# Proposal coverage and interpretation

Audit date: October 30, 2026. The expanded six-month GPU analysis and full raw-file
pipeline experiment have completed and passed validation. Run
`20261030T164900-a9f2d0b4` supplies the expanded analysis; experiment
`pipeline-eb6b177f59b1` supplies the repeated timing/resource evidence. The proposal's
listed analysis components are implemented and exercised, subject to the limits
below. Completed computation does not by itself establish deployment of a new
public snapshot. Earlier runs remain separate historical evidence.

| Proposal component | Implementation and evidence | Interpretation or verification boundary |
|---|---|---|
| Collect TLC records | January–June 2024 yellow taxis: 20,332,093 source rows with source checksums. | Further months and taxi types are extensions. |
| Clean and prepare | Schema and timestamp validation; independent endpoint eligibility; distance, duration, time and quality flags. Clustering and duration analysis apply explicit study limits while preserving raw rows and endpoint counts. | Limits define the analysis population; they cannot establish every retained record's accuracy. |
| CUDA and cuDF | Custom CUDA numerical features, cuDF filtering/grouping and aggregates ran on the TITAN RTX. | Raw schema normalization, date/DST checks and primary quality masks still run on the CPU. |
| cuML clustering | The completed six-month GPU run fits a deterministic 200,000-trip training sample, selects six clusters using April, and predicts all 20,230,079 study-eligible trips. CPU/sklearn and GPU/cuML paths retain comparable inputs and explicit initial centers. | The May–June evaluation is retrospective under study bounds revised after inspecting historical data. Cluster identities can change across fits; they are not known passenger categories. |
| Anomaly detection | Separate frozen baselines score pickup counts and hourly mean trip duration. The completed six-month duration model produces 550 May–June alert hours and 374 candidate incidents. | Both detectors run on CPU. Recorded pickups are observed activity, not total or unmet demand. Duration changes can reflect route/distance composition. Alerts remain unreviewed retrospective findings. |
| cuGraph | Actual cuGraph PageRank and Louvain executed on 263 mapped zones. The selected run has seven communities, including isolated-zone labels; maximum PageRank absolute error versus NetworkX is 2.43 × 10⁻¹⁷. | A graph of taxi zones is small; no graph-specific GPU speed advantage is claimed. Community IDs are arbitrary and valid CPU/GPU partitions can differ. |
| Visualization | Six-section public and local dashboards support endpoint maps, sampled patterns, OD flows, separate count/duration alerts, PageRank/community views, stage/full-pipeline performance and methods. | A selected immutable snapshot shows only the artifacts it contains. Public exports are bounded; graph/model summaries keep their fixed period while supported hourly views honor filters. |
| CPU/GPU evaluation | Full raw-file experiment passed all 18 measured trials at 1/3/6 months, with one discarded full warmup and three measured trials per backend and size. Six-month CPU/GPU medians are 342.734/334.459 s, ratio 1.025×. The earlier 182 validated stage trials retain their own scope. | The measured overall advantage is modest. Three repetitions on this server do not establish a universal winner or statistical significance. Sampled process RSS/NVML memory are not exact peaks. |

Verification for the expansion: 159 local tests passed, seven GPU-only tests were
skipped on the Mac, and 36 dedicated server checks passed. The
[scaling report](pipeline-scaling-results.md) records the full experiment;
[release results](release-results.md) retain model IDs and the earlier measurements.

## What runs where

GPU execution uses custom CUDA/CuPy numerical features, cuDF grouping and
aggregation, cuML KMeans, and cuGraph PageRank/Louvain. Source/schema normalization,
timestamp and DST checks, primary quality masks, canonical-file handling,
deterministic model sampling/scaling, and both seasonal anomaly detectors remain
CPU work. GPU graph validation also executes a CPU reference. The implementation
is therefore hybrid; selecting the GPU backend does not move every pipeline stage
onto the GPU. Missing GPU dependencies fail explicitly rather than triggering an
unreported CPU fallback.

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
The completed GPU run includes 43,501 directed cross-zone pairs and three isolated
zones; 991,723 within-zone trips remain separately accounted for. Graph validation
checks counts, PageRank values and modularity rather than assuming that CPU and GPU
community labels must be identical.

## Timing scopes and resource measurements

The performance ratio is CPU seconds divided by GPU seconds. It is positive
because both durations are positive; above one favors GPU, below one favors CPU.
For example, 0.004186 CPU seconds versus 0.004763 GPU seconds gives about 0.879,
meaning the CPU is about 1.14 times faster for that measured scope. Six decimal
places and an explicit winner prevent the previous rounding from hiding this.

The earlier stage experiment and the completed full-pipeline experiment answer different
questions. The full stopwatch includes raw-source checksums/read/normalization,
features, aggregation, model fitting/evaluation/prediction, duration and graph
analysis, artifact writes and built-in reconciliation. It excludes acquisition,
publication, worker bootstrap, GPU preflight and later cross-trial comparisons.
Lazy initialization reached inside the pipeline and the graph's CPU reference
remain timed. Matching aggregate counts and source/model-input identities are
checked before trials qualify for reported comparisons.

The 1/3-month prefixes do not contain April calibration or May–June test data.
Six-month trials add that evaluation work, so their increase measures more than
just row-count growth. Host process-tree RSS and NVML memory are sampled at requested
10 ms and 20 ms intervals respectively. Short spikes can be missed, shared pages
can appear more than once in RSS, and unavailable GPU readings are not zero usage.

The completed full-pipeline measurements are:

| Source prefix | Rows | CPU median seconds | GPU median seconds | CPU/GPU ratio |
|---|---:|---:|---:|---:|
| January | 2,964,624 | 55.309 | 54.144 | 1.022× |
| January–March | 9,554,778 | 172.874 | 169.288 | 1.021× |
| January–June | 20,332,093 | 342.734 | 334.459 | 1.025× |

At six months, the largest sampled CPU host RSS was 2.231 GiB. GPU trials reached
3.009 GiB sampled host RSS and 7.891 GiB sampled process GPU memory. These are maxima
across the measured repetitions, not exact high-water marks. Other GPU processes
were observed, so possible contention remains part of the experiment conditions.
See [pipeline-scaling-results.md](pipeline-scaling-results.md) for raw-trial summaries,
interquartile ranges, sample gaps and model diagnostics. The modest full-pipeline
advantage must not be replaced with a larger speedup from an isolated stage.

The public website serves precomputed results; visiting it does not execute
CUDA. The completed expanded GPU analysis processed all six months; its retained
candidate was the first measured six-month GPU repetition, selected independently
of runtime. The earlier January-only GPU model remains exploratory because later
calibration/test months were absent. See [release-results.md](release-results.md)
for the expanded analysis and original experiment scopes, and
[case-studies.md](case-studies.md) for alerts prepared for human review.

Further months, other taxi types, weather enrichment, prospective alert evaluation,
distributed execution and multiple GPUs remain extensions. The source adapter
continues to reject unsupported families explicitly.
