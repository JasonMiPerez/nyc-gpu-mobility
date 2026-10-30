# Methods and interpretation

This project studies recorded yellow taxi activity. It compares CPU and GPU
implementations of defined computations. Neither a taxi record nor an aggregate
zone connection reveals the street route, total travel demand, or a passenger's
identity.

## Execution modes

The default `native` mode provides separate CPU and GPU implementations of the
same analysis policy. GPU normalization, eligibility, features, bounded sampling,
scaling, aggregations and anomaly baselines/scoring use cuDF/CuPy; clustering uses
cuML and graph algorithms use cuGraph. The CPU path uses CPU libraries throughout.
The manifest records actual stage backends, implementation descriptions and row
counts. Host services remain responsible for checksums, metadata, small
calendar/zone inputs, job control and reduced output finalization. Independent
CPU reference checks are outside timed native GPU processing.

The retained `legacy` mode describes the older hybrid implementation. Its CPU
source preparation, sampling/scaling and anomaly calculations, plus an internal
NetworkX graph reference, were included in GPU pipeline timings. The published
historical 1.025× six-month ratio belongs to that mode. The separate completed
native experiment measured 309.137 seconds CPU versus 36.499 seconds GPU at six
months (8.470×), with three measured repetitions per backend and size. See the
[native report](native-pipeline-results.md); its result applies to the recorded
server, workload and timing scope.

## Sources and eligibility

Monthly TLC Parquet files are retained with their byte count, SHA-256 checksum,
schema, row count, and source URL. The physical row number and source checksum
form a stable record ID. Similar-looking rows are not automatically deduplicated.
The canonical layer preserves records and assigns separate eligibility decisions:

- A pickup needs a usable pickup time and mapped origin zone. Pickups outside the
  filename's source month are flagged and excluded from the default pickup study.
- A drop-off needs its own usable time and mapped destination. A missing pickup
  time does not erase an independently usable drop-off.
- An OD count uses pickup time and valid origin/destination zones. A missing
  drop-off timestamp or distance does not erase a known zone pair.
- Duration statistics additionally require positive duration and a safe time
  interval. Distance statistics require finite, nonnegative distance.
- Clustering requires an eligible pickup, duration, and distance. Zero distance
  is flagged but is not automatically removed from activity or feature analysis.

Counts, sums, and valid-value counts are merged across batches. Means are computed
from the combined sums and counts. For example, two trips totaling 20 minutes and
eight totaling 160 minutes have a combined mean of 18 minutes, not 15.

## Time and coverage

Source timestamps use an explicit New York local-wall-time convention. This is a
documented analysis assumption; per-row UTC offsets are not invented. Ambiguous
fall-back hours and nonexistent spring-forward hours are flagged. Trips whose
duration might cross an offset change are excluded from duration-based features.

Pickups and drop-offs use their respective endpoint hours. OD tables are anchored
to pickup time. Missing source months have null counts and a coverage status;
observed hours with no eligible records have zero counts. DST discontinuities are
not ordinary zero-demand observations.

Drop-off coverage is conservatively marked `partial_source_context`. A file
organized by pickup month cannot establish complete drop-offs without earlier
pickup context and a defensible duration bound. Even an observed pickup bin
describes the available TLC reporting, not complete real-world transportation.

## Chronological evaluation

The default six-month study uses these half-open intervals:

| Role | Included dates |
| --- | --- |
| Training | January 1 through March 31, 2024 |
| Calibration and model selection | April 1 through April 30, 2024 |
| Held-out evaluation | May 1 through June 30, 2024 |

`configs/full-year.yaml` extends the held-out period through December 31, 2024.
The January configuration exercises the software but cannot provide a separate
April calibration or May-onward test. Its clustering is labeled exploratory and
its anomaly model reports insufficient history or unavailable evaluation.

## Clustering

KMeans uses four features: `log1p(distance_miles)`,
`log1p(duration_seconds)`, `sin(2π × pickup_hour / 24)`, and
`cos(2π × pickup_hour / 24)`. Raw zone IDs are not numerical coordinates.

Native execution computes features and training-only population mean/variance
in float64, then forms standardized float32 KMeans matrices. A deterministic
bounded sample keeps the smallest numeric SplitMix64 priorities derived from
the source token, physical row number and seed, with source/row tie-breaking.
CPU and GPU use the same policy, independent of batch boundaries. The saved
policy name is `smallest_splitmix64_source_row_v1`. Sample identities, their hash,
eligible-row count and sampling fraction are retained privately. The default
training cap is 200,000 rows. Scaling is fitted only on those training rows.

Legacy snapshots instead used the smallest 64-bit BLAKE2b hashes of
`seed:record_id`. This policy change can change fitted models, so historical
cluster labels and quality metrics are not assumed to remain identical.

The pipeline fits the configured candidate cluster counts, normally 4, 6, and 8,
using explicit seeded initial centers. In native mode both backends take the
first k rows under the same seeded numeric priority policy
(`seeded_source_row_priority_first_k_v1`), after training-only scaling. The highest silhouette score on one fixed
calibration sample selects the model; smaller k breaks ties. If a calibration
score is unavailable, the smallest feasible k is retained as an exploratory
choice. Test data never selects k or fits the scaler.

The retained native model `kmeans-336273a6b0eac929` selected eight groups from a
200,000-row training sample. Its 2,000-row May–June evaluation has silhouette
0.283492 and mean squared center distance 1.133352 in standardized feature space.
The earlier six-group model used a different sampling/initialization policy.
Cluster IDs and metrics therefore do not carry unchanged across these revisions.
Matched CPU/GPU inputs establish a comparable experiment, not identical fits or
equal model quality; each measured trial's diagnostics are retained and displayed.

Evaluation records held-out silhouette and center distance, cluster proportions,
and adjusted Rand agreement for two additional seeds. Evaluation samples are
bounded, normally at 2,000 rows. Final prediction processes all eligible records
in batches, even though fitting uses a sample.

Cluster counts and means use all predicted eligible trips. The explicitly named
sampled medians and dashboard scatter use a bounded sample. Cluster summaries
describe their fixed scoring period; date and zone filters do not silently
recompute them. Group IDs have no intrinsic ranking or verified passenger-type
meaning. Saved JSON contains feature order, scaler parameters, centers,
initialization, and sample identity; model loading does not require a pickle.

## Seasonal anomaly detector

The baseline groups observed training counts by dataset, zone, and hour of week.
For each group it stores median `m`, median absolute deviation `MAD`, and number of
eligible historical observations. The score is:

```text
scale = max(1.4826 * MAD, scale_floor)
score = (observed_pickups - median_pickups) / scale
```

Defaults require at least eight historical observations of the same hour of the
week, a scale floor of five trips, an absolute score of at least four, and a count
deviation of at least twenty trips. Both deviation conditions must hold. Missing,
DST-ambiguous, and insufficient-history rows retain their status and no score.
The score is not a p-value or a calibrated statistical test.

Calibration examines score thresholds 3, 4, 5, and 6. It selects the smallest
candidate at least the configured threshold whose calibration alert fraction is
at most 1%. If none meets that budget, it uses the largest candidate and records
that the budget was not met. With no supported calibration rows, it retains the
configured threshold and explicitly makes no calibration claim. This controls
alert burden; 1% is not a measured false-positive rate.

The baseline and selected threshold stay frozen during the test period. Adjacent
alerts merge only when dataset, zone, direction, and evaluation split match.
Missing hours are not bridged. `end_hour` is the start of the last affected hourly
bin, and `affected_hours` is the number of flagged bins.

Synthetic evaluation perturbs separate copies of eligible held-out aggregate
bins using spikes and drops of 0.5, 1, or 2 times the baseline, lasting 1, 3, or 6
hours, with three fixed random seeds. Drops are clipped at zero. The saved mask
contains original and perturbed counts, scores, and alert decisions. An event is
detected if any changed bin in its window alerts. Results distinguish previously
alerted bins from newly alerted bins. Untouched real alerts are unmatched
background alerts, not verified negatives. Injection-relative precision/recall
does not establish real-world detection accuracy.

Incident labels initially remain `unresolved`, and metadata says manual review
has not been performed. A case review should record the model ID, dataset, zone,
dated interval, coverage, baseline support, review label, reasoning, evidence,
reviewer, and review date. Context from an external event can support a discussion
but does not prove causation. Preserve the original results when recording a
review; the current dashboard does not write review annotations back to storage.

## Benchmarks

The main comparisons run CPU and GPU on the same host with explicit thread
limits, warmups, repeated trials, randomized operation order, and GPU completion
before stopping the timer. Failures are retained; a failed correctness check
does not produce a valid speedup. Report medians, interquartile ranges, raw trials,
actual row counts, and the exact timing scope.

KMeans fitting and prediction are separate benchmarks. They use identical
transformed float32 inputs. Matched fitting uses the same starting centers,
`n_init=1`, iteration cap, and tolerance. Fit checks include valid centers and
assignments, objective consistency, sampled nearest-center consistency, and a
predeclared 5% objective comparison tolerance. Adjusted Rand agreement is recorded
without requiring arbitrary cluster numbers to match. Matching the numeric
tolerance does not guarantee identical library stopping rules; iteration counts
and final objectives remain part of the comparison.

Prediction uses byte-identical frozen centers for sklearn and cuML. A differing
label is acceptable only within the declared near-boundary distance tolerance;
the mismatch count is saved. Loaded-compute timing excludes file IO, scaling,
sampling, explicit transfers, and validation. It includes output allocation and
completed library work. These timings cannot be described as full-pipeline
speedups. See `docs/gpu.md` for the custom CUDA path and its validation procedure.

## Current analytical limits

The core implements one clustering feature set, separate frozen pickup-count and
mean-duration detectors, and weighted PageRank/Louvain graph analyses. Their
extended definitions and benchmark scopes appear in [extended methods](extended-methods.md).
Location feature ablations, scale-floor sensitivity experiments, walk-forward
updating, HVFHV, weather corroboration and manually reviewed incident cases remain
additional work. Public charts show recorded activity, sampled trip
patterns, and statistical alerts. They do not establish congestion, fraud, latent
demand, actual routes, or the cause of a change.
