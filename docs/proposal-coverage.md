# Proposal coverage and interpretation

Audit date: October 30, 2026. This is an implementation audit, not a claim that
every proposed component is finished.

| Proposal component | Implemented evidence | Remaining work |
|---|---|---|
| Collect TLC records | January–June 2024 yellow taxis: 20,332,093 source rows with source checksums. | Further months and taxi types are extensions. |
| Clean and prepare | Schema and timestamp validation; independent endpoint eligibility; distance, duration, time and quality flags. Clustering now has explicit model-only limits and exclusion counts. | Limits define this study's scope; they cannot establish every retained record's accuracy. |
| CUDA and cuDF | Custom CUDA numerical features, cuDF filtering/grouping and aggregates ran on the TITAN RTX. | Raw schema normalization, date/DST checks and primary quality masks still run on the CPU. |
| cuML clustering | Implemented and executed in the January GPU run and matched benchmarks. | The published six-month model uses sklearn on CPU. A full six-month GPU model run remains to be done. |
| Anomaly detection | Frozen zone/weekday/hour baselines detect unusual hourly pickup counts. April calibrates thresholds; May–June is held out. | Travel-duration anomalies are absent. Recorded pickups measure observed activity, not total or unmet demand. |
| cuGraph | Origin–destination aggregation and visualization exist. | No cuGraph graph algorithm has been implemented or run. Drawing connections alone does not satisfy this component. |
| Visualization | Public six-section dashboard and local Dash application with mapped endpoints, patterns, flows, anomalies, performance and methods. | The public snapshot has bounded sample/alert exports and explicitly limited filters. |
| CPU/GPU evaluation | Same-server matched stage timings, two warmups, seven repetitions and 182 correctness-passing trials. Aggregation/features compare 100,000 and 1,000,000 rows; clustering has separate fit/prediction scopes. | No matched full 20.3-million-row end-to-end comparison. Sampled RSS and post-trial GPU allocation are not exact peak-memory measures. |

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

The refit excludes 18,290 otherwise eligible trips (0.09033%), retaining
20,230,079 for clustering. This revision followed inspection of the published
full-period sample. Revised May–June evaluation is therefore retrospective;
another untouched period is needed for fresh confirmation. Cluster IDs can
change after refitting and do not identify persistent categories across models.

Historical benchmark timings retain their original configuration and inputs.
They were not rerun under the revised clustering policy.

## Unusual activity and timings

The baseline is the January–March median pickup count for a zone and the same
weekday/hour. History is the number of comparable observed training hours.
The score is `(observed - baseline) / max(1.4826 * MAD, 5)`, where MAD is the
median absolute difference from that training baseline. Current alerts require
at least eight history observations, an absolute score of four and an absolute
change of twenty pickups. Scores are neither percentages nor probabilities.

April calibration selects the threshold before evaluation on May–June. The
calibration alert budget limits review volume, not false-positive probability.
"Unreviewed" displays the stored `unresolved` label: no person has confirmed a
cause or classified the alert. Seasonal changes can also trigger a winter-trained
baseline. Travel-duration anomalies are not currently scored.

The performance ratio is CPU seconds divided by GPU seconds. It is positive
because both durations are positive; above one favors GPU, below one favors CPU.
For example, 0.004186 CPU seconds versus 0.004763 GPU seconds gives about 0.879,
meaning the CPU is about 1.14 times faster for that measured scope. Six decimal
places and an explicit winner prevent the previous rounding from hiding this.

The public website serves precomputed results; visiting it does not execute
CUDA. The January GPU run processed 2,964,624 source records using CUDA/cuDF/cuML,
but its model was exploratory because later calibration/test months were absent.
See [release-results.md](release-results.md) for original experiment scopes and
[case-studies.md](case-studies.md) for alerts prepared for human review.
