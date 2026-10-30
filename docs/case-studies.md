# Three recorded-activity alerts

**These are assistant-prepared case studies pending user review.** They describe
real aggregate results; they are not verified incident labels or an estimate of
real-world detection accuracy. The original model metadata remains
`manual_review_status: not_performed`, and incident labels remain `unresolved`.

The source is run `20261030T141931-b2c5ac43`, model
`seasonal-722214e486ccfee6`, built from 20,332,093 yellow taxi records for
January–June 2024. Training uses January–March, calibration uses April, and these
examples come from the May–June test period. Calibration retained score threshold
4 and absolute count-deviation threshold 20. All times below use the documented
New York source-local-time convention; each listed time starts a one-hour bin.

The test period contains 385,032 scored zone-hours, 3,224 alert bins, and 2,463
consolidated incidents. These three examples were selected after inspecting
results: the largest upper alert, the largest lower alert, and an airport example.
They are illustrative cases, not a random sample of alerts.

| Case | Affected hour starts | Recorded pickups | Sum of hourly baselines | Difference | Peak signed score |
| --- | --- | ---: | ---: | ---: | ---: |
| Lincoln Square East, zone 142 | June 16 23:00–June 17 02:00 | 839 | 89 | +750 | +51.80 |
| East Village, zone 79 | May 25 00:00–01:00 | 561 | 1,250 | −689 | −17.09 |
| LaGuardia Airport, zone 138 | June 6 00:00–02:00 | 460 | 68 | +392 | +23.60 |

“Difference” means deviation from this frozen historical baseline. It is not a
count of unmet trips, affected passengers, or journeys caused by an event.

## 1. Lincoln Square East: a sharp overnight increase

The strongest upper alert occurs at **01:00 on June 17**: 265 pickups versus a
baseline of six. The median absolute deviation is one, so the scale floor of five
applies: `(265 − 6) / 5 = 51.8`. Thirteen matching training-period hours support
the baseline.

| Hour start | Pickups | Baseline | Score | Alert |
| --- | ---: | ---: | ---: | --- |
| June 16 21:00 | 142 | 136 | +0.24 | No |
| June 16 22:00 | 123 | 79 | +1.98 | No |
| June 16 23:00 | 232 | 60 | +4.83 | Yes |
| June 17 00:00 | 262 | 17 | +33.05 | Yes |
| June 17 01:00 | 265 | 6 | +51.80 | Yes |
| June 17 02:00 | 80 | 6 | +12.48 | Yes |
| June 17 03:00 | 13 | 2 | +2.20 | No |
| June 17 04:00 | 11 | 4 | +1.40 | No |

The immediately surrounding hours are below the alert threshold. The same Monday
01:00 bin recorded four pickups on June 3, five on June 10, and eight on June 24.
This makes the selected hour stand out even against nearby test-period weeks,
not just the winter training baseline. Lincoln Square West also has an upper
alert at June 16 23:00, with 38 pickups against a baseline of eleven, but does not
share the later sustained increase.

One independently verified piece of context is the 2024 Tony Awards. The
organizer's announcement schedules the ceremony at the David H. Koch Theater at
Lincoln Center on June 16, with the broadcast from 8–11 p.m. Eastern.
[Official Tony Awards announcement](https://www.tonyawards.com/press/academy-award-winner-and-tony-award-nominee-ariana-debose-returns-to-host-the-77th-annual-tony-awards/).

An event-related pickup increase is therefore a hypothesis worth reviewing.
This review has not geocoded the venue against taxi-zone boundaries, identified
any passenger as an attendee, or established why trips occurred. The temporal
coincidence and Lincoln Center location provide context, not a causal finding.

## 2. East Village: fewer recorded pickups over two hours

At **01:00 on May 25**, 271 pickups were recorded against a baseline of 651,
about 58% fewer. The historical MAD is fifteen and the scale is 22.239, giving
`(271 − 651) / 22.239 ≈ −17.09`. Again, thirteen matching training observations
support the baseline.

| Hour start | Pickups | Baseline | Score | Alert |
| --- | ---: | ---: | ---: | --- |
| May 24 22:00 | 182 | 296 | −2.75 | No |
| May 24 23:00 | 242 | 431 | −2.66 | No |
| May 25 00:00 | 290 | 599 | −6.13 | Yes |
| May 25 01:00 | 271 | 651 | −17.09 | Yes |
| May 25 02:00 | 193 | 467 | −3.36 | No |
| May 25 03:00 | 160 | 273 | −1.95 | No |

The decline extends beyond the two flagged hours. The detector only flags the
hours that exceed both thresholds; an unflagged neighboring hour is not evidence
of normal conditions. The corresponding Saturday 01:00 counts were 566 on May
11, 615 on May 18, and 588 on June 1. June 8 also has a smaller lower alert, with
556 pickups and a score of −4.27.

At the May 25 peak, four Manhattan zones have lower alerts. Manhattan's summed
recorded pickups are 2,570, compared with 4,429 when summing the individual zone
baselines. That comparison suggests a broader change worth examining, although
the sum of zone medians is not a separately fitted borough baseline.

These records establish a drop in observed yellow taxi pickups. They do not
distinguish changes in travel activity, available taxis, transport choices,
calendar patterns, or reporting. No external cause has been assigned.

## 3. LaGuardia Airport: activity later than the historical pattern

At **02:00 on June 6**, the airport zone records 119 pickups against a baseline
of one. The scale floor of five applies, giving `(119 − 1) / 5 = 23.6`, supported
by thirteen matching training observations.

| Hour start | Pickups | Baseline | Score | Alert |
| --- | ---: | ---: | ---: | --- |
| June 5 22:00 | 190 | 191 | −0.04 | No |
| June 5 23:00 | 207 | 163 | +1.56 | No |
| June 6 00:00 | 211 | 63 | +6.24 | Yes |
| June 6 01:00 | 130 | 4 | +21.25 | Yes |
| June 6 02:00 | 119 | 1 | +23.60 | Yes |
| June 6 03:00 | 1 | 0 | +0.20 | No |
| June 6 04:00 | 2 | 1 | +0.20 | No |

The pattern is a three-hour extension of recorded activity into hours with a low
historical baseline. The same Thursday 02:00 bin has five pickups on May 23,
thirty on May 30, seven on June 13, and one on June 20. May 30 also alerts, so this
is not the only unusually active airport night in the held-out period.

The large score partly reflects the very low reference count. It is not a
probability and should be read alongside the absolute increase of 118 pickups
at the peak. Flight schedules, arrival delays, weather, and airport operating
records would be needed to investigate an explanation. None was checked for this
case, so a flight-delay explanation would be speculation.

## Coverage and data-quality context

Every listed case and adjacent hour is `coverage=observed` and `status=scored`.
Each baseline has thirteen eligible historical observations, exceeding the
minimum of eight. These May/June intervals are not marked as DST discontinuities,
and neither source month is missing. The source files passed checksum, schema,
and row-count validation.

| Source month | Raw rows | Missing pickup timestamps | Out-of-source-month pickups | Unmapped pickup zones | Invalid durations |
| --- | ---: | ---: | ---: | ---: | ---: |
| May 2024 | 3,723,833 | 0 | 33 | 13,863 | 1,147 |
| June 2024 | 3,539,193 | 0 | 51 | 13,514 | 1,140 |

The pipeline excludes out-of-month and unmapped pickups from this mapped pickup
analysis. Invalid duration does not remove an otherwise eligible pickup. These
quality counts describe whole source months, not these particular zone-hours;
they cannot rule out localized reporting or coding issues. “Observed” means the
configured source was processed, not that every real trip was reported.

The frozen January–March baseline may also differ from ordinary May/June activity
because behavior changes over time. These cases are useful examples for studying
the detector and choosing follow-up checks, not proof that the model identifies
all real incidents or avoids false alerts.

## Reproduce the extracted values

```sh
.venv/bin/python scripts/export_case_studies.py \
  --run artifacts/runs/20261030T141931-b2c5ac43 --output docs
```

The script reads only saved aggregate/model tables and writes
`case-study-summary.csv`, `case-study-hours.csv`, and
`case-study-weekly-context.csv`. It verifies that hourly counts and deviations
reconcile with the saved incidents. It does not change any source, model,
incident label, or human-review status.
