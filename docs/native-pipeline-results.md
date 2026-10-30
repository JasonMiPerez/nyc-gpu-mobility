# Full-pipeline scaling report

Experiment `native-pipeline-484be339070f` · status: **passed**.
Execution mode: **native**; timing scope: `raw_files_to_native_models_and_artifacts_external_reference_validation`. Compare only matching experiment scopes.
Expected measured repetitions per backend and size: 3. Warmups are excluded.

| Months | Source rows | Backend | Status | Repetitions | Median s | Q25–Q75 s |
|---|---:|---|---|---:|---:|---:|
| 1 | 2964624 | CPU | validated | 3 | 47.829293 | 47.588275–47.985958 |
| 1 | 2964624 | GPU | validated | 3 | 7.286902 | 7.250318–7.289330 |
| 3 | 9554778 | CPU | validated | 3 | 148.470483 | 146.781647–149.129945 |
| 3 | 9554778 | GPU | validated | 3 | 17.563515 | 17.531380–17.632732 |
| 6 | 20332093 | CPU | validated | 3 | 309.137353 | 306.963077–311.794481 |
| 6 | 20332093 | GPU | validated | 3 | 36.499323 | 36.337345–36.607983 |

1 months: CPU time / GPU time = **6.564×**; GPU has the faster measured median (6.564×).

3 months: CPU time / GPU time = **8.453×**; GPU has the faster measured median (8.453×).

6 months: CPU time / GPU time = **8.470×**; GPU has the faster measured median (8.470×).

| Months | Backend | Max sampled host RSS GiB | Max sampled GPU GiB | RSS/GPU max gap ms | Sampling errors RSS/GPU | Other GPU processes |
|---|---|---:|---:|---:|---:|---|
| 1 | CPU | 1.498 | 0.000 | 31.2 / 42.1 | 0 / 0 | True |
| 1 | GPU | 1.546 | 7.926 | 29.1 / 43.6 | 0 / 0 | True |
| 3 | CPU | 1.792 | 0.000 | 30.9 / 43.5 | 0 / 0 | True |
| 3 | GPU | 1.567 | 7.924 | 27.7 / 44.3 | 0 / 0 | True |
| 6 | CPU | 2.231 | 0.000 | 31.2 / 43.2 | 0 / 0 | True |
| 6 | GPU | 1.635 | 7.926 | 29.4 / 43.8 | 0 / 0 | True |

Memory values are sampled maxima across measured repetitions, not exact high-water marks. Requested intervals are 10 ms RSS and 20 ms GPU; missing readings and incomplete process-tree coverage remain in the JSON. Other GPU processes indicate possible contention. Device utilization is system-wide.

| Measured trial | Selected k | Cluster selection / test | CPU graph reference s | Communities | Duration test alert hours |
|---|---:|---|---:|---:|---:|
| months-1-measured-0-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | — | 7 | 0 |
| months-1-measured-0-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.133717 | 7 | 0 |
| months-1-measured-1-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.127724 | 7 | 0 |
| months-1-measured-1-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | — | 7 | 0 |
| months-1-measured-2-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | — | 7 | 0 |
| months-1-measured-2-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.128071 | 7 | 0 |
| months-3-measured-0-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | — | 7 | 0 |
| months-3-measured-0-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.167313 | 7 | 0 |
| months-3-measured-1-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | — | 7 | 0 |
| months-3-measured-1-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.165523 | 7 | 0 |
| months-3-measured-2-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | — | 7 | 0 |
| months-3-measured-2-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.164427 | 7 | 0 |
| months-6-measured-0-gpu (passed) | 8 | calibration_selected / evaluated | — | 7 | 550 |
| months-6-measured-0-cpu (passed) | 8 | calibration_selected / evaluated | 0.196644 | 7 | 550 |
| months-6-measured-1-gpu (passed) | 8 | calibration_selected / evaluated | — | 7 | 550 |
| months-6-measured-1-cpu (passed) | 8 | calibration_selected / evaluated | 0.198203 | 7 | 550 |
| months-6-measured-2-gpu (passed) | 8 | calibration_selected / evaluated | — | 7 | 550 |
| months-6-measured-2-cpu (passed) | 8 | calibration_selected / evaluated | 0.195205 | 7 | 550 |

Native GPU trials run all bulk preparation, model and graph stages on the GPU; CPU trials block CUDA package imports and hide CUDA devices. Independent CPU reference checks run after the stopwatch. Per-stage backend evidence and module audits are retained in the JSON. Short chronological prefixes lack later calibration/test data. 3 repetitions describe this experiment, not a universal winner or statistical significance.

Publication candidate: `months-6-measured-0-gpu`, selected by the first measured six-month GPU repetition rule, independently of runtime. It processed 20,332,093 source rows and 20,230,079 clustering-eligible trips. The local JSON retains its artifact location; no host paths are published here.

Candidate graph: 7 communities, 3 isolates; maximum PageRank absolute error versus CPU: 2.42861286636753e-17. Duration model: evaluated; calibration: calibrated_alert_budget; selected threshold: 4.0; test alert hours/incidents: 550/374.
