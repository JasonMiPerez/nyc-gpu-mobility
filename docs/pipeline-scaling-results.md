# Full-pipeline scaling report

Experiment `pipeline-eb6b177f59b1` · status: **passed**.
Expected measured repetitions per backend and size: 3. Warmups are excluded.

| Months | Source rows | Backend | Status | Repetitions | Median s | Q25–Q75 s |
|---|---:|---|---|---:|---:|---:|
| 1 | 2964624 | CPU | validated | 3 | 55.309418 | 55.213898–55.540010 |
| 1 | 2964624 | GPU | validated | 3 | 54.143779 | 54.041623–54.186978 |
| 3 | 9554778 | CPU | validated | 3 | 172.874315 | 172.629248–172.993530 |
| 3 | 9554778 | GPU | validated | 3 | 169.287782 | 168.405250–169.441125 |
| 6 | 20332093 | CPU | validated | 3 | 342.734061 | 342.092621–343.569425 |
| 6 | 20332093 | GPU | validated | 3 | 334.458998 | 332.545322–337.436392 |

1 months: CPU time / GPU time = **1.022×**; GPU has the faster measured median (1.022×).

3 months: CPU time / GPU time = **1.021×**; GPU has the faster measured median (1.021×).

6 months: CPU time / GPU time = **1.025×**; GPU has the faster measured median (1.025×).

| Months | Backend | Max sampled host RSS GiB | Max sampled GPU GiB | RSS/GPU max gap ms | Sampling errors RSS/GPU | Other GPU processes |
|---|---|---:|---:|---:|---:|---|
| 1 | CPU | 1.521 | 0.000 | 31.7 / 42.5 | 0 / 0 | True |
| 1 | GPU | 2.174 | 7.891 | 30.2 / 42.8 | 0 / 0 | True |
| 3 | CPU | 1.770 | 0.000 | 31.6 / 42.5 | 0 / 0 | True |
| 3 | GPU | 2.529 | 7.891 | 31.7 / 44.4 | 0 / 0 | True |
| 6 | CPU | 2.231 | 0.000 | 32.4 / 43.1 | 0 / 0 | True |
| 6 | GPU | 3.009 | 7.891 | 31.6 / 45.0 | 0 / 0 | True |

Memory values are sampled maxima across measured repetitions, not exact high-water marks. Requested intervals are 10 ms RSS and 20 ms GPU; missing readings and incomplete process-tree coverage remain in the JSON. Other GPU processes indicate possible contention. Device utilization is system-wide.

| Measured trial | Selected k | Cluster selection / test | CPU graph reference s | Communities | Duration test alert hours |
|---|---:|---|---:|---:|---:|
| months-1-measured-0-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.127968 | 7 | 0 |
| months-1-measured-0-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.128809 | 7 | 0 |
| months-1-measured-1-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.128346 | 7 | 0 |
| months-1-measured-1-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.128407 | 7 | 0 |
| months-1-measured-2-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.129489 | 7 | 0 |
| months-1-measured-2-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.129334 | 7 | 0 |
| months-3-measured-0-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.167138 | 7 | 0 |
| months-3-measured-0-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.216038 | 7 | 0 |
| months-3-measured-1-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.165016 | 7 | 0 |
| months-3-measured-1-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.215081 | 7 | 0 |
| months-3-measured-2-gpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.168087 | 7 | 0 |
| months-3-measured-2-cpu (passed) | 4 | exploratory_no_calibration / no_held_out_rows | 0.215178 | 7 | 0 |
| months-6-measured-0-gpu (passed) | 6 | calibration_selected / evaluated | 0.197109 | 7 | 550 |
| months-6-measured-0-cpu (passed) | 6 | calibration_selected / evaluated | 0.198496 | 7 | 550 |
| months-6-measured-1-gpu (passed) | 6 | calibration_selected / evaluated | 0.196207 | 7 | 550 |
| months-6-measured-1-cpu (passed) | 6 | calibration_selected / evaluated | 0.197657 | 7 | 550 |
| months-6-measured-2-gpu (passed) | 6 | calibration_selected / evaluated | 0.201802 | 7 | 550 |
| months-6-measured-2-cpu (passed) | 6 | calibration_selected / evaluated | 0.197785 | 7 | 550 |

GPU runs include the CPU graph reference as validation work. In CPU runs, this is the graph computation itself, not additional overhead. Short chronological prefixes lack later calibration/test data. 3 repetitions describe this experiment, not a universal winner or statistical significance.

Publication candidate: `months-6-measured-0-gpu`, selected by the first measured six-month GPU repetition rule, independently of runtime. It processed 20,332,093 source rows and 20,230,079 clustering-eligible trips. The local JSON retains its artifact location; no host paths are published here.

Candidate graph: 7 communities, 3 isolates; maximum PageRank absolute error versus CPU: 2.42861286636753e-17. Duration model: evaluated; calibration: calibrated_alert_budget; selected threshold: 4.0; test alert hours/incidents: 550/374.
