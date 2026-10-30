# Shared implementation contract

The original specification lives one directory above this project. User decisions:
custom CUDA is required; NVIDIA server is primary; scale yellow data beyond six
months when useful and feasible; public website preferred, local fallback. Team
and semester assumptions do not apply.

Python package: `mobility` in `src/`. Configuration crosses module boundaries as
plain dictionaries. Paths passed to functions are pathlib Paths. Optional GPU
imports must be lazy, and missing GPU support must raise, never silently use CPU.

Canonical Parquet parts in `run/canonical/` retain `record_id`, `dataset_type`,
`source_month`, `pickup_local`, `dropoff_local`, `pickup_zone_id`, `dropoff_zone_id`,
`trip_distance_miles`, `duration_seconds`, `quality_flags`, `eligible_pickup`,
`eligible_dropoff`, `eligible_od`, `eligible_duration`, `eligible_ml`.
Null pickup times are retained. Times are naive NYC source wall times. Hours are
complete dated timestamps, not hour-of-day numbers. Finite negative distances are
excluded from distance features, not endpoint counts. DST-unsafe durations are
excluded from duration/ML. Out-of-source-month pickup rows stay canonical but
are excluded from pickup/OD/default ML views.

Published snapshot is an immutable directory. `manifest.json` includes snapshot_id,
created_at, data_kind (`real` or `synthetic`), study_start, study_end (exclusive),
sources, config, tables (name -> relative filename), row_counts, model metadata,
validation and backend. `artifacts/current.json` contains `snapshot_id` and `path`
relative to artifacts root. Root owns publication/integrity checks.

Tables (Parquet unless GeoJSON/JSON):
- `zone_hour`: dataset_type, zone_id, hour, pickup_count, dropoff_count,
  duration_sum_seconds, duration_count, distance_sum_miles, distance_count,
  pickup_coverage, dropoff_coverage, time_policy_version. Missing-source bins
  have null counts; observed bins with no records have zero. DST bins flagged.
- `od_hour`: dataset_type, pickup_zone_id, dropoff_zone_id, hour, trip_count,
  duration_sum_seconds, duration_count, distance_sum_miles, distance_count.
- `zones`: zone_id, zone, borough, longitude, latitude.
- `zones.geojson`: properties.zone_id, EPSG:4326 polygons.
- `quality_summary`: source_id, rule, affected_rows, total_rows, action.
- model modules write `cluster_summary`, `trip_plot_sample`, `anomaly_scores`,
  `anomaly_incidents` plus model metadata. They return table paths and metadata.
- `benchmark_runs`: experiment_id, repeat, backend, stage, timing_scope,
  input_rows, elapsed_s, validation, plus machine/config metadata where available.

Config defaults: data {dataset: yellow, year: 2024, months: [1..6]}, processing
{backend: cpu, batch_rows: 250000}, splits {train_start: 2024-01-01,
train_end: 2024-04-01, calibration_start: 2024-04-01,
calibration_end: 2024-05-01, test_start: 2024-05-01, test_end: 2024-07-01},
clustering {k_values: [4,6,8], training_max_rows: 200000, evaluation_max_rows: 2000,
seed:42}, anomalies {min_history:8, scale_floor:5.0, score_threshold:4.0,
absolute_threshold:20}, dashboard {max_flows:200,max_points:5000,max_export_rows:50000}.

Model API: `mobility.models.analysis.run_models(canonical_dir, zone_hour, config,
output_dir, backend='cpu') -> dict` with `tables` (name -> relative file), `models`
and `status`. Do not imply complete evaluation if history is too short; output
explicit statuses and retain available exploratory clustering.

Dashboard API: `mobility.dashboard.app.create_app(snapshot_dir: Path) -> Dash`.
Read only artifact tables, use Apply filters, bound map/scatter/export, support
explicit no-data/model-not-evaluated/performance-not-measured states. Query layer
must enforce path safety and artifact/filter capabilities.

GPU API: `mobility.backends.gpu.derive_features(frame, config)`,
`aggregate(frame) -> (zone_hour_part, od_hour_part)` using explicit cuDF.
Root sends pandas canonical rows to GPU when selected; describe CPU schema/time
policy stage separately and include transfer in end-to-end timings. Additional
GPU raw-reading support may be added without hiding CPU preprocessing.
Custom CUDA feature interface should accept epoch nanoseconds, distance,
validity masks, and precomputed time-policy masks; no timezone inference in CUDA.

No fabricated results, no dependency on notebooks, no mandatory GPU for serving.
