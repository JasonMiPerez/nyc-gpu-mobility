# Validation and release checks

Run tests from the project directory in the project environment:

```sh
.venv/bin/python -m pytest -q
```

CPU checks require no network downloads. Synthetic fixtures are marked as
synthetic and are not performance evidence. GPU checks must run in the recorded
server environment; a local skip is not evidence that GPU code passed.

On October 30, 2026, the following targeted CPU suite passed **61 tests** in the
local Python 3.13 environment:

```sh
.venv/bin/python -m pytest tests/test_models.py tests/test_quality.py \
  tests/test_pipeline.py tests/test_cluster_benchmarks.py -q
```

That count covers this named subset, not the complete project or GPU execution.

## What the tests check

| Area | Evidence checked |
| --- | --- |
| Source integrity | Required Parquet fields/types, verified re-use without a network fetch, checksum corruption, approved source URL boundaries |
| Record semantics | Independently valid endpoints, missing timestamp with known OD zones, source-month exclusions, preserved duplicate-looking records and stable identities |
| Numeric quality | Null/nonfinite/negative distance, zero distance, invalid integer zones, zero/negative duration, missing optional fields |
| Time | Midnight/month crossing, spring/fall DST uncertainty, separate endpoint hours, rejection of unexpected timezone-aware sources |
| Aggregation | Exact count conservation, chunk equivalence, weighted means, explicit missing coverage, empty and all-invalid sources |
| Models | Training-only scaler and baseline, no test-based model selection, stable bounded sampling, minimum historical support, conjunctive thresholds, incident boundaries |
| Synthetic evaluation | Original observations unchanged, deterministic perturbations, separate unmatched background counts, no real-world accuracy claim |
| Publication | Failed runs and copy errors preserve the active snapshot; traversal and symlink escapes are rejected; published checksums match files |
| Full fixture integration | Six source months run through canonical data, aggregates, models, validation, and publication with exact reconciled counts |
| KMeans benchmark setup | Actual fit/prediction row counts, matching frozen centers, training-only preprocessing, and no future-data substitute |

The six-month integration fixture has 180 trips and reconciles to 180 pickups,
180 drop-offs, and 180 OD records. It exercises model and publication behavior;
it is too small to establish useful predictive quality or runtime scaling.

## Inspect a completed run

```sh
.venv/bin/mobility validate --run artifacts/runs/RUN_ID
```

Read the run's `manifest.json`, quality summary, and model metadata before
publishing. A complete run records source hashes, resolved configuration,
environment, actual row counts, validation results, and model statuses. A
January-only run should explicitly show exploratory clustering and unavailable
multi-month anomaly evaluation.

For benchmarks, inspect `benchmark_runs.parquet` and the environment record.
Keep failed trials, check that scopes and input rows match, and check the recorded
CPU thread limit. No GPU speedup is established by an import check, by a synthetic
fixture, or by a timer that measured only kernel launch.

## GPU verification

Run the GPU smoke/parity tests and benchmark commands documented in `gpu.md` on
the actual server. Verify matched KMeans prediction with frozen centers and
review fit objective/ARI diagnostics. The custom kernel additionally needs
sanitizer and profiling evidence under the installed toolkit. Preserve the raw
results and hardware/software inventory with the experiment.

## Before sharing a snapshot

1. Confirm source scope, data-kind label, and model evaluation status.
2. Run aggregate validation and applicable CPU/GPU checks.
3. Confirm missing coverage, DST exclusions, and partial drop-off context are
   visible and distinct from zero counts.
4. Check every dashboard page and the applied filters, including an empty
   selection and a bounded export.
5. Confirm static cluster summaries and sampled distributions are labeled.
6. Verify the public deployment reads only the intended published snapshot and
   can serve it without access to the GPU processing environment.
7. Preserve a prior snapshot and verify the local fallback before changing the
   active published version.

Manual incident review, independent corroboration, browser/accessibility review,
and deployment verification are separate checks. Unit-test success does not
automatically complete them. Record what was actually checked for each release.
