"""Published comparisons and duration contexts preserve their scientific scope."""

import json

import pandas as pd

from mobility.dashboard.static_export import (
    _anomalies,
    _benchmark_payload,
    clustering_quality_rows,
    cpu_execution_evidence,
)


def test_pipeline_comparisons_match_protocol_and_exclude_warmup():
    class Repository:
        manifest = {}

    rows = pd.DataFrame(
        {
            "experiment_id": ["same"] * 5,
            "stage": ["production_pipeline"] * 5,
            "timing_scope": ["full_pipeline"] * 5,
            "input_rows": [1000] * 5,
            "backend": ["cpu", "gpu", "cpu", "gpu", "gpu"],
            "phase": ["measured", "measured", "warmup", "measured", "measured"],
            "validation": ["passed", "passed", "passed", "failed", "passed"],
            "elapsed_s": [10, 20, 1000, 0.01, 0.1],
            "protocol_hash": ["matched", "matched", "matched", "matched", "different"],
            "peak_gpu_memory_sampled_bytes": [None, 4000, None, None, None],
            "background_gpu_processes": [False] * 5,
        }
    )
    result = _benchmark_payload(Repository(), rows, pipeline=True)
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["speedup"] == 0.5
    assert result["comparisons"][0]["cpu_seconds"] == 10
    assert result["trials"][0]["peak_gpu_memory_sampled_bytes"] is None
    assert result["trials"][1]["peak_gpu_memory_sampled_bytes"] == 4000
    assert result["trials"][1]["background_gpu_processes"] is False


def test_duration_context_keeps_missing_hours_as_gaps():
    scores = pd.DataFrame(
        {
            "zone_id": [1, 1],
            "hour": pd.to_datetime(["2024-05-01 09:00", "2024-05-01 11:00"]),
            "observed": [40.5, 20.5],
            "baseline": [20.5, 20.5],
            "score": [10.0, 0.0],
            "is_alert": [True, False],
            "status": ["scored", "scored"],
            "split": ["test", "test"],
            "eligible_trip_count": [25, 25],
        }
    )

    class Repository:
        def has(self, name):
            return name == "duration_scores"

        def read(self, name, filters=None):
            assert name == "duration_scores"
            return scores

        def default_filters(self):
            return None

    result = _anomalies(Repository(), "duration")
    context = result["contexts"]["0"]
    missing = next(row for row in context if row["hour"].startswith("2024-05-01T10:"))
    assert missing["observed"] is None
    assert missing["baseline"] is None
    assert result["alerts"][0]["observed"] == 40.5
    assert result["alerts"][0]["eligible_trip_count"] == 25


def test_selected_cluster_quality_is_scoped_and_private_model_fields_stay_private():
    model = {
        "clustering": {
            "selected_k": 6,
            "status": "calibration_selected",
            "test_status": "evaluated",
            "candidates": [
                {"k": 4, "training_inertia_per_row": 9.9},
                {
                    "k": 6,
                    "training_rows": 200000,
                    "training_inertia_per_row": 0.8,
                    "calibration": {"rows": 2000, "silhouette": 0.4},
                },
            ],
            "test": {"rows": 2000, "silhouette": 0.35, "mean_squared_center_distance": 0.9},
            "samples": {"train": {"record_ids": ["private-record"]}},
            "parameters_file": "/private/model.json",
        },
        "other_model": {"path": "/private/another-model"},
    }
    base = {
        "phase": "measured",
        "backend": "gpu",
        "months": 6,
        "repeat": 0,
        "experiment_id": "example",
        "protocol_hash": "same",
        "stage": "native_production_pipeline",
        "timing_scope": "native_raw_files",
        "input_rows": 100,
        "elapsed_s": 2,
        "validation": "passed",
        "model_quality_json": json.dumps(model),
    }
    trials = pd.DataFrame(
        [
            base,
            base | {"phase": "warmup"},
            base | {"repeat": 1, "validation": "failed", "model_quality_json": "invalid json"},
        ]
    )
    result = _benchmark_payload(type("Repository", (), {"manifest": {}})(), trials, pipeline=True)
    quality = result["clustering_quality"]
    assert len(quality) == 2  # Retain the failed measured trial; discard the warmup.
    assert quality[0]["selected_k"] == 6
    assert quality[0]["training_inertia_per_row"] == 0.8  # Selected candidate, not first/minimum.
    assert quality[0]["training_rows"] == 200000
    assert quality[0]["calibration_silhouette"] == 0.4
    assert quality[0]["test_silhouette"] == 0.35
    assert quality[0]["test_mean_squared_center_distance"] == 0.9
    assert quality[0]["test_rows"] == 2000
    assert quality[1]["validation"] == "failed"
    assert quality[1]["diagnostics_status"] == "not_recorded"
    assert quality[1]["test_silhouette"] is None
    encoded = json.dumps(result, allow_nan=False)
    assert "private-record" not in encoded and "/private/" not in encoded
    assert "model_quality_json" not in encoded


def test_missing_held_out_quality_and_historical_metadata_are_not_zero_scores():
    model = {
        "clustering": {
            "selected_k": 4,
            "status": "exploratory_no_calibration",
            "test_status": "no_held_out_rows",
            "test": {"rows": 0, "silhouette": None},
            "candidates": [
                {
                    "k": 4,
                    "training_inertia_per_row": float("inf"),
                    "calibration": {"rows": 0, "silhouette": None},
                }
            ],
        }
    }
    result = clustering_quality_rows(
        pd.DataFrame(
            [
                {"phase": "measured", "months": 1, "model_quality_json": json.dumps(model)},
                {"phase": "measured", "months": 3, "selected_k": 4},
            ]
        )
    )
    assert result[0]["selection_status"] == "exploratory_no_calibration"
    assert result[0]["test_status"] == "no_held_out_rows"
    assert result[0]["test_rows"] == 0
    assert result[0]["test_silhouette"] is None
    assert result[0]["training_inertia_per_row"] is None
    assert result[1]["diagnostics_status"] == "not_recorded"
    assert result[1]["test_rows"] is None


def test_cpu_isolation_evidence_requires_recorded_guards_and_observed_memory():
    base = {
        "phase": "measured",
        "backend": "cpu",
        "execution_mode": "native",
        "cpu_import_guard_enabled": True,
        "loaded_gpu_modules_json": "[]",
        "gpu_samples": 10,
        "peak_gpu_memory_sampled_bytes": 0,
    }
    result = cpu_execution_evidence(
        pd.DataFrame(
            [
                base,
                base | {"peak_gpu_memory_sampled_bytes": None},
                base | {"gpu_samples": 0},  # An unobserved zero is unavailable.
                base
                | {"loaded_gpu_modules_json": '["cupy"]', "peak_gpu_memory_sampled_bytes": 100},
                base | {"phase": "warmup"},
                base | {"execution_mode": "legacy"},
                base | {"backend": "gpu"},
            ]
        )
    )
    assert result == {
        "measured_native_cpu_trials": 4,
        "guard_and_no_cuda_imports_confirmed": 3,
        "zero_sampled_gpu_memory_trials": 1,
        "nonzero_sampled_gpu_memory_trials": 1,
        "gpu_memory_unavailable_trials": 2,
    }
