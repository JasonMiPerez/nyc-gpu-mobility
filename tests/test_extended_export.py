"""Published comparisons and duration contexts preserve their scientific scope."""

import pandas as pd

from mobility.dashboard.static_export import _anomalies, _benchmark_payload


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
