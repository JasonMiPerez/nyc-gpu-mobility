"""Public explanations retain model policy without publishing training identities."""

import pytest

from mobility.dashboard.static_export import _model_summary, processing_metadata


def test_public_model_policy_is_explained_without_training_details():
    policy = {
        "thresholds": {"max_distance_miles": 100},
        "model_eligible_rows": 90,
        "excluded_from_canonical_eligible_rows": 10,
    }
    models = {
        "clustering": {
            "backend": "cpu",
            "eligibility": policy,
            "feature_explanation": {"distance": "log(1 + miles)"},
            "samples": {
                "train": {
                    "record_ids": ["private-trip"],
                    "method": "smallest_splitmix64_source_row_v1",
                }
            },
            "initialization_policy": "seeded_source_row_priority_first_k_v1",
            "model_path": "/private/model.pkl",
        },
        "anomalies": {
            "threshold": 4,
            "anomalies_config": {"scale_floor": 5},
            "splits": {"test_start": "2024-05-01"},
            "baseline_sha256": "internal-hash",
        },
    }
    public = _model_summary(models)
    assert public["clustering"]["eligibility"] == policy
    assert public["clustering"]["backend"] == "cpu"
    assert "samples" not in public["clustering"]
    assert "model_path" not in public["clustering"]
    assert public["clustering"]["sampling_policy"] == "smallest_splitmix64_source_row_v1"
    assert public["clustering"]["initialization_policy"] == "seeded_source_row_priority_first_k_v1"
    assert public["anomalies"]["threshold"] == 4
    assert public["anomalies"]["anomalies_config"]["scale_floor"] == 5
    assert "baseline_sha256" not in public["anomalies"]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_native_backend_evidence_overrides_old_summary_defaults(backend):
    plan = {
        stage: {
            "backend": backend,
            "implementation": "recorded work",
            "rows": 100,
            "private_path": "/secret",
        }
        for stage in ["aggregation", "kmeans_fit", "volume_scoring", "duration_scoring", "graph"]
    }
    manifest = {
        "execution_mode": "native",
        "execution_plan": plan,
        "config": {"processing": {"backend": backend}},
        "models": {"anomalies": {"computation_backend": "CPU old summary"}},
    }
    result = processing_metadata(manifest)
    assert result["execution_mode"] == "native"
    assert all(result[key] == backend for key in result if key.endswith("_backend"))
    assert result["execution_plan"]["volume_scoring"] == {
        "backend": backend,
        "implementation": "recorded work",
        "rows": 100,
    }


def test_native_detector_fallback_uses_computation_evidence_not_aggregation():
    manifest = {
        "config": {"processing": {"backend": "gpu", "execution_mode": "native"}},
        "models": {
            "anomalies": {"computation_backend": "GPU cuDF/CuPy; bounded host reporting"},
            "duration_anomalies": {},
        },
    }
    result = processing_metadata(manifest)
    assert result["anomaly_backend"] == "gpu"
    assert result["duration_anomaly_backend"] == "unknown"


def test_old_hybrid_snapshot_preserves_cpu_anomalies_without_inventing_native_work():
    result = processing_metadata(
        {
            "backend": "gpu",
            "models": {"anomalies": {}, "duration_anomalies": {}, "clustering": {"backend": "gpu"}},
        }
    )
    assert result["execution_mode"] == "legacy"
    assert result["execution_plan"] == {}
    assert result["aggregation_backend"] == "gpu"
    assert result["clustering_backend"] == "gpu"
    assert result["anomaly_backend"] == result["duration_anomaly_backend"] == "cpu"
    assert processing_metadata({})["anomaly_backend"] == "unknown"
