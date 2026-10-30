"""Public explanations retain model policy without publishing training identities."""

from mobility.dashboard.static_export import _model_summary


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
            "samples": {"record_ids": ["private-trip"]},
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
    assert public["anomalies"]["threshold"] == 4
    assert public["anomalies"]["anomalies_config"]["scale_floor"] == 5
    assert "baseline_sha256" not in public["anomalies"]
