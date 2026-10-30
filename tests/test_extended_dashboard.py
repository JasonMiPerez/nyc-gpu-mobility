"""Published duration, graph, and full-pipeline results remain distinct and honest."""

import json

import pandas as pd
import pytest
from plotly.utils import PlotlyJSONEncoder

from mobility.dashboard import charts
from mobility.dashboard.app import render_page
from mobility.serving.repository import Filters, SnapshotRepository


def component_text(value):
    if isinstance(value, (list, tuple)):
        return " ".join(component_text(item) for item in value)
    if hasattr(value, "children"):
        return component_text(value.children)
    return str(value) if isinstance(value, (str, int, float)) else ""


@pytest.fixture
def extended_snapshot(tmp_path):
    snapshot = tmp_path / "extended"
    snapshot.mkdir()
    zones = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "zone": ["North", "South"],
            "borough": ["Example"] * 2,
            "longitude": [-73.98, -74.0],
            "latitude": [40.75, 40.72],
        }
    )
    hours = pd.DataFrame(
        {
            "dataset_type": ["yellow"] * 2,
            "zone_id": [1, 2],
            "hour": pd.to_datetime(["2024-05-06 10:00"] * 2),
            "pickup_count": [30, 20],
            "dropoff_count": [20, 30],
            "duration_count": [30, 20],
            "duration_sum_seconds": [18000, 12000],
            "pickup_coverage": ["observed"] * 2,
            "dropoff_coverage": ["observed"] * 2,
        }
    )
    od = pd.DataFrame(
        {
            "dataset_type": ["yellow"] * 2,
            "pickup_zone_id": [1, 2],
            "dropoff_zone_id": [2, 1],
            "hour": pd.to_datetime(["2024-05-06 10:00"] * 2),
            "trip_count": [30, 20],
            "duration_count": [30, 20],
            "duration_sum_seconds": [18000, 12000],
        }
    )
    duration = pd.DataFrame(
        {
            "dataset_type": ["yellow"] * 3,
            "zone_id": [1] * 3,
            "hour": pd.to_datetime(["2024-05-06 10:00", "2024-05-06 12:00", "2024-05-13 10:00"]),
            "observed": [40.0, 25.0, 60.0],
            "baseline": [10.0] * 3,
            "scale": [3.0] * 3,
            "score": [10.0, 5.0, None],
            "deviation": [30.0, 15.0, 50.0],
            "history_count": [9] * 3,
            "threshold": [5.0] * 3,
            "absolute_threshold": [7.0] * 3,
            "eligible_trip_count": [30, 25, 10],
            "is_alert": [True, True, False],
            "status": ["scored", "scored", "insufficient_trips"],
            "split": ["test"] * 3,
            "metric": ["mean_duration_minutes"] * 3,
            "unit": ["minutes"] * 3,
        }
    )
    incidents = pd.DataFrame(
        {
            "dataset_type": ["yellow"],
            "zone_id": [1],
            "split": ["test"],
            "start_hour": pd.to_datetime(["2024-05-06 10:00"]),
            "end_hour": pd.to_datetime(["2024-05-06 10:00"]),
            "affected_hours": [1],
            "peak_score": [10.0],
            "total_deviation": [30.0],
            "label": ["unresolved"],
        }
    )
    nodes = zones.assign(
        pagerank=[0.7, 0.3],
        community=[5, 9],
        in_degree=1,
        out_degree=1,
        degree=2,
        in_trip_strength=[20, 30],
        out_trip_strength=[30, 20],
        self_trip_count=[2, 3],
        is_isolated=False,
    )
    trial_rows = []
    for backend, seconds in [("cpu", 20.0), ("gpu", 10.0)]:
        for repeat in range(2):
            trial_rows.append(
                {
                    "experiment_id": "same-host",
                    "protocol_hash": "protocol-a",
                    "backend": backend,
                    "stage": "full_pipeline",
                    "timing_scope": "raw_files_to_models_and_artifacts_with_reconciliation",
                    "input_rows": 1000,
                    "months": 1,
                    "phase": "measured",
                    "repeat": repeat,
                    "elapsed_s": seconds,
                    "validation": "passed",
                    "peak_host_rss_sampled_bytes": 2 * 1024**3,
                    "peak_gpu_memory_sampled_bytes": 1024**3 if backend == "gpu" else None,
                    "rss_sample_interval_s": 0.01,
                    "gpu_sample_interval_s": 0.02,
                }
            )
    trial_rows.append(
        trial_rows[2] | {"phase": "warmup", "elapsed_s": 0.01, "validation": "warmup"}
    )
    trial_rows.append(trial_rows[0] | {"elapsed_s": 0.001, "validation": "failed"})
    # This CPU row must never be paired with another protocol's GPU result.
    trial_rows.append(trial_rows[0] | {"protocol_hash": "protocol-b", "elapsed_s": 1.0})
    tables = {}
    for name, frame in {
        "zones": zones,
        "zone_hour": hours,
        "od_hour": od,
        "duration_scores": duration,
        "duration_incidents": incidents,
        "graph_nodes": nodes,
        "pipeline_benchmark_runs": pd.DataFrame(trial_rows),
    }.items():
        tables[name] = f"{name}.parquet"
        frame.to_parquet(snapshot / tables[name], index=False)
    splits = {
        "train_start": "2024-01-01",
        "train_end": "2024-04-01",
        "calibration_start": "2024-04-01",
        "calibration_end": "2024-05-01",
        "test_start": "2024-05-01",
        "test_end": "2024-07-01",
    }
    manifest = {
        "snapshot_id": "extended-test",
        "data_kind": "synthetic",
        "backend": "gpu",
        "study_start": "2024-01-01",
        "study_end": "2024-07-01",
        "tables": tables,
        "config": {"data": {"dataset": "yellow"}, "splits": splits},
        "models": {
            "clustering": {"backend": "gpu"},
            "graph": {
                "backend": "gpu",
                "parameters": {"min_trips_per_edge": 10},
                "preparation_backend": "CPU pandas for edge preparation",
            },
            "duration_anomalies": {
                "computation_backend": "CPU pandas/numpy",
                "splits": splits,
                "duration_anomalies_config": {
                    "min_history": 9,
                    "min_trips_per_hour": 25,
                    "scale_floor_minutes": 3.0,
                    "absolute_threshold_minutes": 7.0,
                },
                "calibration": {"selected_threshold": 5.0},
            },
        },
        "pipeline_benchmark_environment": {"hostname": "shared-test-host"},
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    return snapshot


def test_duration_panels_use_recorded_units_support_and_separate_results(extended_snapshot):
    repo = SnapshotRepository(extended_snapshot)
    panels = render_page(repo, "anomalies", repo.default_filters())
    text = component_text(panels)
    assert "Unusual trip durations" in text
    assert "at least 25 eligible trips" in text
    assert "at least 9 comparable training" in text
    assert "3.0 minutes" in text and "7.0 minutes" in text
    assert "observed mean minutes" in text and "eligible trip count" in text
    assert "Unreviewed" in text and "not total passenger delay" in text
    assert "2024-04-01 to 2024-05-01" in text
    assert "CPU pandas/numpy" in text
    json.dumps(panels, cls=PlotlyJSONEncoder)


def test_duration_context_keeps_missing_hour_gaps_and_minute_axis(extended_snapshot):
    repo = SnapshotRepository(extended_snapshot)
    data = repo.read("duration_scores")
    figure = charts.anomaly_context(data, 1, "2024-05-06 10:00", "mean_duration_minutes")
    assert figure.layout.yaxis.title.text == "Mean trip duration (minutes)"
    assert figure.data[0].name == "Mean eligible trip duration"
    assert len(figure.data[0].x) == 3
    assert pd.isna(figure.data[0].y[1])
    assert figure.data[0].connectgaps is False
    assert "Eligible trips" in figure.data[0].hovertemplate


def test_graph_maps_treat_community_numbers_as_categories(extended_snapshot):
    repo = SnapshotRepository(extended_snapshot)
    figure = charts.graph_zone_map(repo.read("graph_nodes"), repo.zones(), None, "community")
    assert {trace.name for trace in figure.data} == {"Group 5", "Group 9"}
    assert all(isinstance(trace.marker.color, str) for trace in figure.data)
    assert figure.layout.coloraxis.colorbar.title.text is None
    filtered = Filters("2024-05-06", "2024-05-07", "yellow", (1,))
    panels = render_page(repo, "zones", filtered, destinations=(2,))
    text = component_text(panels)
    assert "do not recompute these graph results" in text
    assert "South" in text and "North" in text
    assert "Louvain communities" in text and "PageRank across zones" in text
    assert "Community numbers have no ranking" in text
    json.dumps(panels, cls=PlotlyJSONEncoder)


def test_pipeline_ratio_uses_valid_measured_matching_protocols(extended_snapshot):
    repo = SnapshotRepository(extended_snapshot)
    panels = render_page(repo, "performance", repo.default_filters())
    text = component_text(panels)
    assert "Full raw-file pipeline benchmark" in text
    assert "GPU 2.000× faster" in text
    assert "CPU 10.000× faster" not in text
    assert "GPU 2000.000× faster" not in text
    assert "shared-test-host" in text
    assert "not exact peaks" in text
    assert "Missing GPU memory is unavailable, not zero" in text
    assert "largest sampled host RSS GiB" in text
    json.dumps(panels, cls=PlotlyJSONEncoder)


def test_methods_follow_snapshot_capabilities_and_backend(extended_snapshot):
    repo = SnapshotRepository(extended_snapshot)
    text = component_text(render_page(repo, "methods", repo.default_filters()))
    assert "processing backend is gpu" in text
    assert "Graph analysis backend: gpu" in text
    assert "weighted PageRank and Louvain" in text
    assert "Duration computation: CPU pandas/numpy" in text
    assert "six-month published analysis was built on CPU" not in text
    assert "cuGraph algorithms are not implemented" not in text
    assert "does not yet detect unusual trip durations" not in text


def test_older_snapshots_keep_explicit_unpublished_feature_states(extended_snapshot):
    path = extended_snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in ["duration_scores", "duration_incidents", "graph_nodes", "pipeline_benchmark_runs"]:
        del manifest["tables"][name]
    manifest["models"] = {}
    path.write_text(json.dumps(manifest))
    repo = SnapshotRepository(extended_snapshot)
    anomaly_text = component_text(render_page(repo, "anomalies", repo.default_filters()))
    assert "Duration anomaly results have not been published" in anomaly_text
    zone_text = component_text(render_page(repo, "zones", repo.default_filters()))
    assert "PageRank and community results have not been published" in zone_text
    benchmark_text = component_text(render_page(repo, "performance", repo.default_filters()))
    assert "No full raw-file pipeline trials have been published" in benchmark_text
