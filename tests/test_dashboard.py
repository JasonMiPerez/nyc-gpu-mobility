"""Serving contracts and real Dash callbacks, using tiny synthetic artifacts."""

import io
import json

import pandas as pd
import pytest
from plotly.utils import PlotlyJSONEncoder

from mobility.dashboard.app import create_app, render_page
from mobility.serving.repository import Filters, QueryError, SnapshotRepository


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    hours = pd.to_datetime(
        [
            "2024-01-01 10:00",
            "2024-01-01 10:00",
            "2024-01-02 10:00",
            "2024-01-02 10:00",
            "2024-01-03 10:00",
            "2024-01-03 10:00",
        ]
    )
    zone_hour = pd.DataFrame(
        {
            "dataset_type": ["yellow"] * 6,
            "zone_id": [1, 2, 1, 2, 1, 2],
            "hour": hours,
            "pickup_count": [10, 20, 12, 0, None, None],
            "dropoff_count": [8, 22, 9, 3, None, None],
            "duration_sum_seconds": [6000, 12000, 7200, 0, None, None],
            "duration_count": [10, 20, 12, 0, None, None],
            "distance_sum_miles": [20, 40, 24, 0, None, None],
            "distance_count": [10, 20, 12, 0, None, None],
            "pickup_coverage": ["observed"] * 4 + ["missing_source"] * 2,
            "dropoff_coverage": ["observed"] * 4 + ["missing_source"] * 2,
            "time_policy_version": ["nyc_local_wall_time_v1"] * 6,
        }
    )
    od_hour = pd.DataFrame(
        {
            "dataset_type": ["yellow"] * 3,
            "pickup_zone_id": [1, 1, 2],
            "dropoff_zone_id": [2, 2, 1],
            "hour": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-01 10:00"]),
            "trip_count": [10, 12, 20],
            "duration_sum_seconds": [6000, 7200, 12000],
            "duration_count": [10, 12, 20],
        }
    )
    zones = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "zone": ["Test north", "Test south"],
            "borough": ["Test", "Test"],
            "latitude": [40.75, 40.72],
            "longitude": [-73.98, -74.00],
        }
    )
    sample = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "dataset_type": ["yellow", "yellow"],
            "pickup_local": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"]),
            "pickup_zone_id": [1, 2],
            "dropoff_zone_id": [2, 1],
            "trip_distance_miles": [2.0, 3.0],
            "duration_seconds": [600.0, 1200.0],
            "cluster": [0, 1],
            "model_id": ["test"] * 2,
            "inclusion_fraction": [0.1, 0.1],
        }
    )
    summary = pd.DataFrame(
        {
            "model_id": ["test", "test"],
            "cluster": [0, 1],
            "trip_count": [10, 20],
            "share": [1 / 3, 2 / 3],
            "mean_distance_miles": [2.0, 3.0],
            "mean_duration_minutes": [10.0, 20.0],
            "scoring_start": ["2024-01-01"] * 2,
            "scoring_end": ["2024-01-04"] * 2,
        }
    )
    scores = pd.DataFrame(
        {
            "model_id": ["test"] * 2,
            "dataset_type": ["yellow"] * 2,
            "zone_id": [1, 2],
            "hour": pd.to_datetime(["2024-01-02 10:00"] * 2),
            "observed": [12, 0],
            "baseline": [2.0, 3.0],
            "score": [5.0, -0.5],
            "scale": [2.0, 6.0],
            "history_count": [8, 8],
            "threshold": [4.0, 4.0],
            "absolute_threshold": [5, 5],
            "is_alert": [True, False],
            "direction": ["high", "low"],
            "status": ["scored"] * 2,
            "split": ["test"] * 2,
        }
    )
    tables = {}
    for name, data in {
        "zone_hour": zone_hour,
        "od_hour": od_hour,
        "zones": zones,
        "trip_plot_sample": sample,
        "cluster_summary": summary,
        "anomaly_scores": scores,
    }.items():
        data.to_parquet(root / f"{name}.parquet", index=False)
        tables[name] = f"{name}.parquet"
    manifest = {
        "snapshot_id": "test-snapshot",
        "data_kind": "synthetic",
        "study_start": "2024-01-01",
        "study_end": "2024-01-04",
        "tables": tables,
        "sources": [],
        "config": {"data": {"dataset": "yellow"}, "dashboard": {"max_export_rows": 3}},
        "models": {},
        "validation": {"status": "passed"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_half_open_date_and_zone_filter(snapshot):
    repo = SnapshotRepository(snapshot)
    filters = Filters.from_dict({"start": "2024-01-01", "end": "2024-01-02", "zones": [1]})
    data = repo.read("zone_hour", filters)
    assert data.pickup_count.tolist() == [10]
    assert data.zone_id.tolist() == [1]


def test_missing_source_counts_are_not_replaced_with_zero(snapshot):
    repo = SnapshotRepository(snapshot)
    data = repo.read("zone_hour", repo.default_filters())
    assert data.loc[data.pickup_coverage.eq("missing_source"), "pickup_count"].isna().all()
    assert (
        data.loc[
            (data.zone_id == 2) & data.hour.eq(pd.Timestamp("2024-01-02 10:00")), "pickup_count"
        ].iloc[0]
        == 0
    )


def test_cached_frames_do_not_share_mutable_user_state(snapshot):
    repo = SnapshotRepository(snapshot)
    data = repo.read("zone_hour", repo.default_filters())
    data.loc[:, "pickup_count"] = 9999
    fresh = repo.read("zone_hour", repo.default_filters())
    assert fresh.pickup_count.iloc[0] == 10


def test_od_reduces_counts_and_uses_weighted_duration(snapshot):
    repo = SnapshotRepository(snapshot)
    data = repo.od_totals(repo.default_filters(), [2])
    assert len(data) == 1
    assert data.trip_count.iloc[0] == 22
    assert data.mean_duration_minutes.iloc[0] == 10


def test_export_obeys_filter_and_declares_truncation(snapshot):
    repo = SnapshotRepository(snapshot)
    csv, count, truncated = repo.export_zone_hours(repo.default_filters())
    result = pd.read_csv(io.StringIO(csv))
    assert count == 3 and truncated
    assert len(result) == 3
    assert result.export_truncated.all()
    csv, count, truncated = repo.export_zone_hours(
        Filters.from_dict({"start": "2024-01-01", "end": "2024-01-02", "zones": [2]})
    )
    result = pd.read_csv(io.StringIO(csv))
    assert count == 1 and not truncated
    assert result.zone_id.tolist() == [2]


def test_raw_paths_and_unsupported_filters_are_rejected(snapshot):
    repo = SnapshotRepository(snapshot)
    with pytest.raises(QueryError):
        repo.read("canonical")
    with pytest.raises(QueryError):
        repo.read("cluster_summary", repo.default_filters())
    with pytest.raises(QueryError):
        repo._path("../outside.parquet")
    with pytest.raises(QueryError):
        repo._path("/etc/passwd")


def test_snapshot_symlink_cannot_escape(snapshot, tmp_path):
    external = tmp_path / "outside.parquet"
    pd.DataFrame({"secret": [1]}).to_parquet(external)
    (snapshot / "escaped.parquet").symlink_to(external)
    repo = SnapshotRepository(snapshot)
    with pytest.raises(QueryError):
        repo._path("escaped.parquet")


def test_partitioned_od_dataset_can_be_queried(snapshot):
    original = pd.read_parquet(snapshot / "od_hour.parquet")
    directory = snapshot / "od_hour"
    directory.mkdir()
    original.iloc[:1].to_parquet(directory / "january.parquet", index=False)
    original.iloc[1:].to_parquet(directory / "february.parquet", index=False)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["tables"]["od_hour"] = "od_hour"
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    repo = SnapshotRepository(snapshot)
    assert repo.od_totals(repo.default_filters()).trip_count.sum() == 42
    (directory / "unsafe.txt").write_text("not a published Parquet table")
    with pytest.raises(QueryError):
        SnapshotRepository(snapshot)


@pytest.mark.parametrize(
    "page", ["overview", "patterns", "zones", "anomalies", "performance", "methods"]
)
def test_each_page_renders_real_components(snapshot, page):
    repo = SnapshotRepository(snapshot)
    components = render_page(repo, page, repo.default_filters())
    serialized = json.dumps(components, cls=PlotlyJSONEncoder)
    assert serialized
    if page == "performance":
        assert "No benchmark measurements" in serialized


def test_dash_routes_and_applied_filter_callback(snapshot):
    app = create_app(snapshot)
    client = app.server.test_client()
    for route in ["/", "/_dash-layout", "/_dash-dependencies", "/assets/style.css"]:
        assert client.get(route).status_code == 200
    assert client.get("/healthz").json["snapshot_id"] == "test-snapshot"
    callback_key = next(key for key in app.callback_map if "page-content.children" in key)
    result = client.post(
        "/_dash-update-component",
        json={
            "output": callback_key,
            "outputs": [
                {"id": "page-content", "property": "children"},
                {"id": "page-title", "property": "children"},
                {"id": "page-subtitle", "property": "children"},
            ],
            "inputs": [
                {"id": name, "property": prop, "value": value}
                for name, prop, value in [
                    ("page", "value", "overview"),
                    (
                        "applied-filters",
                        "data",
                        {
                            "start": "2024-01-01",
                            "end": "2024-01-02",
                            "dataset": "yellow",
                            "zones": [1],
                        },
                    ),
                    ("aggregation", "value", "day"),
                    ("destinations", "value", []),
                    ("map-metric", "value", "pickup_count"),
                    ("map-tiles", "value", []),
                    ("selected-cluster", "value", "all"),
                    ("timing-scope", "value", "all"),
                    ("benchmark-backend", "value", "all"),
                ]
            ],
            "state": [],
            "changedPropIds": ["applied-filters.data"],
        },
    )
    assert result.status_code == 200
    content = result.json["response"]["page-content"]["children"]
    assert "could not be loaded" not in json.dumps(content)
    assert result.json["response"]["page-title"]["children"] == "Overview"


def test_invalid_date_range_fails_before_scan(snapshot):
    with pytest.raises(QueryError):
        Filters.from_dict({"start": "2024-01-03", "end": "2024-01-02"})


def test_empty_null_typed_model_tables_remain_queryable(snapshot):
    empty = pd.DataFrame(columns=["pickup_local", "pickup_zone_id", "dataset_type"])
    empty.to_parquet(snapshot / "trip_plot_sample.parquet", index=False)
    repo = SnapshotRepository(snapshot)
    assert repo.read("trip_plot_sample", repo.default_filters()).empty


def test_sample_scatter_accepts_nullable_zone_ids(snapshot):
    from mobility.dashboard.charts import scatter_sample

    sample = pd.read_parquet(snapshot / "trip_plot_sample.parquet")
    sample["pickup_zone_id"] = sample.pickup_zone_id.astype("Int32")
    sample["dropoff_zone_id"] = sample.dropoff_zone_id.astype("Int32")
    assert len(scatter_sample(sample, 5000).data) == 2


def test_public_export_is_bounded_and_omits_source_paths(snapshot, tmp_path):
    from mobility.dashboard.static_export import export_public

    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["sources"] = [
        {
            "source_id": "official",
            "url": "https://example.org/test",
            "local_path": "/private/secret/path",
        }
    ]
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    destination = export_public(snapshot, tmp_path / "public")
    assert (destination / "index.html").is_file()
    assert (destination / "plotly.min.js").stat().st_size > 1_000_000
    summary = json.loads((destination / "data/summary.json").read_text())
    assert sum(row["pickups"] or 0 for row in summary["zone_days"]) == 42
    patterns = json.loads((destination / "data/patterns.json").read_text())
    assert all("record_id" not in row for row in patterns["samples"])
    assert "/private/secret/path" not in (destination / "data/methods.json").read_text()
    assert not (destination / "canonical").exists()
    export_manifest = json.loads((destination / "export-manifest.json").read_text())
    assert export_manifest["json_bytes"] < 50_000_000


def test_public_export_refuses_failed_snapshot(snapshot, tmp_path):
    from mobility.dashboard.static_export import export_public

    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["validation"] = {"status": "failed"}
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="passed validation"):
        export_public(snapshot, tmp_path / "public")


def test_public_benchmarks_pair_actual_backend_names_and_redact_paths(snapshot):
    from mobility.dashboard.static_export import _benchmark_payload

    repo = SnapshotRepository(snapshot)
    repo.manifest["benchmark_environment"] = {
        "hostname": "private-server",
        "platform": "Linux",
        "cpu_logical_cores": 8,
        "threadpools": [{"num_threads": 4, "filepath": "/private/library.so"}],
    }
    trials = pd.DataFrame(
        {
            "experiment_id": ["same-workload"] * 3,
            "repeat": [0] * 3,
            "backend": ["cpu", "gpu_library", "gpu_custom"],
            "stage": ["features"] * 3,
            "timing_scope": ["loaded_compute"] * 3,
            "input_rows": [1000] * 3,
            "elapsed_s": [0.1, 0.05, 0.02],
            "validation": ["passed"] * 3,
        }
    )
    payload = _benchmark_payload(repo, trials)
    assert {row["gpu_backend"] for row in payload["comparisons"]} == {"gpu_library", "gpu_custom"}
    assert {row["speedup"] for row in payload["comparisons"]} == {2.0, 5.0}
    assert "private-server" not in json.dumps(payload)
    assert "/private/library.so" not in json.dumps(payload)


def test_histogram_filters_dates_but_rejects_zone_filter(snapshot):
    data = pd.DataFrame(
        {
            "dataset_type": ["yellow", "yellow"],
            "pickup_date": ["2024-01-01", "2024-01-02"],
            "feature": ["distance_miles"] * 2,
            "bin_index": [0, 0],
            "bin_label": ["0–1", "0–1"],
            "count": [10, 20],
        }
    )
    data.to_parquet(snapshot / "trip_histogram.parquet", index=False)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["tables"]["trip_histogram"] = "trip_histogram.parquet"
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    repo = SnapshotRepository(snapshot)
    result = repo.read("trip_histogram", Filters("2024-01-01", "2024-01-02"))
    assert result["count"].sum() == 10
    with pytest.raises(QueryError):
        repo.read("trip_histogram", Filters("2024-01-01", "2024-01-02", zones=(1,)))
