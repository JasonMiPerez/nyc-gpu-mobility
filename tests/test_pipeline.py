"""Local fixture integration, integrity failures, and atomic publication."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobility.aggregate import aggregate_cpu
from mobility.config import ProjectConfig, load_config
from mobility.data.acquire import BASE, fetch_file, inspect_parquet
from mobility.data.quality import normalize_yellow
from mobility.io import atomic_json, read_json, sha256_file
from mobility.pipeline import (
    add_coverage,
    current_snapshot,
    publish_run,
    run_pipeline,
    source_list,
    validate_outputs,
)


def raw_fixture(month=1, rows=30):
    pickup = pd.date_range(pd.Timestamp(2024, month, 1), periods=rows, freq="h")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(np.arange(rows) % 20 + 5, unit="min"),
            "PULocationID": np.ones(rows, dtype=np.int32),
            "DOLocationID": np.full(rows, 2, dtype=np.int32),
            "trip_distance": 1.0 + (np.arange(rows) % 10) / 2,
        }
    )


def install_source(root: Path, raw: pd.DataFrame, month=1):
    path = root / "data" / "raw" / "yellow" / f"yellow_tripdata_2024-{month:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(path, index=False)
    metadata = {
        "sha256": sha256_file(path),
        "source_id": f"fixture-{month}",
        "nominal_month": f"2024-{month:02d}",
        "dataset_type": "yellow",
        "url": f"{BASE}/trip-data/{path.name}",
        "data_kind": "synthetic",
        "local_path": str(path),
        "byte_count": path.stat().st_size,
        "validation_status": "valid",
        **inspect_parquet(path),
    }
    atomic_json(path.with_suffix(".parquet.json"), metadata)
    return path, metadata


def install_zones(root: Path):
    directory = root / "data" / "zones"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "zone_id": [1, 2],
            "zone": ["Fixture A", "Fixture B"],
            "borough": ["Test", "Test"],
            "longitude": [-74.0, -73.99],
            "latitude": [40.7, 40.71],
        }
    ).to_parquet(directory / "zones.parquet", index=False)
    (directory / "zones.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )


def config_for(months=(1,), **processing):
    config = load_config()
    config["data"]["months"] = list(months)
    config["processing"].update({"batch_rows": 100, **processing})
    config["clustering"].update(
        {"k_values": [2, 3], "training_max_rows": 40, "evaluation_max_rows": 20}
    )
    config["dashboard"]["max_points"] = 10
    return config


@pytest.fixture
def completed_run(tmp_path):
    install_source(tmp_path, raw_fixture())
    install_zones(tmp_path)
    return tmp_path, run_pipeline(tmp_path, config_for(), fit_models=False)


def test_interrupted_run_records_canceled_and_preserves_published_snapshot(
    completed_run, monkeypatch
):
    root, prior_run = completed_run
    snapshot = publish_run(prior_run, root / "artifacts")

    def interrupt(_frame):
        raise KeyboardInterrupt

    monkeypatch.setattr("mobility.pipeline.aggregate_cpu", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(root, config_for(), fit_models=False)
    manifests = [read_json(path) for path in (root / "artifacts" / "runs").glob("*/manifest.json")]
    assert sum(item["status"] == "canceled" for item in manifests) == 1
    assert current_snapshot(root / "artifacts") == snapshot


def test_unsatisfiable_partial_download_restarts_safely(tmp_path, monkeypatch):
    import io

    buffer = io.BytesIO()
    raw_fixture().to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    target = tmp_path / "yellow.parquet"
    partial = target.with_suffix(".parquet.part")
    partial.write_bytes(payload)
    atomic_json(partial.with_suffix(".part.json"), {"etag": '"v1"'})
    requests_made = []

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"ETag": '"v1"', "Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def raise_for_status(self):
            assert self.status_code == 200

        def iter_content(self, _size):
            yield payload

    class Session:
        headers = {}

        def get(self, _url, **kwargs):
            requests_made.append(kwargs["headers"])
            return Response(416 if len(requests_made) == 1 else 200)

    monkeypatch.setattr("mobility.data.acquire.requests.Session", Session)
    monkeypatch.setattr("mobility.data.acquire.time.sleep", lambda _seconds: None)
    metadata = fetch_file(f"{BASE}/trip-data/yellow_tripdata_2024-01.parquet", target, "parquet")
    assert requests_made[0]["Range"] == f"bytes={len(payload)}-"
    assert requests_made[1] == {}
    assert target.read_bytes() == payload
    assert metadata["parquet_row_count"] == 30


def test_small_six_month_pipeline_models_reconcile_and_publish(tmp_path):
    for month in range(1, 7):
        install_source(tmp_path, raw_fixture(month), month)
    install_zones(tmp_path)
    run = run_pipeline(tmp_path, config_for(range(1, 7)))
    manifest = read_json(run / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["data_kind"] == "synthetic"
    assert manifest["raw_rows"] == manifest["canonical_rows"] == 180
    assert manifest["validation"]["counts"] == {"pickup": 180, "dropoff": 180, "od": 180}
    assert manifest["models"]["clustering"]["test"]["rows"] == 20
    assert manifest["models"]["anomalies"]["test_scored_bins"] > 0
    assignments = pd.read_parquet(run / "tables" / manifest["tables"]["trip_cluster_assignment"])
    assert len(assignments) == 180
    snapshot = publish_run(run, tmp_path / "artifacts")
    published = read_json(snapshot / "manifest.json")
    assert current_snapshot(tmp_path / "artifacts") == snapshot
    assert "trip_cluster_assignment" not in published["tables"]
    assert all("local_path" not in source for source in published["sources"])
    assert published["validation"]["counts"] == manifest["validation"]["counts"]
    for relative, expected in published["file_hashes"].items():
        assert sha256_file(snapshot / relative) == expected


def test_source_checksum_corruption_fails_before_processing(tmp_path):
    path, _ = install_source(tmp_path, raw_fixture())
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        source_list(tmp_path / "data", config_for())
    assert not (tmp_path / "artifacts").exists()


def test_existing_download_is_verified_without_network(tmp_path, monkeypatch):
    path, source = install_source(tmp_path, raw_fixture())
    monkeypatch.setattr(
        "requests.Session", lambda: pytest.fail("Existing verified file must not fetch")
    )
    assert fetch_file(source["url"], path, "parquet")["sha256"] == source["sha256"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        fetch_file(source["url"], path, "parquet")


@pytest.mark.parametrize(
    "url",
    [
        "http://d37ci6vzurychx.cloudfront.net/trip-data/x.parquet",
        "https://example.com/data.parquet",
        "file:///etc/passwd",
    ],
)
def test_downloader_rejects_unapproved_urls(tmp_path, url):
    with pytest.raises(ValueError, match="official TLC HTTPS"):
        fetch_file(url, tmp_path / "data.parquet", "parquet")


def test_required_schema_drift_is_rejected(tmp_path):
    path = tmp_path / "drift.parquet"
    raw_fixture().drop(columns="PULocationID").to_parquet(path)
    with pytest.raises(ValueError, match="Required yellow columns"):
        inspect_parquet(path)
    raw = raw_fixture()
    raw["PULocationID"] = "one"
    raw.to_parquet(path)
    with pytest.raises(ValueError, match="incompatible type"):
        inspect_parquet(path)
    raw = raw_fixture()
    raw["tpep_pickup_datetime"] = raw.tpep_pickup_datetime.astype(str)
    raw.to_parquet(path)
    with pytest.raises(ValueError, match="Parquet timestamp"):
        inspect_parquet(path)


def test_missing_source_coverage_stays_null_not_zero(tmp_path):
    _, source = install_source(tmp_path, raw_fixture())
    canonical, _ = normalize_yellow(raw_fixture(), source, {1, 2})
    zone, _ = aggregate_cpu(canonical)
    covered = add_coverage(zone, [1, 2], [source], config_for((1, 2)))
    january = covered.hour.dt.month.eq(1)
    february = covered.hour.dt.month.eq(2)
    assert covered.loc[january, "pickup_coverage"].eq("observed").all()
    assert covered.loc[january, "pickup_count"].notna().all()
    assert covered.loc[february, "pickup_coverage"].eq("missing_source").all()
    assert covered.loc[february, ["pickup_count", "dropoff_count"]].isna().all().all()
    assert covered.loc[january, "dropoff_coverage"].eq("partial_source_context").all()


def test_missing_month_fails_unless_explicitly_allowed(tmp_path):
    install_source(tmp_path, raw_fixture())
    with pytest.raises(FileNotFoundError, match="Missing source"):
        source_list(tmp_path / "data", config_for((1, 2)))
    sources = source_list(tmp_path / "data", config_for((1, 2), allow_missing_sources=True))
    assert len(sources) == 1


def test_dst_coverage_marks_discontinuity_instead_of_demand_zero(tmp_path):
    _, source = install_source(tmp_path, raw_fixture(3), 3)
    canonical, _ = normalize_yellow(raw_fixture(3), source, {1, 2})
    zone, _ = aggregate_cpu(canonical)
    covered = add_coverage(zone, [1, 2], [source], config_for((3,)))
    assert (
        covered.loc[covered.hour.eq(pd.Timestamp("2024-03-10 02:00")), "pickup_coverage"]
        .eq("dst_ambiguous")
        .all()
    )
    assert (
        covered.loc[covered.hour.eq(pd.Timestamp("2024-03-10 03:00")), "pickup_coverage"]
        .eq("observed")
        .all()
    )


def test_empty_source_produces_zero_count_grid_and_valid_artifacts(tmp_path):
    install_source(tmp_path, raw_fixture().iloc[:0])
    install_zones(tmp_path)
    run = run_pipeline(tmp_path, config_for())
    manifest = read_json(run / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["raw_rows"] == 0
    assert manifest["validation"]["counts"] == {"pickup": 0, "dropoff": 0, "od": 0}
    assert manifest["models"]["clustering"]["status"] == "insufficient_training_rows"
    quality = pd.read_parquet(run / "tables" / "quality_summary.parquet")
    assert set(["rule", "affected_rows", "total_rows"]).issubset(quality)
    publish_run(run, tmp_path / "artifacts")


def test_all_invalid_source_keeps_rows_and_quality_report(tmp_path):
    raw = raw_fixture(1, 5)
    raw[["tpep_pickup_datetime", "tpep_dropoff_datetime"]] = pd.NaT
    raw[["PULocationID", "DOLocationID"]] = -1
    raw["trip_distance"] = np.nan
    install_source(tmp_path, raw)
    install_zones(tmp_path)
    run = run_pipeline(tmp_path, config_for())
    manifest = read_json(run / "manifest.json")
    assert manifest["canonical_rows"] == 5
    assert manifest["validation"]["counts"] == {"pickup": 0, "dropoff": 0, "od": 0}
    quality = pd.read_parquet(run / "tables" / "quality_summary.parquet").set_index("rule")
    assert quality.loc["missing_pickup", "affected_rows"] == 5
    assert quality.loc["invalid_distance", "affected_rows"] == 5


def test_incomplete_run_cannot_replace_published_snapshot(completed_run):
    root, run = completed_run
    previous = publish_run(run, root / "artifacts")
    pointer_before = (root / "artifacts" / "current.json").read_bytes()
    manifest = read_json(run / "manifest.json")
    manifest["status"] = "failed"
    atomic_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="completed"):
        publish_run(run, root / "artifacts")
    assert (root / "artifacts" / "current.json").read_bytes() == pointer_before
    assert current_snapshot(root / "artifacts") == previous


def test_copy_failure_leaves_previous_snapshot_and_no_staging_dir(completed_run):
    root, run = completed_run
    previous = publish_run(run, root / "artifacts")
    pointer_before = (root / "artifacts" / "current.json").read_bytes()
    (run / "tables" / "quality_summary.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        publish_run(run, root / "artifacts")
    assert (root / "artifacts" / "current.json").read_bytes() == pointer_before
    assert current_snapshot(root / "artifacts") == previous
    assert not list((root / "artifacts" / "snapshots").glob(".staging-*"))


@pytest.mark.parametrize("relative", ["../../outside.parquet", "../tables/zone_hour.parquet"])
def test_publication_rejects_source_or_destination_path_escape(completed_run, relative):
    root, run = completed_run
    previous = publish_run(run, root / "artifacts")
    pointer_before = (root / "artifacts" / "current.json").read_bytes()
    manifest = read_json(run / "manifest.json")
    manifest["tables"]["suspicious"] = relative
    atomic_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        publish_run(run, root / "artifacts")
    assert (root / "artifacts" / "current.json").read_bytes() == pointer_before
    assert current_snapshot(root / "artifacts") == previous
    assert not (root / "artifacts" / "snapshots" / "tables").exists()


def test_publication_rejects_symlinked_directory_descendant(completed_run):
    root, run = completed_run
    source = run / "tables" / "od_hour" / "2024-01.parquet"
    outside = root / "outside.parquet"
    shutil.copy2(source, outside)
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="[Ss]ymlink|outside|escape"):
        publish_run(run, root / "artifacts")


def test_current_pointer_cannot_escape_artifacts_root(tmp_path):
    artifacts = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()
    atomic_json(outside / "manifest.json", {"snapshot_id": "bad"})
    atomic_json(artifacts / "current.json", {"path": "../outside", "snapshot_id": "bad"})
    with pytest.raises(ValueError, match="Invalid active snapshot"):
        current_snapshot(artifacts)


def test_validator_rejects_duplicate_od_keys_across_parts(completed_run):
    _, run = completed_run
    directory = run / "tables" / "od_hour"
    shutil.copy2(directory / "2024-01.parquet", directory / "duplicate.parquet")
    with pytest.raises(ValueError, match="[Dd]uplicate|[Ii]nvalid|partition"):
        validate_outputs(run / "tables")


@pytest.mark.parametrize("value", [0, 1])
def test_invalid_calibration_budget_rejected_at_config_boundary(value):
    # The model requires a strictly interior fraction, so config must match it.
    with pytest.raises(ValueError):
        ProjectConfig.model_validate({"anomalies": {"calibration_max_alert_fraction": value}})
