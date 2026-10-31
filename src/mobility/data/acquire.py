"""Fetch official TLC files, preserving checksums and resumable partial downloads."""

import hashlib
import logging
import os
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
import requests

from mobility.io import atomic_json, read_json, sha256_file, utc_now

LOGGER = logging.getLogger(__name__)
BASE = "https://d37ci6vzurychx.cloudfront.net"
REQUIRED_YELLOW = {
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
    "DOLocationID", "trip_distance",
}


def inspect_parquet(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    missing = REQUIRED_YELLOW - set(schema.names)
    if missing:
        raise ValueError(f"Required yellow columns missing: {sorted(missing)}")
    import pyarrow as pa
    for name in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        if not pa.types.is_timestamp(schema.field(name).type):
            raise ValueError(f"{name} must be a Parquet timestamp")
    for name in ("PULocationID", "DOLocationID", "trip_distance"):
        dtype = schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(f"{name} has incompatible type {dtype}")
    time_bounds = {}
    for name in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        index = schema.names.index(name)
        statistics = [parquet.metadata.row_group(group).column(index).statistics
                      for group in range(parquet.metadata.num_row_groups)]
        available = [stat for stat in statistics if stat is not None and stat.has_min_max]
        complete = len(available) == len(statistics) and bool(available)
        time_bounds[name] = {"min": str(min(stat.min for stat in available)) if complete else None,
                             "max": str(max(stat.max for stat in available)) if complete else None,
                             "metadata_complete": complete}
    return {
        "parquet_row_count": parquet.metadata.num_rows,
        "parquet_row_groups": parquet.metadata.num_row_groups,
        "schema": str(schema),
        "schema_fingerprint": hashlib.sha256(str(schema).encode()).hexdigest(),
        "optional_columns": sorted(set(schema.names) - REQUIRED_YELLOW),
        "source_timestamp_bounds": time_bounds,
    }


def fetch_file(url: str, destination: Path, kind: str) -> dict:
    """Only configured official URLs are accepted. Existing snapshots are verified."""
    if urlparse(url).scheme != "https" or urlparse(url).hostname != urlparse(BASE).hostname:
        raise ValueError("Only the configured official TLC HTTPS host is allowed")
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    if destination.exists():
        if not metadata_path.exists():
            raise ValueError(f"Untracked file at {destination}; move it before fetching")
        metadata = read_json(metadata_path)
        if metadata.get("url") != url or sha256_file(destination) != metadata["sha256"]:
            raise ValueError(f"Frozen source failed checksum verification: {destination}")
        LOGGER.info("Verified existing %s", destination.name)
        return metadata

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    resume_info = partial.with_suffix(partial.suffix + ".json")
    session = requests.Session()
    session.headers["User-Agent"] = "nyc-mobility-research/0.1"
    for attempt in range(4):
        try:
            headers = {}
            offset = partial.stat().st_size if partial.exists() else 0
            prior = read_json(resume_info) if resume_info.exists() else {}
            if offset and prior.get("etag"):
                headers = {"Range": f"bytes={offset}-", "If-Range": prior["etag"]}
            with session.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code == 416 and headers:
                    # A previous process may have received every byte but stopped
                    # before validating/renaming the partial file. Restart cleanly.
                    partial.unlink(missing_ok=True)
                    resume_info.unlink(missing_ok=True)
                    raise ValueError("Resume range is unavailable; restarting the partial download")
                response.raise_for_status()
                resumed = bool(headers and response.status_code == 206)
                if resumed:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an invalid resume range")
                    if response.headers.get("ETag") != prior.get("etag"):
                        raise ValueError("Source changed during resumed acquisition")
                else:
                    offset = 0
                resume = {"etag": response.headers.get("ETag"), "url": url}
                atomic_json(resume_info, resume)
                expected = int(response.headers.get("Content-Length", 0))
                received = 0
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
                            received += len(block)
                if expected and received != expected:
                    raise ValueError("Incomplete source download")
                metadata = {
                    "url": url, "downloaded_at_utc": utc_now(), "local_path": str(destination),
                    "byte_count": partial.stat().st_size, "sha256": sha256_file(partial),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "validation_status": "valid", "data_kind": "real",
                }
                if kind == "parquet":
                    metadata.update(inspect_parquet(partial))
                elif kind == "csv":
                    lookup = pd.read_csv(partial)
                    if not {"LocationID", "Borough", "Zone"}.issubset(lookup.columns):
                        raise ValueError("Unexpected zone lookup schema")
                    if lookup.LocationID.duplicated().any():
                        raise ValueError("Duplicate zone IDs in official lookup")
                elif kind == "zip":
                    with zipfile.ZipFile(partial) as archive:
                        for name in archive.namelist():
                            member = Path(name)
                            if member.is_absolute() or ".." in member.parts:
                                raise ValueError("Unsafe geometry archive path")
                        if archive.testzip() is not None:
                            raise ValueError("Corrupt geometry archive")
                os.replace(partial, destination)
                atomic_json(metadata_path, metadata)
                resume_info.unlink(missing_ok=True)
                LOGGER.info("Downloaded and validated %s (%.1f MB)", destination.name,
                            metadata["byte_count"] / 1e6)
                return metadata
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            LOGGER.warning("Acquisition retry %d for %s", attempt + 1, destination.name)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Unreachable acquisition state")


def fetch_month(data_root: Path, year: int, month: int) -> dict:
    if year < 2009 or year > 2100 or month not in range(1, 13):
        raise ValueError("Invalid TLC source month")
    name = f"yellow_tripdata_{year}-{month:02d}.parquet"
    path = data_root / "raw" / "yellow" / name
    metadata = fetch_file(f"{BASE}/trip-data/{name}", path, "parquet")
    metadata.update(dataset_type="yellow", nominal_month=f"{year}-{month:02d}",
                    source_id=f"yellow-{year}-{month:02d}-{metadata['sha256'][:12]}")
    atomic_json(path.with_suffix(".parquet.json"), metadata)
    return metadata


def prepare_zones(data_root: Path) -> Path:
    """Prepare EPSG:4326 display geometry; preserve original official sources."""
    import geopandas as gpd

    directory = data_root / "zones"
    directory.mkdir(parents=True, exist_ok=True)
    lookup_path = directory / "taxi_zone_lookup.csv"
    zip_path = directory / "taxi_zones.zip"
    lookup_meta = fetch_file(f"{BASE}/misc/taxi_zone_lookup.csv", lookup_path, "csv")
    shape_meta = fetch_file(f"{BASE}/misc/taxi_zones.zip", zip_path, "zip")
    lookup = pd.read_csv(lookup_path).rename(
        columns={"LocationID": "zone_id", "Borough": "borough", "Zone": "zone"})
    with zipfile.ZipFile(zip_path) as archive:
        shapefiles = [name for name in archive.namelist() if name.endswith(".shp")]
    if len(shapefiles) != 1:
        raise ValueError("Expected exactly one zone shapefile in the official archive")
    geometry = gpd.read_file(f"zip://{zip_path.resolve()}!{shapefiles[0]}")
    if geometry.crs is None or geometry.LocationID.duplicated().any():
        raise ValueError("Zone geometry requires a declared CRS and unique IDs")
    repaired = int((~geometry.geometry.is_valid).sum())
    geometry.geometry = geometry.geometry.make_valid()
    projected = geometry.to_crs("EPSG:32618")
    anchors = gpd.GeoSeries(projected.geometry.centroid, crs=projected.crs).to_crs(4326)
    anchors = pd.DataFrame({"zone_id": geometry.LocationID.astype(int),
                            "longitude": anchors.x, "latitude": anchors.y})
    lookup = lookup.merge(anchors, on="zone_id", how="left", validate="one_to_one")
    lookup.to_parquet(directory / "zones.parquet", index=False)
    projected.geometry = projected.geometry.simplify(10, preserve_topology=True)
    web = projected.to_crs(4326).rename(columns={"LocationID": "zone_id"})
    (directory / "zones.geojson").write_text(web[["zone_id", "geometry"]].to_json())
    atomic_json(directory / "manifest.json", {
        "sources": [lookup_meta, shape_meta], "source_crs": str(geometry.crs),
        "display_crs": "EPSG:4326", "simplification_meters": 10,
        "repaired_geometries": repaired, "geometry_zone_ids": sorted(web.zone_id.tolist()),
        "lookup_rows": len(lookup), "geometry_rows": len(web),
    })
    return directory
