"""Deterministic offline fixtures. These are never represented as real TLC data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobility.data.acquire import inspect_parquet
from mobility.io import atomic_json, sha256_file, utc_now


def create_demo_sources(root: Path, rows_per_month: int = 12_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    zone_dir = root / "data" / "zones"
    raw_dir = root / "data" / "raw" / "yellow"
    zone_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(1, 9))
    zones = pd.DataFrame(
        {
            "zone_id": ids,
            "zone": [f"Fixture zone {i}" for i in ids],
            "borough": "Synthetic area",
            "longitude": np.linspace(-74.04, -73.92, 8),
            "latitude": np.linspace(40.69, 40.84, 8),
        }
    )
    zones.to_parquet(zone_dir / "zones.parquet", index=False)
    features = []
    for zone in zones.itertuples():
        x, y = zone.longitude, zone.latitude
        ring = [
            [x - 0.004, y - 0.004],
            [x + 0.004, y - 0.004],
            [x + 0.004, y + 0.004],
            [x - 0.004, y + 0.004],
            [x - 0.004, y - 0.004],
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {"zone_id": zone.zone_id},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    (zone_dir / "zones.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    for month in range(1, 7):
        start = pd.Timestamp(2024, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        pu = start + pd.to_timedelta(
            rng.integers(0, int((end - start).total_seconds()), rows_per_month), unit="s"
        )
        duration = rng.lognormal(6.5, 0.65, rows_per_month).clip(30, 10_000)
        distance = duration / 3600 * rng.uniform(8, 23, rows_per_month)
        raw = pd.DataFrame(
            {
                "tpep_pickup_datetime": pu,
                "tpep_dropoff_datetime": pu + pd.to_timedelta(duration, unit="s"),
                "PULocationID": rng.choice(ids, rows_per_month),
                "DOLocationID": rng.choice(ids, rows_per_month),
                "trip_distance": distance,
                "passenger_count": 1,
                "fare_amount": distance * 3 + 3,
                "total_amount": distance * 3 + 7,
            }
        )
        raw.loc[0, "tpep_pickup_datetime"] = pd.NaT
        raw.loc[1, "tpep_dropoff_datetime"] = pd.NaT
        raw.loc[2, "trip_distance"] = -1
        raw.loc[3, "PULocationID"] = 265
        path = raw_dir / f"yellow_tripdata_2024-{month:02d}.parquet"
        raw.to_parquet(path, index=False)
        checksum = sha256_file(path)
        atomic_json(
            path.with_suffix(".parquet.json"),
            {
                "source_id": f"synthetic-{month}-{checksum[:12]}",
                "sha256": checksum,
                "local_path": str(path.resolve()),
                "byte_count": path.stat().st_size,
                "nominal_month": f"2024-{month:02d}",
                "dataset_type": "yellow",
                "data_kind": "synthetic",
                "url": None,
                "seed": seed,
                "downloaded_at_utc": utc_now(),
                "validation_status": "valid",
                **inspect_parquet(path),
            },
        )
