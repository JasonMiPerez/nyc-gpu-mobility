"""Small shared helpers for model inputs and reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEATURE_NAMES = ["log_distance_miles", "log_duration_seconds", "hour_sin", "hour_cos"]
TRIP_COLUMNS = [
    "record_id",
    "dataset_type",
    "pickup_local",
    "pickup_zone_id",
    "dropoff_zone_id",
    "trip_distance_miles",
    "duration_seconds",
    "eligible_ml",
]
DEFAULT_SPLITS = {
    "train_start": "2024-01-01",
    "train_end": "2024-04-01",
    "calibration_start": "2024-04-01",
    "calibration_end": "2024-05-01",
    "test_start": "2024-05-01",
    "test_end": "2024-07-01",
}


def split_bounds(config: dict) -> dict[str, pd.Timestamp]:
    bounds = {
        name: pd.Timestamp(value)
        for name, value in (DEFAULT_SPLITS | config.get("splits", {})).items()
    }
    for name in ("train", "calibration", "test"):
        if bounds[f"{name}_start"] >= bounds[f"{name}_end"]:
            raise ValueError(f"The {name} interval must have positive length.")
    if (
        bounds["train_end"] > bounds["calibration_start"]
        or bounds["calibration_end"] > bounds["test_start"]
    ):
        raise ValueError("Training, calibration, and test intervals must not overlap.")
    return bounds


def in_period(times: pd.Series, bounds: dict, period: str) -> pd.Series:
    return times.ge(bounds[f"{period}_start"]) & times.lt(bounds[f"{period}_end"])


def canonical_batches(directory: Path, batch_rows: int) -> Iterator[pd.DataFrame]:
    """Scan only model columns; never load the whole trip dataset into memory."""
    for path in sorted(Path(directory).rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        missing = set(TRIP_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Canonical file {path.name} lacks {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=TRIP_COLUMNS):
            frame = batch.to_pandas()
            frame["pickup_local"] = pd.to_datetime(frame["pickup_local"])
            yield frame


def valid_ml_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The canonical eligibility decision remains authoritative; guard math too."""
    distance = pd.to_numeric(frame["trip_distance_miles"], errors="coerce")
    duration = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    valid = (
        frame["eligible_ml"].fillna(False).astype(bool)
        & frame["pickup_local"].notna()
        & np.isfinite(distance)
        & distance.ge(0)
        & np.isfinite(duration)
        & duration.gt(0)
    )
    return frame.loc[valid].copy()


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    hour = frame["pickup_local"].dt.hour.to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.log1p(frame["trip_distance_miles"].to_numpy(dtype=np.float64)),
            np.log1p(frame["duration_seconds"].to_numpy(dtype=np.float64)),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ]
    ).astype(np.float32)


class HashSample:
    """Keep the smallest seeded BLAKE2b hashes, independent of batch boundaries."""

    def __init__(self, limit: int, seed: int):
        if limit < 1:
            raise ValueError("Sample limit must be positive.")
        self.limit = limit
        self.seed = seed
        self.rows_seen = 0
        self.frame = pd.DataFrame()

    def add(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.rows_seen += len(frame)
        candidate = frame.copy()
        candidate["_sample_hash"] = np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(f"{self.seed}:{record}".encode(), digest_size=8).digest(), "big"
                )
                for record in candidate["record_id"]
            ),
            dtype=np.uint64,
            count=len(candidate),
        )
        if not self.frame.empty:
            candidate = pd.concat([self.frame, candidate], ignore_index=True)
        # Include ties, then resolve them by record ID for a stable exact bound.
        candidate = candidate.nsmallest(self.limit, "_sample_hash", keep="all")
        self.frame = candidate.sort_values(["_sample_hash", "record_id"]).head(self.limit)

    def result(self) -> pd.DataFrame:
        return self.frame.drop(columns="_sample_hash", errors="ignore").reset_index(drop=True)

    def metadata(self) -> dict:
        frame = self.result()
        identities = "\n".join(frame.get("record_id", pd.Series(dtype=str)).astype(str))
        return {
            "method": "smallest_blake2b_64_seed_colon_record_id_v1",
            "seed": self.seed,
            "eligible_rows": self.rows_seen,
            "sample_rows": len(frame),
            "sample_ids_sha256": hashlib.sha256(identities.encode()).hexdigest(),
            "inclusion_fraction": len(frame) / self.rows_seen if self.rows_seen else 0.0,
        }


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def stable_id(prefix: str, value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"
