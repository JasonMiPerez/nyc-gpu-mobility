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
DEFAULT_ML_FILTERS = {
    "max_distance_miles": 100.0,
    "max_duration_minutes": 180.0,
    "max_mean_speed_mph": 100.0,
}
ML_EXCLUSION_RULES = (
    "invalid_features",
    "distance_exceeds_limit",
    "duration_exceeds_limit",
    "mean_speed_exceeds_limit",
)


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
        columns = TRIP_COLUMNS + [
            name for name in ("source_token", "source_row") if name in parquet.schema_arrow.names
        ]
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            frame["pickup_local"] = pd.to_datetime(frame["pickup_local"])
            yield frame


def ml_filter_parameters(parameters: dict | None = None) -> dict[str, float]:
    """Fixed study bounds, chosen before fitting and never estimated on future trips."""
    parameters = parameters or {}
    limits = {
        name: float(parameters.get(name, default)) for name, default in DEFAULT_ML_FILTERS.items()
    }
    if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError("Clustering plausibility limits must be finite and positive.")
    return limits


def ml_row_masks(frame: pd.DataFrame, parameters: dict | None = None) -> dict[str, pd.Series]:
    """Audit model exclusions without changing canonical rows or their eligibility flags.

    Exclusion masks apply only to canonical-eligible rows and may overlap. Limits
    are inclusive study choices, not proof that an excluded trip did not occur.
    Mean speed uses reported distance divided by elapsed time, including stops.
    """
    limits = ml_filter_parameters(parameters)
    distance = pd.to_numeric(frame["trip_distance_miles"], errors="coerce")
    duration = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    canonical = frame["eligible_ml"].fillna(False).astype(bool)
    valid_features = (
        frame["pickup_local"].notna()
        & np.isfinite(distance)
        & distance.ge(0)
        & np.isfinite(duration)
        & duration.gt(0)
    )
    mean_speed = distance / (duration.where(duration.gt(0)) / 3600)
    masks = {
        "canonical_eligible": canonical,
        "invalid_features": canonical & ~valid_features,
        "distance_exceeds_limit": canonical
        & np.isfinite(distance)
        & distance.gt(limits["max_distance_miles"]),
        "duration_exceeds_limit": canonical
        & np.isfinite(duration)
        & duration.gt(limits["max_duration_minutes"] * 60),
        "mean_speed_exceeds_limit": canonical
        & np.isfinite(distance)
        & np.isfinite(duration)
        & duration.gt(0)
        & mean_speed.gt(limits["max_mean_speed_mph"]),
    }
    excluded = masks[ML_EXCLUSION_RULES[0]].copy()
    for name in ML_EXCLUSION_RULES[1:]:
        excluded |= masks[name]
    masks["model_eligible"] = canonical & ~excluded
    return masks


def valid_ml_rows(frame: pd.DataFrame, parameters: dict | None = None) -> pd.DataFrame:
    """Return canonical-eligible trips within the fixed clustering study bounds."""
    return frame.loc[ml_row_masks(frame, parameters)["model_eligible"]].copy()


def feature_matrix(frame: pd.DataFrame, dtype=np.float32) -> np.ndarray:
    hour = frame["pickup_local"].dt.hour.to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.log1p(frame["trip_distance_miles"].to_numpy(dtype=np.float64)),
            np.log1p(frame["duration_seconds"].to_numpy(dtype=np.float64)),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ]
    ).astype(dtype)


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


def priority_values(tokens, rows, seed):
    """Matched CPU/GPU sampling priorities, with explicit uint64 wraparound."""
    tokens, rows = np.asarray(tokens, dtype=np.uint64), np.asarray(rows, dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = (tokens ^ rows ^ np.uint64(seed % 2**64)) + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def initial_center_indices(training, k, seed):
    priorities = priority_values(training.source_token, training.source_row, seed)
    return np.lexsort(
        (training.source_row.to_numpy(), training.source_token.to_numpy(), priorities)
    )[:k]


class PrioritySample(HashSample):
    """Bounded numeric sampling with the same algorithm on CPU and GPU."""

    def add(self, frame):
        if frame.empty:
            return
        if not {"source_token", "source_row"} <= set(frame):
            raise ValueError("Native sampling requires canonical source_token and source_row.")
        self.rows_seen += len(frame)
        candidate = frame.copy()
        candidate["_sample_hash"] = priority_values(
            candidate.source_token, candidate.source_row, self.seed
        )
        if not self.frame.empty:
            candidate = pd.concat([self.frame, candidate], ignore_index=True)
        candidate = candidate.nsmallest(self.limit, "_sample_hash", keep="all")
        self.frame = candidate.sort_values(["_sample_hash", "source_token", "source_row"]).head(
            self.limit
        )

    def metadata(self):
        return super().metadata() | {"method": "smallest_splitmix64_source_row_v1"}


class PopulationScaler:
    """CPU counterpart of the native GPU float64 population reductions."""

    def fit_transform(self, matrix):
        self.mean_ = matrix.mean(axis=0, dtype=np.float64)
        variance = np.square(matrix - self.mean_).mean(axis=0, dtype=np.float64)
        self.scale_ = np.sqrt(variance)
        constant = (matrix.max(axis=0) == matrix.min(axis=0)) | (self.scale_ == 0)
        self.mean_[constant] = matrix[0, constant]
        self.scale_[constant] = 1.0
        return self.transform(matrix)

    def transform(self, matrix):
        return (matrix - self.mean_) / self.scale_


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def stable_id(prefix: str, value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"
