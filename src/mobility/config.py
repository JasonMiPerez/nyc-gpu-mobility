"""Validate configuration before fetching data or allocating large tables."""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    dataset: Literal["yellow"] = "yellow"
    year: int = Field(2024, ge=2009, le=2100)
    months: list[int] = Field(default_factory=lambda: list(range(1, 7)))

    @model_validator(mode="after")
    def valid_months(self):
        if not self.months or any(m < 1 or m > 12 for m in self.months):
            raise ValueError("months must contain month numbers from 1 to 12")
        if len(set(self.months)) != len(self.months):
            raise ValueError("months must not repeat")
        self.months = sorted(self.months)
        return self


class ProcessingConfig(StrictModel):
    backend: Literal["cpu", "gpu"] = "cpu"
    cpu_threads: int = Field(4, ge=1, strict=True)
    batch_rows: int = Field(250_000, ge=100, le=5_000_000)
    gpu_memory_fraction: float = Field(0.65, gt=0, lt=1)
    allow_missing_sources: bool = False
    custom_cuda: bool = False
    verify_gpu_batches: bool = True


class SplitConfig(StrictModel):
    train_start: date = date(2024, 1, 1)
    train_end: date = date(2024, 4, 1)
    calibration_start: date = date(2024, 4, 1)
    calibration_end: date = date(2024, 5, 1)
    test_start: date = date(2024, 5, 1)
    test_end: date = date(2024, 7, 1)

    @model_validator(mode="after")
    def chronological(self):
        if not (
            self.train_start
            < self.train_end
            <= self.calibration_start
            < self.calibration_end
            <= self.test_start
            < self.test_end
        ):
            raise ValueError("Use ordered, non-overlapping, nonempty half-open time splits")
        return self


class ClusterConfig(StrictModel):
    k_values: list[int] = Field(default_factory=lambda: [4, 6, 8])
    training_max_rows: int = Field(200_000, ge=10, le=5_000_000)
    evaluation_max_rows: int = Field(2000, ge=10, le=5000)
    # Study restrictions for clustering; endpoint counts retain their own eligibility.
    max_distance_miles: float = Field(100.0, gt=0)
    max_duration_minutes: float = Field(180.0, gt=0)
    max_mean_speed_mph: float = Field(100.0, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def valid_clusters(self):
        if not self.k_values or any(k < 2 or k > 50 for k in self.k_values):
            raise ValueError("k_values must contain values between 2 and 50")
        return self


class AnomalyConfig(StrictModel):
    min_history: int = Field(8, ge=3)
    scale_floor: float = Field(5.0, gt=0)
    score_threshold: float = Field(4.0, gt=0)
    absolute_threshold: int = Field(20, ge=0)
    threshold_candidates: list[float] = Field(default_factory=lambda: [3, 4, 5, 6])
    calibration_max_alert_fraction: float = Field(0.01, gt=0, lt=1)


class BenchmarkConfig(StrictModel):
    warmups: int = Field(2, ge=0)
    repeats: int = Field(7, ge=1)
    max_rows: int = Field(1_000_000, ge=100)
    cpu_threads: int = Field(4, ge=1)
    seed: int = 42


class DurationAnomalyConfig(StrictModel):
    enabled: bool = True
    min_history: int = Field(8, ge=3)
    min_trips_per_hour: int = Field(20, ge=1)
    scale_floor_minutes: float = Field(2.0, gt=0)
    absolute_threshold_minutes: float = Field(5.0, gt=0)
    score_threshold: float = Field(4.0, gt=0)
    threshold_candidates: list[float] = Field(default_factory=lambda: [3, 4, 5, 6])
    calibration_max_alert_fraction: float = Field(0.01, gt=0, lt=1)


class GraphConfig(StrictModel):
    enabled: bool = True
    min_trips_per_edge: int = Field(1, ge=1)
    pagerank_alpha: float = Field(0.85, gt=0, lt=1)
    pagerank_tol: float = Field(1e-10, gt=0)
    pagerank_max_iter: int = Field(1000, ge=1)
    pagerank_atol: float = Field(1e-7, gt=0)
    pagerank_rtol: float = Field(1e-5, gt=0)
    louvain_resolution: float = Field(1.0, gt=0)
    louvain_threshold: float = Field(1e-7, gt=0)
    louvain_max_level: int = Field(100, ge=1)
    seed: int = 42


class DashboardConfig(StrictModel):
    max_flows: int = Field(200, ge=1, le=1000)
    max_points: int = Field(5000, ge=1, le=10_000)
    max_export_rows: int = Field(50_000, ge=1, le=100_000)


class ProjectConfig(StrictModel):
    schema_version: int = 1
    time_policy: Literal["nyc_local_wall_time_v1"] = "nyc_local_wall_time_v1"
    data: DataConfig = Field(default_factory=DataConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    clustering: ClusterConfig = Field(default_factory=ClusterConfig)
    anomalies: AnomalyConfig = Field(default_factory=AnomalyConfig)
    duration_anomalies: DurationAnomalyConfig = Field(default_factory=DurationAnomalyConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(path: Path | None = None) -> dict:
    raw = yaml.safe_load(path.read_text()) if path else {}
    return ProjectConfig.model_validate(raw or {}).model_dump(mode="json")
