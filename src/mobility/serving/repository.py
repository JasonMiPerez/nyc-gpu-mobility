"""Small, bounded queries over published data, never over raw trip files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


class QueryError(ValueError):
    """A safe, user-readable explanation of an unsupported or oversized query."""


@dataclass(frozen=True)
class Filters:
    """Half-open local-time interval; zones refer to origins for OD queries."""

    start: str
    end: str
    dataset: str = "yellow"
    zones: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["zones"] = list(self.zones)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "Filters":
        start, end = pd.Timestamp(value["start"]), pd.Timestamp(value["end"])
        if start.tzinfo or end.tzinfo or not start < end:
            raise QueryError("Choose a valid local date range with an end after its start.")
        dataset = str(value.get("dataset", "yellow"))
        if dataset not in {"yellow", "hvfhv"}:
            raise QueryError("That source family is not supported.")
        zones = tuple(sorted(set(int(zone) for zone in value.get("zones", []))))
        if len(zones) > 300 or any(zone < 1 or zone > 999 for zone in zones):
            raise QueryError("Choose valid taxi zones.")
        return cls(start.isoformat(), end.isoformat(), dataset, zones)


CAPABILITIES = {
    "zone_hour": ("hour", "zone_id"),
    "od_hour": ("hour", "pickup_zone_id"),
    "trip_plot_sample": ("pickup_local", "pickup_zone_id"),
    "anomaly_scores": ("hour", "zone_id"),
    "anomaly_incidents": ("start_hour", "zone_id"),
    "duration_scores": ("hour", "zone_id"),
    "duration_incidents": ("start_hour", "zone_id"),
    "duration_hour": ("hour", "zone_id"),
    "trip_histogram": ("pickup_date", None),
}


class SnapshotRepository:
    """One immutable snapshot per app instance; cached results are copied on return."""

    allowed_tables = frozenset(
        {
            "zone_hour",
            "od_hour",
            "zones",
            "quality_summary",
            "cluster_summary",
            "trip_plot_sample",
            "trip_histogram",
            "anomaly_scores",
            "anomaly_incidents",
            "benchmark_runs",
            "zone_profile",
            "anomaly_baseline",
            "anomaly_evaluation",
            "duration_scores",
            "duration_incidents",
            "duration_hour",
            "duration_baseline",
            "graph_nodes",
            "graph_edges",
            "graph_undirected_edges",
            "pipeline_benchmark_runs",
            "pipeline_benchmark_summary",
        }
    )

    def __init__(self, snapshot_dir: Path):
        self.root = Path(snapshot_dir).resolve(strict=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.stat().st_size > 5_000_000:
            raise QueryError("The snapshot manifest is unexpectedly large.")
        self.manifest = json.loads(manifest_path.read_text())
        self.snapshot_id = str(self.manifest["snapshot_id"])
        self.tables = self.manifest.get("tables", {})
        self.options = self.manifest.get("config", {}).get("dashboard", {})
        self.max_export = min(int(self.options.get("max_export_rows", 50_000)), 50_000)
        self.max_points = min(int(self.options.get("max_points", 5_000)), 5_000)
        self.max_flows = min(int(self.options.get("max_flows", 200)), 200)
        self.max_query_rows = 3_000_000
        self.query_timeout = 20.0
        for name in self.allowed_tables.intersection(self.tables):
            self._table_path(name)

    def _path(self, value: str, *, allow_dataset=False) -> Path:
        """Reject absolute paths, traversal, missing files, and escaping symlinks."""
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise QueryError("The published snapshot contains an unsafe artifact path.")
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise QueryError("The published artifact is outside this snapshot.")
        if path.is_dir() and allow_dataset:
            for child in path.rglob("*"):
                if child.is_symlink() or not child.resolve().is_relative_to(self.root):
                    raise QueryError("A published dataset contains an unsafe linked path.")
                if child.is_file() and child.suffix != ".parquet":
                    raise QueryError("A published dataset may contain only Parquet files.")
            return path
        if not path.is_file():
            raise QueryError("The published artifact is not a regular file.")
        return path

    def _table_path(self, name: str) -> Path:
        if name not in self.allowed_tables or name not in self.tables:
            raise QueryError("This table is not available to dashboard queries.")
        return self._path(self.tables[name], allow_dataset=True)

    def has(self, name: str) -> bool:
        return name in self.tables and name in self.allowed_tables

    def default_filters(self) -> Filters:
        data = self.manifest.get("config", {}).get("data", {})
        return Filters.from_dict(
            {
                "start": self.manifest["study_start"],
                "end": self.manifest["study_end"],
                "dataset": data.get("dataset", "yellow"),
                "zones": [],
            }
        )

    def _expression(self, schema: pa.Schema, name: str, filters: Filters | None):
        if filters is None:
            return None
        if name not in CAPABILITIES:
            raise QueryError(
                f"{name.replace('_', ' ')} uses a fixed period and does not support these filters."
            )
        time_column, zone_column = CAPABILITIES[name]
        if time_column not in schema.names:
            raise QueryError("This artifact does not support the requested date filter.")
        field_type = schema.field(time_column).type
        if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
            start = pa.scalar(pd.Timestamp(filters.start).date().isoformat(), type=field_type)
            end = pa.scalar(pd.Timestamp(filters.end).date().isoformat(), type=field_type)
        else:
            start = pa.scalar(pd.Timestamp(filters.start).to_pydatetime(), type=field_type)
            end = pa.scalar(pd.Timestamp(filters.end).to_pydatetime(), type=field_type)
        if name in {"anomaly_incidents", "duration_incidents"} and "end_hour" in schema.names:
            # Include events crossing the beginning of the selected interval.
            expression = (ds.field("end_hour") >= start) & (ds.field(time_column) < end)
        else:
            expression = (ds.field(time_column) >= start) & (ds.field(time_column) < end)
        if "dataset_type" in schema.names:
            expression &= ds.field("dataset_type") == filters.dataset
        if filters.zones:
            if zone_column not in schema.names:
                raise QueryError("This artifact does not support a zone filter.")
            expression &= ds.field(zone_column).isin(list(filters.zones))
        return expression

    def _batches(self, name: str, filters: Filters | None = None, columns=None):
        if name not in self.allowed_tables:
            raise QueryError("This table is not available to dashboard queries.")
        if not self.has(name):
            return
        dataset = ds.dataset(self._table_path(name), format="parquet")
        if not dataset.files:
            return
        if filters is not None and name in CAPABILITIES:
            time_column, _ = CAPABILITIES[name]
            if time_column in dataset.schema.names and pa.types.is_null(
                dataset.schema.field(time_column).type
            ):
                if dataset.count_rows() == 0:
                    return
        expression = self._expression(dataset.schema, name, filters)
        started = time.monotonic()
        scanner = dataset.scanner(columns=columns, filter=expression, batch_size=65_536)
        for batch in scanner.to_batches():
            if time.monotonic() - started > self.query_timeout:
                raise QueryError(
                    "This selection takes too long. Choose a shorter date range or fewer zones."
                )
            yield batch

    @lru_cache(maxsize=6)
    def _read_cached(self, name: str, filters: Filters | None) -> pd.DataFrame:
        batches, rows = [], 0
        for batch in self._batches(name, filters):
            rows += batch.num_rows
            if rows > self.max_query_rows:
                raise QueryError(
                    "This selection is too large. Choose a shorter date range or fewer zones."
                )
            batches.append(batch)
        if not batches:
            if self.has(name):
                return ds.dataset(self._table_path(name), format="parquet").head(0).to_pandas()
            return pd.DataFrame()
        return pa.Table.from_batches(batches).to_pandas()

    def read(self, name: str, filters: Filters | None = None) -> pd.DataFrame:
        return self._read_cached(name, filters).copy(deep=True)

    def zones(self) -> pd.DataFrame:
        return self.read("zones")

    @lru_cache(maxsize=1)
    def geojson(self) -> dict | None:
        name = self.tables.get("zones_geojson", self.tables.get("zones.geojson"))
        if name is None and (self.root / "zones.geojson").is_file():
            name = "zones.geojson"
        if name is None:
            return None
        path = self._path(name)
        if path.stat().st_size > 15_000_000:
            return None
        return json.loads(path.read_text())

    @lru_cache(maxsize=12)
    def _od_totals_cached(self, filters: Filters, destinations: tuple[int, ...]):
        """Stream many hourly rows into at most one entry per zone pair."""
        pieces = []
        keys = ["pickup_zone_id", "dropoff_zone_id"]
        for batch in self._batches("od_hour", filters):
            frame = batch.to_pandas()
            if destinations:
                frame = frame[frame.dropoff_zone_id.isin(destinations)]
            if frame.empty:
                continue
            sums = [
                c for c in ("trip_count", "duration_sum_seconds", "duration_count") if c in frame
            ]
            pieces.append(frame.groupby(keys, as_index=False)[sums].sum())
            if len(pieces) >= 8:
                pieces = [pd.concat(pieces).groupby(keys, as_index=False)[sums].sum()]
        if not pieces:
            return pd.DataFrame(columns=keys + ["trip_count"])
        combined = pd.concat(pieces)
        sums = [c for c in combined if c not in keys]
        result = combined.groupby(keys, as_index=False)[sums].sum()
        if "duration_count" in result:
            result["mean_duration_minutes"] = (
                result.duration_sum_seconds.div(result.duration_count.replace(0, float("nan"))) / 60
            )
        return result.sort_values(["trip_count", *keys], ascending=[False, True, True])

    def od_totals(self, filters: Filters, destinations=()) -> pd.DataFrame:
        return self._od_totals_cached(filters, tuple(sorted(map(int, destinations)))).copy(
            deep=True
        )

    def export_zone_hours(self, filters: Filters) -> tuple[str, int, bool]:
        """Read only the bounded prefix; the UI and filename declare truncation."""
        pieces, count, truncated = [], 0, False
        for batch in self._batches("zone_hour", filters):
            frame = batch.to_pandas()
            remaining = self.max_export - count
            if len(frame) > remaining:
                pieces.append(frame.iloc[:remaining])
                truncated = True
                break
            pieces.append(frame)
            count += len(frame)
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        frame.insert(0, "snapshot_id", self.snapshot_id)
        frame.insert(1, "export_truncated", truncated)
        return frame.to_csv(index=False), len(frame), truncated
