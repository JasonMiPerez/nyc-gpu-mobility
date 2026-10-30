"""One public entry point for optional modeling after aggregate validation."""

from pathlib import Path

import pandas as pd

from .anomalies import fit_anomalies
from .clustering import fit_clusters


def run_models(
    canonical_dir: Path,
    zone_hour: pd.DataFrame,
    config: dict,
    output_dir: Path,
    backend: str = "cpu",
) -> dict:
    """Write models and compact serving tables without hiding incomplete evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = fit_clusters(Path(canonical_dir), config, output_dir, backend)
    anomalies = fit_anomalies(zone_hour, config, output_dir)
    tables = clusters["tables"] | anomalies["tables"]
    models = {"clustering": clusters["metadata"], "anomalies": anomalies["metadata"]}
    if config.get("duration_anomalies", {}).get("enabled", True):
        from .duration_anomalies import fit_duration_anomalies

        duration = fit_duration_anomalies(canonical_dir, config, output_dir)
        tables.update(duration["tables"])
        models["duration_anomalies"] = duration["metadata"]
    if config.get("graph", {}).get("enabled", True):
        from .graph_analysis import analyze_graph

        # Reduce each saved month before combining; never load all hourly edges at once.
        keys = ["pickup_zone_id", "dropoff_zone_id"]
        parts = []
        for path in sorted((output_dir / "od_hour").glob("*.parquet")):
            frame = pd.read_parquet(path, columns=keys + ["trip_count"])
            parts.append(frame.groupby(keys, as_index=False).trip_count.sum())
        edges = (
            pd.concat(parts).groupby(keys, as_index=False).trip_count.sum()
            if parts
            else pd.DataFrame(columns=keys + ["trip_count"])
        )
        zones = pd.read_parquet(output_dir / "zones.parquet")
        graph = analyze_graph(edges, zones, config, output_dir, backend=backend)
        tables.update(graph["tables"])
        models["graph"] = graph["metadata"]
    return {
        "tables": tables,
        "models": models,
        "status": {name: metadata["status"] for name, metadata in models.items()},
    }
