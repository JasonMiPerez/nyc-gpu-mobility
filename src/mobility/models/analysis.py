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
    return {
        "tables": clusters["tables"] | anomalies["tables"],
        "models": {"clustering": clusters["metadata"], "anomalies": anomalies["metadata"]},
        "status": {
            "clustering": clusters["metadata"]["status"],
            "anomalies": anomalies["metadata"]["status"],
        },
    }
