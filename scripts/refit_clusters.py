"""Refit clustering in a new run while retaining existing aggregates and benchmarks.

Usage:
    python scripts/refit_clusters.py --source-run artifacts/runs/RUN_ID --publish

Canonical files are read from the source run. Nothing in the source run or its
snapshots is rewritten, and historical anomaly and benchmark results are copied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from mobility.config import ProjectConfig
from mobility.io import atomic_json, read_json, sha256_file, utc_now
from mobility.models.clustering import fit_clusters
from mobility.pipeline import environment_record, publish_run, validate_outputs


CLUSTER_TABLES = {"cluster_summary", "trip_plot_sample", "trip_cluster_assignment"}
CLUSTER_FILES = {
    "clustering.json",
    "cluster_parameters.json",
    "cluster_train_sample_ids.parquet",
    "cluster_calibration_sample_ids.parquet",
    "cluster_test_sample_ids.parquet",
    "cluster_summary.parquet",
    "trip_plot_sample.parquet",
    "trip_cluster_assignment.parquet",
}
POLICY_REVISION_NOTE = (
    "These fixed plausibility limits were introduced after visual review of a sample spanning "
    "the study period. They are a data-quality revision, not thresholds optimized for silhouette "
    "scores. Training, calibration, and test periods remain chronological, but the revised-policy "
    "test results are not a fresh, entirely unseen confirmation of the whole study design."
)


def refit_clusters(source_run: Path, artifacts_root: Path) -> Path:
    """Create a complete run; publication is a separate final step."""
    started = time.perf_counter()
    source_run, artifacts_root = source_run.resolve(), artifacts_root.resolve()
    source_manifest_path = source_run / "manifest.json"
    source_manifest_hash = sha256_file(source_manifest_path)
    source = read_json(source_manifest_path)
    if source.get("status") != "complete":
        raise ValueError("Refitting requires a completed source run.")
    canonical = source_run / "canonical"
    canonical_paths = sorted(canonical.rglob("*.parquet"))
    if not canonical_paths:
        raise ValueError("The source run must contain its original canonical Parquet files.")
    config = ProjectConfig.model_validate(source["config"]).model_dump(mode="json")
    config["processing"]["backend"] = "cpu"
    config["processing"]["custom_cuda"] = False
    canonical_files = {
        str(path.relative_to(canonical)): {
            "sha256": sha256_file(path),
            "rows": pq.ParquetFile(path).metadata.num_rows,
        }
        for path in canonical_paths
    }
    if sum(item["rows"] for item in canonical_files.values()) != source["canonical_rows"]:
        raise ValueError("Canonical files do not reconcile to the source manifest.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = artifacts_root / "runs" / run_id
    output = run_dir / "tables"
    output.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(source)
    manifest.update(
        run_id=run_id,
        created_at=utc_now(),
        status="processing",
        config=config,
        config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        backend="cpu",
        environment=environment_record(),
        tables={
            name: path for name, path in source["tables"].items() if name not in CLUSTER_TABLES
        },
        refit={
            "operation": "clustering_only",
            "source_run_id": source["run_id"],
            "source_manifest_sha256": source_manifest_hash,
            "canonical_source": f"runs/{source['run_id']}/canonical",
            "canonical_files": canonical_files,
            "canonical_storage": "Referenced from the source run, not copied or modified.",
            "source_environment": source.get("environment"),
            "source_elapsed_s": source.get("elapsed_s"),
            "inherited_pipeline_timings": (
                "These are the original source ingestion timings; ingestion was not rerun."
            ),
            "anomaly_results": "Copied unchanged; the anomaly model was not rerun.",
            "historical_benchmarks": (
                "Copied unchanged from the source run. They measure the earlier implementation "
                "and its recorded inputs, not this new clustering filter policy or model fit."
            ),
            "policy_revision_context": POLICY_REVISION_NOTE,
        },
    )
    manifest.pop("completed_at", None)
    manifest["models"]["clustering"] = {"status": "pending_refit"}
    manifest["model_status"]["clustering"] = "pending_refit"
    atomic_json(run_dir / "manifest.json", manifest)
    print(f"New run: {run_dir}", flush=True)
    try:
        # Copy independent files; never hardlink outputs that a fit could overwrite.
        for path in sorted((source_run / "tables").iterdir()):
            if path.name in CLUSTER_FILES:
                continue
            target = output / path.name
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
        inherited_hashes = {
            str(path.relative_to(output)): sha256_file(path)
            for path in output.rglob("*")
            if path.is_file()
        }
        print(f"Refitting CPU clustering from {len(canonical_files)} original files...", flush=True)
        fitted = fit_clusters(canonical, config, output, backend="cpu")
        if fitted["metadata"]["status"] == "insufficient_training_rows":
            raise ValueError("Clustering refit has insufficient training rows.")
        for relative, digest in inherited_hashes.items():
            if sha256_file(output / relative) != digest:
                raise ValueError(f"Refit changed an inherited table: {relative}")
        if sha256_file(source_manifest_path) != source_manifest_hash:
            raise ValueError("The source manifest changed during the refit.")
        validation = validate_outputs(output, source["validation"]["counts"])
        manifest["tables"].update(fitted["tables"])
        manifest["models"]["clustering"] = fitted["metadata"]
        manifest["model_status"]["clustering"] = fitted["metadata"]["status"]
        manifest["refit"]["inherited_table_sha256"] = inherited_hashes
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            validation=validation,
            row_counts=validation["counts"],
            elapsed_s=time.perf_counter() - started,
        )
        atomic_json(run_dir / "manifest.json", manifest)
        print(json.dumps(fitted["metadata"]["eligibility"], indent=2), flush=True)
        print(f"Selected clusters: {fitted['metadata']['selected_k']}", flush=True)
        return run_dir
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="canceled" if isinstance(error, KeyboardInterrupt) else "failed",
            failed_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(run_dir / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--publish", action="store_true", help="Publish and update current.json.")
    args = parser.parse_args()
    artifacts_root = args.artifacts_root or args.source_run.resolve().parent.parent
    run_dir = refit_clusters(args.source_run, artifacts_root)
    if args.publish:
        snapshot = publish_run(run_dir, artifacts_root)
        print(f"Published snapshot: {snapshot}", flush=True)


if __name__ == "__main__":
    main()
