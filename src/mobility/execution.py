"""Record execution locations for the separate native implementations."""


def execution_plan(rows, models, backend, hourly_rows):
    implementation = "cuDF/CuPy device columns" if backend == "gpu" else "pandas/NumPy CPU columns"
    plan = {
        name: {"backend": backend, "implementation": implementation, "rows": rows}
        for name in ("normalization", "eligibility", "aggregation")
    }
    cluster = models.get("clustering", {})
    for name in (
        "features",
        "sampling",
        "scaler_fit",
        "scaler_transform",
        "kmeans_fit",
        "kmeans_predict",
    ):
        if not cluster:
            continue
        count = cluster.get("eligible_rows", 0)
        if name in ("scaler_fit", "kmeans_fit"):
            count = cluster.get("samples", {}).get("train", {}).get("sample_rows", 0)
        evidence = cluster.get("execution_evidence", {}).get(name)
        if not evidence or evidence.get("backend") != backend:
            raise ValueError(f"Missing actual {backend} clustering evidence for {name}")
        plan[name] = {**evidence, "rows": count}
    for name, model_name, count in (
        ("volume_scoring", "anomalies", hourly_rows),
        ("duration_aggregation", "duration_anomalies", rows),
        (
            "duration_scoring",
            "duration_anomalies",
            models.get("duration_anomalies", {}).get("aggregation", {}).get("hourly_rows", 0),
        ),
        ("graph", "graph", models.get("graph", {}).get("directed_edge_count", 0)),
    ):
        if model_name in models:
            plan[name] = {
                "backend": backend,
                "implementation": "cuDF/CuPy/cuGraph"
                if backend == "gpu"
                else "pandas/NumPy/NetworkX",
                "rows": count,
            }
            if (
                name == "graph"
                and backend == "gpu"
                and not models["graph"].get("gpu_algorithms_executed", False)
            ):
                plan[name]["implementation"] = "Empty graph: no GPU algorithm required"
                plan[name]["algorithms_executed"] = False
    if backend == "gpu":
        plan["host_services"] = {
            "backend": "host_io",
            "rows": 0,
            "implementation": "File discovery/checksums, Parquet metadata, small zone/time calendar, JSON and job control",
        }
        plan["reduced_output_finalization"] = {
            "backend": "host_finalize",
            "rows": sum(m.get("test_alert_bins", 0) for m in models.values()),
            "implementation": "Small sample identity hashes, model metadata, alert-incident grouping, bounded graph input preparation, labels and summaries",
        }
    return plan
