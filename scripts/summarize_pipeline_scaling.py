#!/usr/bin/env python3
"""Read a scaling experiment and write local JSON plus a public-safe Markdown report."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def read_json(path):
    return json.loads(Path(path).read_text())


def maximum(values):
    valid = [float(value) for value in values if value is not None and pd.notna(value)]
    return max(valid) if valid else None


def describe_trial(experiment, row):
    directory = experiment / "trials" / row["trial_id"]
    manifest = (
        read_json(directory / "run_manifest.json")
        if (directory / "run_manifest.json").exists()
        else {}
    )
    resources = (
        read_json(directory / "resources.json") if (directory / "resources.json").exists() else {}
    )
    worker = (
        read_json(directory / "worker_result.json")
        if (directory / "worker_result.json").exists()
        else {}
    )
    comparison = (
        read_json(directory / "comparison.json") if (directory / "comparison.json").exists() else {}
    )
    models = manifest.get("models", {})
    cluster, graph, duration = [
        models.get(name, {}) for name in ("clustering", "graph", "duration_anomalies")
    ]
    reference = graph.get("cpu_reference_elapsed_s")
    elapsed = row.get("elapsed_s")
    return {
        "trial_id": row["trial_id"],
        "backend": row["backend"],
        "months": int(row["months"]),
        "repeat": int(row["repeat"]),
        "phase": row["phase"],
        "validation": row["validation"],
        "elapsed_s": float(elapsed) if pd.notna(elapsed) else None,
        "manifest_available": bool(manifest),
        "execution_mode": manifest.get("execution_mode", "legacy"),
        "execution_plan": manifest.get("execution_plan", {}),
        "gpu_batch_reference_checks": manifest.get("gpu_batch_reference_checks", False),
        "models": {"graph": {"cpu_reference_elapsed_s": graph.get("cpu_reference_elapsed_s")}},
        "loaded_gpu_modules": worker.get("loaded_gpu_modules"),
        "cpu_import_guard_enabled": worker.get("cpu_import_guard_enabled"),
        "graph_external_validation": comparison.get("graph_validation", {}),
        "resources": resources,
        "selected_k": cluster.get("selected_k"),
        "cluster_status": cluster.get("status"),
        "cluster_test_status": cluster.get("test_status"),
        "cluster_eligible_rows": cluster.get("eligible_rows"),
        "canonical_eligible_rows": cluster.get("canonical_eligible_rows"),
        "source_rows": manifest.get("raw_rows"),
        "endpoint_counts": manifest.get("row_counts"),
        "graph_reference_s": reference,
        "extra_graph_reference_s": reference if row["backend"] == "gpu" else 0.0,
        "extra_graph_reference_pct": 100 * reference / elapsed
        if row["backend"] == "gpu" and reference is not None and pd.notna(elapsed) and elapsed > 0
        else 0.0,
        "pagerank_max_abs_error": comparison.get("graph_validation", {})
        .get("actual", {})
        .get("pagerank_max_abs_error", graph.get("validation", {}).get("pagerank_max_abs_error")),
        "graph_communities": graph.get("community_count"),
        "graph_isolates": graph.get("isolated_zone_count"),
        "graph_modularity": graph.get("modularity"),
        "duration_status": duration.get("status"),
        "duration_calibration_status": duration.get("calibration", {}).get("status"),
        "duration_threshold": duration.get("threshold"),
        "duration_test_alert_bins": duration.get("test_alert_bins"),
        "duration_test_incidents": duration.get("test_incidents"),
    }


def build_report(experiment):
    experiment = Path(experiment).resolve()
    environment = read_json(experiment / "benchmark_environment.json")
    trials = pd.read_parquet(experiment / "benchmark_runs.parquet")
    summary = pd.read_parquet(experiment / "benchmark_summary.parquet")
    measured = trials.loc[trials.phase.eq("measured")]
    expected = int(environment["repeats"])
    details = [describe_trial(experiment, row) for row in trials.to_dict("records")]
    report = {
        "experiment_id": environment["experiment_id"],
        "experiment_status": environment["status"],
        "expected_repetitions": expected,
        "timing_scope": environment.get("timing_scope"),
        "execution_mode": environment.get("execution_mode", "legacy"),
        "sizes": [],
        "trials": details,
        "publication_candidate": None,
        "failed_trials": trials.loc[trials.validation.eq("failed"), "trial_id"].tolist(),
    }
    for months in environment["requested_month_counts"]:
        selected = measured.loc[measured.months.eq(months)]
        size = {"months": months, "status": "validated", "backends": {}, "issues": []}
        for backend in environment["backends"]:
            rows = selected.loc[selected.backend.eq(backend)]
            valid = (
                len(rows) == expected
                and sorted(rows.repeat.tolist()) == list(range(expected))
                and rows.validation.eq("passed").all()
                and rows.elapsed_s.gt(0).all()
                and rows.elapsed_s.map(lambda value: pd.notna(value) and math.isfinite(value)).all()
            )
            if not valid:
                size["issues"].append(
                    f"{backend}: missing, duplicate, invalid, or failed measured repetitions"
                )
            subset = [
                item
                for item in details
                if item["months"] == months
                and item["backend"] == backend
                and item["phase"] == "measured"
            ]
            if any(not item["manifest_available"] for item in subset):
                size["issues"].append(f"{backend}: a trial manifest is missing")
            if any(item["execution_mode"] != report["execution_mode"] for item in subset):
                size["issues"].append(f"{backend}: execution mode differs from the experiment")
            if report["execution_mode"] == "native":
                from mobility.benchmarks.pipeline_scaling import validate_execution_plan

                for item in subset:
                    try:
                        validate_execution_plan(item, backend)
                        if backend == "cpu" and (
                            item["loaded_gpu_modules"] != []
                            or item["cpu_import_guard_enabled"] is not True
                        ):
                            raise ValueError("CPU CUDA-import exclusion was not established")
                        memory = item["resources"].get("gpu", {}).get("sampled_process_peak_bytes")
                        if backend == "cpu" and memory is not None and memory > 0:
                            raise ValueError("CPU trial used GPU process memory")
                        if item["graph_external_validation"].get("status") != "passed":
                            raise ValueError("Independent post-timing graph validation is missing")
                    except ValueError as error:
                        size["issues"].append(f"{item['trial_id']}: {error}")
            metrics = {
                "repetitions_present": len(rows),
                "validated": bool(valid),
                "median_s": float(rows.elapsed_s.median()) if valid else None,
                "q25_s": float(rows.elapsed_s.quantile(0.25)) if valid else None,
                "q75_s": float(rows.elapsed_s.quantile(0.75)) if valid else None,
                "max_sampled_host_rss_bytes": maximum(
                    item["resources"].get("rss", {}).get("sampled_peak_bytes") for item in subset
                ),
                "max_sampled_gpu_bytes": maximum(
                    item["resources"].get("gpu", {}).get("sampled_process_peak_bytes")
                    for item in subset
                ),
            }
            for kind in ("rss", "gpu"):
                samples = [item["resources"].get(kind, {}) for item in subset]
                metrics[kind] = {
                    "samples": sum(item.get("samples", 0) for item in samples),
                    "max_observed_gap_s": maximum(
                        item.get("max_observed_gap_s") for item in samples
                    ),
                    "sampling_errors": sum(item.get("sampling_errors", 0) for item in samples),
                    "unavailable_trials": sum(not item.get("samples", 0) for item in samples),
                }
            metrics["rss"]["incomplete_process_tree_trials"] = sum(
                not item["resources"].get("rss", {}).get("process_tree_complete", False)
                for item in subset
            )
            metrics["background_gpu_processes_observed"] = any(
                item["resources"].get("gpu", {}).get("background_activity_observed", False)
                for item in subset
            )
            if valid:
                saved = summary.loc[
                    summary.backend.eq(backend) & summary.input_rows.eq(rows.input_rows.iloc[0])
                ]
                if (
                    len(saved) != 1
                    or int(saved.iloc[0].repetitions) != expected
                    or any(
                        not math.isclose(
                            float(saved.iloc[0][key]), metrics[key], rel_tol=1e-10, abs_tol=1e-10
                        )
                        for key in ("median_s", "q25_s", "q75_s")
                    )
                ):
                    size["issues"].append(f"{backend}: saved summary disagrees with raw trials")
            size["backends"][backend] = metrics
        for key in ("protocol_hash", "source_set_sha256", "input_rows", "timing_scope", "stage"):
            if selected[key].nunique(dropna=False) != 1:
                size["issues"].append(f"Measured trials do not share one {key}")
        if report["execution_mode"] == "native" and (
            not selected.stage.eq("native_production_pipeline").all()
            or not selected.timing_scope.eq(environment["timing_scope"]).all()
        ):
            size["issues"].append("Native and earlier hybrid benchmark scopes cannot be combined")
        size["input_rows"] = (
            int(selected.input_rows.iloc[0])
            if len(selected) and selected.input_rows.nunique() == 1
            else None
        )
        if size["issues"] or environment["status"] != "passed" or report["failed_trials"]:
            size["status"] = "not_validated"
        if size["status"] == "validated" and {"cpu", "gpu"} <= set(size["backends"]):
            ratio = size["backends"]["cpu"]["median_s"] / size["backends"]["gpu"]["median_s"]
            size.update(
                cpu_time_over_gpu_time=ratio,
                faster_median="GPU" if ratio > 1 else "CPU" if ratio < 1 else "Equal",
                winner_factor=max(ratio, 1 / ratio),
            )
        report["sizes"].append(size)
    if all(size["status"] == "validated" for size in report["sizes"]):
        candidates = measured.loc[
            measured.months.eq(6) & measured.backend.eq("gpu") & measured.repeat.eq(0)
        ]
        if len(candidates) == 1:
            row = candidates.iloc[0]
            retained = row.get("retained_run")
            report["publication_candidate"] = {
                "selection_rule": "First measured six-month GPU repetition (repeat 0), independent of speed",
                "trial_id": row.trial_id,
                "retained_run": retained if pd.notna(retained) else None,
                "metrics": next(item for item in details if item["trial_id"] == row.trial_id),
            }
    return report


def markdown(report):
    lines = [
        "# Full-pipeline scaling report",
        "",
        f"Experiment `{report['experiment_id']}` · status: **{report['experiment_status']}**.",
        f"Execution mode: **{report['execution_mode']}**; timing scope: `{report['timing_scope']}`. Compare only matching experiment scopes.",
        f"Expected measured repetitions per backend and size: {report['expected_repetitions']}. Warmups are excluded.",
        "",
        "| Months | Source rows | Backend | Status | Repetitions | Median s | Q25–Q75 s |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for size in report["sizes"]:
        for backend, row in size["backends"].items():
            times = (
                f"{row['median_s']:.6f} | {row['q25_s']:.6f}–{row['q75_s']:.6f}"
                if size["status"] == "validated"
                else "— | —"
            )
            lines.append(
                f"| {size['months']} | {size['input_rows']} | {backend.upper()} | {size['status']} | {row['repetitions_present']} | {times} |"
            )
    for size in report["sizes"]:
        if "cpu_time_over_gpu_time" in size:
            winner = (
                "Equal measured medians"
                if size["faster_median"] == "Equal"
                else f"{size['faster_median']} has the faster measured median ({size['winner_factor']:.3f}×)"
            )
            lines += [
                "",
                f"{size['months']} months: CPU time / GPU time = **{size['cpu_time_over_gpu_time']:.3f}×**; {winner}.",
            ]
        elif size["issues"]:
            lines += [
                "",
                f"{size['months']} months: not validated. " + "; ".join(size["issues"]) + ".",
            ]
    lines += [
        "",
        "| Months | Backend | Max sampled host RSS GiB | Max sampled GPU GiB | RSS/GPU max gap ms | Sampling errors RSS/GPU | Other GPU processes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for size in report["sizes"]:
        for backend, row in size["backends"].items():
            memory = [
                "—" if row[key] is None else f"{row[key] / 2**30:.3f}"
                for key in ("max_sampled_host_rss_bytes", "max_sampled_gpu_bytes")
            ]
            gaps = [
                "—"
                if row[kind]["max_observed_gap_s"] is None
                else f"{1000 * row[kind]['max_observed_gap_s']:.1f}"
                for kind in ("rss", "gpu")
            ]
            lines.append(
                f"| {size['months']} | {backend.upper()} | {' | '.join(memory)} | {' / '.join(gaps)} | {row['rss']['sampling_errors']} / {row['gpu']['sampling_errors']} | {row['background_gpu_processes_observed']} |"
            )
    lines += [
        "",
        "Memory values are sampled maxima across measured repetitions, not exact high-water marks. Requested intervals are 10 ms RSS and 20 ms GPU; missing readings and incomplete process-tree coverage remain in the JSON. Other GPU processes indicate possible contention. Device utilization is system-wide.",
        "",
        "| Measured trial | Selected k | Cluster selection / test | CPU graph reference s | Communities | Duration test alert hours |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for trial in report["trials"]:
        if trial["phase"] == "measured":
            reference = (
                "—" if trial["graph_reference_s"] is None else f"{trial['graph_reference_s']:.6f}"
            )
            lines.append(
                f"| {trial['trial_id']} ({trial['validation']}) | {trial['selected_k']} | {trial['cluster_status']} / {trial['cluster_test_status']} | {reference} | {trial['graph_communities']} | {trial['duration_test_alert_bins']} |"
            )
    lines += [
        "",
        (
            "Native GPU trials run all bulk preparation, model and graph stages on the GPU; CPU trials block CUDA package imports and hide CUDA devices. Independent CPU reference checks run after the stopwatch. Per-stage backend evidence and module audits are retained in the JSON. "
            if report["execution_mode"] == "native"
            else "Legacy GPU runs include the CPU graph reference as validation work. In CPU runs, this is the graph computation itself, not additional overhead. "
        )
        + f"Short chronological prefixes lack later calibration/test data. {report['expected_repetitions']} repetitions describe this experiment, not a universal winner or statistical significance.",
    ]
    candidate = report["publication_candidate"]
    if candidate:
        metrics = candidate["metrics"]
        lines += [
            "",
            f"Publication candidate: `{candidate['trial_id']}`, selected by the first measured six-month GPU repetition rule, independently of runtime. It processed {metrics['source_rows']:,} source rows and {metrics['cluster_eligible_rows']:,} clustering-eligible trips. The local JSON retains its artifact location; no host paths are published here.",
            "",
            f"Candidate graph: {metrics['graph_communities']} communities, {metrics['graph_isolates']} isolates; maximum PageRank absolute error versus CPU: {metrics['pagerank_max_abs_error']}. Duration model: {metrics['duration_status']}; calibration: {metrics['duration_calibration_status']}; selected threshold: {metrics['duration_threshold']}; test alert hours/incidents: {metrics['duration_test_alert_bins']}/{metrics['duration_test_incidents']}.",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--json", type=Path, required=True, dest="json_output")
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    for destination in (args.json_output, args.markdown):
        if destination.resolve().is_relative_to(args.experiment.resolve()):
            parser.error("Report outputs must be outside the read-only experiment directory")
    report = build_report(args.experiment)
    for path, content in (
        (args.json_output, json.dumps(report, indent=2, allow_nan=False) + "\n"),
        (args.markdown, markdown(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(markdown(report))


if __name__ == "__main__":
    main()
