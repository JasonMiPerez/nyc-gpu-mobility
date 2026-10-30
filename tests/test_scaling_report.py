"""Report all repetitions and choose publication artifacts independently of speed."""

import importlib.util
import json
from pathlib import Path

import pandas as pd


SPEC = importlib.util.spec_from_file_location(
    "scaling_report", Path(__file__).parents[1] / "scripts/summarize_pipeline_scaling.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def experiment(tmp_path):
    write_json(
        tmp_path / "benchmark_environment.json",
        {
            "experiment_id": "fixture",
            "status": "passed",
            "repeats": 3,
            "requested_month_counts": [6],
            "backends": ["cpu", "gpu"],
            "timing_scope": "full_pipeline",
        },
    )
    rows, summaries = [], []
    for backend, times in (("cpu", [12.0, 10.0, 14.0]), ("gpu", [9.0, 5.0, 7.0])):
        for repeat, elapsed in enumerate(times):
            trial_id = f"months-6-measured-{repeat}-{backend}"
            rows.append(
                {
                    "trial_id": trial_id,
                    "backend": backend,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "validation": "passed",
                    "phase": "measured",
                    "months": 6,
                    "input_rows": 100,
                    "protocol_hash": "same",
                    "source_set_sha256": "source",
                    "timing_scope": "full_pipeline",
                    "stage": "pipeline",
                    "retained_run": f"/private/server/runs/{trial_id}",
                }
            )
            write_json(
                tmp_path / "trials" / trial_id / "run_manifest.json",
                {
                    "raw_rows": 100,
                    "row_counts": {"pickup": 99},
                    "models": {
                        "clustering": {
                            "selected_k": 6,
                            "status": "calibration_selected",
                            "test_status": "evaluated",
                            "eligible_rows": 90,
                        },
                        "graph": {
                            "cpu_reference_elapsed_s": 0.25,
                            "community_count": 4,
                            "validation": {"pagerank_max_abs_error": 1e-10},
                        },
                        "duration_anomalies": {"test_alert_bins": 3, "threshold": 5},
                    },
                },
            )
            write_json(
                tmp_path / "trials" / trial_id / "resources.json",
                {
                    "rss": {
                        "sampled_peak_bytes": 2**30 * (repeat + 1),
                        "samples": 50,
                        "max_observed_gap_s": 0.015,
                        "sampling_errors": 0,
                        "process_tree_complete": True,
                    },
                    "gpu": {
                        "sampled_process_peak_bytes": 2**30 if backend == "gpu" else 0,
                        "samples": 25,
                        "max_observed_gap_s": 0.025,
                        "sampling_errors": 1,
                        "background_activity_observed": True,
                    },
                },
            )
        series = pd.Series(times)
        summaries.append(
            {
                "backend": backend,
                "input_rows": 100,
                "repetitions": 3,
                "median_s": series.median(),
                "q25_s": series.quantile(0.25),
                "q75_s": series.quantile(0.75),
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "benchmark_runs.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(tmp_path / "benchmark_summary.parquet", index=False)
    return tmp_path


def test_complete_report_recomputes_medians_and_chooses_repeat_zero(tmp_path):
    report = REPORT.build_report(experiment(tmp_path))
    size = report["sizes"][0]
    assert size["status"] == "validated"
    assert size["backends"]["cpu"]["median_s"] == 12
    assert size["backends"]["gpu"]["q25_s"] == 6
    assert size["cpu_time_over_gpu_time"] == 12 / 7
    assert size["backends"]["cpu"]["max_sampled_host_rss_bytes"] == 3 * 2**30
    assert size["backends"]["gpu"]["gpu"]["sampling_errors"] == 3
    candidate = report["publication_candidate"]
    assert candidate["trial_id"] == "months-6-measured-0-gpu"  # Slowest, still the fixed choice.
    assert candidate["metrics"]["elapsed_s"] == 9
    assert candidate["retained_run"].startswith("/private/server")
    markdown = REPORT.markdown(report)
    assert "/private/server" not in markdown
    assert "12.000000" in markdown and "7.000000" in markdown
    assert "Other GPU processes" in markdown
    json.dumps(report, allow_nan=False)


def test_missing_failed_or_mismatched_trials_never_produce_a_ratio(tmp_path):
    root = experiment(tmp_path)
    path = root / "benchmark_runs.parquet"
    trials = pd.read_parquet(path)
    trials.loc[trials.backend.eq("gpu") & trials.repeat.eq(1), "validation"] = "failed"
    trials.to_parquet(path, index=False)
    report = REPORT.build_report(root)
    assert report["sizes"][0]["status"] == "not_validated"
    assert "cpu_time_over_gpu_time" not in report["sizes"][0]
    assert report["publication_candidate"] is None
    assert "not_validated" in REPORT.markdown(report)


def test_saved_summary_disagreement_is_reported(tmp_path):
    root = experiment(tmp_path)
    path = root / "benchmark_summary.parquet"
    summary = pd.read_parquet(path)
    summary.loc[summary.backend.eq("gpu"), "median_s"] = 0.01
    summary.to_parquet(path, index=False)
    report = REPORT.build_report(root)
    assert report["sizes"][0]["status"] == "not_validated"
    assert "saved summary disagrees" in " ".join(report["sizes"][0]["issues"])


def test_native_report_requires_cpu_import_guard_and_complete_gpu_stage_plan(tmp_path):
    from mobility.benchmarks.pipeline_scaling import BULK_STAGES, SCOPE, STAGE

    root = experiment(tmp_path)
    environment = json.loads((root / "benchmark_environment.json").read_text())
    environment.update(execution_mode="native", timing_scope=SCOPE)
    write_json(root / "benchmark_environment.json", environment)
    trials = pd.read_parquet(root / "benchmark_runs.parquet")
    trials["timing_scope"], trials["stage"] = SCOPE, STAGE
    trials.to_parquet(root / "benchmark_runs.parquet", index=False)
    for row in trials.to_dict("records"):
        directory = root / "trials" / row["trial_id"]
        manifest = json.loads((directory / "run_manifest.json").read_text())
        manifest.update(
            execution_mode="native",
            execution_plan={
                name: {"backend": row["backend"], "implementation": "fixture", "rows": 100}
                for name in BULK_STAGES
            },
        )
        if row["backend"] == "gpu":
            manifest["models"]["graph"]["cpu_reference_elapsed_s"] = None
        write_json(directory / "run_manifest.json", manifest)
        write_json(
            directory / "worker_result.json",
            {
                "loaded_gpu_modules": []
                if row["backend"] == "cpu"
                else ["cupy", "cudf", "cuml", "cugraph"],
                "cpu_import_guard_enabled": row["backend"] == "cpu",
            },
        )
        write_json(
            directory / "comparison.json",
            {"graph_validation": {"status": "passed", "actual": {"pagerank_max_abs_error": 1e-12}}},
        )
    valid = REPORT.build_report(root)
    assert valid["sizes"][0]["status"] == "validated"
    assert "Independent CPU reference checks run after" in REPORT.markdown(valid)
    corrupted = root / "trials/months-6-measured-0-gpu/run_manifest.json"
    manifest = json.loads(corrupted.read_text())
    manifest["execution_plan"]["scaler_fit"]["backend"] = "cpu"
    write_json(corrupted, manifest)
    failed = REPORT.build_report(root)
    assert failed["sizes"][0]["status"] == "not_validated"
    assert failed["publication_candidate"] is None
    assert "scaler_fit" in " ".join(failed["sizes"][0]["issues"])
