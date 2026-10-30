"""The command line keeps long-running compute jobs outside the public website."""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from mobility.config import load_config
from mobility.io import atomic_json, read_json


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(
        prog="mobility", description="NYC taxi analytics and GPU experiments"
    )
    app.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project/data root (default: current directory)",
    )
    app.add_argument("--config", type=Path, help="validated YAML configuration")
    commands = app.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check packages, storage, and optional GPU")
    doctor.add_argument("--gpu", action="store_true")
    data = commands.add_parser("data").add_subparsers(dest="action", required=True)
    data.add_parser("fetch", help="download and verify configured official months")
    inspect = data.add_parser("inspect", help="inspect a local source Parquet")
    inspect.add_argument("path", type=Path)
    zones = commands.add_parser("zones").add_subparsers(dest="action", required=True)
    zones.add_parser("prepare", help="prepare official zone lookup and map geometry")
    pipeline = commands.add_parser("pipeline").add_subparsers(dest="action", required=True)
    run = pipeline.add_parser("run", help="process configured sources and fit models")
    run.add_argument("--backend", choices=["cpu", "gpu"])
    run.add_argument("--no-models", action="store_true")
    run.add_argument("--publish", action="store_true")
    benchmark = commands.add_parser("benchmark").add_subparsers(dest="action", required=True)
    bench = benchmark.add_parser("run")
    bench.add_argument("--run", type=Path, required=True, help="completed run directory")
    bench.add_argument("--gpu", action="store_true")
    bench.add_argument("--publish", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--run", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--run", type=Path, required=True)
    dashboard = commands.add_parser("dashboard", help="serve only validated published results")
    dashboard.add_argument("--snapshot", type=Path)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8050)
    demo = commands.add_parser(
        "demo", help="build explicitly labeled offline synthetic fixture results"
    )
    demo.add_argument("--rows-per-month", type=int, default=12000)
    commands.add_parser("status", help="list completed and failed runs")
    return app


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root.resolve()
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            from mobility.pipeline import environment_record

            info = environment_record()
            info["free_disk_gb"] = round(shutil.disk_usage(root).free / 1e9, 2)
            info["gpu_requested"] = args.gpu
            if args.gpu:
                from mobility.backends.gpu import doctor

                info["gpu"] = doctor()
            print(json.dumps(info, indent=2, default=str))
        elif args.command == "data":
            from mobility.data.acquire import fetch_month, inspect_parquet

            if args.action == "fetch":
                for month in config["data"]["months"]:
                    result = fetch_month(root / "data", config["data"]["year"], month)
                    print(f"{result['source_id']}: {result['parquet_row_count']:,} verified rows")
            else:
                print(json.dumps(inspect_parquet(args.path), indent=2))
        elif args.command == "zones":
            from mobility.data.acquire import prepare_zones

            print(prepare_zones(root / "data"))
        elif args.command == "pipeline":
            from mobility.pipeline import run_pipeline, publish_run

            if args.backend:
                config["processing"]["backend"] = args.backend
            run = run_pipeline(root, config, fit_models=not args.no_models)
            print(f"Run: {run}")
            if args.publish:
                print(f"Published snapshot: {publish_run(run, root / 'artifacts')}")
        elif args.command == "benchmark":
            from mobility.benchmarks.runner import run_benchmarks
            from mobility.pipeline import publish_run

            run = args.run.resolve()
            manifest = read_json(run / "manifest.json")
            if manifest.get("status") != "complete":
                raise ValueError("Benchmark input must be a completed run")
            result = run_benchmarks(run / "canonical", run / "tables", config, include_gpu=args.gpu)
            manifest["tables"].update(result["tables"])
            manifest["benchmark"] = {k: v for k, v in result.items() if k != "tables"}
            manifest["benchmark_environment"] = read_json(run / "tables" / result["environment"])
            atomic_json(run / "manifest.json", manifest)
            print(json.dumps(result, indent=2, default=str))
            if args.publish:
                print(publish_run(run, root / "artifacts"))
        elif args.command in {"validate", "publish"}:
            from mobility.pipeline import publish_run, validate_outputs

            run = args.run.resolve()
            if args.command == "publish":
                print(publish_run(run, root / "artifacts"))
            else:
                manifest = read_json(run / "manifest.json")
                print(
                    json.dumps(
                        validate_outputs(run / "tables", manifest["validation"]["counts"]), indent=2
                    )
                )
        elif args.command == "dashboard":
            from mobility.pipeline import current_snapshot
            from mobility.dashboard.app import create_app

            snapshot = args.snapshot or current_snapshot(root / "artifacts")
            create_app(snapshot).run(host=args.host, port=args.port, debug=False)
        elif args.command == "demo":
            from mobility.demo import create_demo_sources
            from mobility.pipeline import publish_run, run_pipeline

            workspace = root / "artifacts" / "demo-workspace"
            if args.rows_per_month < 10:
                raise ValueError("Use at least 10 fixture rows per month")
            create_demo_sources(workspace, args.rows_per_month)
            run = run_pipeline(workspace, config)
            print(f"Synthetic demonstration: {publish_run(run, root / 'artifacts')}")
        elif args.command == "status":
            for path in sorted((root / "artifacts" / "runs").glob("*/manifest.json")):
                manifest = read_json(path)
                print(manifest["run_id"], manifest["status"], manifest.get("raw_rows", ""))
        return 0
    except KeyboardInterrupt:
        print("Canceled; the last published snapshot is unchanged.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"Configuration/data error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        logging.exception("Operation failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
