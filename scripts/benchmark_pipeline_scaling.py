#!/usr/bin/env python3
"""Run same-host full-pipeline measurements against already acquired TLC files."""

import argparse
from pathlib import Path

from mobility.benchmarks.pipeline_scaling import run_pipeline_scaling
from mobility.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Existing project/data root")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="New empty experiment directory")
    parser.add_argument(
        "--month-counts",
        type=int,
        nargs="+",
        default=[1, 3, 6],
        help="Lengths of chronological prefixes of configured months",
    )
    parser.add_argument("--backends", choices=["cpu", "gpu"], nargs="+", default=["cpu", "gpu"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Keep generated canonical/model artifacts; can use substantial disk",
    )
    args = parser.parse_args()
    result = run_pipeline_scaling(
        args.root,
        args.output,
        load_config(args.config),
        month_counts=args.month_counts,
        backends=args.backends,
        repeats=args.repeats,
        warmups=args.warmups,
        cpu_threads=args.cpu_threads,
        keep_runs=args.keep_runs,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
