"""Trace real canonical records; profiling output is not benchmark timing data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobility.backends.cuda_features import require_cupy, to_device, fused_features, library_features
from mobility.backends.features import inputs_from_pandas
from mobility.benchmarks.aggregation import canonical_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical", type=Path)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--custom-only", action="store_true")
    args = parser.parse_args()
    frame = canonical_prefix(args.canonical, args.rows)
    device = to_device(inputs_from_pandas(frame))
    cp = require_cupy()
    for _ in range(2):
        fused_features(device)
        cp.cuda.runtime.deviceSynchronize()
    operations = [("custom", fused_features)]
    if not args.custom_only:
        operations.append(("library", library_features))
    for name, operation in operations:
        cp.cuda.nvtx.RangePush(name)
        for _ in range(5):
            operation(device)
        cp.cuda.runtime.deviceSynchronize()
        cp.cuda.nvtx.RangePop()
    print(f"Profiled {len(frame):,} actual canonical rows; these are instrumentation runs, not benchmark results.")


if __name__ == "__main__":
    main()
