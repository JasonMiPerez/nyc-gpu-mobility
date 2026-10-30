"""Run from the repository root on the NVIDIA server: python scripts/gpu_smoke.py."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility.backends.gpu import doctor
from mobility.backends.cuda_features import (
    require_cupy,
    to_device,
    fused_features,
    library_features,
)
from mobility.backends.features import reference_features
import numpy as np


def main():
    report = doctor()
    cp = require_cupy()
    for size in [0, 1, 513, 100_003]:
        inputs = {
            "pickup_ns": np.full(size, 1705312800000000000, dtype="int64"),
            "dropoff_ns": np.full(size, 1705313700000000000, dtype="int64"),
            "distance": np.full(size, 3.0, dtype="float64"),
            **{
                name: np.ones(size, dtype="uint8")
                for name in [
                    "pickup_valid",
                    "dropoff_valid",
                    "time_allowed",
                    "pickup_zone_valid",
                    "dropoff_zone_valid",
                ]
            },
        }
        if size:
            inputs["time_allowed"][::11] = 0
            inputs["distance"][::7] = np.nan
        expected = reference_features(inputs)
        device_inputs = to_device(inputs)
        for implementation in [library_features, fused_features]:
            actual = implementation(device_inputs)
            cp.cuda.runtime.deviceSynchronize()
            for key in expected:
                np.testing.assert_allclose(
                    cp.asnumpy(actual[key]), expected[key], rtol=1e-10, atol=1e-9, equal_nan=True
                )
    report["custom_cuda_validation"] = {"status": "passed", "sizes": [0, 1, 513, 100_003]}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
