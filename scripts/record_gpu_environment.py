"""Record GPU hardware/package versions without allocating a CUDA context."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pynvml

    pynvml.nvmlInit()
    try:
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            devices.append(
                {
                    "index": index,
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "memory_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                }
            )
        gpu = {
            "name": ", ".join(item["name"] for item in devices),
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "devices": devices,
        }
    finally:
        pynvml.nvmlShutdown()
    packages = {}
    for name in ["cudf", "cuml", "cugraph", "cupy", "rmm", "networkx", "nvidia-ml-py"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "packages": packages,
        "method": "NVML hardware query and installed-package metadata; no CUDA context allocated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
