"""Compile and launch our CUDA C++ source through CuPy's NVRTC interface."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .features import INPUT_DTYPES, OUTPUT_DTYPES, NAT, validate_inputs


def require_cupy():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device is visible")
        return cp
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "A working NVIDIA CUDA/CuPy environment is required; CPU fallback is disabled"
        ) from exc


@lru_cache(maxsize=4)
def _kernel(device_id):
    cp = require_cupy()
    source = Path(__file__).with_name("kernels") / "fused_trip_features.cu"
    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=source.read_text(), options=("--std=c++17",))
        return module.get_function("fused_trip_features")


def to_device(inputs):
    cp = require_cupy()
    validate_inputs(inputs)
    return {name: cp.asarray(inputs[name]) for name in INPUT_DTYPES}


def allocate_outputs(size):
    cp = require_cupy()
    return {name: cp.empty(size, dtype=dtype) for name, dtype in OUTPUT_DTYPES.items()}


def fused_features(inputs, outputs=None, threads=256):
    """Launch on CuPy's current stream; caller synchronizes before reading/timing.

    Arrays retain their ownership in the caller. Supplying output buffers permits
    a true kernel-only CUDA-event experiment without timing allocations.
    """
    cp = require_cupy()
    size = validate_inputs(inputs, cp.ndarray)
    if threads < 32 or threads > 1024 or threads % 32:
        raise ValueError("threads must be a multiple of 32 between 32 and 1024")
    device = inputs["pickup_ns"].device.id
    if device != cp.cuda.Device().id:
        raise ValueError("Input device differs from the current CUDA device")
    outputs = allocate_outputs(size) if outputs is None else outputs
    if set(outputs) != set(OUTPUT_DTYPES):
        raise ValueError("Unexpected output buffer names")
    for name, dtype in OUTPUT_DTYPES.items():
        value = outputs[name]
        if (
            not isinstance(value, cp.ndarray)
            or value.dtype != dtype
            or value.shape != (size,)
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"Invalid output buffer: {name}")
        if value.device.id != device:
            raise ValueError("Input and output buffers must use the same CUDA device")
        if size:
            begin, end = value.data.ptr, value.data.ptr + value.nbytes
            for other_name, other in [*inputs.items(), *outputs.items()]:
                if other_name == name:
                    continue
                if begin < other.data.ptr + other.nbytes and other.data.ptr < end:
                    raise ValueError("Output buffers must not overlap other buffers")
    if size:
        blocks = min((size + threads - 1) // threads, 65535)
        args = tuple(inputs[name] for name in INPUT_DTYPES)
        args += tuple(outputs[name] for name in OUTPUT_DTYPES) + (np.int64(size),)
        _kernel(device)((blocks,), (threads,), args, stream=cp.cuda.get_current_stream())
        # CuPy checks the driver launch result. Deferred execution errors surface
        # at the caller's explicit stream/device synchronization boundary.
    return outputs


def prepare_fused_launch(inputs, outputs, threads=256):
    """Validate/compile once, then return a minimal launch for CUDA-event timing.

    The returned callable owns references to all buffers and is bound to the
    stream/device active during preparation. Do not mutate their shapes or
    change streams between event recording and launch.
    """
    cp = require_cupy()
    fused_features(inputs, outputs, threads)
    size = len(inputs["pickup_ns"])
    stream = cp.cuda.get_current_stream()
    kernel = _kernel(inputs["pickup_ns"].device.id)
    blocks = min((size + threads - 1) // threads, 65535)
    args = tuple(inputs[name] for name in INPUT_DTYPES)
    args += tuple(outputs[name] for name in OUTPUT_DTYPES) + (np.int64(size),)

    def launch():
        if size:
            kernel((blocks,), (threads,), args, stream=stream)
        return outputs

    return launch


def library_features(inputs):
    """Separate CuPy array operations implementing the identical kernel contract."""
    cp = require_cupy()
    size = validate_inputs(inputs, cp.ndarray)
    pickup, dropoff, distance = inputs["pickup_ns"], inputs["dropoff_ns"], inputs["distance"]
    pu_ok = (inputs["pickup_valid"] != 0) & (pickup != NAT)
    do_ok = (inputs["dropoff_valid"] != 0) & (dropoff != NAT)
    allowed = inputs["time_allowed"] != 0
    zones_ok = (inputs["pickup_zone_valid"] != 0) & (inputs["dropoff_zone_valid"] != 0)
    distance_ok = cp.isfinite(distance) & (distance >= 0)
    valid_times = pu_ok & do_ok & allowed
    # Replace invalid timestamps before arithmetic, including NaT sentinels.
    safe_p = cp.where(valid_times, pickup, cp.int64(0))
    safe_d = cp.where(valid_times, dropoff, cp.int64(0))
    duration = (safe_d // 1_000_000_000 - safe_p // 1_000_000_000).astype(cp.float64)
    duration += (safe_d % 1_000_000_000 - safe_p % 1_000_000_000) * 1e-9
    positive = valid_times & (duration > 0)
    denominator = cp.where(positive, duration, cp.float64(1))
    speed = cp.where(positive & distance_ok, distance / denominator * 3600.0, cp.nan)
    flags = cp.zeros(size, dtype=cp.uint32)
    for mask, flag in [
        (~pu_ok, 1),
        (~do_ok, 2),
        (~allowed, 4),
        (valid_times & ~positive, 8),
        (~distance_ok, 16),
        (distance_ok & (distance == 0), 32),
        (~zones_ok, 64),
    ]:
        flags |= cp.where(mask, cp.uint32(flag), cp.uint32(0))
    return {
        "duration_seconds": cp.where(positive, duration, cp.nan),
        "mean_speed_mph": speed,
        "feature_eligible": (positive & distance_ok & zones_ok).astype(cp.uint8),
        "feature_flags": flags,
    }
