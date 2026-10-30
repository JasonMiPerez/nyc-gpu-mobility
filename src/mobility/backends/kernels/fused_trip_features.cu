// One thread per trip, with a grid-stride loop for large batches.
// Inputs are explicit byte masks, not Arrow bit-packed validity buffers.
// Timezone/DST eligibility is computed before this kernel is called.
#ifdef __CUDACC_RTC__
// NVRTC intentionally does not search a host compiler's standard headers.
using int64_t = long long;
using uint8_t = unsigned char;
using uint32_t = unsigned int;
#define INT64_MIN (-9223372036854775807LL - 1LL)
#define NAN __longlong_as_double(0x7ff8000000000000LL)
#else
#include <math.h>
#include <stdint.h>
#include <limits.h>
#endif

extern "C" __global__ void fused_trip_features(
    const int64_t* pickup_ns, const int64_t* dropoff_ns,
    const double* distance, const uint8_t* pickup_valid,
    const uint8_t* dropoff_valid, const uint8_t* time_allowed,
    const uint8_t* pickup_zone_valid, const uint8_t* dropoff_zone_valid,
    double* duration_seconds, double* mean_speed_mph,
    uint8_t* feature_eligible, uint32_t* feature_flags, const int64_t n) {
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const bool pu_ok = pickup_valid[i] && pickup_ns[i] != INT64_MIN;
        const bool do_ok = dropoff_valid[i] && dropoff_ns[i] != INT64_MIN;
        const bool allowed = time_allowed[i] != 0;
        const bool zones_ok = pickup_zone_valid[i] && dropoff_zone_valid[i];
        const double miles = distance[i];
        const bool distance_ok = isfinite(miles) && miles >= 0.0;
        uint32_t flags = (!pu_ok ? 1u : 0u) | (!do_ok ? 2u : 0u)
            | (!allowed ? 4u : 0u) | (!distance_ok ? 16u : 0u)
            | (distance_ok && miles == 0.0 ? 32u : 0u)
            | (!zones_ok ? 64u : 0u);
        double seconds = NAN;
        bool duration_ok = false;
        if (pu_ok && do_ok && allowed) {
            // Quotients fit safely in int64 even for opposite-sign endpoints.
            // This also avoids precision loss from casting whole nanoseconds.
            const int64_t billion = 1000000000LL;
            int64_t p_seconds = pickup_ns[i] / billion;
            int64_t d_seconds = dropoff_ns[i] / billion;
            int64_t p_remainder = pickup_ns[i] % billion;
            int64_t d_remainder = dropoff_ns[i] % billion;
            // Use floor division, matching NumPy/CuPy even before 1970.
            if (p_remainder < 0) { --p_seconds; p_remainder += billion; }
            if (d_remainder < 0) { --d_seconds; d_remainder += billion; }
            seconds = (double)(d_seconds - p_seconds)
                + (double)(d_remainder - p_remainder) * 1e-9;
            duration_ok = seconds > 0.0;
            if (!duration_ok) {
                flags |= 8u;
                seconds = NAN;
            }
        }
        duration_seconds[i] = seconds;
        mean_speed_mph[i] = duration_ok && distance_ok ? miles / seconds * 3600.0 : NAN;
        feature_eligible[i] = duration_ok && distance_ok && zones_ok ? 1 : 0;
        feature_flags[i] = flags;
    }
}
