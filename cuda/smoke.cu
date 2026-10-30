// Standalone test of the exact same CUDA source loaded by Python.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "fused_trip_features.cu"

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> class DeviceArray {
public:
    T* data = nullptr;
    explicit DeviceArray(size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void upload(const std::vector<T>& values) {
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download(size_t n) {
        std::vector<T> values(n);
        check(cudaMemcpy(values.data(), data, n * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

int main() {
    try {
        // 513 exercises two full blocks plus a tail. Null, reversed, zero,
        // nonfinite-distance, time-policy and invalid-zone cases repeat.
        constexpr int64_t n = 513;
        std::vector<int64_t> p(n, 1705312800000000000LL), d(n, 1705313700000000000LL);
        std::vector<double> miles(n, 3.0);
        std::vector<uint8_t> pv(n, 1), dv(n, 1), time_ok(n, 1), pz(n, 1), dz(n, 1);
        for (int64_t i = 0; i < n; ++i) {
            switch (i % 8) {
                case 1: p[i] = INT64_MIN; pv[i] = 0; break;
                case 2: d[i] = p[i]; break;
                case 3: d[i] = p[i] - 1000000000LL; break;
                case 4: miles[i] = INFINITY; break;
                case 5: time_ok[i] = 0; break;
                case 6: dz[i] = 0; break;
                case 7: miles[i] = 0; break;
            }
        }
        DeviceArray<int64_t> dp(n), dd(n);
        DeviceArray<double> dm(n), duration(n), speed(n);
        DeviceArray<uint8_t> dpv(n), ddv(n), dt(n), dpz(n), ddz(n), eligible(n);
        DeviceArray<uint32_t> flags(n);
        dp.upload(p); dd.upload(d); dm.upload(miles); dpv.upload(pv); ddv.upload(dv);
        dt.upload(time_ok); dpz.upload(pz); ddz.upload(dz);
        fused_trip_features<<<3, 256>>>(dp.data, dd.data, dm.data, dpv.data, ddv.data,
            dt.data, dpz.data, ddz.data, duration.data, speed.data, eligible.data, flags.data, n);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        auto hs = speed.download(n);
        auto hd = duration.download(n);
        auto he = eligible.download(n);
        auto hf = flags.download(n);
        const uint32_t expected_flags[] = {0, 1, 8, 8, 16, 4, 64, 32};
        for (int64_t i = 0; i < n; ++i) {
            const int category = i % 8;
            const bool duration_ok = category == 0 || category == 4 || category == 6 || category == 7;
            const bool speed_ok = duration_ok && category != 4;
            const bool eligible_ok = category == 0 || category == 7;
            if (hf[i] != expected_flags[category] || he[i] != eligible_ok)
                throw std::runtime_error("Flag/eligibility mismatch at row " + std::to_string(i));
            if (duration_ok ? std::abs(hd[i] - 900.0) > 1e-10 : !std::isnan(hd[i]))
                throw std::runtime_error("Duration mismatch");
            const double expected_speed = category == 7 ? 0.0 : 12.0;
            if (speed_ok ? std::abs(hs[i] - expected_speed) > 1e-10 : !std::isnan(hs[i]))
                throw std::runtime_error("Speed mismatch");
        }
        std::cout << "CUDA smoke passed: " << n << " rows, including irregular tail and invalid inputs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA smoke failed: " << error.what() << '\n';
        return 1;
    }
}
