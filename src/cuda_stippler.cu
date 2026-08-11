#include "stipple.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

namespace stipple {
namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(e_));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__ void assignKernel(const float* __restrict__ xs,
                             const float* __restrict__ ys,
                             const float* __restrict__ density,
                             float* __restrict__ sw,
                             float* __restrict__ sxw,
                             float* __restrict__ syw,
                             int w, int P, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    float d = density[i];
    if (d <= 0.0f) return;

    float px = (float)(i % w) + 0.5f;
    float py = (float)(i / w) + 0.5f;

    float best = INFINITY;
    int bi = 0;
    for (int j = 0; j < n; ++j) {
        float dx = px - xs[j];
        float dy = py - ys[j];
        float d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; bi = j; }
    }

    atomicAdd(&sw[bi], d);
    atomicAdd(&sxw[bi], px * d);
    atomicAdd(&syw[bi], py * d);
}

}

RunResult runCUDA(const Image& img, const Params& p,
                  std::vector<float>& xs, std::vector<float>& ys,
                  const std::function<void(int, float)>& onIter) {
    RunResult r;
    r.impl = "cuda";
    const int n = p.numPoints;
    const int P = img.w * img.h;

    float* d_density = nullptr;
    float* d_xs = nullptr;
    float* d_ys = nullptr;
    float* d_sw = nullptr;
    float* d_sxw = nullptr;
    float* d_syw = nullptr;

    CUDA_CHECK(cudaMalloc(&d_density, (size_t)P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xs, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ys, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sw, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sxw, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_syw, (size_t)n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_density, img.density.data(), (size_t)P * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;

    std::vector<float> sw(n), sxw(n), syw(n);
    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < p.maxIter; ++iter) {
        auto ts = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemset(d_sw, 0, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sxw, 0, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_syw, 0, (size_t)n * sizeof(float)));

        assignKernel<<<blocks, threads>>>(d_xs, d_ys, d_density, d_sw, d_sxw, d_syw,
                                          img.w, P, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(sw.data(), d_sw, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sxw.data(), d_sxw, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(syw.data(), d_syw, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToHost));

        float disp = AssignCore::updateCentroids(n, sw, sxw, syw, xs, ys, r.cellWeight);

        CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), (size_t)n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), (size_t)n * sizeof(float),
                              cudaMemcpyHostToDevice));

        auto te = std::chrono::steady_clock::now();
        r.perIterMs.push_back(std::chrono::duration<double, std::milli>(te - ts).count());

        if (onIter) onIter(iter, disp);
        r.maxDisp = disp;
        r.iterations = iter + 1;
        if (disp < p.epsilon) { r.converged = true; break; }
    }

    r.totalMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    r.avgIterMs = r.totalMs / std::max(1, r.iterations);

    cudaFree(d_density);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_sw);
    cudaFree(d_sxw);
    cudaFree(d_syw);
    return r;
}

}
