#include "stipple.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace stipple {
namespace {

inline int nearestPointSimd(float px, float py, int n,
                            const float* xs, const float* ys) {
    const __m256 vpx = _mm256_set1_ps(px);
    const __m256 vpy = _mm256_set1_ps(py);
    const __m256 vinf = _mm256_set1_ps(INFINITY);
    const __m256 vlane = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

    __m256 vmin = vinf;
    __m256 vmini = vlane;

    int j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m256 vx = _mm256_loadu_ps(xs + j);
        const __m256 vy = _mm256_loadu_ps(ys + j);
        const __m256 dx = _mm256_sub_ps(vpx, vx);
        const __m256 dy = _mm256_sub_ps(vpy, vy);
        const __m256 d2 = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));

        const __m256 maskf = _mm256_cmp_ps(d2, vmin, _CMP_LT_OQ);
        vmin = _mm256_blendv_ps(vmin, d2, maskf);
        const __m256 vidx = _mm256_add_ps(vlane, _mm256_set1_ps((float)j));
        vmini = _mm256_blendv_ps(vmini, vidx, maskf);
    }

    alignas(32) float mins[8];
    alignas(32) float idxs[8];
    _mm256_store_ps(mins, vmin);
    _mm256_store_ps(idxs, vmini);
    float best = INFINITY;
    int bi = 0;
    for (int k = 0; k < 8; ++k) {
        if (mins[k] < best) { best = mins[k]; bi = (int)idxs[k]; }
    }

    for (; j < n; ++j) {
        float dx = px - xs[j], dy = py - ys[j];
        float d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; bi = j; }
    }
    return bi;
}

void assignSimd(const Image& img, int n, const float* xs, const float* ys,
                std::vector<float>& sw, std::vector<float>& sxw, std::vector<float>& syw) {
    const int w = img.w;
    const int P = w * img.h;
    const float* density = img.density.data();
    sw.assign(n, 0.0f);
    sxw.assign(n, 0.0f);
    syw.assign(n, 0.0f);

    const int T = omp_get_max_threads();
    std::vector<std::vector<float>> lsw(T, std::vector<float>(n, 0.0f));
    std::vector<std::vector<float>> lsxw(T, std::vector<float>(n, 0.0f));
    std::vector<std::vector<float>> lsyw(T, std::vector<float>(n, 0.0f));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* mySw = lsw[tid].data();
        float* mySxw = lsxw[tid].data();
        float* mySyw = lsyw[tid].data();

#pragma omp for schedule(static)
        for (int i = 0; i < P; ++i) {
            float d = density[i];
            if (d <= 0.0f) continue;
            float px = (float)(i % w) + 0.5f;
            float py = (float)(i / w) + 0.5f;
            int bi = nearestPointSimd(px, py, n, xs, ys);
            mySw[bi] += d;
            mySxw[bi] += px * d;
            mySyw[bi] += py * d;
        }
    }

    for (int t = 0; t < T; ++t)
        for (int j = 0; j < n; ++j) {
            sw[j] += lsw[t][j];
            sxw[j] += lsxw[t][j];
            syw[j] += lsyw[t][j];
        }
}

}

RunResult runSimd(const Image& img, const Params& p,
                  std::vector<float>& xs, std::vector<float>& ys,
                  const std::function<void(int, float)>& onIter) {
    RunResult r;
    r.impl = "simd";
    const int n = p.numPoints;

    std::vector<float> sw, sxw, syw;
    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < p.maxIter; ++iter) {
        auto ts = std::chrono::steady_clock::now();
        assignSimd(img, n, xs.data(), ys.data(), sw, sxw, syw);
        float disp = AssignCore::updateCentroids(n, sw, sxw, syw, xs, ys, r.cellWeight);
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
    return r;
}

}
