#include "stipple.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>

namespace stipple {
namespace {

void assignOpenMP(const Image& img, int n, const float* xs, const float* ys,
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
            int bi = AssignCore::nearestPoint(px, py, n,
                [&](int j, float px_, float py_) {
                    float dx = px_ - xs[j], dy = py_ - ys[j];
                    return dx * dx + dy * dy;
                });
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

RunResult runOpenMP(const Image& img, const Params& p,
                    std::vector<float>& xs, std::vector<float>& ys,
                    const std::function<void(int, float)>& onIter) {
    RunResult r;
    r.impl = "openmp";
    const int n = p.numPoints;

    std::vector<float> sw, sxw, syw;
    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < p.maxIter; ++iter) {
        auto ts = std::chrono::steady_clock::now();
        assignOpenMP(img, n, xs.data(), ys.data(), sw, sxw, syw);
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
