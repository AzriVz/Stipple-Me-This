#include "stipple.hpp"

#include <chrono>
#include <cmath>

namespace stipple {
namespace {

void assignSerial(const Image& img, int n, const float* xs, const float* ys,
                  std::vector<float>& sw, std::vector<float>& sxw, std::vector<float>& syw) {
    const int w = img.w;
    const float* density = img.density.data();
    sw.assign(n, 0.0f);
    sxw.assign(n, 0.0f);
    syw.assign(n, 0.0f);
    for (int i = 0; i < w * img.h; ++i) {
        float d = density[i];
        if (d <= 0.0f) continue;
        float px = (float)(i % w) + 0.5f;
        float py = (float)(i / w) + 0.5f;
        int bi = AssignCore::nearestPoint(px, py, n,
            [&](int j, float px_, float py_) {
                float dx = px_ - xs[j], dy = py_ - ys[j];
                return dx * dx + dy * dy;
            });
        sw[bi] += d;
        sxw[bi] += px * d;
        syw[bi] += py * d;
    }
}

}

RunResult runSerial(const Image& img, const Params& p,
                    std::vector<float>& xs, std::vector<float>& ys,
                    const std::function<void(int, float)>& onIter) {
    RunResult r;
    r.impl = "serial";
    const int n = p.numPoints;

    std::vector<float> sw, sxw, syw;
    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < p.maxIter; ++iter) {
        auto ts = std::chrono::steady_clock::now();
        assignSerial(img, n, xs.data(), ys.data(), sw, sxw, syw);
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
