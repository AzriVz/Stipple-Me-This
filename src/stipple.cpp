#include "stipple.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace stipple {

const char* implName(Impl i) {
    switch (i) {
        case Impl::Serial: return "serial";
        case Impl::OpenMP: return "openmp";
        case Impl::Simd:   return "simd";
        case Impl::CUDA:   return "cuda";
    }
    return "?";
}

Impl implFromString(const std::string& s) {
    if (s == "serial") return Impl::Serial;
    if (s == "openmp") return Impl::OpenMP;
    if (s == "simd")   return Impl::Simd;
    if (s == "cuda")   return Impl::CUDA;
    return Impl::Serial;
}

void initPoints(const Image& img, int n, std::vector<float>& xs, std::vector<float>& ys,
                uint64_t seed) {
    const int P = img.w * img.h;

    std::vector<uint64_t> prefix(P);
    uint64_t total = 0;
    for (int i = 0; i < P; ++i) {
        prefix[i] = total;
        total += (uint64_t)(img.lum[i] == 255 ? 0u : (uint32_t)(img.density[i] + 0.5f));
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> jitter(0.0f, 1.0f);

    xs.resize(n);
    ys.resize(n);
    for (int i = 0; i < n; ++i) {
        uint64_t r = rng() % total;

        auto it = std::upper_bound(prefix.begin(), prefix.end(), r);
        if (it == prefix.begin()) it = prefix.begin() + 1;
        if (it == prefix.end()) it = prefix.end() - 1;
        int px = (int)(it - prefix.begin());
        xs[i] = (float)(px % img.w) + jitter(rng);
        ys[i] = (float)(px / img.w) + jitter(rng);
    }
}

float AssignCore::updateCentroids(int n, const std::vector<float>& sw,
                                  const std::vector<float>& sxw,
                                  const std::vector<float>& syw,
                                  std::vector<float>& xs, std::vector<float>& ys,
                                  std::vector<float>& cellWeightOut) {
    float maxDisp = 0.0f;
    if (cellWeightOut.size() != (size_t)n) cellWeightOut.assign(n, 0.0f);
    for (int j = 0; j < n; ++j) {
        cellWeightOut[j] = sw[j];
        if (sw[j] > 1e-6f) {
            float nx = sxw[j] / sw[j];
            float ny = syw[j] / sw[j];
            float dx = nx - xs[j];
            float dy = ny - ys[j];
            maxDisp = std::max(maxDisp, dx * dx + dy * dy);
            xs[j] = nx;
            ys[j] = ny;
        }
    }
    return std::sqrt(maxDisp);
}

void renderStipple(const Image& img, const std::vector<float>& xs,
                   const std::vector<float>& ys,
                   const std::vector<float>& cellWeight,
                   std::vector<uint8_t>& rgbOut,
                   float radiusScale) {
    const int w = img.w, h = img.h;
    const int P = w * h;
    std::vector<float> canvas(P, 1.0f);

    const int n = (int)xs.size();
    if (n == 0 || img.darkArea <= 0.0) {
        rgbOut.assign((size_t)P * 3, 255);
        return;
    }

    const double r0 = std::sqrt(img.darkArea / (n * M_PI)) * (double)radiusScale;

    double meanW = 1.0;
    if (cellWeight.size() == (size_t)n) {
        double s = 0.0;
        for (float wgt : cellWeight) s += wgt;
        meanW = s / n;
    }

    for (int j = 0; j < n; ++j) {
        float fx = xs[j], fy = ys[j];
        double r = r0;
        if (cellWeight.size() == (size_t)n && meanW > 0.0) {
            double scale = std::sqrt((double)cellWeight[j] / meanW);

            r = r0 * std::clamp(scale, 0.4, 1.7);
        }
        if (r < 0.35) r = 0.35;

        int cx = (int)fx, cy = (int)fy;
        int ri = (int)std::ceil(r) + 1;
        for (int dy = -ri; dy <= ri; ++dy) {
            int yy = cy + dy;
            if (yy < 0 || yy >= h) continue;
            for (int dx = -ri; dx <= ri; ++dx) {
                int xx = cx + dx;
                if (xx < 0 || xx >= w) continue;
                float ddx = (float)xx + 0.5f - fx;
                float ddy = (float)yy + 0.5f - fy;
                float dd = std::sqrt(ddx * ddx + ddy * ddy);
                if (dd > (float)r + 0.5f) continue;

                float darkness;
                if (dd <= (float)r - 0.5f) {
                    darkness = 1.0f;
                } else {
                    darkness = 1.0f - (dd - ((float)r - 0.5f));
                }
                darkness = std::clamp(darkness, 0.0f, 1.0f);
                int pidx = yy * w + xx;
                canvas[pidx] *= (1.0f - darkness);
            }
        }
    }

    rgbOut.resize((size_t)P * 3);
    for (int i = 0; i < P; ++i) {
        uint8_t v = (uint8_t)std::clamp((int)(canvas[i] * 255.0f + 0.5f), 0, 255);
        rgbOut[i * 3]     = v;
        rgbOut[i * 3 + 1] = v;
        rgbOut[i * 3 + 2] = v;
    }
}

}
