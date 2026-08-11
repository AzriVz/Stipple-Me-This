#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace stipple {

struct Image {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> lum;
    std::vector<float> density;
    double darkArea = 0.0;
};

struct Params {
    int numPoints = 0;
    int maxIter = 100;
    float epsilon = 0.5f;
    float gamma = 1.4f;
    float radiusScale = 1.0f;
    float edgeWeight = 0.35f;
};

struct RunResult {
    std::string impl;
    int iterations = 0;
    bool converged = false;
    double totalMs = 0.0;
    double avgIterMs = 0.0;
    double maxDisp = 0.0;
    std::vector<double> perIterMs;
    std::vector<float> cellWeight;
};

enum class Impl { Serial, OpenMP, Simd, CUDA };

const char* implName(Impl i);
Impl implFromString(const std::string& s);

void initPoints(const Image& img, int n, std::vector<float>& xs, std::vector<float>& ys,
                uint64_t seed);

RunResult runSerial(const Image& img, const Params& p,
                    std::vector<float>& xs, std::vector<float>& ys,
                    const std::function<void(int iter, float maxDisp)>& onIter);
RunResult runOpenMP(const Image& img, const Params& p,
                    std::vector<float>& xs, std::vector<float>& ys,
                    const std::function<void(int iter, float maxDisp)>& onIter);
RunResult runSimd(const Image& img, const Params& p,
                  std::vector<float>& xs, std::vector<float>& ys,
                  const std::function<void(int iter, float maxDisp)>& onIter);
RunResult runCUDA(const Image& img, const Params& p,
                  std::vector<float>& xs, std::vector<float>& ys,
                  const std::function<void(int iter, float maxDisp)>& onIter);

void renderStipple(const Image& img, const std::vector<float>& xs,
                   const std::vector<float>& ys,
                   const std::vector<float>& cellWeight,
                   std::vector<uint8_t>& rgbOut,
                   float radiusScale = 1.0f);

struct AssignCore {

    template <typename F>
    static int nearestPoint(float px, float py, int n, F&& dist2) {
        float best = INFINITY;
        int bi = 0;
        for (int j = 0; j < n; ++j) {
            float d2 = dist2(j, px, py);
            if (d2 < best) { best = d2; bi = j; }
        }
        return bi;
    }

    static float updateCentroids(int n, const std::vector<float>& sw,
                                 const std::vector<float>& sxw,
                                 const std::vector<float>& syw,
                                 std::vector<float>& xs, std::vector<float>& ys,
                                 std::vector<float>& cellWeightOut);
};

}
