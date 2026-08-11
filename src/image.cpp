#include "image.hpp"
#include "gif_writer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace stipple {

bool loadImage(const std::string& path, Image& out, float gamma, float edgeWeight) {
    int comp = 0;
    unsigned char* data = stbi_load(path.c_str(), &out.w, &out.h, &comp, 1);
    if (!data) {
        std::fprintf(stderr, "Gagal memuat gambar '%s': %s\n", path.c_str(),
                     stbi_failure_reason());
        return false;
    }
    const int w = out.w, h = out.h;
    const int P = w * h;
    out.lum.assign(data, data + P);
    out.density.resize(P);

    std::vector<float> baseD(P);
    for (int i = 0; i < P; ++i) {
        float d = 255.0f - (float)out.lum[i];
        if (gamma != 1.0f && d > 0.0f) {
            d = 255.0f * std::pow(d / 255.0f, gamma);
        }
        baseD[i] = d;
    }

    double total = 0.0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            float d = baseD[i];
            if (edgeWeight > 0.0f && y > 0 && y < h - 1 && x > 0 && x < w - 1) {
                float gx = (out.lum[(y-1)*w + (x+1)] + 2.0f*out.lum[y*w + (x+1)] + out.lum[(y+1)*w + (x+1)]) -
                           (out.lum[(y-1)*w + (x-1)] + 2.0f*out.lum[y*w + (x-1)] + out.lum[(y+1)*w + (x-1)]);
                float gy = (out.lum[(y+1)*w + (x-1)] + 2.0f*out.lum[(y+1)*w + x] + out.lum[(y+1)*w + (x+1)]) -
                           (out.lum[(y-1)*w + (x-1)] + 2.0f*out.lum[(y-1)*w + x] + out.lum[(y-1)*w + (x+1)]);
                float edge = std::sqrt(gx * gx + gy * gy) / 1020.0f;
                d += edgeWeight * edge * baseD[i];
                d = std::min(255.0f, d);
            }
            out.density[i] = d;
            total += d;
        }
    }
    out.darkArea = total / 255.0;
    stbi_image_free(data);
    return true;
}

bool savePNG(const std::string& path, int w, int h, const uint8_t* rgb) {
    return stbi_write_png(path.c_str(), w, h, 3, rgb, w * 3) != 0;
}

bool savePPM(const std::string& path, int w, int h, const uint8_t* rgb) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::fwrite(rgb, 1, (size_t)w * h * 3, f);
    std::fclose(f);
    return true;
}

bool saveGIF(const std::string& path, int w, int h,
             const std::vector<std::vector<uint8_t>>& frames, int delayMs) {
    if (frames.empty()) return false;
    gif::Writer wtr(path, w, h, delayMs);
    if (!wtr) return false;
    for (const auto& f : frames) wtr.addFrame(f.data());
    wtr.close();
    return true;
}

}
