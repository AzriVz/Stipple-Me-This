#pragma once
#include "stipple.hpp"

namespace stipple {

bool loadImage(const std::string& path, Image& out, float gamma = 1.0f, float edgeWeight = 0.35f);

bool savePNG(const std::string& path, int w, int h, const uint8_t* rgb);

bool savePPM(const std::string& path, int w, int h, const uint8_t* rgb);

bool saveGIF(const std::string& path, int w, int h,
             const std::vector<std::vector<uint8_t>>& frames, int delayMs);

}
