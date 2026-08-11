#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gif {

class Writer {
public:

    Writer(const std::string& path, int w, int h, int delayMs = 80)
        : f_(std::fopen(path.c_str(), "wb")), w_(w), h_(h), delay_(delayMs / 10) {
        if (!f_) return;
        writeHeader();
    }

    ~Writer() { close(); }

    explicit operator bool() const { return f_ != nullptr; }

    bool addFrame(const uint8_t* rgb) {
        if (!f_) return false;

        writeByte(0x21);
        writeByte(0xF9);
        writeByte(0x04);
        writeByte(0x04);
        writeByte((uint8_t)(delay_ & 0xFF));
        writeByte((uint8_t)((delay_ >> 8) & 0xFF));
        writeByte(0x00);
        writeByte(0x00);

        writeByte(0x2C);
        writeLE16(0);
        writeLE16(0);
        writeLE16((uint16_t)w_);
        writeLE16((uint16_t)h_);
        writeByte(0x00);

        writeByte(8);

        std::vector<uint8_t> idx((size_t)w_ * h_);
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = (uint8_t)((299u * rgb[i * 3] + 587u * rgb[i * 3 + 1] +
                                114u * rgb[i * 3 + 2]) / 1000u);
        }
        encodeLZW(idx);
        return true;
    }

    void close() {
        if (f_) {
            writeByte(0x3B);
            std::fclose(f_);
            f_ = nullptr;
        }
    }

private:
    void writeHeader() {
        std::fwrite("GIF89a", 1, 6, f_);
        writeLE16((uint16_t)w_);
        writeLE16((uint16_t)h_);
        writeByte(0xF7);
        writeByte(0x00);
        writeByte(0x00);
        for (int i = 0; i < 256; ++i) {
            writeByte((uint8_t)i);
            writeByte((uint8_t)i);
            writeByte((uint8_t)i);
        }
    }

    void writeByte(uint8_t b) { std::fputc(b, f_); }

    void writeLE16(uint16_t v) {
        writeByte((uint8_t)(v & 0xFF));
        writeByte((uint8_t)((v >> 8) & 0xFF));
    }

    void encodeLZW(const std::vector<uint8_t>& data) {
        constexpr int CLEAR = 256;
        constexpr int EOI = 257;
        constexpr int MAX_CODES = 4096;

        std::vector<int32_t> dict(MAX_CODES, -1);
        std::vector<int32_t> dictPrefix(MAX_CODES, -1);
        std::vector<uint8_t> dictByte(MAX_CODES, 0);
        int nextCode = 258;
        int codeSize = 9;

        auto resetDict = [&]() {
            nextCode = 258;
            codeSize = 9;
            std::fill(dict.begin(), dict.end(), -1);
        };

        uint32_t bitBuf = 0;
        int bitCnt = 0;
        std::vector<uint8_t> block;

        auto emitOut = [&](uint8_t b) {
            block.push_back(b);
            if (block.size() == 255) {
                writeByte(255);
                std::fwrite(block.data(), 1, 255, f_);
                block.clear();
            }
        };

        auto emitCode = [&](int code) {
            bitBuf |= (uint32_t)code << bitCnt;
            bitCnt += codeSize;
            while (bitCnt >= 8) {
                emitOut((uint8_t)(bitBuf & 0xFF));
                bitBuf >>= 8;
                bitCnt -= 8;
            }
        };

        auto emitEnd = [&]() {
            if (bitCnt > 0) {
                emitOut((uint8_t)(bitBuf & 0xFF));
                bitBuf = 0;
                bitCnt = 0;
            }
            if (!block.empty()) {
                writeByte((uint8_t)block.size());
                std::fwrite(block.data(), 1, block.size(), f_);
                block.clear();
            }
            writeByte(0x00);
        };

        emitCode(CLEAR);

        if (data.empty()) {
            emitCode(EOI);
            emitEnd();
            return;
        }

        int prefix = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            uint8_t k = data[i];
            int hash = ((prefix << 8) | k) & (MAX_CODES - 1);
            int found = -1;
            for (int probe = 0; probe < 64; ++probe) {
                int h = (hash + probe) & (MAX_CODES - 1);
                if (dict[h] < 0) break;
                if (dictPrefix[h] == prefix && dictByte[h] == k) { found = dict[h]; break; }
            }
            if (found >= 0) {
                prefix = found;
            } else {
                emitCode(prefix);
                if (nextCode < MAX_CODES) {

                    if (nextCode == (1 << codeSize) && codeSize < 12) ++codeSize;
                    for (int probe = 0; probe < 64; ++probe) {
                        int h = (hash + probe) & (MAX_CODES - 1);
                        if (dict[h] < 0) {
                            dict[h] = nextCode;
                            dictPrefix[h] = prefix;
                            dictByte[h] = k;
                            break;
                        }
                    }
                    ++nextCode;
                    if (nextCode == MAX_CODES) {
                        emitCode(CLEAR);
                        resetDict();
                    }
                }
                prefix = k;
            }
        }
        emitCode(prefix);
        emitCode(EOI);
        emitEnd();
    }

    FILE* f_ = nullptr;
    int w_ = 0, h_ = 0;
    int delay_ = 0;
};

}
