#include "stipple.hpp"
#include "image.hpp"
#include "gui.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace stipple;

static void mkdirParent(const std::string& path) {
    std::string dir;
    for (char c : path) {
        if (c == '/' || c == '\\') {
            if (!dir.empty()) ::mkdir(dir.c_str(), 0755);
        }
        dir += c;
    }
}

static void printUsage() {
    std::printf(
        "Stipple Me This - stippling otomatis dengan Lloyd's Algorithm\n"
        "Serial / OpenMP / SIMD-AVX2 / CUDA + benchmarking\n\n"
        "Usage:\n"
        "  stipple --input <gambar> --points <N> --iters <N> --eps <E> --output <file> [opsi]\n\n"
        "Parameter wajib:\n"
        "  --input  <path>   Lokasi file gambar sumber\n"
        "  --points <int>    Jumlah titik stipple\n"
        "  --iters  <int>    Batas maksimal iterasi\n"
        "  --eps    <float>  Toleransi pergeseran titik (epsilon)\n"
        "  --output <path>   Lokasi penyimpanan gambar hasil\n\n"
        "Opsi:\n"
        "  --impl <name>     serial | openmp | simd | cuda  (default: serial)\n"
        "  --gamma <float>   Faktor gamma kontras densitas (default: 1.4)\n"
        "  --radius-scale <f> Skala radius titik stipple (default: 1.0)\n"
        "  --edge <float>    Bobot penajaman kontur/tepi Sobel (default: 0.35)\n"
        "  --bench           Benchmark ketiga implementasi + tabel speedup\n"
        "  --threads <int>   Jumlah thread OpenMP (default: semua core)\n"
        "  --seed <int>      Seed RNG (default: 42)\n"
        "  --gif <path>      Simpan animasi tiap iterasi ke file GIF\n"
        "  --gif-every <k>   Rekam 1 frame tiap k iterasi (default: 1)\n"
        "  --gui             Buka antarmuka grafis\n"
        "  --quiet           Tanpa output progress\n"
        "\n"
        "Tanpa argumen juga akan membuka antarmuka grafis (GUI).\n");
}

static std::string fmtMs(double ms) {
    if (ms >= 1000.0) return std::to_string(ms / 1000.0).substr(0, 6) + " s";
    return std::to_string(ms).substr(0, 5) + " ms";
}

static bool isTty() { return isatty(fileno(stderr)); }

static void printProgress(int iter, int maxIter, float disp, double iterMs,
                          double elapsedMs, bool quiet) {
    if (quiet) return;
    if (!isTty()) {
        std::fprintf(stderr, "iter %4d/%d  max_disp=%.4f  %.1f ms/iter  elapsed=%s\n",
                     iter + 1, maxIter, disp, iterMs, fmtMs(elapsedMs).c_str());
        return;
    }
    const int barW = 28;
    int filled = (int)((iter + 1) / (double)maxIter * barW);
    if (filled > barW) filled = barW;
    std::string bar;
    for (int i = 0; i < barW; ++i) bar += (i < filled) ? '#' : '-';
    double eta = elapsedMs / (iter + 1) * (maxIter - iter - 1);
    int percent = (int)std::round((iter + 1) * 100.0 / maxIter);
    std::fprintf(stderr,
                 "\r\033[K[%s] %3d%% | iter %d/%d | max_disp %.4f | %.1f ms | ETA %s",
                 bar.c_str(), percent, iter + 1, maxIter, disp, iterMs,
                 fmtMs(eta).c_str());
    std::fflush(stderr);
}

struct Config {
    std::string input, output, implName = "serial", gifPath;
    int numPoints = 0;
    int maxIter = 0;
    int gifEvery = 1;
    int threads = 0;
    float epsilon = 0.0f;
    float gamma = 1.4f;
    float radiusScale = 1.0f;
    float edgeWeight = 0.35f;
    uint64_t seed = 42;
    bool bench = false;
    bool quiet = false;
    bool gui = false;
};

static bool parseArgs(int argc, char** argv, Config& cfg, bool& printHelp) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: '%s' butuh nilai.\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--input") cfg.input = next("--input");
        else if (a == "--output") cfg.output = next("--output");
        else if (a == "--points") cfg.numPoints = std::atoi(next("--points"));
        else if (a == "--iters") cfg.maxIter = std::atoi(next("--iters"));
        else if (a == "--eps") cfg.epsilon = (float)std::atof(next("--eps"));
        else if (a == "--gamma") cfg.gamma = (float)std::atof(next("--gamma"));
        else if (a == "--radius-scale") cfg.radiusScale = (float)std::atof(next("--radius-scale"));
        else if (a == "--edge") cfg.edgeWeight = (float)std::atof(next("--edge"));
        else if (a == "--impl") cfg.implName = next("--impl");
        else if (a == "--threads") cfg.threads = std::atoi(next("--threads"));
        else if (a == "--seed") cfg.seed = (uint64_t)std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--gif") cfg.gifPath = next("--gif");
        else if (a == "--gif-every") cfg.gifEvery = std::atoi(next("--gif-every"));
        else if (a == "--bench") cfg.bench = true;
        else if (a == "--gui") cfg.gui = true;
        else if (a == "--quiet") cfg.quiet = true;
        else if (a == "--help" || a == "-h") { printHelp = true; return true; }
        else {
            std::fprintf(stderr, "Error: argumen tidak dikenal '%s'\n", a.c_str());
            printHelp = true;
            return false;
        }
    }
    return true;
}

static bool validate(const Config& cfg) {
    if (cfg.input.empty() || cfg.numPoints <= 0 || cfg.maxIter <= 0 ||
        !std::isfinite(cfg.epsilon) || cfg.epsilon < 0.0f || cfg.output.empty()) {
        std::fprintf(stderr, "Error: semua 5 parameter wajib harus diisi.\n\n");
        return false;
    }
    if (cfg.implName != "serial" && cfg.implName != "openmp" &&
        cfg.implName != "simd" && cfg.implName != "cuda") {
        std::fprintf(stderr,
                     "Error: implementasi '%s' tidak valid (pilih serial/openmp/simd/cuda).\n",
                     cfg.implName.c_str());
        return false;
    }
    if (!std::isfinite(cfg.gamma) || cfg.gamma <= 0.0f ||
        !std::isfinite(cfg.radiusScale) || cfg.radiusScale <= 0.0f ||
        !std::isfinite(cfg.edgeWeight) || cfg.edgeWeight < 0.0f ||
        cfg.gifEvery <= 0 || cfg.threads < 0) {
        std::fprintf(stderr, "Error: nilai opsi numerik tidak valid.\n");
        return false;
    }
    return true;
}

struct RunArgs {
    const Image* img;
    Params p;
    uint64_t seed;
    Impl impl;
    bool quiet;
    const Config* cfg;
};

static RunResult runOnce(const RunArgs& a, std::vector<float>& xs,
                         std::vector<float>& ys,
                         std::vector<std::vector<uint8_t>>* gifFrames = nullptr) {
    initPoints(*a.img, a.p.numPoints, xs, ys, a.seed);

    auto tStart = std::chrono::steady_clock::now();
    double last = 0.0;
    auto onIter = [&](int iter, float disp) {
        (void)disp;
        double now = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - tStart).count();
        printProgress(iter, a.p.maxIter, disp, now - last, now, a.quiet);
        last = now;

        if (gifFrames) {
            bool capture = (iter == 0) || ((iter + 1) % a.cfg->gifEvery == 0);
            if (capture) {
                std::vector<uint8_t> frame;
                renderStipple(*a.img, xs, ys, {}, frame, a.cfg->radiusScale);
                gifFrames->push_back(std::move(frame));
            }
        }
    };

    RunResult r;
    switch (a.impl) {
        case Impl::Serial: r = runSerial(*a.img, a.p, xs, ys, onIter); break;
        case Impl::OpenMP: r = runOpenMP(*a.img, a.p, xs, ys, onIter); break;
        case Impl::Simd:   r = runSimd(*a.img, a.p, xs, ys, onIter); break;
        case Impl::CUDA:   r = runCUDA(*a.img, a.p, xs, ys, onIter); break;
    }

    if (a.p.maxIter > 0 && !a.quiet) {
        if (isTty()) std::fprintf(stderr, "\r\033[K");
        std::fprintf(stderr,
                     "%s: %d iterasi, %s, %.2f ms/iter rata-rata, "
                     "max_disp akhir=%.4f (%s)\n",
                     implName(a.impl), r.iterations, fmtMs(r.totalMs).c_str(),
                     r.avgIterMs, r.maxDisp, r.converged ? "konvergen" : "maks iterasi");
        std::fflush(stderr);
    }
    return r;
}

static void runBenchmark(const Config& cfg, const Image& img) {
    const int REPS = 3;
    const int WARMUP = 1;
    const Impl all[] = {Impl::Serial, Impl::OpenMP, Impl::Simd, Impl::CUDA};

    struct Row { RunResult best; double devFromSerial = 0.0; };
    std::vector<Row> rows(4);
    std::vector<float> refX, refY;

    std::fprintf(stderr, "\nBenchmark: %s (%dx%d), %d titik, max %d iterasi, eps=%.3f, "
                         "seed=%llu, %d reps + %d warmup\n",
                 cfg.input.c_str(), img.w, img.h, cfg.numPoints, cfg.maxIter,
                 cfg.epsilon, (unsigned long long)cfg.seed, REPS, WARMUP);

    for (int r = 0; r < (int)4; ++r) {
        Impl impl = all[r];
        std::fprintf(stderr, "  [%s] ", implName(impl));
        std::fflush(stderr);

        std::vector<float> xs, ys;
        RunArgs args{&img, {cfg.numPoints, cfg.maxIter, cfg.epsilon}, cfg.seed, impl, true, &cfg};

        for (int w = 0; w < WARMUP; ++w) runOnce(args, xs, ys);

        double bestMs = 1e30;
        for (int rep = 0; rep < REPS; ++rep) {
            RunResult rr = runOnce(args, xs, ys);
            std::fprintf(stderr, "%s ", fmtMs(rr.totalMs).c_str());
            std::fflush(stderr);
            if (rr.totalMs < bestMs) { bestMs = rr.totalMs; rows[r].best = rr; }
        }

        if (impl == Impl::Serial) { refX = xs; refY = ys; }
        else {
            float maxDev = 0.0f;
            for (int i = 0; i < cfg.numPoints; ++i) {
                float dx = xs[i] - refX[i], dy = ys[i] - refY[i];
                maxDev = std::max(maxDev, std::sqrt(dx * dx + dy * dy));
            }
            rows[r].devFromSerial = maxDev;
        }
        std::fprintf(stderr, "\n");
    }

    const double serialMs = rows[0].best.totalMs;
    std::printf("\n%-10s | %7s | %9s | %12s | %9s | %7s | %14s\n",
                "impl", "iterasi", "konvergen", "total (ms)", "ms/iter", "speedup", "deviasi vs serial");
    std::printf("%-10s-+-%7s-+-%9s-+-%12s-+-%9s-+-%7s-+-%14s\n",
                "----------", "-------", "---------", "------------", "---------", "-------", "--------------");
    for (int r = 0; r < 4; ++r) {
        const RunResult& b = rows[r].best;
        std::printf("%-10s | %7d | %9s | %12.2f | %9.2f | %6.2fx | %14s\n",
                    implName(all[r]), b.iterations, b.converged ? "ya" : "tidak",
                    b.totalMs, b.avgIterMs, serialMs / b.totalMs,
                    r == 0 ? "-" : (std::to_string(rows[r].devFromSerial) + " px").c_str());
    }
    std::printf("\n");
}

int main(int argc, char** argv) {
    Config cfg;
    bool help = false;
    if (!parseArgs(argc, argv, cfg, help)) return 1;
    if (help) { printUsage(); return 0; }

    if (argc == 1 || cfg.gui) return runGui(argc, argv);

    if (!validate(cfg)) { printUsage(); return 1; }

#ifdef _OPENMP

    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#else
    if (cfg.threads > 0) {
        std::fprintf(stderr, "Peringatan: --threads diabaikan (build tanpa OpenMP).\n");
    }
#endif

    Image img;
    if (!loadImage(cfg.input, img, cfg.gamma, cfg.edgeWeight)) return 1;
    if (img.w <= 0 || img.h <= 0 || img.darkArea <= 0.0) {
        std::fprintf(stderr, "Error: gambar kosong atau seluruhnya putih.\n");
        return 1;
    }
    std::fprintf(stderr, "Input: %s (%dx%d), luas tinta=%.0f px\n",
                 cfg.input.c_str(), img.w, img.h, img.darkArea);

    if (cfg.bench) {
        runBenchmark(cfg, img);
        return 0;
    }

    Params params{cfg.numPoints, cfg.maxIter, cfg.epsilon, cfg.gamma, cfg.radiusScale, cfg.edgeWeight};
    RunArgs args{&img, params, cfg.seed, implFromString(cfg.implName), cfg.quiet, &cfg};

    std::vector<float> xs, ys;
    std::vector<std::vector<uint8_t>> gifFrames;
    if (!cfg.gifPath.empty()) {
        std::vector<uint8_t> frame0;
        initPoints(img, cfg.numPoints, xs, ys, cfg.seed);
        renderStipple(img, xs, ys, {}, frame0, cfg.radiusScale);
        gifFrames.push_back(std::move(frame0));
    }

    RunResult r = runOnce(args, xs, ys, cfg.gifPath.empty() ? nullptr : &gifFrames);

    std::vector<uint8_t> out;
    renderStipple(img, xs, ys, r.cellWeight, out, cfg.radiusScale);
    mkdirParent(cfg.output);
    if (!savePNG(cfg.output, img.w, img.h, out.data())) {
        std::fprintf(stderr, "Gagal menyimpan output '%s'\n", cfg.output.c_str());
        return 1;
    }
    std::fprintf(stderr, "Output tersimpan: %s (%d titik, %d iterasi)\n",
                 cfg.output.c_str(), cfg.numPoints, r.iterations);

    if (!cfg.gifPath.empty()) {
        int delay = 100;
        mkdirParent(cfg.gifPath);
        if (saveGIF(cfg.gifPath, img.w, img.h, gifFrames, delay)) {
            std::fprintf(stderr, "Animasi tersimpan: %s (%zu frame)\n",
                         cfg.gifPath.c_str(), gifFrames.size());
        } else {
            std::fprintf(stderr, "Gagal menyimpan GIF '%s'\n", cfg.gifPath.c_str());
            return 1;
        }
    }
    return 0;
}
