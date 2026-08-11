#!/usr/bin/env bash
# Jalur cepat: build + benchmark lengkap semua implementasi.
# (total waktu jalan ~1-2 menit)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== [1/3] Build ==="
make -j$(nproc)

echo
echo "=== [2/3] Generate sample images ==="
python3 scripts/make_sample.py sample
mkdir -p output

echo
echo "=== [3/3] Benchmark (512x512, 2000 titik, 60 iterasi, eps 0.05) ==="
./stipple --bench \
    --input sample/target.png \
    --points 2000 \
    --iters 60 \
    --eps 0.05 \
    --output output/bench_target.png \
    --gif output/bench_anim.gif \
    --gif-every 3
