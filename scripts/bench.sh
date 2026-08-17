#!/usr/bin/env bash
# Build + benchmark lengkap semua implementasi menggunakan contoh Umaru.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== [1/2] Build ==="
make -j$(nproc)

if [[ ! -f output/umaru_stipple.png ]]; then
    echo "Error: output/umaru_stipple.png tidak ditemukan." >&2
    exit 1
fi

echo
echo "=== [2/2] Benchmark Umaru (1920x1080, 1000 titik, 10 iterasi, eps 0.2) ==="
./stipple --bench \
    --input output/umaru_stipple.png \
    --points 1000 \
    --iters 10 \
    --eps 0.2 \
    --output output/bench_umaru.png
