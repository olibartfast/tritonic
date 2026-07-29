#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
volume="tritonic-dali-25-12-build"
docker volume create "${volume}" >/dev/null
docker run --rm -v "${volume}:/export" nvcr.io/nvidia/tritonserver:25.12-py3 \
  sh -lc 'cp -a /opt/tritonserver/backends/dali/wheel/dali/nvidia/dali/. /export/'
docker run --rm --gpus all \
  -v "${volume}:/dali:ro" -v "${script_dir}:/src" -w /src \
  nvcr.io/nvidia/tensorrt:25.12-py3 \
  sh -lc 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2'
echo "${script_dir}/build/libyolo26_seg_dali.so"
