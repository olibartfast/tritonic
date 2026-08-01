#!/usr/bin/env bash
set -euo pipefail

# Overridable defaults. Set once here; override from the environment, e.g.
#   TRITON_IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3 CUDA_ARCH="86;89" ./build_plugin.sh
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.12-py3}"
TENSORRT_IMAGE="${TENSORRT_IMAGE:-nvcr.io/nvidia/tensorrt:25.12-py3}"
# Semicolon-separated CUDA architectures; 86 is the RTX 30-series default.
CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_JOBS="${BUILD_JOBS:-2}"
# Cache of DALI headers/libs extracted from the Triton image.
DALI_VOLUME="${DALI_VOLUME:-tritonic-dali-25-12-build}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker volume create "${DALI_VOLUME}" >/dev/null
docker run --rm -v "${DALI_VOLUME}:/export" "${TRITON_IMAGE}" \
  sh -lc 'cp -a /opt/tritonserver/backends/dali/wheel/dali/nvidia/dali/. /export/'
docker run --rm --gpus all \
  -v "${DALI_VOLUME}:/dali:ro" -v "${script_dir}:/src" -w /src \
  "${TENSORRT_IMAGE}" \
  sh -lc "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES='${CUDA_ARCH}' \
          && cmake --build build -j${BUILD_JOBS}"
echo "${script_dir}/build/libyolo26_seg_dali.so"
