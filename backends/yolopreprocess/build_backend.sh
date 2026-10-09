#!/usr/bin/env bash
# Builds libtriton_yolopreprocess.so in the TensorRT image of the pinned Triton release
# (specs/2026-10-01-gpu-preprocess-alternatives, plan T-15). No GPU is needed to build.
#
#   backends/yolopreprocess/build_backend.sh
#
# Overridable defaults, e.g. CUDA_ARCH="86;89" BUILD_JOBS=8 ./build_backend.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/scripts/versions.sh"

# Semicolon-separated CUDA architectures; 86 is the RTX 30-series default.
CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_JOBS="${BUILD_JOBS:-2}"

# The backend compiles src/cuda/letterbox.cu from the repository root, so mount it whole.
docker run --rm \
  -v "${repo_root}:/workspace" -w /workspace/backends/yolopreprocess \
  "${TENSORRT_IMAGE}" \
  bash -c "set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null && apt-get install -y -qq git rapidjson-dev >/dev/null
    # The r26.08 common repo requires CMake >= 3.31.8.
    pip install --quiet --no-cache-dir 'cmake>=3.31.8' >/dev/null
    hash -r; cmake --version | head -1
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='${CUDA_ARCH}'
    cmake --build build -j${BUILD_JOBS}
    chown -R $(id -u):$(id -g) build"
echo "${script_dir}/build/libtriton_yolopreprocess.so"
