#!/usr/bin/env bash
# Phase 0 spikes for specs/2026-10-01-gpu-preprocess-alternatives (plan T-2, T-3).
# Requires an NVIDIA GPU host with the Docker GPU runtime.
#
#   integration-tests/gpu-preprocess/run_spikes.sh [output-dir]
#
# T-3 runs first: it writes the DALI tensors that T-2 compares TensorRT against.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/versions.sh"

OUTPUT_DIR="$(mkdir -p "${1:-/tmp/tritonic-spikes}" && cd "${1:-/tmp/tritonic-spikes}" && pwd)"
PIP_DEPS="numpy opencv-python-headless onnx"

run_in() {
  local image="$1"
  shift
  docker run --rm --gpus all \
    -v "${REPO_ROOT}:/workspace:ro" \
    -v "${OUTPUT_DIR}:/spikes" \
    -w /workspace \
    "${image}" \
    bash -c "pip install --quiet --no-cache-dir ${PIP_DEPS} >/dev/null && $*"
}

echo "== T-3: DALI letterbox conventions (${TRITON_IMAGE})"
run_in "${TRITON_IMAGE}" \
  python3 integration-tests/gpu-preprocess/dali_letterbox_spike.py \
    --tensor-dir /spikes/dali --output /spikes/dali_letterbox.json

PAD_ROUNDING="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["pad_rounding"]; print(r if r in ("floor","ceil") else "floor")' "${OUTPUT_DIR}/dali_letterbox.json")"
echo "   measured pad rounding: ${PAD_ROUNDING}"

echo "== T-2: TensorRT letterbox graph (${TENSORRT_IMAGE})"
run_in "${TENSORRT_IMAGE}" \
  python3 integration-tests/gpu-preprocess/trt_letterbox_spike.py \
    --pad-rounding "${PAD_ROUNDING}" --dali-dir /spikes/dali --output /spikes/trt_letterbox.json

echo "Reports: ${OUTPUT_DIR}/dali_letterbox.json ${OUTPUT_DIR}/trt_letterbox.json"
