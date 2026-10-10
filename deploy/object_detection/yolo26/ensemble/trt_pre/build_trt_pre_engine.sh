#!/usr/bin/env bash
# Builds the V13 engine yolo26det_trt_pre: the R-2 letterbox prepended to a YOLO26
# detection ONNX model, compiled with one dynamic-shape profile.
#
#   build_trt_pre_engine.sh --onnx /path/yolo26.onnx --output /path/model.plan
#
# The profile range must cover the decode model's max_side (4096 in
# yolo26det_nvjpeg_decode/config.pbtxt). PRECISION=fp32 (default) keeps the whole graph in FP32;
# TensorRT 11 builds strongly typed networks only, and the end2end YOLO26 export mixes
# Float and Half Concat inputs, which TensorRT's parser rejects (2026-10-09).
# PRECISION=fp16 converts the YOLO graph to FP16 compute before the FP32 letterbox is prepended.
# This is tracked in specs/roadmap.md as experimental pending a YOLO export fix.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
source "${repo_root}/scripts/versions.sh"
PRECISION="${PRECISION:-fp32}"
PAD_ROUNDING="${PAD_ROUNDING:-floor}"
MIN_SHAPE="${MIN_SHAPE:-1x1x1x3}"
OPT_SHAPE="${OPT_SHAPE:-1x1080x1920x3}"
MAX_SHAPE="${MAX_SHAPE:-1x4096x4096x3}"

usage() { echo "Usage: $0 --onnx /path/yolo26.onnx --output /path/model.plan" >&2; }
onnx=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) onnx="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -s "${onnx}" && -n "${output}" ]] || { usage; exit 2; }

onnx="$(realpath "${onnx}")"
mkdir -p "$(dirname "${output}")"
output="$(realpath -m "${output}")"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cp "${onnx}" "${work}/yolo26.onnx"


docker run --rm --gpus all \
  -v "${script_dir}:/trt_pre:ro" -v "${work}:/work" -w /work \
  "${TENSORRT_IMAGE}" \
  bash -c "set -e
    pip install --quiet --no-cache-dir onnx numpy onnxconverter-common >/dev/null
    yolo=yolo26.onnx
    if [[ \"${PRECISION}\" == fp16 ]]; then
      python3 /trt_pre/to_fp16_onnx.py --input yolo26.onnx --output yolo26.fp16.onnx
      yolo=yolo26.fp16.onnx
    fi
    python3 /trt_pre/generate_trt_pre_onnx.py --yolo-onnx \"\${yolo}\" \
      --output yolo26det_trt_pre.onnx --pad-rounding ${PAD_ROUNDING}
    trtexec=\$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
    \"\${trtexec}\" --onnx=yolo26det_trt_pre.onnx --saveEngine=model.plan \
      --minShapes=image:${MIN_SHAPE} --optShapes=image:${OPT_SHAPE} --maxShapes=image:${MAX_SHAPE}
    chown $(id -u):$(id -g) model.plan yolo26det_trt_pre.onnx"
cp "${work}/model.plan" "${output}"
echo "${output}"
