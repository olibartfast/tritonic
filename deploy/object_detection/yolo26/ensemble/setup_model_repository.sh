#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --engine /path/to/yolo26-det.engine [--repository path] [--triton-image ref]" >&2
  echo "          [--with-cuda-backend] [--trt-pre-onnx /path/to/yolo26-det.onnx]" >&2
  echo "  --with-cuda-backend   also stage the custom CUDA preprocessing backend (variant V12)" >&2
  echo "  --trt-pre-onnx PATH   also build the TensorRT-graph letterbox engine (variant V13);" >&2
  echo "                        implies --with-cuda-backend for its nvJPEG decode step" >&2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
repository="${script_dir}/model_repository"
# Overridable default: env var, or --triton-image on the command line.
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.08-py3}"
engine=""
with_cuda_backend=false
trt_pre_onnx=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) engine="$2"; shift 2 ;;
    --repository) repository="$2"; shift 2 ;;
    --triton-image) TRITON_IMAGE="$2"; shift 2 ;;
    --with-cuda-backend) with_cuda_backend=true; shift ;;
    --trt-pre-onnx) trt_pre_onnx="$2"; with_cuda_backend=true; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${engine}" || ! -s "${engine}" ]]; then
  usage
  exit 2
fi
if [[ -n "${trt_pre_onnx}" && ! -s "${trt_pre_onnx}" ]]; then
  echo "--trt-pre-onnx file not found: ${trt_pre_onnx}" >&2
  exit 2
fi

repository="$(realpath -m "${repository}")"
case "${repository}/" in
  "${repo_root}/"*) ;;
  *) echo "--repository must be inside ${repo_root}" >&2; exit 2 ;;
esac
relative_repository="${repository#"${repo_root}/"}"
models=(
  yolo26det_trt
  yolo26det_dali_preprocess
  yolo26det_dali_postprocess
  yolo26det_gpu_pre_cpu_post
  yolo26det_gpu_pre_gpu_post
)
# GPU preprocessing alternatives (specs/2026-10-01-gpu-preprocess-alternatives).
if [[ "${with_cuda_backend}" == true ]]; then
  models+=(yolo26det_cuda_preprocess yolo26det_cudapre_cpu_post yolo26det_cudapre_gpu_post)
fi
if [[ -n "${trt_pre_onnx}" ]]; then
  models+=(yolo26det_nvjpeg_decode yolo26det_trt_pre yolo26det_trtpre_cpu_post
           yolo26det_trtpre_gpu_post)
fi
source_repository="${script_dir}/model_repository"
for model in "${models[@]}"; do
  mkdir -p "${repository}/${model}/1"
  # config.pbtxt is source, not a generated artifact: copy it so --repository
  # pointing at a fresh path produces a repository Triton can actually load.
  # Variant models live outside model_repository/ so that a repository staged without
  # their flags never contains a model Triton cannot load.
  if [[ -f "${script_dir}/variants/${model}/config.pbtxt" ]]; then
    cp "${script_dir}/variants/${model}/config.pbtxt" "${repository}/${model}/config.pbtxt"
  elif [[ "${repository}" != "${source_repository}" ]]; then
    cp "${source_repository}/${model}/config.pbtxt" "${repository}/${model}/config.pbtxt"
  fi
done
cp "${engine}" "${repository}/yolo26det_trt/1/model.plan"
"${script_dir}/dali_plugin/build_plugin.sh"
cp "${script_dir}/dali_plugin/build/libyolo26_det_dali.so" \
  "${repository}/libyolo26_det_dali.so"

docker run --gpus all --rm \
  -v "${repo_root}:/workspace" -w /workspace \
  "${TRITON_IMAGE}" \
  python3 deploy/object_detection/yolo26/ensemble/generate_pipeline.py \
    --output "${relative_repository}/yolo26det_dali_preprocess/1/model.dali"

docker run --gpus all --rm \
  -v "${repo_root}:/workspace" -w /workspace \
  "${TRITON_IMAGE}" \
  python3 deploy/object_detection/yolo26/ensemble/generate_postprocess_pipeline.py \
    --plugin "${relative_repository}/libyolo26_det_dali.so" \
    --output "${relative_repository}/yolo26det_dali_postprocess/1/model.dali"

if [[ "${with_cuda_backend}" == true ]]; then
  "${repo_root}/backends/yolopreprocess/build_backend.sh"
  # Triton looks for libtriton_<backend>.so in the model directory before the global
  # backend directory, so no extra server mount is needed.
  for model in yolo26det_cuda_preprocess yolo26det_nvjpeg_decode; do
    if [[ -d "${repository}/${model}" ]]; then
      cp "${repo_root}/backends/yolopreprocess/build/libtriton_yolopreprocess.so" \
        "${repository}/${model}/libtriton_yolopreprocess.so"
    fi
  done
fi
if [[ -n "${trt_pre_onnx}" ]]; then
  "${script_dir}/trt_pre/build_trt_pre_engine.sh" \
    --onnx "${trt_pre_onnx}" --output "${repository}/yolo26det_trt_pre/1/model.plan"
fi

engine_sha256="$(sha256sum "${repository}/yolo26det_trt/1/model.plan" | awk '{print $1}')"
cat >"${repository}/reference_model.yaml" <<EOF
model: yolo26-det
engine_sha256: ${engine_sha256}
triton_image: ${TRITON_IMAGE}
input: {name: images, shape: [1, 3, 640, 640], datatype: FP32}
outputs:
  - {name: output0, shape: [1, 300, 6], datatype: FP32}
confidence_threshold: 0.5
EOF
echo "YOLO26 detection model repository prepared at ${repository}"
