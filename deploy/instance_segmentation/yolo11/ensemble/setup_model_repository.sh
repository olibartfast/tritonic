#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --engine /path/to/yolo11m-seg.engine [--repository path]" >&2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
repository="${script_dir}/model_repository"
engine=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) engine="$2"; shift 2 ;;
    --repository) repository="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${engine}" || ! -s "${engine}" ]]; then
  usage
  exit 2
fi

repository="$(realpath -m "${repository}")"
case "${repository}/" in
  "${repo_root}/"*) ;;
  *) echo "--repository must be inside ${repo_root}" >&2; exit 2 ;;
esac
relative_repository="${repository#"${repo_root}/"}"
mkdir -p \
  "${repository}/yolo11seg_trt/1" \
  "${repository}/yolo11seg_dali_preprocess/1" \
  "${repository}/yolo11seg_dali_postprocess/1" \
  "${repository}/yolo11seg_dali_mask_postprocess/1" \
  "${repository}/yolo11seg_gpu_pre_cpu_post/1" \
  "${repository}/yolo11seg_gpu_pre_gpu_post/1" \
  "${repository}/yolo11seg_gpu_pre_gpu_mask_post/1"
cp "${engine}" "${repository}/yolo11seg_trt/1/model.plan"
"${script_dir}/dali_plugin/build_plugin.sh"
cp "${script_dir}/dali_plugin/build/libyolo11_seg_dali.so" \
  "${repository}/libyolo11_seg_dali.so"

docker run --gpus all --rm \
  -v "${repo_root}:/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 deploy/instance_segmentation/yolo11/ensemble/generate_pipeline.py \
    --output "${relative_repository}/yolo11seg_dali_preprocess/1/model.dali"

docker run --gpus all --rm \
  -v "${repo_root}:/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 deploy/instance_segmentation/yolo11/ensemble/generate_postprocess_pipeline.py \
    --plugin "${relative_repository}/libyolo11_seg_dali.so" \
    --output "${relative_repository}/yolo11seg_dali_postprocess/1/model.dali"

docker run --gpus all --rm \
  -v "${repo_root}:/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 deploy/instance_segmentation/yolo11/ensemble/generate_mask_postprocess_pipeline.py \
    --plugin "${relative_repository}/libyolo11_seg_dali.so" \
    --output "${relative_repository}/yolo11seg_dali_mask_postprocess/1/model.dali"

engine_sha256="$(sha256sum "${repository}/yolo11seg_trt/1/model.plan" | awk '{print $1}')"
cat >"${repository}/reference_model.yaml" <<EOF
model: yolo11m-seg
engine_sha256: ${engine_sha256}
triton_image: nvcr.io/nvidia/tritonserver:25.12-py3
input: {name: images, shape: [1, 3, 640, 640], datatype: FP32}
outputs:
  - {name: output0, shape: [1, 116, 8400], datatype: FP32}
  - {name: output1, shape: [1, 32, 160, 160], datatype: FP32}
confidence_threshold: 0.5
mask_threshold: 0.5
segmentation_outputs: [mask, polygon]
EOF
echo "YOLO11m-seg model repository prepared at ${repository}"
