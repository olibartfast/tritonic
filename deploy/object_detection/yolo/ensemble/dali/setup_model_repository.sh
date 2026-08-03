#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --engine /path/to/model.plan [--repository path/inside/checkout]" >&2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
repository="${script_dir}/model_repository"
engine=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      engine="$2"
      shift 2
      ;;
    --repository)
      repository="$2"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${engine}" || ! -f "${engine}" ]]; then
  usage
  exit 2
fi

repository="$(realpath -m "${repository}")"
case "${repository}/" in
  "${repo_root}/"*) ;;
  *)
    echo "--repository must be inside the Tritonic checkout: ${repo_root}" >&2
    exit 2
    ;;
esac
relative_repository="${repository#"${repo_root}/"}"

mkdir -p "${repository}/yolo_dali_preprocess/1" "${repository}/yolo_trt/1"
cp "${engine}" "${repository}/yolo_trt/1/model.plan"

docker run --gpus all --rm \
  -v "${repo_root}:/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 deploy/object_detection/yolo/ensemble/dali/generate_pipeline.py \
    --output "${relative_repository}/yolo_dali_preprocess/1/model.dali"

engine_sha256="$(sha256sum "${repository}/yolo_trt/1/model.plan" | awk '{print $1}')"
cat >"${repository}/reference_model.yaml" <<EOF
engine_sha256: ${engine_sha256}
triton_image: nvcr.io/nvidia/tritonserver:25.12-py3
input_name: images
input_shape: [1, 3, 640, 640]
input_datatype: FP32
input_format: NCHW
output_names: [output0]
output_shapes: [[1, 84, 8400]]
output_datatypes: [FP32]
confidence_threshold: 0.5
nms_threshold: 0.4
EOF

echo "Model repository prepared at ${repository}"
