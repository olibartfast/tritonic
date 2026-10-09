#!/usr/bin/env bash
# compute-sanitizer gate for the CUDA code of the GPU preprocessing variants (spec V-14).
#
#   integration-tests/gpu-preprocess/sanitize.sh [kernel|client|server|all] [build-dir]
#
#   kernel  memcheck + racecheck on cuda_preprocess_dump (shared letterbox kernel)
#   client  memcheck on tritonic with --preprocess_device=cuda (needs Triton serving
#           yolo26det_trt on localhost with CUDA shared memory reachable)
#   server  memcheck on tritonserver loading only the V12/V13 preprocessing models from
#           the staged repository, while sending fixture requests
#
# kernel and client need compute-sanitizer on PATH (CUDA toolkit) and a build configured
# with -DTRITONIC_ENABLE_CUDA=ON -DBUILD_TESTING=ON. server copies compute-sanitizer out
# of the TensorRT image, since the Triton server image ships only the CUDA runtime.
set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/versions.sh"

BUILD_DIR="$(realpath -m "${2:-${REPO_ROOT}/build-cuda}")"
OUT="${OUT:-/tmp/tritonic-gpu-preprocess/sanitizer}"
SANITIZER_VOLUME="${SANITIZER_VOLUME:-tritonic-compute-sanitizer-${NGC_CONTAINER_TAG//./-}}"
REPOSITORY="${REPOSITORY:-${REPO_ROOT}/deploy/object_detection/yolo26/ensemble/model_repository}"
FIXTURES=(data/images/bus.jpg data/images/horses.jpg data/images/person.jpg data/images/mug.jpg)
mkdir -p "${OUT}"
failures=0

# compute-sanitizer exits non-zero when it reports errors (--error-exitcode).
record() {
  local name="$1"
  shift
  if "$@" >"${OUT}/${name}.log" 2>&1; then
    echo "PASS ${name}"
  else
    echo "FAIL ${name} (see ${OUT}/${name}.log)"
    failures=$((failures + 1))
  fi
}

run_kernel() {
  for tool in memcheck racecheck; do
    record "kernel_${tool}" compute-sanitizer --tool "${tool}" --error-exitcode 1 \
      "${BUILD_DIR}/cuda_preprocess_dump" "${OUT}/kernel_tensors" "${FIXTURES[@]/#/${REPO_ROOT}/}"
  done
}

run_client() {
  record client_memcheck compute-sanitizer --tool memcheck --error-exitcode 1 \
    "${BUILD_DIR}/tritonic" --source="${REPO_ROOT}/data/images/bus.jpg" --model_type=yolo26 \
    --model=yolo26det_trt --labelsFile="${REPO_ROOT}/labels/coco.txt" --protocol=grpc \
    --port=8001 --write_frame=false --preprocess_device=cuda --shared_memory_type=cuda
}

run_server() {
  docker volume create "${SANITIZER_VOLUME}" >/dev/null
  docker run --rm -v "${SANITIZER_VOLUME}:/export" "${TENSORRT_IMAGE}" \
    sh -c 'd="$(dirname "$(readlink -f "$(command -v compute-sanitizer)")")"
           # On PATH is a wrapper that execs ../compute-sanitizer/compute-sanitizer.
           [ -d "${d}/../compute-sanitizer" ] && d="${d}/../compute-sanitizer"
           cp -a "${d}/." /export/'

  local models=()
  for model in yolo26det_cuda_preprocess yolo26det_nvjpeg_decode; do
    [[ -d "${REPOSITORY}/${model}" ]] && models+=("--load-model=${model}")
  done
  if [[ ${#models[@]} -eq 0 ]]; then
    echo "FAIL server: no staged yolopreprocess models in ${REPOSITORY}"
    failures=$((failures + 1))
    return
  fi

  local container="tritonic-sanitizer-$$"
  docker run -d --rm --name "${container}" --gpus all --shm-size=1g --network host \
    -v "${SANITIZER_VOLUME}:/sanitizer:ro" -v "${REPOSITORY}:/models:ro" -v "${OUT}:/out" \
    "${TRITON_IMAGE}" \
    /sanitizer/compute-sanitizer --tool memcheck --error-exitcode 1 \
    --log-file /out/server_memcheck.log \
    tritonserver --model-repository=/models --model-control-mode=explicit "${models[@]}" \
    --disable-auto-complete-config --allow-client-shm=true >/dev/null

  # The sanitizer slows startup considerably.
  for _ in $(seq 1 120); do
    curl -sf localhost:8000/v2/health/ready >/dev/null && break
    sleep 2
  done
  for model in yolo26det_cuda_preprocess yolo26det_nvjpeg_decode; do
    [[ -d "${REPOSITORY}/${model}" ]] || continue
    for fixture in "${FIXTURES[@]}"; do
      python3 - "${model}" "${REPO_ROOT}/${fixture}" <<'EOF' >>"${OUT}/server_requests.log" 2>&1 || true
import sys
import numpy as np
import tritonclient.http as httpclient
model, path = sys.argv[1:]
data = np.fromfile(path, dtype=np.uint8).reshape(1, -1)
tensor = httpclient.InferInput("IMAGE", list(data.shape), "UINT8")
tensor.set_data_from_numpy(data)
httpclient.InferenceServerClient("localhost:8000").infer(model, [tensor])
print(model, path, "ok")
EOF
    done
  done
  # Stop the server so the sanitizer writes its final summary to the mounted log, then
  # wait for the container (whose main process is the sanitizer) to exit.
  docker exec "${container}" sh -c 'kill -INT $(pidof tritonserver)' || true
  timeout 180 docker wait "${container}" >/dev/null 2>&1 || docker stop "${container}" >/dev/null 2>&1 || true
  if grep -q "ERROR SUMMARY: 0 errors" "${OUT}/server_memcheck.log" 2>/dev/null; then
    echo "PASS server_memcheck"
  else
    echo "FAIL server_memcheck (see ${OUT}/server_memcheck.log)"
    failures=$((failures + 1))
  fi
}

case "${MODE}" in
  kernel) run_kernel ;;
  client) run_client ;;
  server) run_server ;;
  all) run_kernel; run_client; run_server ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
echo "Logs: ${OUT}"
exit "${failures}"
