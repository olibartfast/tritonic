#!/usr/bin/env bash
# Full GPU gate for specs/2026-10-01-gpu-preprocess-alternatives on a Docker GPU host
# (e.g. an NVIDIA Brev VM). Everything except Docker itself runs in the NGC 26.08 images.
#
#   integration-tests/gpu-preprocess/run_gpu_host_gate.sh [results-dir]
#
# Steps (each logs to <results>/logs/<step>.log; a failing step is recorded, not fatal):
#   env      environment identity (V-0)
#   libs     the long-lived SDK work container (client libraries are in it at /workspace/install)
#   build    tritonic default + TRITONIC_ENABLE_CUDA builds with unit tests (V-3, V-4, V-6)
#   model    YOLO26n export to ONNX and FP16 TensorRT engine (no model is checked in)
#   stage    setup_model_repository.sh with every variant (DALI, V12, V13)
#   triton   start tritonserver:26.08, wait until every model is ready
#   spikes   T-2/T-3 again on this host (run_spikes.sh)
#   tensors  V-5: kernel tool and V12 server output vs the DALI preprocess model
#   gates    V-7..V-11, V-13, B1-vs-B0 detection parity
#   bench    V-12: run.py (HTTP+gRPC) and perf.py
#   sanitize V-14: kernel + V2 client in the SDK container, server on the host
#   report   compare.py -> comparison.md
#
# Env overrides: STEPS (space-separated subset), CUDA_ARCH (default: detected), YOLO_MODEL
# (default yolo26n.pt), SOAK_SEQUENTIAL / SOAK_CONCURRENT, BENCH_ITERATIONS, LABEL.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO}/scripts/versions.sh"

RESULTS="$(mkdir -p "${1:-${REPO}/gate-results}" && cd "${1:-${REPO}/gate-results}" && pwd)"
TAG="${NGC_CONTAINER_TAG}"
SDK_IMAGE="${TRITON_SDK_IMAGE}"
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')"
LABEL="${LABEL:-${GPU_NAME}-${TAG}}"
YOLO_MODEL="${YOLO_MODEL:-yolo26n.pt}"
MODELS="${REPO}/gate-models"
REPOSITORY="${REPO}/deploy/object_detection/yolo26/ensemble/model_repository"
STEPS="${STEPS:-env libs build model stage triton spikes tensors gates bench sanitize report}"
WORK=tritonic-gate-sdk
SERVER=tritonic-gate-triton
# FP32 by default: onnxconverter-common leaves the YOLO26 end2end head with mixed Half/Float
# Concat inputs, which TensorRT 11's strongly typed parser rejects (2026-10-09). Every
# variant uses the same precision, so the preprocessing comparison stays fair.
PRECISION="${PRECISION:-fp32}"
export TRITON_IMAGE TENSORRT_IMAGE CUDA_ARCH PRECISION
mkdir -p "${RESULTS}/logs" "${RESULTS}/gates" "${MODELS}"
SUMMARY="${RESULTS}/STATUS"
: > "${SUMMARY}"

note() { echo "$*" | tee -a "${SUMMARY}"; }
step() {
  local name="$1"; shift
  [[ " ${STEPS} " == *" ${name} "* ]] || return 0
  echo "=== ${name} ($(date -Is))"
  if "$@" > "${RESULTS}/logs/${name}.log" 2>&1; then note "PASS ${name}"; else note "FAIL ${name} (logs/${name}.log)"; fi
}
# Commands inside the long-lived SDK container (host network and IPC for CUDA shared memory).
sdk() { docker exec -w /w "${WORK}" bash -lc "$*"; }

start_sdk() {
  docker rm -f "${WORK}" >/dev/null 2>&1 || true
  docker run -d --name "${WORK}" --gpus all --network host --ipc host --pid host \
    -v "${REPO}:/w" -e TritonClientBuild_DIR=/workspace/install "${SDK_IMAGE}" sleep infinity >/dev/null
  sdk "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq cmake ninja-build \
       libcurl4-openssl-dev rapidjson-dev libprotobuf-dev protobuf-compiler >/dev/null && \
       pip install -q opencv-python-headless numpy 'tritonclient[http,grpc]' \"nvidia-dali-cuda130==${NGC_DALI_VERSION}\" && git config --global --add safe.directory /w"
}

do_env() {
  for image in "${TRITON_IMAGE}" "${SDK_IMAGE}" "${TENSORRT_IMAGE}"; do docker pull -q "${image}"; done
  nvidia-smi
  "${REPO}/benchmarks/yolo26-det/env_probe.sh" "${RESULTS}/environment.json"
}

do_libs() { start_sdk; }

do_build() {
  sdk "cmake -S . -B build-sdk -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON && cmake --build build-sdk && ctest --test-dir build-sdk --output-on-failure" &&
  sdk "cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DTRITONIC_ENABLE_CUDA=ON -DCUDA_ARCH=${CUDA_ARCH} && cmake --build build-cuda && ctest --test-dir build-cuda --output-on-failure"
}

do_model() {
  # end2end=True exports the NMS-free YOLO26 head as output0 [1,300,6], the yolo26det_trt contract;
  # without it Ultralytics 8.4 exports the raw [1,84,8400] head.
  # TensorRT 11 has no --fp16: FP16 comes from an FP16-typed ONNX with FP32 I/O (to_fp16_onnx.py).
  docker run --rm --gpus all -v "${MODELS}:/m" -v "${REPO}/deploy/object_detection/yolo26/ensemble/trt_pre:/trt_pre:ro" \
    -w /m "${TENSORRT_IMAGE}" bash -c "
    set -e
    pip install -q ultralytics onnx onnxslim onnxconverter-common >/dev/null
    # ultralytics pulls GUI OpenCV, which needs libxcb (absent from the TensorRT image).
    pip uninstall -y -q opencv-python >/dev/null 2>&1 || true
    pip install -q --force-reinstall --no-deps opencv-python-headless >/dev/null
    yolo export model=${YOLO_MODEL} format=onnx imgsz=640 opset=17 end2end=True > export.log 2>&1 || (cat export.log; exit 1)
    mv \"\$(basename ${YOLO_MODEL} .pt).onnx\" yolo26-det.onnx
    trtexec=\$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
    onnx=yolo26-det.onnx
    if [[ ${PRECISION} == fp16 ]]; then
      python3 /trt_pre/to_fp16_onnx.py --input yolo26-det.onnx --output yolo26-det.fp16.onnx
      onnx=yolo26-det.fp16.onnx
    fi
    \"\${trtexec}\" --onnx=\"\${onnx}\" --saveEngine=yolo26-det.engine > trtexec.log 2>&1 || (tail -30 trtexec.log; exit 1)
    python3 -c \"import onnx, sys; m = onnx.load('yolo26-det.onnx'); outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]; print(outs); sys.exit(outs != [('output0', [1, 300, 6])])\"
    chown -R $(id -u):$(id -g) /m"
}

do_stage() {
  "${REPO}/deploy/object_detection/yolo26/ensemble/setup_model_repository.sh" \
    --engine "${MODELS}/yolo26-det.engine" --trt-pre-onnx "${MODELS}/yolo26-det.onnx" || return 1
  find "${REPOSITORY}" -maxdepth 3 -type f | sort
}

start_triton() {
  docker rm -f "${SERVER}" >/dev/null 2>&1 || true
  docker run -d --name "${SERVER}" --gpus all --network host --ipc host --pid host --shm-size=1g \
    -v "${REPOSITORY}:/models:ro" "${TRITON_IMAGE}" \
    tritonserver --model-repository=/models --disable-auto-complete-config \
      --allow-client-shm=true --backend-config=dali,plugin_libs=/models/libyolo26_det_dali.so >/dev/null
  for _ in $(seq 1 120); do curl -sf localhost:8000/v2/health/ready >/dev/null && break; sleep 2; done
  docker logs "${SERVER}" 2>&1 | tail -40
  curl -s -X POST localhost:8000/v2/repository/index | python3 -c "
import json, sys
models = json.load(sys.stdin)
for m in models: print(m.get('name'), m.get('state'), m.get('reason', ''))
sys.exit(0 if models and all(m.get('state') == 'READY' for m in models) else 1)"
}

do_spikes() { "${REPO}/integration-tests/gpu-preprocess/run_spikes.sh" "${RESULTS}/spikes"; }

do_tensors() {
  local fixtures="data/images/bus.jpg data/images/horses.jpg data/images/person.jpg data/images/mug.jpg"
  sdk "python3 - <<'EOF'
import pathlib, numpy as np, tritonclient.http as httpclient
client = httpclient.InferenceServerClient('localhost:8000')
for model, out in (('yolo26det_dali_preprocess', 'dali_server'), ('yolo26det_cuda_preprocess', 'v12_server')):
    d = pathlib.Path('/w/gate-results-tmp') / out; d.mkdir(parents=True, exist_ok=True)
    for name in ('bus', 'horses', 'person', 'mug'):
        data = np.fromfile(f'data/images/{name}.jpg', dtype=np.uint8).reshape(1, -1)
        t = httpclient.InferInput('IMAGE', list(data.shape), 'UINT8'); t.set_data_from_numpy(data)
        r = client.infer(model, [t], outputs=[httpclient.InferRequestedOutput('PREPROCESSED'), httpclient.InferRequestedOutput('ORIGINAL_SIZE')])
        r.as_numpy('PREPROCESSED')[0].astype(np.float32).tofile(d / f'{name}.fp32')
print('server tensors written')
EOF
build-cuda/cuda_preprocess_dump gate-results-tmp/kernel ${fixtures}" &&
  sdk "python3 integration-tests/gpu-preprocess/compare_tensors.py --reference gate-results-tmp/dali_server --actual gate-results-tmp/v12_server --output gate-results-tmp/tensors_V12.json;
       r1=\$?; python3 integration-tests/gpu-preprocess/compare_tensors.py --reference gate-results-tmp/dali_server --actual gate-results-tmp/kernel --output gate-results-tmp/tensors_V2.json; r2=\$?;
       exit \$((r1 + r2))"
  local rc=$?
  cp "${REPO}"/gate-results-tmp/tensors_*.json "${RESULTS}/gates/" 2>/dev/null
  return "${rc}"
}

do_gates() {
  local rc=0 t=build-cuda/tritonic g=/w/gate-results-tmp
  sdk "python3 integration-tests/gpu-preprocess/compare_detections.py --tritonic ${t} --candidates B1,B1-gpupost --reference B0 --output ${g}/detections_B1_vs_B0.json" || rc=1
  sdk "python3 integration-tests/gpu-preprocess/compare_detections.py --tritonic ${t} --candidates V2,V12,V12-gpupost,V13,V13-gpupost --output ${g}/detections.json" || rc=1
  sdk "python3 integration-tests/gpu-preprocess/compare_decode.py --output ${g}/decode.json" || rc=1
  sdk "python3 integration-tests/gpu-preprocess/invalid_inputs.py --tritonic ${t} --output ${g}/invalid_inputs.json" || rc=1
  sdk "python3 integration-tests/gpu-preprocess/soak.py --sequential ${SOAK_SEQUENTIAL:-10000} --concurrent-requests ${SOAK_CONCURRENT:-1000} --output ${g}/soak.json" || rc=1
  cp "${REPO}"/gate-results-tmp/{detections*,decode,invalid_inputs,soak}.json "${RESULTS}/gates/" 2>/dev/null
  return "${rc}"
}

do_bench() {
  local out=/w/gate-results-tmp/bench
  cp "${RESULTS}/environment.json" "${REPO}/gate-results-tmp/environment.json"
  sdk "python3 benchmarks/yolo26-det/run.py --tritonic build-cuda/tritonic --label ${LABEL} --variants B0,B1,B1-gpupost,V12,V12-gpupost,V13,V13-gpupost,V2 --iterations ${BENCH_ITERATIONS:-100} --environment /w/gate-results-tmp/environment.json --output-dir ${out}" &&
  sdk "python3 benchmarks/yolo26-det/perf.py --variants B0,B1,V12,V13,V2 --label ${LABEL} --output-dir ${out}"
  local rc=$?
  cp -r "${REPO}/gate-results-tmp/bench/." "${RESULTS}/"
  return "${rc}"
}

do_sanitize() {
  sdk "integration-tests/gpu-preprocess/sanitize.sh kernel build-cuda; k=\$?; integration-tests/gpu-preprocess/sanitize.sh client build-cuda; exit \$((k + \$?))"
  local rc=$?
  # The server sanitizer runs its own tritonserver on the same ports, and sends its requests
  # from the host Python.
  docker rm -f "${SERVER}" >/dev/null 2>&1
  python3 -m pip install -q --user --break-system-packages numpy 'tritonclient[http]' >/dev/null 2>&1 ||
    python3 -m pip install -q --user numpy 'tritonclient[http]' >/dev/null 2>&1
  REPOSITORY="${REPOSITORY}" "${REPO}/integration-tests/gpu-preprocess/sanitize.sh" server || rc=1
  cp -r /tmp/tritonic-gpu-preprocess/sanitizer "${RESULTS}/" 2>/dev/null
  sdk "mkdir -p /w/gate-results-tmp/sanitizer && cp -r /tmp/tritonic-gpu-preprocess/sanitizer/. /w/gate-results-tmp/sanitizer/ 2>/dev/null; true"
  cp -r "${REPO}/gate-results-tmp/sanitizer/." "${RESULTS}/sanitizer/" 2>/dev/null
  return "${rc}"
}

do_report() {
  (cd "${REPO}" && python3 benchmarks/yolo26-det/compare.py "${RESULTS}")
}

mkdir -p "${REPO}/gate-results-tmp"
step env do_env
step libs do_libs
[[ " ${STEPS} " == *" libs "* ]] || start_sdk > "${RESULTS}/logs/sdk.log" 2>&1
step build do_build
step model do_model
step stage do_stage
step triton start_triton
step spikes do_spikes
step tensors do_tensors
step gates do_gates
step bench do_bench
step sanitize do_sanitize
step report do_report
docker rm -f "${WORK}" "${SERVER}" >/dev/null 2>&1
sudo chown -R "$(id -u):$(id -g)" "${REPO}" 2>/dev/null || true
tar -czf "${RESULTS}.tar.gz" -C "$(dirname "${RESULTS}")" "$(basename "${RESULTS}")"
note "DONE $(date -Is) -> ${RESULTS}.tar.gz"
cat "${SUMMARY}"
