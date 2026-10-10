#!/usr/bin/env bash
# Environment identity for GPU preprocessing evidence (spec V-0, plan T-1).
#
#   benchmarks/yolo26-det/env_probe.sh [output.json]
#
# Prints JSON with the GPU, driver, client CPU, git commit, and the digest plus
# CUDA/TensorRT versions of the server and TensorRT images. Versions come from
# the image metadata, so no container is started. "images_consistent" checks
# assumption A-4: both images must ship the same CUDA and TensorRT versions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/versions.sh"

image_json() {
  if docker image inspect "$1" >/dev/null 2>&1; then
    docker image inspect "$1" --format '{"digest": {{json .RepoDigests}}, "env": {{json .Config.Env}}}'
  else
    echo '{"digest": null, "env": []}'
  fi
}

gpu_csv="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)"

python3 - "$TRITON_IMAGE" "$(image_json "$TRITON_IMAGE")" \
  "$TENSORRT_IMAGE" "$(image_json "$TENSORRT_IMAGE")" \
  "$gpu_csv" "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || head -1 "$REPO_ROOT/COMMIT" 2>/dev/null || echo unknown)" \
  "$( if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then (git -C "$REPO_ROOT" status --porcelain 2>/dev/null || echo) | grep -c . ; else echo unknown; fi )" \
  "$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)" \
  "${1:-}" <<'EOF'
import json
import sys

(triton_image, triton_json, trt_image, trt_json, gpu_csv, commit, dirty, cpu, output) = sys.argv[1:]

KEYS = ("CUDA_VERSION", "TRT_VERSION", "TENSORRT_VERSION", "NVIDIA_TRITON_SERVER_VERSION",
        "TRITON_SERVER_VERSION", "DALI_VERSION", "NVIDIA_BUILD_ID")


def describe(name, raw):
    data = json.loads(raw)
    env = dict(item.split("=", 1) for item in data["env"] if "=" in item)
    return {"image": name, "digest": data["digest"], **{k.lower(): env[k] for k in KEYS if k in env}}


def tensorrt(info):
    return info.get("trt_version") or info.get("tensorrt_version")


gpus = []
for line in filter(None, gpu_csv.splitlines()):
    gpu_name, driver, memory = (part.strip() for part in line.split(","))
    gpus.append({"name": gpu_name, "driver": driver, "memory": memory})

server = describe(triton_image, triton_json)
builder = describe(trt_image, trt_json)
# dirty_files is None if git is unavailable (no .git directory); int otherwise
dirty_files = None if dirty == "unknown" else int(dirty)
report = {
    "gpus": gpus,
    "client_cpu": cpu,
    "commit": commit,
    "dirty_files": dirty_files,
    "server_image": server,
    "tensorrt_image": builder,
    "images_consistent": bool(server["digest"] and builder["digest"])
    and server.get("cuda_version") == builder.get("cuda_version")
    and (tensorrt(server) is None or tensorrt(server) == tensorrt(builder)),
}
text = json.dumps(report, indent=2)
if output:
    with open(output, "w", encoding="utf-8") as stream:
        stream.write(text + "\n")
print(text)
EOF
