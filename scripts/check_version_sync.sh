#!/usr/bin/env bash
# scripts/check_version_sync.sh — guards the pins that cannot read versions.env.
#
# CMake and the shell scripts consume versions.env directly (cmake/versions.cmake,
# scripts/versions.sh). Some formats cannot: Dockerfile ARG defaults, k8s manifests,
# Helm charts, and the Colab notebook are plain data files with no include mechanism,
# and they restate the values; this script fails when the restatement drifts.
#
#   ./scripts/check_version_sync.sh
#
# Run by .github/workflows/ci.yml, and by hand after editing versions.env.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/versions.sh
source "${repo_root}/scripts/versions.sh"

failures=0

# expect <file> <what> <value> <expected-line-regex>
# Passes when at least one line of <file> matches the regex.
expect() {
    local file="$1" what="$2" value="$3" pattern="$4"
    if [[ ! -f "${repo_root}/${file}" ]]; then
        printf '  MISSING %-26s %-28s %s\n' "${what}" "${value}" "${file}"
        failures=$((failures + 1))
        return
    fi
    if grep -Eq -- "${pattern}" "${repo_root}/${file}"; then
        printf '  ok      %-26s %-28s %s\n' "${what}" "${value}" "${file}"
    else
        printf '  DRIFT   %-26s %-28s %s\n' "${what}" "${value}" "${file}"
        printf '          versions.env implies a line matching: %s\n' "${pattern}"
        printf '          found instead:\n'
        # Show the nearest lines so the fix is obvious without opening the file.
        grep -nE -- "$(printf '%s' "${pattern}" | sed -E 's/=.*/=/; s/\^//')" \
            "${repo_root}/${file}" | sed 's/^/          /' || true
        failures=$((failures + 1))
    fi
}

# expect_all <what> <value> <expected-line-regex> <file> [<file> ...]
# The same pin restated in several files (e.g. a Triton image tag in each k8s manifest).
expect_all() {
    local what="$1" value="$2" pattern="$3"
    shift 3
    local f
    for f in "$@"; do expect "$f" "$what" "$value" "$pattern"; done
}

# Escape a version so it is a literal inside an extended regex.
lit() { printf '%s' "$1" | sed -e 's/[][\.*^$+?(){}|/]/\\&/g'; }

echo "Checking pins that cannot read versions.env:"

# --- Docker ARG defaults and derived base images ------------------
expect docker/Dockerfile "NGC_CONTAINER_TAG" "${NGC_CONTAINER_TAG}" \
    "^ARG TRITON_VERSION=$(lit "${NGC_CONTAINER_TAG}")$"

# --- Kubernetes deployment image tags ------------------
expect_all "NGC_CONTAINER_TAG [k8s image]" "${NGC_CONTAINER_TAG}" \
    "image: nvcr.io/nvidia/tritonserver:$(lit "${NGC_CONTAINER_TAG}")-py3" \
    k8s/deployment-cpu.yaml k8s/deployment-gpu.yaml

# --- Helm Chart version and image tag ------------------
expect helm/triton-server/Chart.yaml "NGC_CONTAINER_TAG [appVersion]" "${NGC_CONTAINER_TAG}" \
    "^appVersion: \"$(lit "${NGC_CONTAINER_TAG}")\"$"
expect helm/triton-server/values.yaml "NGC_CONTAINER_TAG [helm tag]" "${NGC_CONTAINER_TAG}" \
    "tag: \"$(lit "${NGC_CONTAINER_TAG}")-py3\"$"

# --- CI cache keys ------------------
expect .github/workflows/ci.yml "TRITON_REPO_TAG [cache key]" "${TRITON_REPO_TAG}" \
    "key: triton-client-$(lit "${TRITON_REPO_TAG}")-"

# --- README "Versions at a Glance" table ---------------
# Prose that states pins: only this table, so it is checked row by row.
expect README.md "NGC_CONTAINER_TAG [README]" "${NGC_CONTAINER_TAG}" \
    "^\| \*\*Triton server and client libraries\*\* \| \*\*$(lit "${NGC_CONTAINER_TAG}")\*\* \(\`tritonserver:$(lit "${NGC_CONTAINER_TAG}")-py3\`, client \`$(lit "${TRITON_REPO_TAG}")\`\)"
expect README.md "NEURIPLO_TASKS_VERSION [README]" "${NEURIPLO_TASKS_VERSION}" \
    "^\| \*\*neuriplo-tasks\*\* \| \*\*$(lit "${NEURIPLO_TASKS_VERSION}")\*\*"
expect README.md "NGC_TENSORRT_VERSION [README]" "${NGC_TENSORRT_VERSION}" \
    "^\| TensorRT \| $(lit "${NGC_TENSORRT_VERSION}") \(in \`tensorrt:$(lit "${NGC_CONTAINER_TAG}")-py3\`\)"
expect README.md "NGC_DALI_VERSION [README]" "${NGC_DALI_VERSION}" \
    "^\| NVIDIA DALI \| $(lit "${NGC_DALI_VERSION}") \(in \`tritonserver:$(lit "${NGC_CONTAINER_TAG}")-py3\`\)"
expect README.md "NGC_CUDA_VERSION [README]" "${NGC_CUDA_VERSION}" \
    "^\| CUDA Toolkit \| $(lit "${NGC_CUDA_VERSION}") series"
expect README.md "NGC_CONTAINER_TAG [README quick start]" "${NGC_CONTAINER_TAG}" \
    "docker_triton_run\.sh \"\\\$PWD/model_repository\" $(lit "${NGC_CONTAINER_TAG}") gpu"
expect README.md "GTEST_VERSION [README]" "${GTEST_VERSION}" \
    "^\| GoogleTest \| $(lit "${GTEST_VERSION#v}") "

# --- Colab notebook version literals ------------------
# The notebook ships standalone to Colab and cannot read versions.env.
# These are Python string assignments in a JSON notebook file (with \n escape at line end).
# The notebook keeps its own variable names (TENSORRT_VERSION, DALI_VERSION); only the values are checked.
expect integration-tests/gpu-preprocess/colab_validation.ipynb "NGC_TENSORRT_VERSION [notebook]" "${NGC_TENSORRT_VERSION}" \
    "TENSORRT_VERSION, DALI_VERSION = \\\\\"$(lit "${NGC_TENSORRT_VERSION}")\\\\\", \\\\\"$(lit "${NGC_DALI_VERSION}")\\\\\".*"

echo
if [[ "${failures}" -ne 0 ]]; then
    echo "${failures} pin(s) drifted from versions.env." >&2
    echo "Update the file(s) above, or change versions.env if the bump is intended." >&2
    exit 1
fi
echo "version pins in sync"
