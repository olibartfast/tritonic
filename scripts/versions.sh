#!/usr/bin/env bash
# scripts/versions.sh — loads versions.env into the environment.
#
# Source it, never execute it:
#     source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"
#
# Values already set in the environment are left alone, so every pin stays
# overridable on the command line exactly as before this file existed:
#     TRITON_IMAGE=nvcr.io/nvidia/tritonserver:26.07-py3 integration-tests/gpu-preprocess/run_spikes.sh
#
# The repo root is resolved from this file's own location, so callers at any
# depth (scripts/, scripts/ci/, the root) work without passing anything.

TRITONIC_REPO_ROOT="${TRITONIC_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRITONIC_VERSIONS_ENV="${TRITONIC_VERSIONS_ENV:-${TRITONIC_REPO_ROOT}/versions.env}"

if [[ ! -f "${TRITONIC_VERSIONS_ENV}" ]]; then
    echo "versions.sh: ${TRITONIC_VERSIONS_ENV} not found" >&2
    return 1 2>/dev/null || exit 1
fi

while IFS= read -r _line || [[ -n "${_line}" ]]; do
    # Strip surrounding whitespace, then skip blanks and comments.
    _line="${_line#"${_line%%[![:space:]]*}"}"
    _line="${_line%"${_line##*[![:space:]]}"}"
    [[ -z "${_line}" || "${_line}" == \#* ]] && continue
    [[ "${_line}" == *=* ]] || continue

    _key="${_line%%=*}"
    _value="${_line#*=}"
    [[ "${_key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    # Strip one layer of surrounding quotes, as a shell would.
    if [[ "${_value}" == \"*\" || "${_value}" == \'*\' ]]; then
        _value="${_value:1:${#_value}-2}"
    fi

    # Environment wins. `+x` distinguishes "unset" from "set but empty".
    if [[ -z "${!_key+x}" ]]; then
        export "${_key}=${_value}"
    fi
done < "${TRITONIC_VERSIONS_ENV}"
unset _line _key _value

# --- Derived coordinates ------------------------------------------------------
# Container images derived from NGC_CONTAINER_TAG.
: "${TRITON_IMAGE:=nvcr.io/nvidia/tritonserver:${NGC_CONTAINER_TAG}-py3}"
: "${TRITON_SDK_IMAGE:=nvcr.io/nvidia/tritonserver:${NGC_CONTAINER_TAG}-py3-sdk}"
: "${TENSORRT_IMAGE:=nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3}"
export TRITON_IMAGE TRITON_SDK_IMAGE TENSORRT_IMAGE
