#!/usr/bin/env bash

if [[ -n "${TRITON_K8S_LIB_LOADED:-}" ]]; then
  return 0
fi
TRITON_K8S_LIB_LOADED=1

set -o errexit
set -o nounset
set -o pipefail

readonly LOG_PREFIX="[triton-k8s]"

log_info() {
  printf '%s [INFO] %s\n' "$LOG_PREFIX" "$*"
}

log_warn() {
  printf '%s [WARN] %s\n' "$LOG_PREFIX" "$*" >&2
}

log_error() {
  printf '%s [ERROR] %s\n' "$LOG_PREFIX" "$*" >&2
}

fail() {
  log_error "$*"
  return 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_cmd() {
  log_info "Running: $*"
  "$@"
}

script_dir() {
  local source_path
  source_path="${BASH_SOURCE[1]}"
  cd "$(dirname "$source_path")" && pwd
}
