#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

nvidia_gpu_available() {
  local allocatable
  allocatable="$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"

  if [[ -z "${allocatable// /}" ]]; then
    log_warn "No allocatable nvidia.com/gpu resources found on nodes"
    return 1
  fi

  if ! grep -Eq '[1-9]' <<<"$allocatable"; then
    log_warn "nvidia.com/gpu present but value appears to be zero: $allocatable"
    return 1
  fi

  log_info "Node GPU allocatable values: $allocatable"

  if kubectl get pods -A 2>/dev/null | grep -Eqi 'nvidia|gpu-operator|device-plugin'; then
    log_info "NVIDIA-related pods detected in the cluster"
  else
    log_warn "No NVIDIA operator/device-plugin pods detected"
  fi

  return 0
}

print_gpu_debug_info() {
  log_info "NVIDIA daemonsets:"
  kubectl get ds -A | grep -Ei 'nvidia|device-plugin' || true
  log_info "NVIDIA pods:"
  kubectl get pods -A | grep -Ei 'nvidia|gpu-operator|device-plugin' || true
}
