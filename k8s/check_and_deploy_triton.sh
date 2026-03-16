#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=./scripts/lib.sh
source "$SCRIPT_DIR/scripts/lib.sh"
# shellcheck source=./scripts/kubectl.sh
source "$SCRIPT_DIR/scripts/kubectl.sh"
# shellcheck source=./scripts/cluster.sh
source "$SCRIPT_DIR/scripts/cluster.sh"
# shellcheck source=./scripts/gpu.sh
source "$SCRIPT_DIR/scripts/gpu.sh"
# shellcheck source=./scripts/triton.sh
source "$SCRIPT_DIR/scripts/triton.sh"

main() {
  local cluster_status gpu_status triton_status deployment_name use_gpu
  cluster_status="NOT_ALIVE"
  gpu_status="NOT_AVAILABLE"
  triton_status="NOT_INSTALLED"
  use_gpu="false"

  install_kubectl_if_missing

  if cluster_is_alive; then
    cluster_status="ALIVE"
    print_cluster_status
  else
    cat <<STATUS

Summary:
- Cluster: ${cluster_status}
- NVIDIA GPU: ${gpu_status}
- Triton: ${triton_status}

Cluster is not reachable. Fix kubeconfig/context and rerun.
STATUS
    exit 1
  fi

  if nvidia_gpu_available; then
    gpu_status="AVAILABLE"
    use_gpu="true"
  else
    print_gpu_debug_info
  fi

  if deployment_name="$(triton_deployment_exists)"; then
    log_info "Triton deployment already present: ${deployment_name}"
    wait_for_triton_ready "$deployment_name"
    triton_status="RUNNING"
  else
    log_info "No Triton deployment found. Deploying now."
    deploy_triton_stack "$REPO_ROOT" "$use_gpu"

    if [[ "$use_gpu" == "true" ]]; then
      deployment_name="triton-server-gpu"
    else
      deployment_name="triton-server-cpu"
    fi

    wait_for_triton_ready "$deployment_name"
    triton_status="DEPLOYED_AND_RUNNING"
  fi

  print_triton_status

  cat <<STATUS

Summary:
- Cluster: ${cluster_status}
- NVIDIA GPU: ${gpu_status}
- Triton: ${triton_status}
STATUS
}

main "$@"
