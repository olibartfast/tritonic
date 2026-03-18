#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=./lib.sh
source "$REPO_ROOT/k8s/scripts/lib.sh"
# shellcheck source=./kubectl.sh
source "$REPO_ROOT/k8s/scripts/kubectl.sh"
# shellcheck source=./cluster.sh
source "$REPO_ROOT/k8s/scripts/cluster.sh"
# shellcheck source=./gpu.sh
source "$REPO_ROOT/k8s/scripts/gpu.sh"
# shellcheck source=./triton.sh
source "$REPO_ROOT/k8s/scripts/triton.sh"

main() {
  local cluster_status gpu_status triton_status deployment_name use_gpu
  cluster_status="NOT_ALIVE"
  gpu_status="NOT_AVAILABLE"
  triton_status="NOT_INSTALLED"
  use_gpu="false"

  install_kubectl_if_missing

  if ensure_cluster_alive; then
    cluster_status="ALIVE"
    print_cluster_status
  else
    cat <<STATUS

Summary:
- Cluster: ${cluster_status}
- NVIDIA GPU: ${gpu_status}
- Triton: ${triton_status}

Cluster is not reachable and could not be started automatically.
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
    log_info "Reconciling Triton deployment with the current manifests."
  else
    log_info "No Triton deployment found. Deploying now."
  fi

  deploy_triton_stack "$REPO_ROOT" "$use_gpu"

  # Helm path: --wait already handled readiness; resolve deployment name for status display
  if [[ "$use_gpu" == "true" ]]; then
    deployment_name="triton-server-gpu"
  else
    deployment_name="triton-server-cpu"
  fi

  if ! command_exists helm; then
    wait_for_triton_ready "$deployment_name"
  fi

  triton_status="DEPLOYED_AND_RUNNING"

  print_triton_status

  cat <<STATUS

Summary:
- Cluster: ${cluster_status}
- NVIDIA GPU: ${gpu_status}
- Triton: ${triton_status}
STATUS
}

main "$@"
