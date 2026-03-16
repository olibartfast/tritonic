#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

triton_deployment_exists() {
  if kubectl get deployment triton-server-gpu -n triton >/dev/null 2>&1; then
    echo "triton-server-gpu"
    return 0
  fi

  if kubectl get deployment triton-server-cpu -n triton >/dev/null 2>&1; then
    echo "triton-server-cpu"
    return 0
  fi

  return 1
}

deploy_triton_stack() {
  local repo_root use_gpu deployment_file
  repo_root="$1"
  use_gpu="$2"

  run_cmd kubectl apply -f "$repo_root/k8s/namespace.yaml"
  run_cmd kubectl apply -f "$repo_root/k8s/configmap.yaml"
  run_cmd kubectl apply -f "$repo_root/k8s/persistent-volume.yaml"

  if [[ "$use_gpu" == "true" ]]; then
    deployment_file="$repo_root/k8s/deployment-gpu.yaml"
  else
    deployment_file="$repo_root/k8s/deployment-cpu.yaml"
  fi

  run_cmd kubectl apply -f "$deployment_file"
  run_cmd kubectl apply -f "$repo_root/k8s/service.yaml"
}

wait_for_triton_ready() {
  local deployment_name
  deployment_name="$1"

  run_cmd kubectl rollout status "deployment/${deployment_name}" -n triton --timeout=300s
  run_cmd kubectl wait --for=condition=ready pod -l app=triton-server -n triton --timeout=300s
}

print_triton_status() {
  log_info "Triton deployments:"
  kubectl get deployments -n triton -l app=triton-server || true
  log_info "Triton pods:"
  kubectl get pods -n triton -l app=triton-server -o wide || true
  log_info "Triton services:"
  kubectl get svc -n triton | grep -E 'triton-service' || true
}
