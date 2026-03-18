#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# Helm release name used for all Helm operations
readonly HELM_RELEASE="triton-server"
readonly HELM_NAMESPACE="triton"

_running_on_minikube() {
  local ctx
  ctx="$(kubectl config current-context 2>/dev/null || true)"
  [[ "$ctx" == minikube* ]]
}

_minikube_image_present() {
  local image_ref
  image_ref="$1"
  minikube ssh -- "sudo crictl inspecti '$image_ref' >/dev/null 2>&1" >/dev/null 2>&1
}

_wait_for_minikube_image_load() {
  local image_ref elapsed
  image_ref="$1"
  elapsed=0

  log_info "A minikube image load is already in progress for ${image_ref}; waiting for it to finish..."
  while pgrep -f "minikube image load ${image_ref}" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 30 == 0 )); then
      log_info "Still waiting for minikube image load (${elapsed}s elapsed)"
    fi
  done
}

_preload_minikube_image_if_cached() {
  local image_ref
  image_ref="$1"

  if ! _running_on_minikube || ! command_exists minikube || ! command_exists docker; then
    return 0
  fi

  if ! docker image inspect "$image_ref" >/dev/null 2>&1; then
    log_info "Image ${image_ref} is not present in host Docker cache; minikube will pull it directly."
    return 0
  fi

  if _minikube_image_present "$image_ref"; then
    log_info "Image ${image_ref} is already present in minikube; skipping image load."
    return 0
  fi

  if pgrep -f "minikube image load ${image_ref}" >/dev/null 2>&1; then
    _wait_for_minikube_image_load "$image_ref"
    if _minikube_image_present "$image_ref"; then
      log_info "Image ${image_ref} is now present in minikube."
      return 0
    fi
  fi

  log_info "Loading cached image into minikube: ${image_ref}"
  run_cmd minikube image load "$image_ref"
}

_kubectl_image_ref() {
  local repo_root use_gpu deployment_file
  repo_root="$1"
  use_gpu="$2"

  if [[ "$use_gpu" == "true" ]]; then
    deployment_file="$repo_root/k8s/deployment-gpu.yaml"
  else
    deployment_file="$repo_root/k8s/deployment-cpu.yaml"
  fi

  sed -n 's/^[[:space:]]*image:[[:space:]]*//p' "$deployment_file" | head -n 1
}

_helm_image_ref() {
  local repo_root values_file repository tag
  repo_root="$1"
  values_file="$repo_root/helm/triton-server/values.yaml"

  repository="$(sed -n 's/^[[:space:]]*repository:[[:space:]]*//p' "$values_file" | head -n 1 | tr -d '"')"
  tag="$(sed -n 's/^[[:space:]]*tag:[[:space:]]*//p' "$values_file" | head -n 1 | tr -d '"')"

  if [[ -n "$repository" && -n "$tag" ]]; then
    printf '%s:%s\n' "$repository" "$tag"
  fi
}

triton_deployment_exists() {
  # Check Helm release first
  if command_exists helm && helm status "$HELM_RELEASE" -n "$HELM_NAMESPACE" >/dev/null 2>&1; then
    # Return the underlying deployment name
    if kubectl get deployment triton-server-gpu -n "$HELM_NAMESPACE" >/dev/null 2>&1; then
      echo "triton-server-gpu"
    else
      echo "triton-server-cpu"
    fi
    return 0
  fi

  # Fall back to raw kubectl check
  if kubectl get deployment triton-server-gpu -n "$HELM_NAMESPACE" >/dev/null 2>&1; then
    echo "triton-server-gpu"
    return 0
  fi

  if kubectl get deployment triton-server-cpu -n "$HELM_NAMESPACE" >/dev/null 2>&1; then
    echo "triton-server-cpu"
    return 0
  fi

  return 1
}

deploy_triton_stack() {
  local repo_root use_gpu
  repo_root="$1"
  use_gpu="$2"

  if command_exists helm; then
    _deploy_triton_helm "$repo_root" "$use_gpu"
  else
    _deploy_triton_kubectl "$repo_root" "$use_gpu"
  fi
}

_deploy_triton_helm() {
  local repo_root use_gpu gpu_flag image_ref
  repo_root="$1"
  use_gpu="$2"
  gpu_flag="false"
  [[ "$use_gpu" == "true" ]] && gpu_flag="true"

  local chart_dir="$repo_root/helm/triton-server"
  image_ref="$(_helm_image_ref "$repo_root")"
  if [[ -n "$image_ref" ]]; then
    _preload_minikube_image_if_cached "$image_ref"
  fi

  log_info "Deploying Triton via Helm (gpu.enabled=${gpu_flag})..."
  run_cmd helm upgrade --install "$HELM_RELEASE" "$chart_dir" \
    --namespace "$HELM_NAMESPACE" \
    --create-namespace \
    --set gpu.enabled="$gpu_flag" \
    --wait \
    --timeout 5m
}

_deploy_triton_kubectl() {
  local repo_root use_gpu deployment_file image_ref
  repo_root="$1"
  use_gpu="$2"

  log_info "Deploying Triton via kubectl (helm not found)..."
  run_cmd kubectl apply -f "$repo_root/k8s/namespace.yaml"
  run_cmd kubectl apply -f "$repo_root/k8s/configmap.yaml"
  run_cmd kubectl apply -f "$repo_root/k8s/persistent-volume.yaml"

  if [[ "$use_gpu" == "true" ]]; then
    deployment_file="$repo_root/k8s/deployment-gpu.yaml"
  else
    deployment_file="$repo_root/k8s/deployment-cpu.yaml"
  fi

  image_ref="$(_kubectl_image_ref "$repo_root" "$use_gpu")"
  if [[ -n "$image_ref" ]]; then
    _preload_minikube_image_if_cached "$image_ref"
  fi

  run_cmd kubectl apply -f "$deployment_file"
  run_cmd kubectl apply -f "$repo_root/k8s/service.yaml"
}

wait_for_triton_ready() {
  local deployment_name
  deployment_name="$1"

  if ! run_cmd kubectl rollout status "deployment/${deployment_name}" -n "$HELM_NAMESPACE" --timeout=300s; then
    log_warn "Triton rollout failed for ${deployment_name}; collecting diagnostics..."
    print_triton_failure_debug "$deployment_name"
    return 1
  fi
  run_cmd kubectl wait --for=condition=ready pod -l app=triton-server -n "$HELM_NAMESPACE" --timeout=300s
}

print_triton_status() {
  log_info "Triton deployments:"
  kubectl get deployments -n "$HELM_NAMESPACE" -l app=triton-server || true
  log_info "Triton pods:"
  kubectl get pods -n "$HELM_NAMESPACE" -l app=triton-server -o wide || true
  log_info "Triton services:"
  kubectl get svc -n "$HELM_NAMESPACE" | grep -E 'triton-service' || true
  print_triton_access_info
}

print_triton_failure_debug() {
  local deployment_name pod_name
  deployment_name="$1"

  log_info "Deployment details:"
  kubectl describe deployment "$deployment_name" -n "$HELM_NAMESPACE" || true

  pod_name="$(kubectl get pods -n "$HELM_NAMESPACE" -l app=triton-server -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "$pod_name" ]]; then
    log_info "Pod details: ${pod_name}"
    kubectl describe pod "$pod_name" -n "$HELM_NAMESPACE" || true
    log_info "Pod logs: ${pod_name}"
    kubectl logs "$pod_name" -n "$HELM_NAMESPACE" --tail=200 || true
  fi
}

print_triton_access_info() {
  local node_ip http_port grpc_port metrics_port

  http_port="$(kubectl get svc triton-service-nodeport -n "$HELM_NAMESPACE" -o jsonpath='{.spec.ports[?(@.name=="http")].nodePort}' 2>/dev/null || true)"
  grpc_port="$(kubectl get svc triton-service-nodeport -n "$HELM_NAMESPACE" -o jsonpath='{.spec.ports[?(@.name=="grpc")].nodePort}' 2>/dev/null || true)"
  metrics_port="$(kubectl get svc triton-service-nodeport -n "$HELM_NAMESPACE" -o jsonpath='{.spec.ports[?(@.name=="metrics")].nodePort}' 2>/dev/null || true)"

  if [[ -z "$http_port" || -z "$grpc_port" || -z "$metrics_port" ]]; then
    return 0
  fi

  node_ip="$(minikube ip 2>/dev/null || true)"
  if [[ -z "$node_ip" ]]; then
    node_ip="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || true)"
  fi
  if [[ -z "$node_ip" ]]; then
    node_ip="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || true)"
  fi

  if [[ -n "$node_ip" ]]; then
    log_info "External Triton endpoints:"
    printf '  HTTP:    http://%s:%s\n' "$node_ip" "$http_port"
    printf '  gRPC:    %s:%s\n' "$node_ip" "$grpc_port"
    printf '  Metrics: http://%s:%s/metrics\n' "$node_ip" "$metrics_port"
  fi
}
