#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

_host_has_nvidia_gpu() {
  command_exists nvidia-smi && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1
}

# Install nvidia-container-toolkit (if missing) and set nvidia as Docker's default runtime.
# Idempotent: skips if already configured.
_configure_nvidia_runtime() {
  # Already configured if daemon.json already lists nvidia as default runtime
  if docker info 2>/dev/null | grep -q 'Default Runtime: nvidia'; then
    log_info "NVIDIA is already the Docker default runtime — skipping reconfiguration"
    return 0
  fi
  if ! command_exists nvidia-ctk; then
    log_info "nvidia-container-toolkit not found — installing..."
    if command_exists curl; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    elif command_exists wget; then
      wget -qO- https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      wget -qO- https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    else
      log_warn "Neither curl nor wget available — cannot install nvidia-container-toolkit"
      return 1
    fi
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
  fi

  log_info "Configuring NVIDIA runtime as Docker default..."
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl restart docker
  log_info "Docker restarted with NVIDIA runtime as default"
}

# Install the NVIDIA k8s device plugin and wait for nvidia.com/gpu to be allocatable.
_install_nvidia_device_plugin() {
  local plugin_url="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml"
  log_info "Installing NVIDIA device plugin into the cluster..."
  kubectl apply -f "$plugin_url" >/dev/null 2>&1 || { log_warn "Failed to apply NVIDIA device plugin manifest"; return 1; }

  log_info "Waiting up to 120s for nvidia.com/gpu to become allocatable..."
  local elapsed=0
  while [[ $elapsed -lt 120 ]]; do
    local allocatable
    allocatable="$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"
    if grep -Eq '[1-9]' <<<"$allocatable"; then
      log_info "nvidia.com/gpu is now allocatable: $allocatable"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done

  log_warn "nvidia.com/gpu not allocatable after device plugin install"
  return 1
}

# Configure the cluster to expose the host GPU.
# Strategy by cluster driver:
#   minikube — GPU passthrough is handled by --gpus all at start time; only the
#              device plugin needs to be installed.
#   kind     — kind does not support native GPU passthrough; the cluster is deleted
#              and recreated via minikube (preferred) or the user is warned.
#   other    — install device plugin and hope the runtime is already set up.
configure_cluster_for_gpu() {
  local driver
  driver="$(_cluster_driver 2>/dev/null || echo other)"

  case "$driver" in
    minikube)
      log_info "minikube cluster detected — installing NVIDIA device plugin..."
      _install_nvidia_device_plugin
      return
      ;;
    kind)
      log_info "kind cluster detected — kind does not support native GPU passthrough."
      log_info "Migrating to minikube for proper GPU support..."
      kind delete cluster --name kind 2>/dev/null || true
      if command_exists minikube || _install_minikube; then
        _start_minikube_gpu && _install_nvidia_device_plugin && return 0
      fi
      log_warn "Could not migrate to minikube; GPU will not be available"
      return 1
      ;;
    *)
      log_info "Unknown cluster driver — attempting device plugin install..."
      _install_nvidia_device_plugin
      ;;
  esac
}

nvidia_gpu_available() {
  local allocatable
  allocatable="$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"

  if [[ -n "${allocatable// /}" ]] && grep -Eq '[1-9]' <<<"$allocatable"; then
    log_info "Node GPU allocatable values: $allocatable"
    return 0
  fi

  if _host_has_nvidia_gpu; then
    log_info "Host has NVIDIA GPU but cluster does not expose it — configuring now..."
    configure_cluster_for_gpu || true
    # Re-check after configuration
    allocatable="$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"
    if [[ -n "${allocatable// /}" ]] && grep -Eq '[1-9]' <<<"$allocatable"; then
      log_info "Node GPU allocatable values: $allocatable"
      return 0
    fi
    log_warn "GPU not allocatable after cluster configuration"
  else
    log_warn "No allocatable nvidia.com/gpu resources found and no host GPU detected"
  fi

  return 1
}

print_gpu_debug_info() {
  log_info "NVIDIA daemonsets:"
  kubectl get ds -A | grep -Ei 'nvidia|device-plugin' || true
  log_info "NVIDIA pods:"
  kubectl get pods -A | grep -Ei 'nvidia|gpu-operator|device-plugin' || true
}
