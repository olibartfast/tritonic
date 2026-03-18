#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

cluster_is_alive() {
  if ! kubectl cluster-info >/dev/null 2>&1; then
    log_warn "kubectl cannot reach a cluster (cluster-info failed)"
    return 1
  fi

  local ready_nodes
  ready_nodes="$(kubectl get nodes --no-headers 2>/dev/null | awk '$2 == "Ready" {count++} END {print count+0}')"

  if [[ "$ready_nodes" -lt 1 ]]; then
    log_warn "No Ready nodes found in the cluster"
    return 1
  fi

  log_info "Cluster is alive with ${ready_nodes} Ready node(s)"
  return 0
}

# Returns the local cluster tool driving the current context: "minikube", "kind", or "other".
_cluster_driver() {
  local ctx
  ctx="$(kubectl config current-context 2>/dev/null || true)"
  if [[ "$ctx" == minikube* ]]; then
    echo "minikube"
  elif [[ "$ctx" == kind-* ]]; then
    echo "kind"
  else
    echo "other"
  fi
}

# Try to start a local cluster using whichever tool is available.
# When the host has an NVIDIA GPU, minikube is strongly preferred because it
# passes through GPU devices natively (--gpus all); kind requires complex manual
# containerd configuration that is unreliable across versions.
ensure_cluster_alive() {
  # If cluster is already alive, check whether we need to migrate a kind cluster
  # to minikube so that the GPU becomes usable.
  if cluster_is_alive; then
    if _host_has_nvidia_gpu && [[ "$(_cluster_driver)" == "kind" ]]; then
      log_info "Running cluster is kind but host has GPU — migrating to minikube for native GPU support..."
      kind delete cluster --name kind 2>/dev/null || true
      # Ensure minikube is installed before attempting to start
      if command_exists minikube || _install_minikube; then
        _start_minikube_gpu && return 0
      fi
      log_warn "minikube migration failed; falling through to normal startup"
    else
      return 0
    fi
  fi

  log_info "Attempting to start a local cluster automatically..."

  # --- GPU host: prefer minikube ---
  if _host_has_nvidia_gpu; then
    if command_exists minikube || _install_minikube; then
      _start_minikube_gpu && return 0
    fi
    log_warn "Could not start minikube with GPU support; falling back to kind (no GPU)"
  fi

  # --- No-GPU / minikube-failed path ---
  if command_exists minikube; then
    log_info "Starting minikube (CPU)..."
    minikube start --driver=docker || { log_warn "minikube start failed"; }
    _wait_nodes_ready 120
    cluster_is_alive && return 0
  fi

  if command_exists kind; then
    _start_kind_cluster && return 0
  fi

  if command_exists k3s; then
    log_info "Found k3s — starting server in background"
    sudo k3s server --disable=traefik &>/tmp/k3s-server.log &
    log_info "k3s started; waiting up to 60 s for nodes to become Ready"
    local elapsed=0
    while [[ $elapsed -lt 60 ]]; do
      export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
      cluster_is_alive && return 0
      sleep 5; elapsed=$((elapsed + 5))
    done
    log_warn "k3s did not become ready in time"
  fi

  # Last resort: install kind (requires Docker)
  log_info "No local cluster tool found. Attempting to install kind..."
  if _install_kind; then
    _start_kind_cluster && return 0
  fi

  log_error "Could not bring up a reachable cluster automatically."
  log_error "Install minikube (recommended for GPU) or kind/k3s, then rerun."
  return 1
}

# Start minikube with GPU passthrough.
_start_minikube_gpu() {
  _configure_nvidia_runtime  # ensure Docker uses nvidia runtime
  log_info "Starting minikube with --gpus all (Docker driver)..."
  minikube start --driver=docker --gpus all || {
    log_warn "minikube start --gpus all failed"
    return 1
  }
  _wait_nodes_ready 120
  cluster_is_alive
}

# Create (or reuse) a kind cluster.
_start_kind_cluster() {
  log_info "Setting up kind cluster..."
  if ! kind get clusters 2>/dev/null | grep -q '^kind$'; then
    kind create cluster --name kind || { log_warn "kind create cluster failed"; return 1; }
  else
    log_info "kind cluster already exists; exporting kubeconfig"
    kind export kubeconfig --name kind || true
  fi
  _wait_nodes_ready 120
  cluster_is_alive
}

_install_minikube() {
  if ! command_exists docker; then
    log_warn "Docker is not available — cannot install minikube"
    return 1
  fi

  local os arch
  os="$(uname | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$arch" in
    x86_64)       arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) log_warn "Unsupported architecture for minikube: $arch"; return 1 ;;
  esac

  local url="https://storage.googleapis.com/minikube/releases/latest/minikube-${os}-${arch}"
  log_info "Downloading minikube from ${url}..."
  local bin
  bin="$(mktemp)"
  if command_exists curl; then
    curl -fsSL "$url" -o "$bin"
  elif command_exists wget; then
    wget -qO "$bin" "$url"
  else
    log_warn "Neither curl nor wget available"; return 1
  fi
  chmod +x "$bin"

  if [[ -w /usr/local/bin ]]; then
    mv "$bin" /usr/local/bin/minikube
    log_info "minikube installed at /usr/local/bin/minikube"
  else
    mkdir -p "$HOME/.local/bin"
    mv "$bin" "$HOME/.local/bin/minikube"
    export PATH="$HOME/.local/bin:$PATH"
    log_info "minikube installed at $HOME/.local/bin/minikube"
  fi
  command_exists minikube
}

_install_kind() {
  if ! command_exists docker; then
    log_warn "Docker is not available — cannot install kind"
    return 1
  fi

  local os arch version kind_bin
  os="$(uname | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$arch" in
    x86_64)       arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) log_warn "Unsupported architecture for kind: $arch"; return 1 ;;
  esac

  if command_exists curl; then
    version="$(curl -fsSL https://api.github.com/repos/kubernetes-sigs/kind/releases/latest \
      | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
  elif command_exists wget; then
    version="$(wget -qO- https://api.github.com/repos/kubernetes-sigs/kind/releases/latest \
      | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
  else
    log_warn "Neither curl nor wget available to download kind"; return 1
  fi

  [[ -z "$version" ]] && { log_warn "Could not determine latest kind version"; return 1; }

  local url="https://github.com/kubernetes-sigs/kind/releases/download/${version}/kind-${os}-${arch}"
  log_info "Downloading kind ${version} from ${url}"
  kind_bin="$(mktemp)"
  if command_exists curl; then
    curl -fsSL "$url" -o "$kind_bin"
  else
    wget -qO "$kind_bin" "$url"
  fi
  chmod +x "$kind_bin"

  if [[ -w /usr/local/bin ]]; then
    mv "$kind_bin" /usr/local/bin/kind
    log_info "kind installed at /usr/local/bin/kind"
  else
    mkdir -p "$HOME/.local/bin"
    mv "$kind_bin" "$HOME/.local/bin/kind"
    export PATH="$HOME/.local/bin:$PATH"
    log_info "kind installed at $HOME/.local/bin/kind"
  fi
  command_exists kind
}

# Wait up to <timeout> seconds for at least one node to reach Ready status.
_wait_nodes_ready() {
  local timeout="${1:-120}"
  log_info "Waiting up to ${timeout}s for nodes to become Ready..."
  kubectl wait --for=condition=Ready node --all --timeout="${timeout}s" 2>/dev/null && return 0
  log_warn "Timed out waiting for nodes to become Ready"
  return 1
}

print_cluster_status() {
  log_info "Cluster summary:"
  kubectl get nodes -o wide
}
