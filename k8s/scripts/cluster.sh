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

# Try to start a local cluster using whichever tool is available.
# Returns 0 if a cluster becomes reachable, 1 otherwise.
ensure_cluster_alive() {
  if cluster_is_alive; then
    return 0
  fi

  log_info "Attempting to start a local cluster automatically..."

  if command_exists minikube; then
    log_info "Found minikube — running 'minikube start'"
    minikube start || { log_warn "minikube start failed"; }
    _wait_nodes_ready 120
    if cluster_is_alive; then
      return 0
    fi
  fi

  if command_exists kind; then
    log_info "Found kind — preparing cluster"
    # If host has a GPU, configure the nvidia runtime before cluster creation so
    # kind node containers inherit GPU device access. Delete any existing cluster
    # first because it was created without the nvidia runtime.
    if _host_has_nvidia_gpu; then
      _configure_nvidia_runtime
      if kind get clusters 2>/dev/null | grep -q '^kind$'; then
        log_info "Deleting existing kind cluster to recreate with GPU support..."
        kind delete cluster --name kind
      fi
    fi

    if ! kind get clusters 2>/dev/null | grep -q '^kind$'; then
      kind create cluster --name kind || { log_warn "kind create cluster failed"; }
    else
      log_info "kind cluster already exists; merging kubeconfig"
      kind export kubeconfig --name kind || true
    fi
    _wait_nodes_ready 120
    if cluster_is_alive; then
      return 0
    fi
  fi

  if command_exists k3s; then
    log_info "Found k3s — starting server in background"
    sudo k3s server --disable=traefik &>/tmp/k3s-server.log &
    local k3s_pid=$!
    log_info "k3s started (pid ${k3s_pid}); waiting up to 60 s for nodes to become Ready"
    local elapsed=0
    while [[ $elapsed -lt 60 ]]; do
      export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
      if cluster_is_alive; then
        return 0
      fi
      sleep 5
      elapsed=$((elapsed + 5))
    done
    log_warn "k3s did not become ready in time"
  fi

  # No tool found — try to install kind (requires Docker)
  log_info "No local cluster tool found. Attempting to install kind..."
  if _install_kind; then
    log_info "kind installed."
    if _host_has_nvidia_gpu; then
      _configure_nvidia_runtime
    fi
    log_info "Creating cluster..."
    kind create cluster --name kind || { log_warn "kind create cluster failed"; return 1; }
    _wait_nodes_ready 120
    if cluster_is_alive; then
      return 0
    fi
  fi

  log_error "Could not bring up a reachable cluster automatically."
  log_error "Install minikube, kind, or k3s (or fix your kubeconfig) and rerun."
  return 1
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
    x86_64)  arch="amd64" ;;
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
    log_warn "Neither curl nor wget available to download kind"
    return 1
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
