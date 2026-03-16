#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

cluster_is_alive() {
  if ! kubectl cluster-info >/dev/null 2>&1; then
    log_warn "kubectl cannot reach a cluster (cluster-info failed)"
    return 1
  fi

  local ready_nodes
  ready_nodes="$(kubectl get nodes --no-headers 2>/dev/null | awk '$2 ~ /Ready/ {count++} END {print count+0}')"

  if [[ "$ready_nodes" -lt 1 ]]; then
    log_warn "No Ready nodes found in the cluster"
    return 1
  fi

  log_info "Cluster is alive with ${ready_nodes} Ready node(s)"
  return 0
}

print_cluster_status() {
  log_info "Cluster summary:"
  kubectl get nodes -o wide
}
