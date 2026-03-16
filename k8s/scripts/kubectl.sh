#!/usr/bin/env bash

# shellcheck source=./lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

install_kubectl_if_missing() {
  if command_exists kubectl; then
    log_info "kubectl is already installed: $(kubectl version --client --output=yaml 2>/dev/null | awk '/gitVersion/ {print $2; exit}' || echo 'unknown')"
    return 0
  fi

  local os arch version tmp_dir kubectl_bin
  os="$(uname | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "$arch" in
    x86_64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) fail "Unsupported architecture: $arch" || return 1 ;;
  esac

  if command_exists curl; then
    version="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"
  elif command_exists wget; then
    version="$(wget -qO- https://dl.k8s.io/release/stable.txt)"
  else
    fail "Neither curl nor wget is available to download kubectl" || return 1
  fi

  tmp_dir="$(mktemp -d)"
  kubectl_bin="$tmp_dir/kubectl"

  log_info "Downloading kubectl ${version} for ${os}/${arch}"
  if command_exists curl; then
    curl -fsSL "https://dl.k8s.io/release/${version}/bin/${os}/${arch}/kubectl" -o "$kubectl_bin"
  else
    wget -qO "$kubectl_bin" "https://dl.k8s.io/release/${version}/bin/${os}/${arch}/kubectl"
  fi

  chmod +x "$kubectl_bin"

  if [[ -w /usr/local/bin ]]; then
    mv "$kubectl_bin" /usr/local/bin/kubectl
    log_info "kubectl installed at /usr/local/bin/kubectl"
  else
    mkdir -p "$HOME/.local/bin"
    mv "$kubectl_bin" "$HOME/.local/bin/kubectl"
    export PATH="$HOME/.local/bin:$PATH"
    log_warn "Installed kubectl in $HOME/.local/bin. Ensure this is in your PATH."
  fi

  rm -rf "$tmp_dir"
  kubectl version --client >/dev/null
  log_info "kubectl installation verified"
}
