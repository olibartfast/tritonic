# Triton Kubernetes Scripts

This folder contains modular Bash scripts used by `k8s/check_and_deploy_triton.sh`.

## Scripts Overview

- `lib.sh`: Shared logging/helpers.
- `kubectl.sh`: Installs `kubectl` if missing.
- `cluster.sh`: Validates Kubernetes cluster reachability and node readiness.
- `gpu.sh`: Checks NVIDIA GPU exposure and NVIDIA-related components.
- `triton.sh`: Checks/deploys Triton resources and waits for readiness.

## Main Entrypoint

Run from repository root:

```bash
./k8s/check_and_deploy_triton.sh
```

What it does in order:

1. Ensures `kubectl` exists (installs if missing).
2. Checks cluster status.
3. Checks NVIDIA GPU status in cluster.
4. Checks if Triton deployment exists.
5. Deploys Triton if not installed.
6. Prints final status summary.

## Prerequisites

- A Kubernetes cluster configured in your kubeconfig/context.
- Network access (only needed if `kubectl` must be downloaded).
- Permissions to apply manifests in `k8s/`.
- For GPU deployment: NVIDIA device plugin/operator correctly installed.

## Deployment Behavior

- If GPU resources are detected (`nvidia.com/gpu` > 0), script uses:
  - `k8s/deployment-gpu.yaml`
- Otherwise it falls back to:
  - `k8s/deployment-cpu.yaml`

In both cases it also applies:

- `k8s/namespace.yaml`
- `k8s/configmap.yaml`
- `k8s/persistent-volume.yaml`
- `k8s/service.yaml`

## Useful Commands

Check cluster:

```bash
kubectl cluster-info
kubectl get nodes -o wide
```

Check Triton resources:

```bash
kubectl get deploy,pods,svc -n triton
kubectl logs -n triton deployment/triton-server-cpu
kubectl logs -n triton deployment/triton-server-gpu
```

Check NVIDIA components:

```bash
kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
kubectl get ds -A | grep -Ei 'nvidia|device-plugin'
kubectl get pods -A | grep -Ei 'nvidia|gpu-operator|device-plugin'
```

## Notes

- If `kubectl` is installed to `~/.local/bin`, ensure it is in your `PATH`.
- If cluster is not reachable, the script exits early with a summary.
- Scripts are modular and can be sourced individually for testing.
