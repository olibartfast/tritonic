# Triton Kubernetes Scripts

This folder contains modular Bash scripts used by `k8s/check_and_deploy_triton.sh`.

## Scripts Overview

- `lib.sh`: Shared logging/helpers.
- `kubectl.sh`: Installs `kubectl` if missing.
- `cluster.sh`: Validates Kubernetes cluster reachability and node readiness. Automatically starts or installs a local cluster (minikube → kind → k3s) if none is reachable.
- `gpu.sh`: Checks NVIDIA GPU exposure and NVIDIA-related components.
- `triton.sh`: Checks/deploys Triton resources and waits for readiness. Uses Helm (`helm/triton-server/`) when available; falls back to raw `kubectl apply` on `k8s/*.yaml`.

## Main Entrypoint

Run from repository root:

```bash
./k8s/check_and_deploy_triton.sh
```

What it does in order:

1. Ensures `kubectl` exists (installs if missing).
2. Checks cluster reachability; if unreachable, attempts to start or install a local cluster automatically (minikube → kind → k3s; installs `kind` if none present and Docker is available).
3. Checks NVIDIA GPU status in cluster.
4. Checks if Triton deployment exists.
5. Deploys Triton if not installed.
6. Prints final status summary.

## Prerequisites

- Network access (required for downloading `kubectl`, `kind`, or cluster images on first run).
- Permissions to apply manifests in `k8s/`.
- For GPU deployment: NVIDIA device plugin/operator correctly installed.

**A pre-configured Kubernetes cluster is optional.** If no cluster is reachable, the script will attempt to start one automatically:

| Tool available | Action taken |
|---|---|
| `minikube` | `minikube start` |
| `kind` | `kind create cluster` (or kubeconfig export if cluster exists) |
| `k3s` | `sudo k3s server` (background, waits up to 60 s) |
| none + Docker | Downloads and installs the latest `kind` binary, then creates a cluster |
| none + no Docker | Exits with an error — install one of the above manually |

## Deployment Behavior

GPU resources detected (`nvidia.com/gpu` > 0) → GPU variant; otherwise CPU variant.

### Helm (preferred — used automatically when `helm` is in PATH)

```bash
helm upgrade --install triton-server helm/triton-server \
  --namespace triton --create-namespace \
  --set gpu.enabled=true   # or false
```

Chart location: `helm/triton-server/`. Key `values.yaml` overrides:

| Value | Default | Description |
|---|---|---|
| `gpu.enabled` | `false` | Enable GPU variant |
| `image.tag` | `25.12-py3` | Triton image tag |
| `storage.type` | `pvc` | `pvc` or `hostPath` |
| `storage.hostPath.path` | `/tmp/triton-models` | Model repo path (hostPath) |
| `autoscaling.enabled` | `false` | Enable HPA |
| `ingress.enabled` | `false` | Enable Ingress |

### kubectl fallback (used when `helm` is not installed)

Applies in order:
- `k8s/namespace.yaml`
- `k8s/configmap.yaml`
- `k8s/persistent-volume.yaml`
- `k8s/deployment-gpu.yaml` or `k8s/deployment-cpu.yaml`
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

- If `kubectl` or `kind` is installed to `~/.local/bin`, ensure it is in your `PATH` (the script exports it for the current session automatically).
- If the cluster cannot be started automatically (no Docker, no supported tool), the script exits with a clear error.
- Scripts are modular and can be sourced individually for testing.
