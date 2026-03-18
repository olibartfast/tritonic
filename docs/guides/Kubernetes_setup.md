# Kubernetes Deployment Guide for Triton Inference Server

This guide covers installing Kubernetes and deploying NVIDIA Triton Inference Server for the TritonIC project.

## Automated Setup and Health Check (Recommended)

Use the modular deployment/check scripts:

```bash
./k8s/scripts/check_and_deploy_triton.sh
```

The script will:
1. Install `kubectl` if it is missing.
2. Check that the cluster is reachable and has Ready nodes. If not reachable, automatically starts or installs a local cluster (see below).
3. Check NVIDIA GPU availability in the cluster.
4. Check if Triton is already deployed.
5. Reconcile the existing Triton deployment to the current manifests or Helm chart.
6. Print external `NodePort` endpoints for HTTP, gRPC, and metrics.

Module documentation:
- `k8s/scripts/README.md`

### Automatic Cluster Recovery

If no Kubernetes cluster is reachable, the script tries the following in order:

| Condition | Action |
|---|---|
| `minikube` installed | `minikube start` |
| `kind` installed | `kind create cluster` (or re-exports kubeconfig if cluster exists) |
| `k3s` installed | starts `k3s server` in background, waits up to 60 s |
| None installed, Docker available | downloads and installs the latest `kind` binary, then creates a cluster |
| None installed, no Docker | exits with an error |

## Prerequisites

- Ubuntu/Debian Linux system
- Docker installed (required for automatic `kind` cluster setup if no other tool is present)
- At least 2 CPU cores and 4 GB RAM
- For GPU workloads: NVIDIA GPU with drivers installed

## Kubernetes Installation

### 1. Install kubectl (Manual Alternative)

```bash
# Download kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install kubectl
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify installation
kubectl version --client
```

### 2. Install and Start Minikube (Local Development)

```bash
# Install Minikube
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube
sudo mv minikube /usr/local/bin/

# Start Minikube with Docker driver
minikube start --driver=docker --memory=4096 --cpus=2

# For GPU passthrough on supported hosts
minikube start --driver=docker --gpus all

# Verify cluster is running
kubectl get nodes
```

### 3. Alternative: Kind (Kubernetes in Docker)

```bash
# Install Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Create cluster
kind create cluster --name triton-cluster

# Verify cluster
kubectl cluster-info --context kind-triton-cluster
```

## GPU Support Setup

For GPU workloads, the deployment script installs the NVIDIA device plugin when the host GPU is detected but `nvidia.com/gpu` is not yet exposed in the cluster. You can still install the GPU Operator manually if your environment requires it, but the local minikube path is designed around the device plugin plus Docker GPU passthrough.

## Triton Server Deployment

### 1. Prepare Model Repository

Create your model repository directory and copy models:

```bash
# Create model repository
sudo mkdir -p /tmp/triton-models

# Copy your models (example structure)
# /tmp/triton-models/
# ├── yolov8m/
# │   ├── config.pbtxt
# │   └── 1/
# │       └── model.onnx
# └── resnet50/
#     ├── config.pbtxt
#     └── 1/
#         └── model.savedmodel/

# Update the path in k8s/persistent-volume.yaml if different
```

### 2. Deploy to Kubernetes (Manual Alternative)

Apply the Kubernetes manifests:

```bash
# Create namespace and basic resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volume.yaml

# Deploy CPU version (recommended for testing)
kubectl apply -f k8s/deployment-cpu.yaml

# OR deploy GPU version (requires cluster GPU support)
# kubectl apply -f k8s/deployment-gpu.yaml

# Create services
kubectl apply -f k8s/service.yaml

# Optional: Apply ingress (requires nginx ingress controller)
# kubectl apply -f k8s/ingress.yaml

# Optional: Enable auto-scaling
# kubectl apply -f k8s/hpa.yaml
```

Notes:
- The Kubernetes manifests use `nvcr.io/nvidia/tritonserver:25.12-py3`.
- GPU deployments use `strategy: Recreate` so single-node clusters with one allocatable GPU can update cleanly.
- On minikube, if the Triton image already exists in host Docker, `k8s/scripts/check_and_deploy_triton.sh` loads it into the node before applying the deployment.

### 3. Verify Deployment

Check deployment status:
```bash
kubectl get pods -n triton -l app=triton-server
kubectl get services -n triton
kubectl logs -n triton deployment/triton-server-cpu
```

Wait for the pod to be ready:
```bash
kubectl wait --for=condition=ready pod -l app=triton-server -n triton --timeout=300s
```

### 4. Test the Deployment

Use the default `NodePort` service for external access:
```bash
export TRITON_IP=$(minikube ip)
curl "http://${TRITON_IP}:30800/v2/health/ready"
curl "http://${TRITON_IP}:30800/v2/models"
curl "http://${TRITON_IP}:30802/metrics"
```

Port-forwarding remains useful for ad hoc local testing:
```bash
kubectl port-forward -n triton svc/triton-service 8000:8000 8001:8001 8002:8002
```

Test the server:
```bash
# Health check
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Get model metadata (replace 'yolov8m' with your model name)
curl http://localhost:8000/v2/models/yolov8m

# View metrics
curl http://localhost:8002/metrics
```

## Accessing the Service

### Internal Access (within cluster)
- HTTP API: `http://triton-service.triton.svc.cluster.local:8000`
- GRPC API: `triton-service.triton.svc.cluster.local:8001`
- Metrics: `http://triton-service.triton.svc.cluster.local:8002/metrics`

### External Access Options

#### 1. NodePort (Default)
```bash
minikube ip

# Access via NodePort
# HTTP:   http://<node-ip>:30800
# gRPC:   <node-ip>:30801
# Metrics: http://<node-ip>:30802/metrics
```

#### 2. Port Forwarding (Testing)
```bash
kubectl port-forward -n triton svc/triton-service 8000:8000 8001:8001
```

#### 3. Ingress (Production)
```bash
# Install nginx ingress controller first
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml

# Add to /etc/hosts: <ingress-ip> triton.local
# Access: http://triton.local
```

## Running Your TritonIC Client

Update your client configuration to use the Kubernetes service:

```bash
# Example with port-forwarding
./tritonic \
    --source=/path/to/image.jpg \
    --model_type=yolov8 \
    --model=yolov8m \
    --labelsFile=labels/coco.txt \
    --protocol=http \
    --serverAddress=localhost \
    --port=8000

# Example with NodePort
./tritonic \
    --source=/path/to/image.jpg \
    --model_type=yolov8 \
    --model=yolov8m \
    --labelsFile=labels/coco.txt \
    --protocol=http \
    --serverAddress=$(minikube ip) \
    --port=30800
```

## Scaling and Management

### Manual Scaling
```bash
# Scale CPU deployment
kubectl scale deployment triton-server-cpu --replicas=3 -n triton

# Scale GPU deployment
kubectl scale deployment triton-server-gpu --replicas=2 -n triton
```

### Auto-scaling
```bash
# Enable horizontal pod autoscaler
kubectl apply -f k8s/hpa.yaml

# Check HPA status
kubectl get hpa -n triton
```

### Rolling Updates
```bash
# Update image version
kubectl set image deployment/triton-server-cpu triton-server=nvcr.io/nvidia/tritonserver:25.07-py3 -n triton

# Check rollout status
kubectl rollout status deployment/triton-server-cpu -n triton
```

## Monitoring and Troubleshooting

### View Logs
```bash
# Current logs
kubectl logs -n triton deployment/triton-server-cpu

# Follow logs
kubectl logs -n triton -f deployment/triton-server-cpu

# Previous pod logs (if pod crashed)
kubectl logs -n triton deployment/triton-server-cpu --previous
```

### Debug Pod Issues
```bash
# Describe pod for events
kubectl describe pod -n triton -l app=triton-server

# Get into pod for debugging
kubectl exec -it -n triton deployment/triton-server-cpu -- bash

# Check model repository inside pod
kubectl exec -it -n triton deployment/triton-server-cpu -- ls -la /models
```

### Resource Usage
```bash
# Check resource usage
kubectl top pods -n triton
kubectl top nodes
```

## Cleanup

Remove all resources:
```bash
kubectl delete namespace triton
```

Or remove individual components:
```bash
kubectl delete -f k8s/
```
