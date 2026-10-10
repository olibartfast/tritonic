# Deployment

This page covers model deployment, Triton server setup, Docker, and Kubernetes. See the [project README](../README.md) for quick-start instructions.

## Deploying Models

To deploy models, set up a model repository following the [Triton Model Repository schema](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md). The `config.pbtxt` file is optional unless you're using the OpenVino backend, implementing an Ensemble pipeline, or passing custom inference parameters.

### Model Repository Structure
```
<model_repository>/
    <model_name>/
        config.pbtxt
        <model_version>/
            <model_binary>
```

### Starting Triton Server

Use the provided script for easy setup:
```bash
# Start Triton server with GPU support
./docker/scripts/docker_triton_run.sh /path/to/model_repository 26.08 gpu

# Start with CPU only
./docker/scripts/docker_triton_run.sh /path/to/model_repository 26.08 cpu
```

For client shared memory (`--shared_memory_type`, `--preprocess_device=cuda`) start the server with:
```bash
ALLOW_CLIENT_SHM=true ./docker/scripts/docker_triton_run.sh /path/to/model_repository 26.08 gpu
```

Or manually with Docker:
```bash
docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /full/path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver \
  --model-repository=/models
```

*Omit the `--gpus` flag if using the CPU version.*

## Docker Support
For detailed instructions on installing Docker and the NVIDIA Container Toolkit, refer to the [Docker Setup Document](guides/Docker_setup.md).

### Build

```bash
docker build --rm -t tritonic -f docker/Dockerfile .
```

### Run Container
```bash
docker run --rm \
  -v /path/to/host/data:/app/data \
  tritonic \
  --network host \
  --source=<path_to_source_on_container> \
  --model_type=<model_type> \
  --model=<model_name_folder_on_triton> \
  --labelsFile=<path_to_labels_on_container> \
  --protocol=<http or grpc> \
  --serverAddress=<triton-ip> \
  --port=<8000 for http, 8001 for grpc>
```

## Kubernetes Deployment

For Kubernetes setup and deployment details, see:
- [Kubernetes Deployment Guide](guides/Kubernetes_setup.md)
- [K8s Scripts Usage](../k8s/scripts/README.md)

Quick start:
```bash
./k8s/scripts/check_and_deploy_triton.sh
```

This script performs:
1. `kubectl` installation check (and install if missing)
2. Kubernetes cluster liveness check — automatically starts or installs a local cluster (minikube → kind → k3s) if none is reachable; installs `kind` via Docker if no tool is present
3. NVIDIA GPU availability check inside cluster — installs `nvidia-container-toolkit` and the NVIDIA device plugin automatically if the host has a GPU
4. Triton deployment status check and reconciliation against the current manifests
5. Triton deploy or update (GPU or CPU manifest)
6. External Triton endpoint summary for the `NodePort` service

Default external access on minikube:
- HTTP: `http://$(minikube ip):30800`
- gRPC: `$(minikube ip):30801`
- Metrics: `http://$(minikube ip):30802/metrics`

Notes:
- The deployment uses the Triton `26.08-py3` image by default for Kubernetes.
- On minikube, if the Triton image is already present in host Docker, the deploy script loads it into the node before rollout to avoid long registry pulls.
- GPU deployments use the `Recreate` strategy so updates work on single-node, single-GPU clusters.
