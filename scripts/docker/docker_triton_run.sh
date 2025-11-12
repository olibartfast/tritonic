#!/bin/bash
# chmod +x docker_triton_run.sh
# ./docker_triton_run.sh <path_to_models_host> <triton_version> [cpu|gpu] [python_backend]
# 
# Examples:
# MODEL_REPOSITORY_HOST=$HOME/model_repository
# TRITON_VERSION=25.06
# ./docker_triton_run.sh $MODEL_REPOSITORY_HOST $TRITON_VERSION cpu
# ./docker_triton_run.sh $MODEL_REPOSITORY_HOST $TRITON_VERSION gpu
# ./docker_triton_run.sh $MODEL_REPOSITORY_HOST $TRITON_VERSION gpu python_backend

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Default values
MODEL_REPO=${1:-}
TRITON_VERSION=${2:-25.06}
DEVICE_TYPE=${3:-cpu}
PYTHON_BACKEND_ARG=${4:-false}

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down Triton Server..."
}
trap cleanup EXIT INT TERM

# Validate required parameters
if [ $# -lt 1 ] || [ -z "$MODEL_REPO" ]; then
    echo "Usage: $0 <model_repository_path> [triton_version] [cpu|gpu] [python_backend|true]"
    echo ""
    echo "Examples:"
    echo "  $0 \$HOME/model_repository 25.06 cpu"
    echo "  $0 \$HOME/model_repository 25.06 gpu python_backend"
    echo "  $0 \$HOME/model_repository 25.06 gpu true"
    echo ""
    echo "Parameters:"
    echo "  model_repository_path: Path to your model repository (required)"
    echo "  triton_version: Docker image version (default: 25.06)"
    echo "  device_type: cpu or gpu (default: cpu)"
    echo "  python_backend: python_backend|true|1|yes to enable (default: false)"
    exit 1
fi

# Validate dependencies
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install Docker first."
    exit 1
fi

# Validate model repository
if [ ! -d "$MODEL_REPO" ]; then
    echo "❌ Error: Model repository not found: $MODEL_REPO"
    echo "Please ensure the directory exists and is accessible."
    exit 1
fi

# Normalize device type
case "${DEVICE_TYPE,,}" in  # Convert to lowercase
    gpu|cuda) DEVICE_TYPE="gpu" ;;
    cpu|x86|x64) DEVICE_TYPE="cpu" ;;
    *) 
        echo "❌ Error: Invalid device type '$DEVICE_TYPE'. Use 'cpu' or 'gpu'."
        exit 1
        ;;
esac

# Normalize python backend parameter
case "${PYTHON_BACKEND_ARG,,}" in
    python_backend|true|1|yes|on) PYTHON_BACKEND=true ;;
    *) PYTHON_BACKEND=false ;;
esac

# Validate GPU availability if requested
if [ "$DEVICE_TYPE" = "gpu" ]; then
    echo "🔍 Checking GPU availability..."
    if ! docker run --rm --gpus=all --entrypoint nvidia-smi nvcr.io/nvidia/cuda:12.2-base-ubuntu20.04 &>/dev/null; then
        echo "⚠️  Warning: GPU not available or nvidia-container-toolkit not configured properly"
        echo "💡 Falling back to CPU mode. To fix:"
        echo "   1. Install nvidia-container-toolkit"
        echo "   2. Restart Docker daemon"
        echo "   3. Ensure NVIDIA drivers are installed"
        DEVICE_TYPE="cpu"
        sleep 2
    else
        echo "✅ GPU support confirmed"
    fi
fi

# Display configuration
echo ""
echo "🚀 Starting Triton Inference Server"
echo "📁 Model repository: $MODEL_REPO"
echo "🏷️  Triton version: $TRITON_VERSION"
echo "💻 Device: $DEVICE_TYPE"
echo "🐍 Python backend: $PYTHON_BACKEND"
echo ""

# Determine Docker image and setup
BASE_IMAGE="nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3"

if [ "$PYTHON_BACKEND" = "true" ]; then
    echo "📦 Python backend enabled - packages will be installed at startup"
    echo "💡 Consider creating a custom image for production use"
    PYTHON_INSTALL_CMD="pip install --no-cache-dir torch torchvision transformers pillow numpy scipy && echo '✅ Python packages installed' && "
else
    echo "⚡ Skipping Python backend package installation"
    PYTHON_INSTALL_CMD=""
fi

# Build Docker command components
DOCKER_BASE="docker run --rm --name triton-server-$$"
DOCKER_PORTS="-p8000:8000 -p8001:8001 -p8002:8002"
DOCKER_VOLUME="-v \"$MODEL_REPO\":/models"
DOCKER_IMAGE="$BASE_IMAGE"

# Add GPU support if needed
if [ "$DEVICE_TYPE" = "gpu" ]; then
    echo "🎮 Using GPU acceleration"
    DOCKER_GPU="--gpus=all"
else
    echo "🖥️  Using CPU only"
    DOCKER_GPU=""
fi

# Triton server arguments
TRITON_ARGS="--model-repository=/models"

# Add CPU-specific optimizations
if [ "$DEVICE_TYPE" = "cpu" ]; then
    TRITON_ARGS="$TRITON_ARGS --backend-config=tensorflow,allow_soft_placement=true"
fi

# Build final command
if [ -n "$PYTHON_INSTALL_CMD" ]; then
    FULL_CMD="$DOCKER_BASE $DOCKER_GPU $DOCKER_PORTS $DOCKER_VOLUME $DOCKER_IMAGE bash -c \"$PYTHON_INSTALL_CMD tritonserver $TRITON_ARGS\""
else
    FULL_CMD="$DOCKER_BASE $DOCKER_GPU $DOCKER_PORTS $DOCKER_VOLUME $DOCKER_IMAGE tritonserver $TRITON_ARGS"
fi

# Display the command being executed (for debugging)
echo "🔧 Docker command:"
echo "$FULL_CMD"
echo ""

# Pull the latest image if it doesn't exist locally
echo "🔄 Ensuring Docker image is available..."
if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
    echo "📥 Pulling $BASE_IMAGE..."
    docker pull "$BASE_IMAGE"
fi

echo "🌟 Starting Triton Server..."
echo "📡 Server will be available at:"
echo "   HTTP:  http://localhost:8000"
echo "   GRPC:  localhost:8001"  
echo "   Metrics: http://localhost:8002/metrics"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Execute the command
eval "$FULL_CMD"