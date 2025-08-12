#!/bin/bash 
# chmod +x docker_triton_run.sh
# ./docker_triton_run.sh <path_to_models_host> <triton_version> [cpu|gpu]
# Examples:
# MODEL_REPOSITORY_HOST=$HOME/model_repository
# TRITON_VERSION=25.06
# ./docker_triton_run.sh $MODEL_REPOSITORY_HOST $TRITON_VERSION cpu
# ./docker_triton_run.sh $MODEL_REPOSITORY_HOST $TRITON_VERSION gpu

MODEL_REPO=$1
TRITON_VERSION=${2:-25.06}
DEVICE_TYPE=${3:-cpu}

if [ -z "$MODEL_REPO" ]; then
    echo "Usage: $0 <model_repository_path> [triton_version] [cpu|gpu]"
    echo "Example: $0 \$HOME/model_repository 25.06 cpu"
    exit 1
fi

if [ ! -d "$MODEL_REPO" ]; then
    echo "Error: Model repository not found: $MODEL_REPO"
    exit 1
fi

echo "🚀 Starting Triton Server"
echo "📁 Model repository: $MODEL_REPO"
echo "🏷️  Triton version: $TRITON_VERSION"
echo "💻 Device: $DEVICE_TYPE"

if [ "$DEVICE_TYPE" = "gpu" ]; then
    echo "🎮 Using GPU acceleration"
    docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
        -v "$MODEL_REPO":/models \
        nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3 tritonserver \
        --model-repository=/models \
        --log-verbose=1
else
    echo "🖥️  Using CPU only"
    docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
        -v "$MODEL_REPO":/models \
        nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3 tritonserver \
        --model-repository=/models \
        --log-verbose=1
fi-gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
-v $1:/models \
nvcr.io/nvidia/tritonserver:$2-py3 tritonserver \
--model-repository=/models 
#--log-verbose 1