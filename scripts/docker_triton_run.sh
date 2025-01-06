#!/bin/bash 

# chmod +x docker_triton_run.sh
#./docker_triton_run.sh <path_to_models_host> <triton_version>
#example
#MODEL_REPOSITORY_HOST=$HOME/model_repository
#TRITON_VERSION=24.12
#./docker_triton_run.sh $MODEL_REPOSITORY_HOST $TRITON_VERSION

docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
-v $1:/models \
nvcr.io/nvidia/tritonserver:$2-py3 tritonserver \
--model-repository=/models --log-verbose 1