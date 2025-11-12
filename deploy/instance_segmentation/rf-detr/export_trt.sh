#!/bin/bash
# chmod +x export_trt.sh
export NGC_TAG_VERSION=25.06

# Get absolute path to workspace root to locate the output directory
WORKSPACE_ROOT=$(realpath "$(dirname "$0")/../../..")
OUTPUT_DIR="$WORKSPACE_ROOT/output"
EXPORTS_DIR="$(pwd)/exports"

echo "Using ONNX from: $OUTPUT_DIR/inference_model.sim.onnx"
echo "Saving engine to: $EXPORTS_DIR/model.engine"

docker run --rm -it --gpus=all \
    -v "$EXPORTS_DIR":/exports \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$OUTPUT_DIR/inference_model.sim.onnx":/workspace/model.onnx \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
    /bin/bash -cx "trtexec --onnx=model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            --fp16 \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"