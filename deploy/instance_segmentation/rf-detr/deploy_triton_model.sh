#!/usr/bin/env bash
set -euo pipefail

# deploy_triton_model.sh
# Copies a TensorRT engine file into a Triton model repository layout
# Usage: ./deploy_triton_model.sh [engine_path] [model_name] [triton_model_repo]
# Defaults:
#   engine_path: exports/model.engine
#   model_name: rf_detr
#   triton_model_repo: ./triton_model_repo

ENGINE_PATH=${1:-exports/model.engine}
MODEL_NAME=${2:-rf_detr}
TRITON_REPO=${3:-./triton_model_repo}

usage() {
    echo "Usage: $0 [engine_path] [model_name] [triton_model_repo]"
    echo "Example: $0 exports/model.engine rf_detr /absolute/path/to/triton/model/repo"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ ! -f "$ENGINE_PATH" ]; then
    echo "❌ Error: Engine file not found: $ENGINE_PATH"
    echo "Provide the path to the exported TensorRT engine file (e.g. exports/model.engine)"
    exit 1
fi

# Ensure triton model repo path is absolute (Docker and Triton prefer absolute host paths)
TRITON_REPO=$(realpath -m "$TRITON_REPO")

echo "📦 Deploying engine to Triton model repository"
echo "    Engine: $ENGINE_PATH"
echo "    Model name: $MODEL_NAME"
echo "    Triton model repo: $TRITON_REPO"

# Create model directory and version subdirectory
MODEL_DIR="$TRITON_REPO/$MODEL_NAME"
VERSION_DIR="$MODEL_DIR/1"
mkdir -p "$VERSION_DIR"

# Copy engine into version directory and rename to model.plan (Triton expects a plan file for tensorrt)
TARGET_PLAN="$VERSION_DIR/model.plan"
cp -v -- "$ENGINE_PATH" "$TARGET_PLAN"

# Create a minimal config.pbtxt if one doesn't already exist
CONFIG_FILE="$MODEL_DIR/config.pbtxt"
if [ -f "$CONFIG_FILE" ]; then
    echo "ℹ️  Using existing $CONFIG_FILE"
else
    cat > "$CONFIG_FILE" <<EOF
name: "${MODEL_NAME}"
platform: "tensorrt_plan"
max_batch_size: 1
# NOTE: Inputs/outputs and other model-specific settings are not added here.
# Edit this file to specify input/output names, data types and shapes as required by your model.
EOF
    echo "✅ Created minimal $CONFIG_FILE (please edit inputs/outputs as needed)"
fi

echo "✅ Deployed engine to $TARGET_PLAN"

exit 0
