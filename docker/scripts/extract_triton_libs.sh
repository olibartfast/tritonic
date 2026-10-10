#!/bin/bash

set -e

# =========================
# Configuration
# =========================

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/scripts/versions.sh"

TRITON_VERSION="${1:-${NGC_CONTAINER_TAG}}"
TRITON_IMAGE="${TRITON_SDK_IMAGE}"
if [[ "$TRITON_VERSION" != "${NGC_CONTAINER_TAG}" ]]; then
  TRITON_IMAGE="nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3-sdk"
fi

CONTAINER_NAME="tritonic-extract-$(date +%s)"
HOST_EXTRACT_DIR="./triton_client_libs"

echo "🐳 Extracting Triton client libraries from Docker image:"
echo "   $TRITON_IMAGE"
echo ""

# =========================
# Setup
# =========================

echo "📁 Creating host extraction directory: $HOST_EXTRACT_DIR"
mkdir -p "$HOST_EXTRACT_DIR"

echo "🚀 Starting temporary container: $CONTAINER_NAME"
docker run -d --name "$CONTAINER_NAME" "$TRITON_IMAGE" sleep 3600

# =========================
# Extraction
# =========================

echo "📦 Copying /workspace/install to $HOST_EXTRACT_DIR..."
docker cp "$CONTAINER_NAME:/workspace/install" "$HOST_EXTRACT_DIR/"

if docker exec "$CONTAINER_NAME" test -d "/opt/tritonserver/include"; then
    echo "📦 Copying Triton server includes..."
    docker cp "$CONTAINER_NAME:/opt/tritonserver/include" "$HOST_EXTRACT_DIR/triton_server_include"
fi

if docker exec "$CONTAINER_NAME" test -d "/opt/tritonserver/lib"; then
    echo "📦 Copying Triton server libraries..."
    docker cp "$CONTAINER_NAME:/opt/tritonserver/lib" "$HOST_EXTRACT_DIR/triton_server_lib"
fi

if docker exec "$CONTAINER_NAME" test -d "/workspace"; then
    echo "📦 Copying workspace files..."
    docker cp "$CONTAINER_NAME:/workspace" "$HOST_EXTRACT_DIR/workspace"
fi

# =========================
# Cleanup
# =========================

echo "🧹 Cleaning up temporary container..."
docker stop "$CONTAINER_NAME" > /dev/null
docker rm "$CONTAINER_NAME" > /dev/null

# =========================
# Summary
# =========================

echo "✅ Extraction complete!"
echo ""
echo "📋 Extracted files are in: $HOST_EXTRACT_DIR"
echo "   - install/"
echo "   - triton_server_include/"
echo "   - triton_server_lib/"
echo "   - workspace/"
echo ""
echo "🔧 Usage:"
echo "   export TritonClientBuild_DIR=$(pwd)/$HOST_EXTRACT_DIR/install"
echo "   cmake -DTritonClientBuild_DIR=\$TritonClientBuild_DIR ..."
