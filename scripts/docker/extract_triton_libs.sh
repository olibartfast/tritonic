#!/bin/bash

# Script to extract Triton client libraries from the Docker container to host
# This copies the TritonClientBuild_DIR=/workspace/install contents to the host

set -e

CONTAINER_NAME="tritonic-extract-$(date +%s)"
HOST_EXTRACT_DIR="./triton_client_libs"

echo "🐳 Extracting Triton client libraries from nvcr.io/nvidia/tritonserver:25.07-py3-sdk Docker image..."

# Create extraction directory on host
echo "📁 Creating host extraction directory: $HOST_EXTRACT_DIR"
mkdir -p "$HOST_EXTRACT_DIR"

# Run container in background to keep it alive
echo "🚀 Starting temporary container: $CONTAINER_NAME"
docker run -d --name "$CONTAINER_NAME" nvcr.io/nvidia/tritonserver:25.07-py3-sdk sleep 3600

# Copy the Triton client build directory from container to host
echo "📦 Copying /workspace/install to $HOST_EXTRACT_DIR..."
docker cp "$CONTAINER_NAME:/workspace/install" "$HOST_EXTRACT_DIR/"

# Also copy any additional Triton-related directories
echo "📦 Copying additional Triton directories..."

# Copy Triton includes if they exist
if docker exec "$CONTAINER_NAME" test -d "/opt/tritonserver/include"; then
    echo "📦 Copying Triton server includes..."
    docker cp "$CONTAINER_NAME:/opt/tritonserver/include" "$HOST_EXTRACT_DIR/triton_server_include"
fi

# Copy Triton libraries if they exist  
if docker exec "$CONTAINER_NAME" test -d "/opt/tritonserver/lib"; then
    echo "📦 Copying Triton server libraries..."
    docker cp "$CONTAINER_NAME:/opt/tritonserver/lib" "$HOST_EXTRACT_DIR/triton_server_lib"
fi

# Copy any cmake files
if docker exec "$CONTAINER_NAME" test -d "/workspace"; then
    echo "📦 Copying workspace cmake files..."
    docker cp "$CONTAINER_NAME:/workspace" "$HOST_EXTRACT_DIR/workspace"
fi

# Clean up container
echo "🧹 Cleaning up temporary container..."
docker stop "$CONTAINER_NAME" > /dev/null
docker rm "$CONTAINER_NAME" > /dev/null

echo "✅ Extraction complete!"
echo ""
echo "📋 Extracted files are in: $HOST_EXTRACT_DIR"
echo "   - install/          : Triton client build artifacts"
echo "   - triton_server_include/ : Triton server headers"
echo "   - triton_server_lib/     : Triton server libraries"
echo "   - workspace/             : Additional workspace files"
echo ""
echo "🔧 To use these libraries in your local build:"
echo "   export TritonClientBuild_DIR=$(pwd)/$HOST_EXTRACT_DIR/install"
echo "   cmake -DTritonClientBuild_DIR=\$TritonClientBuild_DIR ..."
echo ""
echo "📚 Directory structure:"
find "$HOST_EXTRACT_DIR" -type d -maxdepth 3 | head -20
