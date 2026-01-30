#!/bin/bash
set -e

# =========================
# Configuration
# =========================
TRITON_VERSION="${1:-25.12}"
BUILD_TYPE="${2:-debug}"

if [ "$BUILD_TYPE" == "debug" ]; then
    echo "🔨 Building Triton C++ Client Libraries in DEBUG mode"
    
    cat > Dockerfile.triton-debug <<EOF
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

# Install build dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    autoconf \
    automake \
    build-essential \
    git \
    libb64-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libtool \
    pkg-config \
    rapidjson-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake (required >= 3.31.8)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.sh && \
    chmod +x cmake-3.31.8-linux-x86_64.sh && \
    ./cmake-3.31.8-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.31.8-linux-x86_64.sh && \
    cmake --version

WORKDIR /workspace

# Clone specific branch
RUN git clone --single-branch --depth=1 -b r${TRITON_VERSION} \
    https://github.com/triton-inference-server/client.git

# Build C++ client libraries
WORKDIR /workspace/client
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=/workspace/install_debug \
          -DTRITON_ENABLE_CC_HTTP=ON \
          -DTRITON_ENABLE_CC_GRPC=ON \
          -DTRITON_ENABLE_PYTHON_HTTP=OFF \
          -DTRITON_ENABLE_PYTHON_GRPC=OFF \
          -DTRITON_ENABLE_PERF_ANALYZER=OFF \
          -DTRITON_ENABLE_EXAMPLES=OFF \
          -DTRITON_ENABLE_TESTS=OFF \
          -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
          .. && \
    make -j\$(nproc) cc-clients

# Verify installation
RUN ls -la /workspace/install_debug/ && \
    ls -la /workspace/install_debug/lib/ && \
    echo "Debug symbols check:" && \
    file /workspace/install_debug/lib/*.so | head -5
EOF

    echo "🐳 Building Docker image with debug C++ libraries..."
    docker build -f Dockerfile.triton-debug \
                 -t triton-client-debug:${TRITON_VERSION} \
                 --progress=plain \
                 .
    
    TRITON_IMAGE="triton-client-debug:${TRITON_VERSION}"
    HOST_EXTRACT_DIR="./triton_client_libs_debug"
    INSTALL_PATH="/workspace/install_debug"
else
    echo "📦 Extracting RELEASE libraries from official image"
    TRITON_IMAGE="nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3-sdk"
    HOST_EXTRACT_DIR="./triton_client_libs"
    INSTALL_PATH="/workspace/install"
fi

CONTAINER_NAME="tritonic-extract-$(date +%s)"

echo "   Image: $TRITON_IMAGE"
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
echo "📦 Copying $INSTALL_PATH to $HOST_EXTRACT_DIR..."
docker cp "$CONTAINER_NAME:$INSTALL_PATH" "$HOST_EXTRACT_DIR/"

if docker exec "$CONTAINER_NAME" test -d "/opt/tritonserver/include"; then
    echo "📦 Copying Triton server includes..."
    docker cp "$CONTAINER_NAME:/opt/tritonserver/include" "$HOST_EXTRACT_DIR/triton_server_include"
fi

if docker exec "$CONTAINER_NAME" test -d "/opt/tritonserver/lib"; then
    echo "📦 Copying Triton server libraries..."
    docker cp "$CONTAINER_NAME:/opt/tritonserver/lib" "$HOST_EXTRACT_DIR/triton_server_lib"
fi

# =========================
# Cleanup
# =========================
echo "🧹 Cleaning up temporary container..."
docker stop "$CONTAINER_NAME" > /dev/null
docker rm "$CONTAINER_NAME" > /dev/null

# Cleanup Dockerfile
[ -f Dockerfile.triton-debug ] && rm Dockerfile.triton-debug

# =========================
# Summary
# =========================
echo "✅ Extraction complete!"
echo ""
echo "📋 Extracted files are in: $HOST_EXTRACT_DIR"
echo ""
if [ "$BUILD_TYPE" == "debug" ]; then
    echo "🔍 Verifying debug symbols in extracted libraries..."
    file "$HOST_EXTRACT_DIR/install_debug/lib/"*.so 2>/dev/null | head -3 || echo "   (Run 'file' command on .so files to verify debug symbols)"
fi
echo ""
echo "🔧 Usage:"
if [ "$BUILD_TYPE" == "debug" ]; then
    echo "   export TritonClientBuild_DIR=$(pwd)/$HOST_EXTRACT_DIR/install_debug"
else
    echo "   export TritonClientBuild_DIR=$(pwd)/$HOST_EXTRACT_DIR/install"
fi
echo "   cmake -DTritonClientBuild_DIR=\$TritonClientBuild_DIR ..."
