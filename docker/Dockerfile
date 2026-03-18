# Production Dockerfile for Computer Vision Triton C++ Client
# Build: docker build --rm -t tritonic .
# Run: docker run --rm tritonic [args]

ARG TRITON_VERSION=25.12
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3-sdk

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    rapidjson-dev \
    libgtest-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y build-essential cmake
RUN apt-get install -y libcurl4-openssl-dev
RUN apt-get install -y rapidjson-dev libprotobuf-dev protobuf-compiler 

# Set environment variables
ENV TritonClientBuild_DIR=/workspace/install

# Copy your C++ source code and CMakeLists.txt into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Build the application with release configuration
RUN rm -rf build && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -GNinja .. && \
    ninja

# Optional: Build and run tests (can be enabled during development)
# RUN cd build && \
#     cmake -DBUILD_TESTING=ON -GNinja .. && \
#     ninja run_tests && \
#     ./tests/run_tests

# Add metadata labels
LABEL maintainer="Computer Vision Triton C++ Client"
LABEL description="C++ client for computer vision inference with Nvidia Triton Server"
LABEL version="1.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
# Ensure data directory and subdirectories are readable and writable
RUN chmod -R 755 /app/data
USER appuser

# Set the entry point for the container
ENTRYPOINT ["/app/build/tritonic"]

# Default command if no arguments are provided
CMD ["--help"]
