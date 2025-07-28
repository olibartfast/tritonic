#!/bin/bash

# Run unit tests via Docker
# This script builds the development Docker image and runs the unit tests

echo "Building development Docker image with testing support..."
docker build -f Dockerfile.dev --rm -t tritonic:latest .

if [ $? -eq 0 ]; then
    echo "Build successful! Running unit tests..."
    docker run --rm tritonic:latest
else
    echo "Build failed! Please check the Docker build output above."
    exit 1
fi 