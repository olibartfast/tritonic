#!/bin/bash
# filepath: /home/oli/repos/tritonic/docker/scripts/run_vit_classifier.sh

# Verify image exists locally first
if [ ! -f "data/images/cat.jpeg" ]; then
    echo "❌ Image not found: data/images/cat.jpeg"
    exit 1
fi

echo "✅ Found image: data/images/cat.jpeg"
echo "📁 Current directory: $(pwd)"

# First test: Check if the container image exists and runs
echo "🔍 Testing container..."
docker run --rm tritonic:latest --help 2>&1 | head -10

echo ""
echo "🔍 Testing volume mount..."
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    tritonic:latest \
    sh -c "ls -la /app/data/images/ && echo 'File test:' && file /app/data/images/cat.jpeg"

echo ""
echo "🔍 Testing Triton server connection..."
curl -s http://localhost:8000/v2/health/ready || echo "❌ Triton server not accessible"

echo ""
echo "🐳 Running ViT classifier..."

# Add verbose output and error capture
docker run --rm \
    --network host \
    -v "$(pwd)/data:/app/data" \
    tritonic:latest \
    --source=/app/data/images/cat.jpeg \
    --model_type=vit-classifier \
    --model=vit_onnx \
    --serverAddress=localhost \
    --input_sizes='3,224,224' \
    --verbose 2>&1

echo "Exit code: $?"
