#!/bin/bash
# Test script for ChatBackend with Gemma 4 model via OpenRouter API
#
# OpenRouter provides free access to various models including Gemma
# Endpoint: https://openrouter.ai/api/v1/chat/completions
#
# Usage:
#   1. Get a free API key from https://openrouter.ai/keys
#   2. Set environment variable: export OPENROUTER_API_KEY="your-key"
#   3. Run this script: ./scripts/test_chat_backend_gemma.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
EXECUTABLE="$BUILD_DIR/tritonic"

# Configuration
API_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
MODEL="google/gemma-2-9b-it"  # Gemma 2 9B Instruct
TEST_IMAGE="$REPO_ROOT/tests/test_data/test_image.jpg"

# Check if API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set"
    echo "Please get a free API key from https://openrouter.ai/keys"
    echo "Then set it: export OPENROUTER_API_KEY='your-key'"
    exit 1
fi

# Check if build exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: tritonic executable not found at $EXECUTABLE"
    echo "Please build the project first:"
    echo "  mkdir -p build && cd build"
    echo "  cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON .."
    echo "  cmake --build . -j\$(nproc)"
    exit 1
fi

echo "=========================================="
echo "Testing ChatBackend with Gemma 4 Model"
echo "=========================================="
echo "Endpoint: $API_ENDPOINT"
echo "Model: $MODEL"
echo ""

# Test 1: Simple text-only conversation
echo "Test 1: Text-only conversation"
echo "Prompt: 'What is C++?'"
echo ""

"$EXECUTABLE" \
    --backend=chat \
    --api_endpoint="$API_ENDPOINT" \
    --api_key_env=OPENROUTER_API_KEY \
    --model="$MODEL" \
    --text_prompt="What is C++? Answer in one sentence." \
    --max_tokens=100

echo ""
echo "Test 1 completed successfully!"
echo ""

# Test 2: Interactive mode (if supported)
echo "=========================================="
echo "Test 2: Interactive mode"
echo "You can now chat with Gemma. Type 'exit' to quit."
echo "=========================================="
echo ""

"$EXECUTABLE" \
    --backend=chat \
    --api_endpoint="$API_ENDPOINT" \
    --api_key_env=OPENROUTER_API_KEY \
    --model="$MODEL" \
    --interactive

echo ""
echo "All tests completed!"
