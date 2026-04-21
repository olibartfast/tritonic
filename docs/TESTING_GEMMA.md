# Chat Backend Testing Guide - Gemma Models

This guide provides step-by-step instructions for testing the tritonic chat backend with Gemma models using public APIs.

## Quick Start

### Option 1: Using OpenRouter (Recommended for Testing)

OpenRouter provides free access to Gemma models with a simple API key.

1. **Get API Key**
   ```bash
   # Visit https://openrouter.ai/keys and create a free account
   # Copy your API key
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

2. **Build tritonic with chat backend**
   ```bash
   # Install dependencies (if not already installed)
   sudo apt-get install -y libopencv-dev libcurl4-openssl-dev

   # Set Triton client directory (or extract libraries)
   export TritonClientBuild_DIR=$(pwd)/triton_client_libs/install

   # Build
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON ..
   cmake --build . -j$(nproc)
   cd ..
   ```

3. **Run the test script**
   ```bash
   ./scripts/test_chat_backend_gemma.sh
   ```

### Option 2: Manual Testing

**Test 1: Simple query**
```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --text_prompt="Explain C++ in one sentence" \
    --max_tokens=100
```

**Test 2: Interactive conversation**
```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --interactive
```

### Option 3: Integration Tests

Run automated integration tests (requires API credentials):

```bash
# Set environment variables
export CHAT_API_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
export CHAT_API_KEY="sk-or-v1-..."
export CHAT_MODEL="google/gemma-2-9b-it"

# Build with tests
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)

# Run integration tests
./tritonic_integration_tests --gtest_filter="ChatBackendIntegration.*"
```

## Available Gemma Models on OpenRouter

| Model | ID | Description |
|-------|----|----|
| Gemma 2 9B Instruct | `google/gemma-2-9b-it` | Fast, efficient instruction-tuned model |
| Gemma 2 27B Instruct | `google/gemma-2-27b-it` | Larger, more capable model |

Check https://openrouter.ai/models for the latest available Gemma models.

## Expected Results

### Successful Test Output

```
==========================================
Testing ChatBackend with Gemma 4 Model
==========================================
Endpoint: https://openrouter.ai/api/v1/chat/completions
Model: google/gemma-2-9b-it

Test 1: Text-only conversation
Prompt: 'What is C++?'

Configuration:
  Backend: chat (OpenAI-compatible)
  Endpoint: https://openrouter.ai/api/v1/chat/completions
  Model: google/gemma-2-9b-it

Response: C++ is a general-purpose programming language that extends C with object-oriented features.

Test 1 completed successfully!
```

### Common Issues and Solutions

1. **"OPENROUTER_API_KEY environment variable is not set"**
   - Solution: Set the API key: `export OPENROUTER_API_KEY="your-key"`

2. **"CURL error: Could not resolve host"**
   - Solution: Check internet connection
   - Verify endpoint URL is correct

3. **"Error: The model 'model-name' does not exist"**
   - Solution: Check available models at https://openrouter.ai/models
   - Use exact model ID (e.g., `google/gemma-2-9b-it`)

4. **"Rate limit exceeded"**
   - Solution: OpenRouter has rate limits on free tier
   - Wait a few minutes and try again
   - Consider upgrading account if testing extensively

## Testing Different Providers

### Google AI Studio (via proxy)

```bash
# Install litellm proxy
pip install litellm

# Get API key from https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-google-api-key"

# Start proxy
litellm --model gemini/gemini-1.5-flash --port 8000

# Test with tritonic (in another terminal)
./build/tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:8000/v1/chat/completions \
    --model=gemini/gemini-1.5-flash \
    --text_prompt="Hello, Gemma!" \
    --interactive
```

### Ollama (Local)

```bash
# Install Ollama from https://ollama.ai/download
ollama pull gemma2:9b

# Test with tritonic
./build/tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:11434/v1/chat/completions \
    --model=gemma2:9b \
    --text_prompt="What is machine learning?" \
    --interactive
```

## Unit Tests

The chat backend also includes unit tests that don't require API access:

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)

# Run all unit tests
./tritonic_unit_tests

# Run only chat backend unit tests
./tritonic_unit_tests --gtest_filter="*Chat*"
```

## Verification Checklist

- [ ] Build succeeds with `-DWITH_CHAT_BACKEND=ON`
- [ ] Unit tests pass
- [ ] Simple text query returns valid response
- [ ] Interactive mode accepts user input and responds
- [ ] Multi-turn conversation maintains context
- [ ] Error handling works (invalid model, network errors)
- [ ] Integration tests pass with real API

## Performance Notes

Typical response times with OpenRouter:
- First token: 1-3 seconds
- Subsequent tokens: ~50-100 tokens/second (Gemma 2 9B)
- Full response (100 tokens): 2-5 seconds total

## Troubleshooting

### Build Issues

If build fails with missing dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install -y libopencv-dev libcurl4-openssl-dev build-essential cmake

# Extract Triton client libraries
./docker/scripts/extract_triton_libs.sh
```

### Runtime Issues

Enable debug logging:
```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --text_prompt="Test" \
    --log_level=debug
```

## References

- [Main Testing Documentation](./TESTING_CHAT_BACKEND.md)
- [Project Architecture](../AGENTS.md)
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [Gemma Models](https://ai.google.dev/gemma)
