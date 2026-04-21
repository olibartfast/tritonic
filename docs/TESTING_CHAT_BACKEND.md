# Testing Chat Backend with Gemma Models

This document describes how to test the tritonic chat backend with Google's Gemma models using public APIs.

## Overview

The chat backend supports any OpenAI-compatible API endpoint. This includes:
- OpenRouter (provides access to many models including Gemma)
- Google AI Studio (direct Gemma access, OpenAI-compatible via proxy)
- Ollama (local deployment)
- Other OpenAI-compatible services

## Testing with OpenRouter

OpenRouter provides free access to various models including Gemma 2 and Gemma 4.

### Prerequisites

1. **Get an API Key**
   - Visit https://openrouter.ai/keys
   - Sign up for a free account
   - Create an API key

2. **Build the Project**
   ```bash
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON ..
   cmake --build . -j$(nproc)
   ```

### Available Gemma Models on OpenRouter

- `google/gemma-2-9b-it` - Gemma 2 9B Instruct
- `google/gemma-2-27b-it` - Gemma 2 27B Instruct
- `google/gemma-3-27b-it` - Gemma 3 27B Instruct (if available)

Note: OpenRouter model availability may change. Check https://openrouter.ai/models for the latest list.

### Quick Test Script

Use the provided test script:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
./scripts/test_chat_backend_gemma.sh
```

### Manual Testing

#### Test 1: Single-turn text conversation

```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --text_prompt="Explain what C++ is in one sentence." \
    --max_tokens=100
```

#### Test 2: Interactive REPL

```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --interactive
```

In interactive mode, type your messages and press Enter. Type `exit` or `quit` to end the session.

#### Test 3: Vision-capable model (if supported)

```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --text_prompt="Describe what you see in this image" \
    --source=/path/to/image.jpg
```

Note: Vision capabilities depend on the specific model variant. Check model documentation.

## Testing with Google AI Studio (Alternative)

Google AI Studio provides direct access to Gemma models, but requires an OpenAI-compatible proxy or adapter.

### Using with litellm (Proxy)

1. Install litellm:
   ```bash
   pip install litellm
   ```

2. Start the proxy:
   ```bash
   export GEMINI_API_KEY="your-google-ai-studio-key"
   litellm --model gemini/gemini-1.5-flash --port 8000
   ```

3. Test with tritonic:
   ```bash
   ./build/tritonic \
       --backend=chat \
       --api_endpoint=http://localhost:8000/v1/chat/completions \
       --model=gemini/gemini-1.5-flash \
       --text_prompt="Hello, world!" \
       --interactive
   ```

## Testing with Ollama (Local)

For local testing without API keys:

1. Install Ollama: https://ollama.ai/download

2. Pull a Gemma model:
   ```bash
   ollama pull gemma2:9b
   ```

3. Test with tritonic:
   ```bash
   ./build/tritonic \
       --backend=chat \
       --api_endpoint=http://localhost:11434/v1/chat/completions \
       --model=gemma2:9b \
       --text_prompt="What is machine learning?" \
       --interactive
   ```

## Expected Behavior

### Successful Test Output

```
Configuration:
  Backend: chat (OpenAI-compatible)
  Endpoint: https://openrouter.ai/api/v1/chat/completions
  Model: google/gemma-2-9b-it

Request sent successfully
Response: [Model's response text here]
```

### Error Cases

1. **Missing API Key**
   ```
   Error: OPENROUTER_API_KEY environment variable is not set
   ```

2. **Invalid Endpoint**
   ```
   CURL error: Could not resolve host
   ```

3. **Model Not Found**
   ```
   Error: The model 'model-name' does not exist
   ```

4. **Rate Limit**
   ```
   Error: Rate limit exceeded
   ```

## Troubleshooting

### Issue: Connection timeout

- Check your internet connection
- Verify the API endpoint is correct
- Ensure the API service is not down

### Issue: Authentication failed

- Verify your API key is correct
- Check that the environment variable is set: `echo $OPENROUTER_API_KEY`
- Ensure your API key has not expired

### Issue: Model not available

- Check the model name is correct
- Verify the model is available on the platform
- Try a different model variant

## Integration with CI/CD

For automated testing in CI/CD pipelines, you can:

1. Store API keys as secrets in your CI system
2. Use the test script in your pipeline:
   ```yaml
   - name: Test Chat Backend
     env:
       OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
     run: ./scripts/test_chat_backend_gemma.sh
   ```

## Performance Expectations

| Model | Expected Latency | Token Throughput |
|-------|------------------|------------------|
| Gemma 2 9B | 1-3s first token | ~50-100 tokens/s |
| Gemma 2 27B | 2-5s first token | ~30-50 tokens/s |

Actual performance depends on:
- API provider infrastructure
- Network latency
- Current server load
- Model configuration (temperature, max_tokens)

## References

- OpenRouter API Docs: https://openrouter.ai/docs
- Google Gemma Models: https://ai.google.dev/gemma
- tritonic Chat Backend: [AGENTS.md](../AGENTS.md)
