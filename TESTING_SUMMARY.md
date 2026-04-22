# Chat Backend Testing Summary

## Overview

This document summarizes the testing infrastructure created for the tritonic chat backend with Gemma models.

## Changes Made

### 1. Test Scripts
- **`scripts/test_chat_backend_gemma.sh`**: Main test script for running chat backend tests with Gemma models via OpenRouter
- **`scripts/verify_gemma_api.py`**: Python pre-flight check script to verify API connectivity before running tests

### 2. Documentation
- **`docs/TESTING_CHAT_BACKEND.md`**: Comprehensive guide for testing the chat backend with various providers
- **`docs/TESTING_GEMMA.md`**: Specific guide for testing with Gemma models
- Both documents cover:
  - Setup instructions
  - Multiple testing providers (OpenRouter, Google AI Studio, Ollama)
  - Troubleshooting guide
  - Expected outputs

### 3. Integration Tests
- **`tests/integration/test_chat_backend_integration.cpp`**: GoogleTest-based integration tests that verify:
  - Simple text queries
  - Multi-turn conversations with context
  - System prompt injection
  - Token limits
  - Error handling
  - All tests are skippable if API credentials are not provided

## How to Use

### Quick Start

1. **Get API Key**
   ```bash
   # Visit https://openrouter.ai/keys
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

2. **Verify API Access (Optional)**
   ```bash
   python3 scripts/verify_gemma_api.py
   ```

3. **Build Project**
   ```bash
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON -DBUILD_TESTING=ON ..
   cmake --build . -j$(nproc)
   cd ..
   ```

4. **Run Tests**
   ```bash
   # Automated test script
   ./scripts/test_chat_backend_gemma.sh

   # Or run integration tests
   export CHAT_API_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
   export CHAT_API_KEY="$OPENROUTER_API_KEY"
   export CHAT_MODEL="google/gemma-2-9b-it"
   ./build/tritonic_integration_tests --gtest_filter="ChatBackendIntegration.*"
   ```

### Manual Testing

```bash
# Text query
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --text_prompt="What is C++?" \
    --max_tokens=100

# Interactive mode
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --interactive
```

## Supported Models

### OpenRouter (Free Tier)
**Gemma 4 (Latest - April 2026):**
- `google/gemma-4-31b:free` - **Recommended** - 31B dense, multimodal
- `google/gemma-4-26b-a4b:free` - 26B MoE (3.8B active), multimodal

**Gemma 2 (Previous generation):**
- `google/gemma-2-9b-it` - Gemma 2 9B Instruct
- `google/gemma-2-27b-it` - Gemma 2 27B Instruct

### Ollama (Local)
- `gemma2:9b`
- `gemma2:27b`

## Test Coverage

### Unit Tests (No API Required)
Located in `tests/unit/test_chat_backend.cpp`:
- ✅ Single-turn success
- ✅ Error propagation
- ✅ Session history management
- ✅ System prompt injection
- ✅ Pinned context
- ✅ History trimming
- ✅ Image attachment

### Integration Tests (API Required)
Located in `tests/integration/test_chat_backend_integration.cpp`:
- ✅ Simple text query (validates response)
- ✅ Multi-turn conversation (tests context retention)
- ✅ System prompt adherence
- ✅ Token limits enforcement
- ✅ Error handling (invalid models)

## Verification Checklist

Before submitting:
- [x] Created test script for automated testing
- [x] Added API verification script
- [x] Documented multiple testing providers
- [x] Created integration tests with real API
- [x] All tests are optional (skipped without credentials)
- [x] Documentation covers troubleshooting
- [x] Examples for all major use cases

To complete verification, a user needs to:
- [ ] Build the project successfully
- [ ] Run verification script with their API key
- [ ] Execute test script and verify responses
- [ ] Run integration tests (optional)

## Architecture Notes

The chat backend implementation:
- Supports any OpenAI-compatible API (`/v1/chat/completions`)
- Uses CURL for HTTP requests (no external HTTP library dependency)
- Implements both `IInferenceBackend` and `IChatBackend` interfaces
- Handles text and multimodal (text + images) requests
- Manages conversation history with sliding window
- Supports system prompts and pinned context

## Future Improvements

Potential enhancements:
1. Add vision model tests (multimodal inference)
2. Add streaming response support
3. Add performance benchmarks
4. Add CI/CD integration with secrets
5. Add more model providers (Anthropic, Cohere, etc.)

## References

- Main documentation: [AGENTS.md](../AGENTS.md)
- Chat backend testing: [docs/TESTING_CHAT_BACKEND.md](../docs/TESTING_CHAT_BACKEND.md)
- Gemma testing: [docs/TESTING_GEMMA.md](../docs/TESTING_GEMMA.md)
- OpenRouter: https://openrouter.ai/docs
- Gemma models: https://ai.google.dev/gemma
