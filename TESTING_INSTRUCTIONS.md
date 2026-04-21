# Testing Chat Backend with Gemma Models - Instructions

This branch contains comprehensive testing infrastructure for the tritonic chat backend using Gemma models via public APIs.

## What Has Been Done

✅ **Test Infrastructure Created:**
- Automated test script for OpenRouter API
- Python verification script for pre-flight checks
- GoogleTest integration tests for automated testing
- Comprehensive documentation for multiple providers

✅ **Documentation Added:**
- `docs/TESTING_CHAT_BACKEND.md` - General chat backend testing guide
- `docs/TESTING_GEMMA.md` - Specific Gemma model testing guide
- `TESTING_SUMMARY.md` - Complete summary of all testing infrastructure

✅ **Code Quality:**
- All tests are optional and skip gracefully without API credentials
- Error handling for common failure cases
- Support for multiple providers (OpenRouter, Ollama, Google AI Studio)

## What Needs Manual Verification

To complete the testing, someone with access to the build environment needs to:

### Step 1: Set Up API Access

Get a free API key from OpenRouter:
```bash
# 1. Visit https://openrouter.ai/keys
# 2. Create a free account
# 3. Generate an API key
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Step 2: Verify API Connectivity

Run the verification script:
```bash
python3 scripts/verify_gemma_api.py
```

Expected output:
```
✅ API key found: sk-or-v1-...
🔄 Testing API connection to OpenRouter...
✅ API test successful!
   Response: Hello from Gemma!
📊 Token usage:
   Prompt tokens: 12
   Completion tokens: 5
   Total tokens: 17
```

### Step 3: Build the Project

```bash
# Set up Triton client libraries
export TritonClientBuild_DIR=$(pwd)/triton_client_libs/install
./docker/scripts/extract_triton_libs.sh  # if needed

# Build with chat backend
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CHAT_BACKEND=ON -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)
cd ..
```

### Step 4: Run Tests

**Option A: Automated test script**
```bash
./scripts/test_chat_backend_gemma.sh
```

**Option B: Manual testing**
```bash
# Simple query
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --text_prompt="Explain C++ in one sentence" \
    --max_tokens=100

# Interactive mode
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-2-9b-it \
    --interactive
```

**Option C: Integration tests**
```bash
export CHAT_API_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
export CHAT_API_KEY="$OPENROUTER_API_KEY"
export CHAT_MODEL="google/gemma-2-9b-it"
./build/tritonic_integration_tests --gtest_filter="ChatBackendIntegration.*"
```

### Step 5: Document Results

After successful testing, document:
1. Build success/failure
2. Verification script output
3. Test script output
4. Any errors encountered
5. Performance observations

## Expected Results

### Successful Test Output

```
==========================================
Testing ChatBackend with Gemma Model
==========================================
Endpoint: https://openrouter.ai/api/v1/chat/completions
Model: google/gemma-2-9b-it

Test 1: Text-only conversation
Prompt: 'What is C++?'

Configuration:
  Backend: chat (OpenAI-compatible)
  Endpoint: https://openrouter.ai/api/v1/chat/completions
  Model: google/gemma-2-9b-it

Response: C++ is a general-purpose programming language that extends C with object-oriented features...

Test 1 completed successfully!

==========================================
Test 2: Interactive mode
You can now chat with Gemma. Type 'exit' to quit.
==========================================
```

## Alternative Testing Methods

### Using Ollama (No API Key Required)

```bash
# Install Ollama
# Visit https://ollama.ai/download

# Pull Gemma model
ollama pull gemma2:9b

# Test with tritonic
./build/tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:11434/v1/chat/completions \
    --model=gemma2:9b \
    --text_prompt="Hello, Gemma!" \
    --interactive
```

### Using Google AI Studio (with litellm proxy)

```bash
# Install proxy
pip install litellm

# Get API key from https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="..."

# Start proxy
litellm --model gemini/gemini-1.5-flash --port 8000

# Test (in another terminal)
./build/tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:8000/v1/chat/completions \
    --model=gemini/gemini-1.5-flash \
    --interactive
```

## Troubleshooting

See the detailed guides:
- [TESTING_CHAT_BACKEND.md](docs/TESTING_CHAT_BACKEND.md) - General troubleshooting
- [TESTING_GEMMA.md](docs/TESTING_GEMMA.md) - Gemma-specific issues

Common issues:
1. **Missing API key**: Set `OPENROUTER_API_KEY` environment variable
2. **Build failures**: Check dependencies and Triton client libraries
3. **Connection errors**: Verify internet connectivity
4. **Rate limits**: Wait and retry, or use Ollama locally

## Branch Information

- **Branch**: `claude/test-chat-backend-gemma-4-model`
- **Base**: `develop`
- **Purpose**: Add testing infrastructure for chat backend with Gemma models
- **Status**: Ready for manual verification

## Files Changed

```
docs/
  ├── TESTING_CHAT_BACKEND.md    (new) - Comprehensive testing guide
  └── TESTING_GEMMA.md            (new) - Gemma-specific guide

scripts/
  ├── test_chat_backend_gemma.sh  (new) - Main test script
  └── verify_gemma_api.py         (new) - API verification tool

tests/
  └── integration/
      └── test_chat_backend_integration.cpp  (new) - Integration tests

TESTING_SUMMARY.md                (new) - Summary of all changes
TESTING_INSTRUCTIONS.md           (new) - This file
```

## Next Steps

1. **For Reviewers**: Review the test infrastructure and documentation
2. **For Testers**: Follow the steps above to run tests with your API key
3. **For Merging**: Once verified, this branch can be merged to develop

## Contact

If you encounter any issues or need clarification:
- Check the documentation in `docs/`
- Review the test scripts in `scripts/`
- Examine the integration tests in `tests/integration/`

---

**Ready to test?** Start with the verification script:
```bash
export OPENROUTER_API_KEY="your-key"
python3 scripts/verify_gemma_api.py
```
