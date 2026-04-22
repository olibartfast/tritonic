# Gemma 4 Update - April 2026

## Summary

The testing infrastructure has been updated to support **Google's new Gemma 4 models**, released in April 2026. These models represent a significant upgrade over Gemma 2 with multimodal capabilities and enhanced features.

## What Changed

### Default Model Updated
- **Previous**: `google/gemma-2-9b-it`
- **New**: `google/gemma-4-31b:free` ⭐ **Recommended**

### New Models Available

| Model | Parameters | Cost | Key Features |
|-------|-----------|------|--------------|
| **google/gemma-4-31b:free** | 30.7B | FREE | Dense, multimodal, 256K context |
| google/gemma-4-31b | 30.7B | $0.13/$0.38/M | Same as free, paid tier |
| **google/gemma-4-26b-a4b:free** | 25.2B (3.8B active) | FREE | MoE, efficient, multimodal |
| google/gemma-4-26b-a4b | 25.2B (3.8B active) | $0.07/$0.35/M | Same as free, paid tier |

### Gemma 4 Key Features

1. **🎨 Multimodal Input**
   - Text (of course)
   - Images (JPEG, PNG, etc.)
   - Video (up to 60 seconds at 1fps)

2. **📚 Massive Context Window**
   - 256K tokens (vs 8K in Gemma 2)
   - Perfect for long documents and conversations

3. **🔧 Native Function Calling**
   - Built-in tool use support
   - Structured output generation

4. **🧠 Thinking/Reasoning Mode**
   - Configurable reasoning capabilities
   - Better for complex problem solving

5. **🌍 Enhanced Multilingual**
   - 140+ languages supported
   - Improved non-English performance

6. **⚡ Efficient MoE Variant**
   - 26B A4B model uses only 3.8B active parameters
   - Near 31B quality at fraction of compute cost

## Usage Examples

### Quick Start with Gemma 4

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Verify connectivity
python3 scripts/verify_gemma_api.py

# Run test suite
./scripts/test_chat_backend_gemma.sh
```

### Text Query

```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-4-31b:free \
    --text_prompt="Explain quantum computing" \
    --max_tokens=200
```

### Image Understanding (NEW in Gemma 4!)

```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-4-31b:free \
    --text_prompt="Analyze this code screenshot and explain what it does" \
    --source=/path/to/code_screenshot.png \
    --max_tokens=300
```

### Interactive Mode

```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=https://openrouter.ai/api/v1/chat/completions \
    --api_key_env=OPENROUTER_API_KEY \
    --model=google/gemma-4-31b:free \
    --interactive
```

## Choosing the Right Model

### For Most Use Cases: `google/gemma-4-31b:free`
- ✅ Best balance of quality and availability
- ✅ Free tier
- ✅ Full multimodal support
- ✅ 256K context window
- ✅ 30.7B parameters

### For Efficiency: `google/gemma-4-26b-a4b:free`
- ✅ Free tier
- ✅ Lower latency (only 3.8B active)
- ✅ Near 31B model quality
- ✅ Full multimodal support
- ✅ Best for resource-constrained environments

### For Development/Testing: `google/gemma-2-9b-it`
- ✅ Still available
- ✅ Fast responses
- ✅ Good for basic testing
- ⚠️ Text-only (no multimodal)
- ⚠️ Smaller context (8K tokens)

## Migration Guide

If you were using Gemma 2, migration is seamless:

1. **No code changes required** - The chat backend is model-agnostic
2. **Just change the model name** in your command:
   - Old: `--model=google/gemma-2-9b-it`
   - New: `--model=google/gemma-4-31b:free`
3. **Optionally add multimodal input** with `--source=image.jpg`

## Performance Expectations

### Response Times (OpenRouter)

| Model | First Token | Throughput | Use Case |
|-------|-------------|------------|----------|
| gemma-4-31b:free | 1-3s | 50-80 tok/s | General use |
| gemma-4-26b-a4b:free | 0.8-2s | 80-120 tok/s | Speed priority |
| gemma-2-9b-it | 0.5-1s | 100-150 tok/s | Quick tests |

*Times vary based on server load and network latency*

## Updated Files

- ✅ `scripts/test_chat_backend_gemma.sh` - Default to Gemma 4 31B
- ✅ `scripts/verify_gemma_api.py` - Test with Gemma 4
- ✅ `docs/TESTING_CHAT_BACKEND.md` - Updated examples
- ✅ `docs/TESTING_GEMMA.md` - Comprehensive Gemma 4 guide
- ✅ `TESTING_SUMMARY.md` - Updated model info
- ✅ `TESTING_INSTRUCTIONS.md` - Current best practices

## Backward Compatibility

✅ All Gemma 2 models remain supported
✅ Existing scripts work with Gemma 2 by setting `GEMMA_MODEL` env var
✅ Integration tests work with both Gemma 2 and Gemma 4

Example with Gemma 2:
```bash
export GEMMA_MODEL="google/gemma-2-9b-it"
./scripts/test_chat_backend_gemma.sh
```

## Cost Comparison

| Model | Input | Output | Free Tier |
|-------|-------|--------|-----------|
| gemma-4-31b:free | $0 | $0 | ✅ Yes |
| gemma-4-31b | $0.13/M | $0.38/M | ❌ No |
| gemma-4-26b-a4b:free | $0 | $0 | ✅ Yes |
| gemma-4-26b-a4b | $0.07/M | $0.35/M | ❌ No |
| gemma-2-9b-it | varies | varies | ⚠️ Limited |

**Recommendation**: Use the `:free` variants for testing and development. The free tier provides excellent quality and features.

## Testing Checklist

When testing Gemma 4, verify:

- [ ] Text-only conversations work correctly
- [ ] Multimodal input (images) is processed
- [ ] Context window handles long conversations
- [ ] Interactive mode maintains context
- [ ] Error handling works as expected
- [ ] Response quality meets expectations

## Resources

- **OpenRouter Models**: https://openrouter.ai/models
- **Gemma Documentation**: https://ai.google.dev/gemma
- **Testing Guide**: See `docs/TESTING_GEMMA.md`
- **Quick Start**: See `TESTING_INSTRUCTIONS.md`

## Questions?

For detailed testing instructions, see:
1. `TESTING_INSTRUCTIONS.md` - Step-by-step setup
2. `docs/TESTING_GEMMA.md` - Comprehensive Gemma guide
3. `docs/TESTING_CHAT_BACKEND.md` - General chat backend testing

---

**Updated**: April 2026
**Gemma 4 Release**: April 2-3, 2026
**Branch**: `claude/test-chat-backend-gemma-4-model`
