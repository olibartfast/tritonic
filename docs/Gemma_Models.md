# Gemma Model Reference

Google's Gemma family of open models, available for inference via the tritonic Chat backend through any OpenAI-compatible endpoint.

For testing setup and usage instructions, see [Chat Backend Testing](Chat_Backend_Testing.md).

## Available Gemma Models on OpenRouter

### Gemma 4 (Latest — April 2026)

| Model | ID | Parameters | Cost (Input/Output) | Features |
|-------|----|-----------:|---------------------|----------|
| Gemma 4 31B (Free) | `google/gemma-4-31b:free` | 30.7B | FREE | Dense, multimodal |
| Gemma 4 31B | `google/gemma-4-31b` | 30.7B | $0.13/$0.38 per M | Dense, multimodal |
| Gemma 4 26B A4B (Free) | `google/gemma-4-26b-a4b:free` | 25.2B (3.8B active) | FREE | MoE, multimodal, efficient |
| Gemma 4 26B A4B | `google/gemma-4-26b-a4b` | 25.2B (3.8B active) | $0.07/$0.35 per M | MoE, multimodal, efficient |

**Gemma 4 Features:**
- Multimodal: text, images, and video (up to 60s at 1fps)
- 256K context window
- Function calling (native tool use)
- Thinking mode (configurable reasoning)
- Multilingual: 140+ languages
- Structured output (JSON, etc.)

### Gemma 2 (Previous Generation)

| Model | ID | Description |
|-------|----|-------------|
| Gemma 2 9B Instruct | `google/gemma-2-9b-it` | Fast, efficient instruction-tuned |
| Gemma 2 27B Instruct | `google/gemma-2-27b-it` | Larger, more capable |

### Ollama (Local)

| Model | Command |
|-------|---------|
| Gemma 2 9B | `ollama pull gemma2:9b` |
| Gemma 2 27B | `ollama pull gemma2:27b` |

Check [OpenRouter models](https://openrouter.ai/models) for the latest availability.

## Performance

Typical response times with OpenRouter:
- First token: 1–3 seconds
- Subsequent tokens: ~50–100 tokens/s (Gemma 2 9B)
- Full response (100 tokens): 2–5 seconds total

## References

- [Chat Backend Testing](Chat_Backend_Testing.md) — setup and testing instructions
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [Google Gemma Models](https://ai.google.dev/gemma)
