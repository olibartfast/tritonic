# Chat Backend (OpenAI-compatible)

Using TritonIC against any OpenAI-compatible `/v1/chat/completions` endpoint,
with the complete flag reference.

Back to the [documentation index](README.md) | [project README](../README.md).
For the test procedure see [Chat_Backend_Testing.md](Chat_Backend_Testing.md).

## Overview

Skip Triton entirely and query any OpenAI-compatible server. Works with Ollama, llama.cpp, SGLang, vLLM, OpenAI, Together AI, OpenRouter, and Z.AI.

**Single-turn with an image:**
```bash
./tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:11434/v1/chat/completions \
    --model=llava:7b \
    --text_prompt="Describe what you see" \
    --source=/path/to/image.jpg
```

**Interactive multi-turn session:**
```bash
./tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:11434/v1/chat/completions \
    --model=llava:7b \
    --text_prompt="You are a helpful assistant" \
    --interactive
```

**OpenRouter multimodal with Kimi K2.6:**
```bash
export OPENROUTER_API_KEY=...

./tritonic \
    --backend=chat \
    --api_service=openrouter \
    --model=moonshotai/kimi-k2.6 \
    --text_prompt="Describe the scene and read any visible text." \
    --source=/path/to/image.jpg
```

**Together AI text-only with GLM-5.1:**
```bash
export TOGETHER_API_KEY=...

./tritonic \
    --backend=chat \
    --api_service=together \
    --model=zai-org/GLM-5.1 \
    --text_prompt="Summarize the design tradeoffs in this architecture."
```

**Z.AI multimodal with GLM-4.6V:**
```bash
export ZAI_API_KEY=...

./tritonic \
    --backend=chat \
    --api_service=zai \
    --model=glm-4.6v \
    --text_prompt="Describe the image and extract the key objects." \
    --source=/path/to/image.jpg
```

`GLM-5.1` is available on Together AI and Z.AI, but it is text-only. For GLM-family image input, use `GLM-4.6V`.

**Chat CLI parameters:**

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--backend` | `be` | `triton` | `triton` or `chat` |
| `--api_endpoint` | `ae` | — | Full URL, e.g. `http://localhost:11434/v1/chat/completions` |
| `--api_service` | `as` | — | Service preset: `openai`, `openrouter`, `together`, `zai` |
| `--api_key_env` | `ak` | — | Env-var name that holds the API key (e.g. `OPENAI_API_KEY`) |
| `--text_prompt` | `tp` | — | System prompt (interactive) or user prompt (single-turn) |
| `--max_tokens` | `mxt` | `256` | Max tokens to generate |
| `--temperature` | `temp` | `1.0` | Sampling temperature |
| `--target_image_size` | `tis` | `512` | Longest edge (px) before base64 encoding |
| `--interactive` | `ia` | `false` | Enable multi-turn REPL |
| `--inference_timeout` | `it` | `0` | Request timeout in milliseconds (`0` keeps backend default) |
