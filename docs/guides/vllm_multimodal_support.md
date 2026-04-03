# vLLM / LLM / Multimodal Inference Support

This document describes the support for LLM and multimodal inference in TritonIC
using Triton's [vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html),
and outlines the changes required in the sibling
[vision-core](https://github.com/olibartfast/vision-core) repository for full
multimodal task integration.

---

## Overview

NVIDIA Triton Inference Server supports the **vLLM backend** for serving Large
Language Models (LLMs). This enables text generation, chat completion, and
multimodal inference (image + text → text) through the same Triton infrastructure
used for computer vision models.

### Supported LLM Model Types

| Model Type Parameter | Description |
|---------------------|-------------|
| `vllm`              | Generic vLLM-served model |
| `llm`               | Generic LLM |
| `llama`             | LLaMA family models |
| `mistral`           | Mistral models |
| `qwen`              | Qwen models |
| `phi`               | Phi models |
| `gemma`             | Gemma models |
| `chatglm`           | ChatGLM models |
| `text-generation`   | Generic text generation |

---

## Tritonic Changes (This Repository)

The following changes have been made to tritonic to support LLM inference:

### 1. String Tensor Support
- `TensorElement` variant now includes `std::string` alongside numeric types
- `getInferResults()` handles `BYTES` output datatype via `StringData()`
- `parseModelHttp()`/`parseModelGrpc()` accept `BYTES` and `BOOL` input datatypes

### 2. New `inferText()` API
- `ITriton::inferText(const std::vector<std::vector<std::string>>& string_inputs)`
- Uses `AppendFromString()` for string tensor inputs (required by vLLM backend)
- Keeps backward compatibility — existing `infer()` for numeric tensors is unchanged

### 3. Output Metadata in TritonModelInfo
- `output_datatypes` and `output_shapes` vectors track output tensor metadata
- Enables the client to know output types before inference

### 4. LLM Generation Parameters
New CLI flags for controlling text generation:

| Flag | Default | Description |
|------|---------|-------------|
| `--max_tokens` / `-mxt` | 256 | Maximum tokens to generate |
| `--temperature` / `-temp` | 1.0 | Sampling temperature |
| `--top_p` | 1.0 | Top-p (nucleus) sampling |
| `--repetition_penalty` / `-rp` | 1.0 | Repetition penalty |
| `--stop_words` / `-sw` | (empty) | Comma-separated stop sequences |

### 5. Text Generation Processing Path
- `App::processTextGeneration()` handles text-only LLM inference
- Bypasses vision-core task system (no image preprocessing needed)
- Automatically maps vLLM model input names (`text_input`, `sampling_parameters`, `stream`, etc.)

---

## Running LLM Inference

### 1. Deploy a vLLM Model on Triton

Create a model repository entry:

```
model_repository/
  vllm_llama/
    1/
      model.json
    config.pbtxt
```

**config.pbtxt**:
```protobuf
name: "vllm_llama"
backend: "vllm"
max_batch_size: 0

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  },
  {
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
```

**model.json** (vLLM engine args):
```json
{
    "model": "meta-llama/Llama-2-7b-hf",
    "disable_log_requests": true,
    "gpu_memory_utilization": 0.85
}
```

### 2. Start Triton Server

```bash
docker run --gpus=1 --rm --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.06-vllm-python-py3 \
  tritonserver --model-repository=/models
```

### 3. Run Tritonic

```bash
./tritonic \
    --model_type=vllm \
    --model=vllm_llama \
    --text_prompt="What is machine learning?" \
    --max_tokens=256 \
    --temperature=0.7 \
    --top_p=0.9 \
    --protocol=http \
    --serverAddress=localhost \
    --port=8000
```

---

## Required Changes in vision-core

For full integration of LLM and multimodal tasks into the vision-core task
framework (enabling consistent pre/postprocessing and result handling), the
following modifications are needed in the
[vision-core](https://github.com/olibartfast/vision-core) repository:

### 1. New Result Types

Add text generation result types in `core/result_types.hpp`:

```cpp
struct TextGeneration {
    std::string generated_text;
    int prompt_tokens{0};
    int completion_tokens{0};
    std::string finish_reason;  // "stop", "length", etc.
};

struct MultimodalGeneration : public TextGeneration {
    // Inherits text output fields, used when image input is provided
};
```

Update the `Result` variant:
```cpp
using Result = std::variant<
    Classification, Detection, InstanceSegmentation,
    OpticalFlow, VideoClassification, PoseEstimation,
    DepthEstimation, TextGeneration, MultimodalGeneration>;
```

### 2. New Task Types

Add to `TaskType` enum:
```cpp
enum class TaskType {
    // ... existing types ...
    TextGeneration,
    MultimodalGeneration
};
```

### 3. New Task Implementations

#### TextGenerationTask
- **Preprocess**: Minimal — pass text prompt as-is (no image resizing/normalization)
- **Postprocess**: Extract generated text from string tensor output
- **Input**: `std::string` text prompt (not `cv::Mat`)
- **Output**: `TextGeneration` result

#### MultimodalGenerationTask
- **Preprocess**: Encode image to base64, format with text prompt for the model's expected input format
- **Postprocess**: Same as TextGenerationTask
- **Input**: `cv::Mat` image + `std::string` text prompt
- **Output**: `MultimodalGeneration` result

### 4. TaskInterface Extension

The current `TaskInterface` assumes image-based input:
```cpp
virtual std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) = 0;
```

For LLM tasks, add a text-based preprocess overload:
```cpp
virtual std::vector<std::vector<std::string>> preprocessText(
    const std::string& text_prompt,
    const std::vector<cv::Mat>& images = {}) {
    throw std::runtime_error("Text preprocessing not supported for this task type");
}
```

### 5. TaskFactory Registration

Register new model types in `TaskFactory::createTaskInstance()`:
```cpp
// Text generation models
if (normalizedType == "vllm" || normalizedType == "llm" ||
    normalizedType == "llama" || normalizedType == "mistral" || ...)
    return std::make_unique<TextGenerationTask>(model_info);
```

### 6. ModelInfo Updates

Consider adding string-aware fields to `ModelInfo`:
```cpp
struct ModelInfo {
    // ... existing fields ...
    std::vector<std::string> output_datatypes;  // To detect BYTES outputs
    bool has_string_inputs{false};               // Quick check for LLM models
};
```

---

## Architecture: LLM vs Vision Pipeline

### Vision Pipeline (existing)
```
Image → task.preprocess() → triton.infer(bytes) → task.postprocess() → Result
```

### LLM Pipeline (new, text-only)
```
Text Prompt → Build sampling params → triton.inferText(strings) → Extract text → Result
```

### Multimodal Pipeline (future, requires vision-core changes)
```
Image + Text → task.preprocessText(prompt, images)
             → triton.inferText(strings)
             → task.postprocess() → MultimodalGeneration Result
```

---

## Limitations

1. **Streaming**: The current implementation does not support streaming responses.
   vLLM on Triton supports streaming via the `stream` input parameter, but
   tritonic currently waits for the complete response.

2. **Shared Memory**: String (BYTES) tensors are not compatible with shared memory
   inference. An error is raised if shared memory is requested with string inputs.

3. **Multimodal Image Input**: Full multimodal support (image + text → text)
   requires vision-core changes for image encoding and prompt formatting. The
   current tritonic implementation provides a placeholder for the image input.

4. **Chat/Conversation**: Multi-turn conversation support would require
   maintaining conversation history, which is not currently implemented.
