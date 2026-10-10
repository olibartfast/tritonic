# Running Inference

This page covers all command-line options and inference modes. See the [project README](../README.md) for quick-start instructions.

## Command-line flags

| Flag | Alias | Default | Description |
|------|-------|---------|-------------|
| `--help` | `-h`, `-?` | — | Show help message |
| `--source` | `-s` | — | Path to input image/video file |
| `--model_type` | `-mt` | — | Type of model (yolov5, yolov8, etc.) |
| `--model` | `-m` | — | Model name on inference server |
| `--labelsFile` | `-lf` | — | Path to labels file |
| `--protocol` | `-p` | `http` | Protocol to use (http or grpc) |
| `--serverAddress` | `-sa` | `localhost` | Inference server address |
| `--port` | `-pt` | `8000` | Inference server port |
| `--input_sizes` | `-is` | — | Input sizes for dynamic axes (format: 'c,h,w;c,h,w') |
| `--input_mode` | `-im` | `preprocessed` | Input transport: preprocessed or encoded-image |
| `--task_model` | `-tm` | — | Inner Triton model used for task metadata in encoded-image mode |
| `--postprocess_mode` | `-pm` | `cpu` | Postprocessing placement: cpu or gpu |
| `--segmentation_output` | `-so` | `mask` | Segmentation representation: mask or polygon |
| `--preprocess_device` | `-pd` | `cpu` | Client preprocessing device: cpu or cuda |
| `--batch_size` | `-bs` | `1` | Batch size |
| `--inference_timeout` | `-it` | `0` | Inference timeout in milliseconds (0 = no timeout) |
| `--benchmark_warmup` | `-bw` | `0` | Benchmark warmup iterations |
| `--benchmark_iterations` | `-bi` | `0` | Measured benchmark iterations |
| `--benchmark_output` | `-bo` | — | Benchmark JSON output path |
| `--show_frame` | `-sf` | `false` | Show processed frames |
| `--write_frame` | `-wf` | `true` | Write processed frames to disk |
| `--confidence_threshold` | `-ct` | `0.5` | Confidence threshold |
| `--nms_threshold` | `-nt` | `0.4` | NMS threshold |
| `--verbose` | `-v` | `false` | Verbose output |
| `--shared_memory_type` | `-smt` | `none` | Shared memory type (none, system, cuda) |
| `--cuda_device_id` | `-cdi` | `0` | CUDA device ID for CUDA shared memory |
| `--log_level` | `-ll` | `info` | Log level (debug, info, warn, error) |
| `--log_file` | `-lf2` | — | Log file path (use `--log_file=path`) |
| `--enable_multimodal` | `-em` | `false` | Enable multimodal model support |
| `--text_input` | `-ti` | — | Path to text input file |
| `--audio_input` | `-ai` | — | Path to audio input file |
| `--text_prompt` | `-tp` | — | Text prompt for multimodal model |
| `--modality_combination` | `-mc` | `concat` | How to combine modalities |
| `--text_weight` | `-tw` | `1.0` | Weight for text modality |
| `--image_weight` | `-iw` | `1.0` | Weight for image modality |
| `--audio_weight` | `-aw` | `1.0` | Weight for audio modality |
| `--max_tokens` | `-mxt` | `256` | Max tokens for LLM generation |
| `--temperature` | `-temp` | `1.0` | Sampling temperature for LLM generation |
| `--top_p` | — | `1.0` | Top-p sampling for LLM generation |
| `--repetition_penalty` | `-rp` | `1.0` | Repetition penalty for LLM generation |
| `--stop_words` | `-sw` | — | Comma-separated stop words for LLM generation |
| `--backend` | `-be` | `triton` | Inference backend: triton or chat |
| `--api_endpoint` | `-ae` | — | Full URL for chat backend (e.g. http://localhost:11434/v1/chat/completions) |
| `--api_service` | `-as` | — | Chat API service preset (openai, openrouter, together, zai) |
| `--api_key_env` | `-ak` | — | Environment variable name containing the API key for chat backend |
| `--target_image_size` | `-tis` | `512` | Image resize size (px) before base64 encoding for chat backend |
| `--interactive` | `-ia` | `false` | Enable multi-turn interactive chat session (--backend=chat only) |

## Running Inference

### Command-Line Inference on Video or Image
```bash
./tritonic \
    --source=/path/to/source.format \
    --model_type=<model_type> \
    --model=<model_name_folder_on_triton> \
    --labelsFile=/path/to/labels/coco.names \
    --protocol=<http or grpc> \
    --serverAddress=<triton-ip> \
    --port=<8000 for http, 8001 for grpc>
```

For dynamic input sizes:
```bash
    --input_sizes="c,h,w"
```

### Shared Memory Support

Tritonic supports shared memory to improve inference performance by reducing data copying between the client and Triton server. Two types of shared memory are available:

> Since Triton 26.08 the server rejects client shared-memory registration unless it runs with `--allow-client-shm=true` (with `docker_triton_run.sh`: `ALLOW_CLIENT_SHM=true`). CUDA shared memory also needs the client and server on the same host.

#### System (POSIX) Shared Memory
Uses CPU-based shared memory for efficient data transfer:
```bash
./tritonic \
    --source=/path/to/source.format \
    --model=<model_name> \
    --shared_memory_type=system \
    ...
```

#### CUDA Shared Memory
Uses GPU memory directly for zero-copy inference (requires GPU support):
```bash
./tritonic \
    --source=/path/to/source.format \
    --model=<model_name> \
    --shared_memory_type=cuda \
    --cuda_device_id=0 \
    ...
```

**Configuration Options:**
- `--shared_memory_type` or `-smt`: Shared memory type (`none`, `system`, or `cuda`). Default: `none`
- `--cuda_device_id` or `-cdi`: CUDA device ID when using CUDA shared memory. Default: `0`

### Ensemble models

Triton ensembles move preprocessing, and optionally postprocessing, onto the GPU: the
client sends the encoded JPEG and the server runs DALI decode/resize/normalize, TensorRT
inference, and — for the `gpu_pre_gpu_*` variants — a custom CUDA postprocess operator.

| Ensemble | Task | Deployment guide |
| --- | --- | --- |
| `yolo_dali_ensemble` | Detection, GPU preprocess only | [YOLO ensemble](../deploy/object_detection/yolo/ensemble/README.md) |
| `yolo26det_*` | Detection, GPU pre/post | [YOLO26 detection ensemble](../deploy/object_detection/yolo26/ensemble/README.md) |
| `yolo26seg_*` | Instance segmentation, GPU pre/post | [YOLO26 segmentation ensemble](../deploy/instance_segmentation/yolo26/ensemble/README.md) |
| `yolo11seg_*` | Instance segmentation, GPU pre/post | [YOLO11 segmentation ensemble](../deploy/instance_segmentation/yolo11/ensemble/README.md) |

For YOLO26 detection there are also non-DALI GPU preprocessing options — a custom CUDA backend, a TensorRT-graph letterbox, and client-side CUDA (`--preprocess_device=cuda`). They were validated on an A10G; DALI stays the server-side default and client-side CUDA is fastest when the client shares the GPU host (3.1 ms vs 9.3 ms per request). See the [variants section](../deploy/object_detection/yolo26/ensemble/README.md#non-dali-gpu-preprocessing-variants) and the [comparison report](../specs/2026-10-01-gpu-preprocess-alternatives/evidence/2026-10-09-brev-a10g/comparison.md).

Instance-segmentation tasks return raster masks by default. Select convex-hull polygon rings
with `--segmentation_output=polygon`; exteriors and holes are returned in image
coordinates. `--postprocess_mode=gpu` supports both representations
through separate encoded-image ensembles, so mask-only requests do not run polygon extraction.
The `yolo26seg` examples below apply unchanged to `yolo11seg` by substituting the model names.

GPU mask output (default):

```bash
./build/tritonic \
  --source=data/images/bus.jpg \
  --model_type=yolo26seg \
  --model=yolo26seg_gpu_pre_gpu_mask_post \
  --task_model=yolo26seg_trt \
  --labelsFile=labels/coco.txt \
  --input_mode=encoded-image \
  --postprocess_mode=gpu
```

GPU polygon output:

```bash
./build/tritonic \
  --source=data/images/bus.jpg \
  --model_type=yolo26seg \
  --model=yolo26seg_gpu_pre_gpu_post \
  --task_model=yolo26seg_trt \
  --labelsFile=labels/coco.txt \
  --input_mode=encoded-image \
  --postprocess_mode=gpu \
  --segmentation_output=polygon
```

The same polygon ensemble accepts video sources. Tritonic JPEG-encodes each
decoded frame in memory for DALI preprocessing and writes the rendered result to
`<video-directory>/output/processed_<model-name>.avi`:

```bash
./build/tritonic \
  --source=/path/to/input.mp4 \
  --model_type=yolo26seg \
  --model=yolo26seg_gpu_pre_gpu_post \
  --task_model=yolo26seg_trt \
  --labelsFile=labels/coco.txt \
  --input_mode=encoded-image \
  --postprocess_mode=gpu \
  --segmentation_output=polygon
```

Deployment instructions and both tensor ABIs are documented in the
[YOLO26 segmentation ensemble guide](../deploy/instance_segmentation/yolo26/ensemble/README.md).
The reproducible CPU/GPU benchmark and semantic parity gate are in the
[YOLO26-seg benchmark](../benchmarks/yolo26-seg/README.md).

### Running with the Docker scripts

Wrapper scripts run TritonIC in a container without a local build. The full
placeholder and model-type-tag reference is in
[Running Inference with the Docker Scripts](Docker_Scripts.md).

### Chat Backend (OpenAI-compatible)

Skip Triton entirely and query any OpenAI-compatible server — Ollama, llama.cpp,
SGLang, vLLM, OpenAI, Together AI, OpenRouter, Z.AI:

```bash
./build/tritonic --backend=chat --api_endpoint=http://localhost:11434/v1/chat/completions \
  --model=gemma3:4b --text_prompt="Describe this image" --source=data/images/bus.jpg
```

Full flag reference in [Chat Backend](Chat_Backend.md); the test procedure is
in [Chat Backend Testing](Chat_Backend_Testing.md).
