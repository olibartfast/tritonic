# Tritonic — Agent Guidelines

TritonIC is a C++20 application for running ML inference (object detection, instance segmentation, classification, video classification, pose estimation, optical flow, depth estimation) via the Nvidia Triton Inference Server **or** any OpenAI-compatible chat endpoint. It communicates with Triton over HTTP or gRPC, and with chat servers via `/v1/chat/completions`.

## Build and Test

**Prerequisites:**
```bash
export TritonClientBuild_DIR=$(pwd)/triton_client_libs/install
./docker/scripts/extract_triton_libs.sh   # extract from Docker if not present
```

**Standard build:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

**CMake options:**
- `-DWITH_SHOW_FRAME=ON` — display frames in a window during inference
- `-DWITH_WRITE_FRAME=ON` — write output frames/video to disk (default ON)
- `-DWITH_CHAT_BACKEND=ON` — OpenAI-compatible chat backend (default ON)
- `-DBUILD_TESTING=ON` — build unit and integration tests

**Build and run tests:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)
ctest --output-on-failure
# or: ./build/tritonic_unit_tests
```

**Code formatting and git hooks:**
```bash
# Install git hook for automatic formatting on commit (RECOMMENDED)
./scripts/setup/install_git_hooks.sh

# Or manually format all files before committing
./scripts/format_all.sh

# Or use the legacy pre-commit setup (requires Python)
./scripts/setup/pre_commit_setup.sh
```

> **Important:** Install the git hook to avoid CI format check failures. The hook automatically formats C++ files before each commit.

## Architecture

### Dual-backend design

| Backend | CLI flag | Use case |
|---------|----------|----------|
| **Triton** (default) | `--backend=triton` | Binary tensor inference — detection, seg, classification, optical flow, pose, depth |
| **Chat** | `--backend=chat` | OpenAI `/v1/chat/completions` — VLMs, LLMs, multimodal chat |

Both backends implement `tritonic::core::IInferenceBackend` (Strategy pattern), enabling dependency injection and mockability in tests.

### Code structure

```
src/
  main/client.cpp          Entry point: parses CLI, selects backend, runs
  main/App.cpp             Inference pipeline (image/video loop, rendering)
  main/Logger.cpp          Logger implementation (namespace tritonic::infra)
  main/ConfigManager.cpp   CLI parsing via cv::CommandLineParser (namespace tritonic::infra)
  triton/Triton.hpp/.cpp   Concrete ITriton (HTTP & gRPC, shared memory)
  triton/ITriton.hpp       Forwarding header → tritonic/triton/itriton.hpp
  triton/TritonModelInfo.hpp  Forwarding header → tritonic/triton/model_info.hpp
  triton/TritonBackend.hpp Forwarding header → tritonic/triton/triton_backend.hpp
  chat/ChatBackend.hpp/.cpp  OpenAI-compatible CURL facade (dual-inherits IChatBackend + IInferenceBackend)
  chat/ChatSession.hpp/.cpp  Stateful multi-turn manager (sliding window, pinned context)
  chat/IChatBackend.hpp    Forwarding header → tritonic/chat/ichat_backend.hpp
  common/IInferenceBackend.hpp  Forwarding header → tritonic/core/interfaces.hpp
include/
  tritonic/core/types.hpp       Tensor, Message, ChatRequest/Response, BackendRequest/Response
  tritonic/core/interfaces.hpp  IInferenceBackend
  tritonic/triton/model_info.hpp   ModelInfo (namespace tritonic::triton)
  tritonic/triton/itriton.hpp      ITriton
  tritonic/triton/triton_backend.hpp  TritonBackend adapter
  tritonic/chat/ichat_backend.hpp  IChatBackend
  tritonic/infra/logger.hpp        Logger, LoggerManager, LogLevel
  tritonic/infra/config.hpp        InferenceConfig
  tritonic/infra/config_manager.hpp  ConfigManager
  App.hpp / CommonTypes.hpp / Logger.hpp / Config.hpp / ConfigManager.hpp
    ↑ backward-compat forwarding headers (kept for existing code)
tests/
  unit/            GoogleTest unit tests
  mocks/           MockTriton.hpp, MockChatBackend.hpp
  test_main.cpp    GoogleTest main runner
```

### Namespace layout

| Namespace | Headers | Contents |
|-----------|---------|----------|
| `tritonic::core` | `include/tritonic/core/` | `Tensor`, `Message`, `ChatRequest/Response`, `IInferenceBackend` |
| `tritonic::triton` | `include/tritonic/triton/` | `ITriton`, `ModelInfo`, `TritonBackend` |
| `tritonic::chat` | `include/tritonic/chat/` | `IChatBackend` |
| `tritonic::infra` | `include/tritonic/infra/` | `InferenceConfig`, `ConfigManager`, `Logger` |

Old flat `include/*.hpp` and `src/*/` headers are forwarding headers with backward-compat `using` aliases — do not revert them to full definitions.

### Data flow (Triton backend)

1. `client.cpp`: `ConfigManager::LoadFromCommandLine()` → creates `Triton` → creates `App`
2. `App::run()`: connects, checks liveness, loads model, fetches `ModelInfo`, creates task via `TaskFactory::createTaskInstance()`
3. Routing by task type: `OpticalFlow` → `processImages()` with pairs; `VideoClassification` → `processVideoClassification()` with deque frame buffer; others → `processVideo()` or `processImages()`
4. Core loop: `task_->preprocess(frames)` → `tritonClient_->infer(input_data)` → `task_->postprocess(size, tensors)`

### Batched image inference (neuriplo-tasks ≥ v0.5.0)

`processImages()` runs the per-image loop by default. When the task is an
independent-image family (classification, detection, instance segmentation,
pose, depth, open-vocab) **and** `ModelInfo.max_batch_size_ > 1`, it dispatches
to `processImagesBatched()`, which uses the neuriplo-tasks Track B helpers:

- `batchPreprocess(*task_, {images})` → per-image buffers + `batch_size`
- `stackBatchBuffers()` concatenates per-image buffers into one Triton input per
  node (Pattern A: classification/YOLO/RF-DETR/depth/pose) or passes the
  already-stacked buffers through (Pattern B: RT-DETR/EdgeCrafter/open-vocab)
- `applyBatchedInputShapes(N)` sets `input_shapes[i][0] = N` via
  `ITriton::setInputShapes()` before a single batched `infer()` call, then
  restores `N = 1` so the subsequent video path stays single-image
- `postprocessBatched()` maps results back per image: classification uses
  `batchPostprocess()` (strict 1:1); variable-count/spatial families slice the
  output tensors along axis 0 and `postprocess()` each image at its own frame size

Temporal/multi-input tasks (optical flow, video classification, gaussian
splatting) are **not** batched — see `neuriplo-tasks` `docs/batch_support_matrix.md`.

### Data flow (Chat backend)

- `client.cpp` instantiates `ChatBackend(endpoint, api_key)`
- Single-turn: builds `ChatRequest{messages:[user_msg]}`, calls `chatBackend->infer(req)`
- Interactive: `ChatSession` accumulates history, calls `session.send(text, images, model, max_tokens)` per turn

### External dependencies (CMake FetchContent)

- **neuriplo-tasks** (`github.com/olibartfast/neuriplo-tasks`) — all model pre/postprocessing and `TaskFactory`. Add new model types there, not in tritonic.

CMake auto-detects offline mode via ping and sets `FETCHCONTENT_FULLY_DISCONNECTED` accordingly.

### Protocol / shared memory

`Triton` holds a `union TritonClient` (HTTP or gRPC). Auto-corrects port 8000→8001 for gRPC. Optional shared memory (`--shared_memory_type=system|cuda`) controlled via `SharedMemoryType` enum in `Triton.hpp`.

### Testing pattern

`ITriton` and `IChatBackend` interfaces enable unit tests without live servers. `tests/mocks/MockTriton.hpp` and `tests/mocks/MockChatBackend.hpp` provide GMock implementations.

## Hyperlink verification

When editing `README.md` or any documentation with hyperlinks:
- Verify all relative links resolve to existing files in the repo.
- Verify absolute GitHub URLs are reachable (use `curl -sI <url>` or a quick fetch).
- Prefer absolute GitHub blob/tree URLs over fragile cross-repo relative paths.

## Conventions

- **C++20**, `CMAKE_CXX_STANDARD 20`
- New code goes in the appropriate `tritonic::` namespace; update the matching `include/tritonic/` header
- New tasks/model types belong in **neuriplo-tasks**, not here
- `BackendRequest = std::variant<TritonInferRequest, ChatRequest>` — use `std::get_if` / `std::visit` to dispatch
- `ChatBackend` has no `nlohmann/json` or `cpp-base64` dependency — keep it that way
- `ChatSession::send()` appends to history **only on success**

## Running the Application

**Triton inference:**
```bash
./build/tritonic \
    --source=/path/to/image_or_video \
    --model_type=<type> \
    --model=<model_name_on_triton> \
    --labelsFile=/path/to/labels.names \
    --protocol=<http|grpc> \
    --serverAddress=<triton-ip> \
    --port=<8000|8001>
```

For dynamic input sizes: `--input_sizes="c,h,w"` (e.g., `"3,640,640"` or `"3,640,640;2"` for two inputs).

**Chat backend — single turn:**
```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:11434/v1/chat/completions \
    --model=llava:7b \
    --text_prompt="Describe what you see" \
    --source=/path/to/image.jpg
```

**Chat backend — interactive REPL:**
```bash
./build/tritonic \
    --backend=chat \
    --api_endpoint=http://localhost:11434/v1/chat/completions \
    --model=llava:7b \
    --interactive
```

**Triton server (Docker):**
```bash
./docker/scripts/docker_triton_run.sh /path/to/model_repository 25.06 gpu
```

## Model Type Tags

| Task | `--model_type` | Notes |
|------|----------------|-------|
| **Object Detection** | `yolo` | YOLOv5–v9, v11, v12 |
| | `yolov7e2e` | YOLOv7 `--grid --end2end` (TRT only) |
| | `yolov10`, `yolo26` | Same output format |
| | `yolonas` | YOLO-NAS |
| | `yolov4` | YOLOv4 |
| | `rtdetr` | RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2 |
| | `rtdetrul` | RT-DETR (Ultralytics) |
| | `rfdetr` | RF-DETR |
| **Instance Segmentation** | `yoloseg` | YOLOv5/v8/v11/v12 seg |
| | `yolov10seg` | YOLOv10 seg |
| | `yolo26seg` | YOLO26 seg |
| | `rfdetrseg` | RF-DETR seg |
| **Classification** | `torchvision-classifier` | ResNet, EfficientNet, etc. |
| | `tensorflow-classifier` | TF/Keras models |
| | `vit-classifier` | Vision Transformers |
| **Video Classification** | `videomae` | VideoMAE (16-frame default) |
| | `vivit` | ViViT |
| | `timesformer` | TimeSformer |
| **Optical Flow** | `raft` | Pass two images as `img1.jpg,img2.jpg` |
| **Pose Estimation** | `vitpose` | ViTPose (COCO 17 keypoints) |
| **Depth Estimation** | `depth_anything_v2` | Depth Anything V2 |
| **Open Vocabulary Detection** | `owlv2` | OWLv2 open-vocabulary detection |
| | `owlvit` | OWL-ViT open-vocabulary detection |

`TaskFactory::normalizeModelType()` strips spaces, hyphens, underscores and lowercases before matching.

## Branch Strategy

Main branch: `master`. Active development on `feature/*` branches, merge via PR.
