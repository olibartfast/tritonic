# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TritonIC is a C++20 application for running ML inference (object detection, instance segmentation, classification, video classification, pose estimation, optical flow, depth estimation) via the Nvidia Triton Inference Server. It communicates with Triton over HTTP or gRPC.

## Build Commands

**Prerequisites:** Set the Triton client libraries path (defaults to `./triton_client_libs/install`):
```bash
export TritonClientBuild_DIR=$(pwd)/triton_client_libs/install
# Extract libraries from Docker if not already present:
./scripts/docker/extract_triton_libs.sh
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
- `-DBUILD_TESTING=ON` — build unit and integration tests

**Build with tests:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)
```

**Run tests:**
```bash
cd build && ctest --output-on-failure
# Or run directly:
./build/tritonic_unit_tests
```

**Pre-commit hooks:**
```bash
./scripts/setup/pre_commit_setup.sh
```

## Running the Application

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

**Triton server (Docker):**
```bash
./scripts/docker/docker_triton_run.sh /path/to/model_repository 25.06 gpu
```

## Architecture

### Code Structure

```
src/
  main/client.cpp   — Entry point: parses CLI config, instantiates Triton + App, runs
  main/App.cpp      — Inference pipeline orchestration (image/video processing loop)
  triton/Triton.cpp — Concrete ITriton implementation (HTTP & gRPC, shared memory)
  triton/Triton.hpp — Triton class + SharedMemoryRegion + TritonClient union
  triton/ITriton.hpp — Abstract interface for Triton client (mockable for tests)
  triton/TritonModelInfo.hpp — Struct holding input/output names, shapes, types
include/
  App.hpp           — App class declaration
  CommonTypes.hpp   — Tensor struct (data + shape, elements as std::variant)
  common.hpp        — Common lightweight includes
tests/
  unit/             — GoogleTest unit tests
  mocks/            — MockTriton.hpp, MockFileSystem.hpp
  test_main.cpp     — GoogleTest main runner
```

### Data Flow

1. `client.cpp`: `ConfigManager` parses CLI → creates `Triton` (implements `ITriton`) → creates `App`
2. `App::run()`: connects to Triton, checks server liveness, verifies/loads model, fetches `TritonModelInfo`, creates task via `TaskFactory::createTaskInstance(model_type, model_info)`. For `VideoClassification`, extracts `num_frames_` from the 5D input shape `[B, T, C, H, W]`.
3. Processing is routed by task type:
   - **OpticalFlow**: `processImages()` pairs consecutive images
   - **VideoClassification** (images): `processImages()` sends all frames as one batch
   - **VideoClassification** (video): `processVideoClassification()` maintains a `std::deque<cv::Mat>` frame buffer of `num_frames_` depth; infers on every full sliding window
   - **All others** (video): `processVideo()` processes frame-by-frame
4. Core pipeline per call: `task_->preprocess(frames)` → `tritonClient_->infer(input_data)` → `task_->postprocess(size, tensors)`
5. Results are `std::variant<Detection, Classification, InstanceSegmentation, OpticalFlow, VideoClassification, PoseEstimation>` — App dispatches rendering via `std::holds_alternative<>`. Pose skeleton uses 16 COCO connections drawn by `drawPose()`.

### External Dependencies (CMake FetchContent)

- **vision-core** (`github.com/olibartfast/vision-core`) — all model pre/postprocessing and `TaskFactory`. New model types are added here, not in tritonic itself.

Fetched at CMake configure time from `master`. CMake auto-detects offline mode via ping and sets `FETCHCONTENT_FULLY_DISCONNECTED` accordingly.

### Local Infra Module (`src/infra/`)

Replaces the former `vision-infra` external dependency. Provides:
- `Logger.hpp/.cpp` — `Logger`, `ILogger`, `LoggerManager`, `LogLevel` in namespace `vision_infra::core`
- `Config.hpp` — `InferenceConfig` in namespace `vision_infra::config`
- `ConfigManager.hpp/.cpp` — `ConfigManager::LoadFromCommandLine()` using `cv::CommandLineParser`
- `infra.hpp` — aggregate include (use `#include "infra/infra.hpp"`)

### Protocol Handling

`Triton` holds a `union TritonClient` that is either an HTTP or gRPC client. Protocol is selected at construction time via `ProtocolType` enum. The app auto-corrects port 8000→8001 when gRPC is requested.

### Shared Memory

Optional shared memory support (`--shared_memory_type=system|cuda`) reduces copy overhead between client and Triton. System uses POSIX shm; CUDA uses GPU memory directly. Controlled via `SharedMemoryType` enum and `SharedMemoryRegion` struct in `Triton.hpp`.

### Testing Pattern

`ITriton` interface enables unit testing without a live Triton server. `tests/mocks/MockTriton.hpp` provides a GMock implementation. Unit tests live in `tests/unit/`, integration tests in `tests/integration/`.

## Model Type Tags

| Task | `--model_type` value(s) | Notes |
|---|---|---|
| **Object Detection** | `yolo` | YOLOv5–v9, v11, v12 (standard YOLO output) |
| | `yolov7e2e` | YOLOv7 exported with `--grid --end2end` (TRT only) |
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
| **Optical Flow** | `raft` | RAFT (pass two images as `img1.jpg,img2.jpg`) |
| **Pose Estimation** | `vitpose` | ViTPose (COCO 17 keypoints) |
| **Depth Estimation** | `depth_anything_v2` | Depth Anything V2 *(requires updated vision-core)* |

`TaskFactory::normalizeModelType()` strips spaces, hyphens, and underscores and lowercases before matching, so `torchvision-classifier` and `torchvision_classifier` are equivalent.

## Branch Strategy

Main branch is `master`. Active development branches (e.g., `feature/k8s`) merge into `master` via PR.
