# Architecture

This page describes the project structure and architecture. See the [project README](../README.md) for quick-start instructions.

## Project Structure

```
tritonic/
├── src/
│   ├── main/                     # Entry point (client.cpp), App, Logger, ConfigManager
│   ├── triton/                   # Triton client (Triton.hpp/.cpp, forwarding headers)
│   ├── chat/                     # OpenAI-compatible backend (ChatBackend, ChatSession)
│   └── common/                   # Shared forwarding headers
├── include/
│   ├── tritonic/                 # Canonical namespaced headers
│   │   ├── core/                 # types.hpp, interfaces.hpp
│   │   ├── triton/               # model_info.hpp, itriton.hpp, triton_backend.hpp
│   │   ├── chat/                 # ichat_backend.hpp
│   │   └── infra/                # logger.hpp, config.hpp, config_manager.hpp
│   └── *.hpp                     # Backward-compat forwarding headers
├── deploy/                       # Model export scripts (per task type)
├── scripts/                      # Docker, setup, and utility scripts
├── config/                       # Configuration files
├── docs/                         # Documentation and guides
├── labels/                       # Label files (COCO, ImageNet, etc.)
├── data/                         # Data files (images, videos, outputs)
└── tests/                        # Unit and integration tests
```

**CMake Fetched Dependencies:**
- [neuriplo-tasks](https://github.com/olibartfast/neuriplo-tasks) (pinned in [versions.env](../versions.env)) — model pre/post processing and task management

## Architecture

TritonIC selects an inference backend at startup via `--backend`:

| `--backend` | Requires | Best for |
|-------------|----------|----------|
| `triton` (default) | NVIDIA Triton server | CV tasks — detection, segmentation, classification, optical flow, pose, depth |
| `chat` | Any OpenAI-compatible server | LLMs, VLMs, multimodal chat |

The two modes are **not competing** — Triton handles binary tensor workloads at real-time throughput, while the Chat backend handles text/image generation over a REST API. Choose based on your model and server.

Both implement the common `tritonic::core::IInferenceBackend` interface (Strategy pattern), enabling clean dependency injection and unit testing without live servers.

> Note: "backend" here refers to *tritonic's* server selection (`--backend=triton` vs `--backend=chat`). This is distinct from the Triton server's own *framework backends* (TensorRT, ONNX Runtime, etc.), which are configured server-side.

For full code structure and namespace layout see [AGENTS.md](../AGENTS.md).

The YOLO DALI ensemble can accept original JPEG bytes and run decode,
letterboxing, normalization, and TensorRT inference on the server. See the
[reproducible GPU-preprocessing workflow](../deploy/object_detection/yolo/ensemble/README.md)
and its [atomic implementation roadmap](ENSEMBLE_INFERENCE_ROADMAP.md).
