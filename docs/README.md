# TritonIC Documentation

Everything under `docs/`, plus the per-model deployment and benchmark guides that
live next to the code they describe. Start from the
[project README](../README.md) for build and quick-start instructions.

## Guides

| Guide | What it covers |
| --- | --- |
| [Docker setup](guides/Docker_setup.md) | Installing Docker and the NVIDIA Container Toolkit |
| [Kubernetes setup](guides/Kubernetes_setup.md) | Deploying Triton and TritonIC on a cluster |
| [Shared memory](guides/SharedMemory.md) | POSIX and CUDA shared-memory transport between client and server |
| [Local CI checks](guides/Local_CI_Checks.md) | Reproducing the CI jobs before pushing |

## Running inference

| Document | What it covers |
| --- | --- |
| [Docker scripts](Docker_Scripts.md) | Container wrapper scripts, placeholder substitutions, model-type tags |
| [Chat backend](Chat_Backend.md) | OpenAI-compatible endpoints and the full flag reference |
| [Chat backend testing](Chat_Backend_Testing.md) | Test procedure for the chat backend |
| [Gemma models](Gemma_Models.md) | Running Gemma and compatible VLMs |

## Deployment and benchmarks

Deployment guides sit beside the model repositories they configure:

| Model | Deployment | Benchmark |
| --- | --- | --- |
| YOLO (detection) | [ensemble](../deploy/object_detection/yolo/ensemble/README.md) | — |
| YOLO26 (detection) | [ensemble](../deploy/object_detection/yolo26/ensemble/README.md) | — |
| YOLO26-seg | [ensemble](../deploy/instance_segmentation/yolo26/ensemble/README.md) | [benchmark](../benchmarks/yolo26-seg/README.md) |
| YOLO11-seg | `../deploy/instance_segmentation/yolo11/ensemble/` | [benchmark](../benchmarks/yolo11-seg/README.md) |
| RF-DETR-seg | [export and deploy](../deploy/instance_segmentation/rf-detr/README.md) | — |
| ViT classifier | [Python backends](../deploy/classifier/vit/README.md) | — |

See [deploy/README.md](../deploy/README.md) for the full directory layout.

## Project direction

| Document | What it covers |
| --- | --- |
| [Feature roadmap](ROADMAP.md) | Async inference, streaming, timeouts, metrics, batching |
| [Ensemble inference roadmap](ENSEMBLE_INFERENCE_ROADMAP.md) | GPU pre/postprocessing milestones for the YOLO ensembles |
| [Versioning](Versioning.md) | Release and version-bump policy |
| [Changelog](../CHANGELOG.md) | Released changes |

## Contributing

| Document | What it covers |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | Code structure, namespaces, conventions for contributors |
