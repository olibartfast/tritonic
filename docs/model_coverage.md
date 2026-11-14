# Coverage matrix — task models × inference paths

Snapshot: 2026-11-14. Default task dependency: `neuriplo-tasks v0.8.2`, as
pinned in [versions.env](../versions.env). Local CMake overrides can change
the available tasks.

This matrix follows the evidence-oriented structure of the
[NERT coverage document](https://github.com/olibartfast/nert/blob/a5d426082a9f50b21e716ed09212a74a646d66a2/specs/2026-10-17-task-models/coverage.md).
TritonIC is a client: model execution belongs to Triton or a chat endpoint.
Consequently, coverage is tracked by client/server path and recorded model
validation, rather than by runtime kernel or device. A CPU preprocessing path
can still execute the model on a server GPU.

## Definitions

| Status | Meaning |
| --- | --- |
| REGISTERED | The pinned task factory recognizes the tag; this alone does not establish a working Triton deployment. |
| PASS | A linked, checked-in evidence packet records a passing gate for the stated model, fixtures, environment, and path. It is historical evidence, not a fresh run of this snapshot. |
| TESTED | Maintainer-confirmed manual inference testing, with configurations recorded in `.vscode/launch.json`; no archived numerical gate is required for this status. |
| REPORTED | A checked-in guide reports validation, but the raw run artifacts are not tracked in the repository. |
| UNRUN | No recorded full-model gate was identified for the stated scope. This does not assert that the model has never been run. |
| GAP | A concrete client integration requirement is missing. |
| FAIL | A recorded gate failed its declared criterion; retain the failure and any later amended criterion in the evidence. |
| OUT-OF-SCOPE | The path does not apply to this backend or task. |

Deployment scripts, unit tests, and upstream runtime gates are separate evidence
from a full TritonIC inference run. Model families below group aliases and sizes;
one validated size does not validate every size or export in the family.

## Task and model inventory

Tags are checked against the pinned
[TaskFactory](https://github.com/olibartfast/neuriplo-tasks/blob/v0.8.2/src/core/task_factory.cpp)
and its [model catalog](https://github.com/olibartfast/neuriplo-tasks/blob/v0.8.2/README.md).
Matching lowercases and removes whitespace, hyphens, and underscores.
All rows except chat are REGISTERED upstream; the last column describes
TritonIC integration or validation, not upstream test results.

| Task | Model family | Representative `--model_type` | TritonIC coverage |
| --- | --- | --- | --- |
| Detection | YOLOv5–v9, YOLO11, YOLOv12 | `yolo` | TESTED for the YOLOv5/v6/v7/v8/v9/v12 exports listed below; YOLO11 detection remains UNRUN. |
| Detection | YOLOv4 | `yolov4` | UNRUN |
| Detection | YOLOv7 end-to-end | `yolov7e2e` | TESTED for `yolov7_e2e_trt`. |
| Detection | YOLOv10 | `yolov10` | TESTED for `yolov10n_onnx`. |
| Detection | YOLO26 | `yolo26` | PASS for the A10G paths in the evidence table below. |
| Detection | YOLO-NAS | `yolonas` | UNRUN |
| Detection | RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2 | `rtdetr`, `rtdetrv2`, `dfine`, `deim`, `deimv2` | TESTED for RT-DETR v1/v4, D-FINE, DEIM, and DEIMv2 configurations below; v2 remains UNRUN. Use `rtdetr` for the compatible v4 contract. |
| Detection | Ultralytics RT-DETR | `rtdetrul`, `rtdetrultralytics` | UNRUN |
| Detection | RF-DETR | `rfdetr` | TESTED for `rfdetr_base`. |
| Detection | EdgeCrafter | `ecdet`, `edgecrafter-det` | UNRUN |
| Instance segmentation | YOLOv5/v8/11/v12 | `yoloseg`, `yolo11seg` | TESTED for YOLOv5/v8/11/12 segmentation configurations below; [YOLO11 benchmark harness](../benchmarks/yolo11-seg/README.md) also exists. |
| Instance segmentation | YOLOv10 | `yolov10seg` | UNRUN |
| Instance segmentation | YOLO26 | `yolo26seg` | TESTED for `yolo26m_seg_onnx`; also REPORTED for TensorRT YOLO26m-seg on RTX 3060 Laptop. |
| Instance segmentation | RF-DETR | `rfdetrseg` | TESTED for `rf-detr-seg-preview_onnx`; [export/deployment scripts](../deploy/instance_segmentation/rf-detr/README.md) also exist. |
| Instance segmentation | EdgeCrafter | `ecseg`, `edgecrafter-seg` | UNRUN |
| Classification | Torchvision, including ResNet | `torchvision-classifier`, `resnet50` | UNRUN |
| Classification | TensorFlow/Keras | `tensorflow-classifier` | TESTED for `resnet50_tensorflow`. |
| Classification | Vision Transformers | `vit-classifier` | TESTED for `vit_onnx`, `vit_standard`, and `vit_pipeline`. |
| Video classification | VideoMAE, ViViT, TimeSformer | `videomae`, `vivit`, `timesformer` | UNRUN; temporal frame-buffer path implemented. |
| Optical flow | RAFT | `raft` | TESTED for RAFT-large ONNX image pairs and TorchScript video configurations. |
| Pose estimation | YOLOv5/v8/11/26 pose | `yolov5pose`, `yolov8pose`, `yolo11pose`, `yolo26pose` | UNRUN |
| Pose estimation | ViTPose | `vitpose` | UNRUN; top-down heatmap contract. |
| Pose estimation | RF-DETR keypoints | `rfdetrpose`, `rfdetrkeypoint`, `rfdetrkpt` | UNRUN |
| Pose estimation | EdgeCrafter | `ecpose`, `edgecrafter-pose` | UNRUN |
| Depth estimation | Depth Anything V2 | `depth_anything_v2` | UNRUN |
| Depth estimation | YOLO26 depth | `yolo-depth`, `yolo26n-depth` | UNRUN |
| Open-vocabulary detection | OWLv2, OWL-ViT, Grounding DINO | `owlv2`, `owlvit`, `groundingdino` | GAP: App does not populate task prompts or tokenizer settings. |
| Gaussian splatting | LGM, LGM-mini, GRM | `lgm`, `lgm-mini`, `grm`, `gaussiansplatting` | UNRUN; upstream registration only, no dedicated TritonIC deployment or full-model gate identified. |
| Image understanding via task tensors | Gemma/llama.cpp task adapter | `gemma4`, `gemma`, `llamacpp`, `imageunderstanding` | GAP: upstream custom tensor contract is not the OpenAI chat path; no corresponding TritonIC deployment/prompt wiring established. |
| Chat / image understanding | Endpoint-provided LLMs and VLMs | Not used: select `--backend=chat --model=<endpoint-model>` | UNRUN per model/endpoint; [chat integration tests](../tests/integration/test_chat_backend_integration.cpp) are conditional. |

## Manually tested launch configurations

The maintainer confirmed on 2026-11-14 that all models in
[launch.json](../.vscode/launch.json) have already been tested. The following
configurations therefore have TESTED status. This records historical manual
inference testing; the confirmation date is not the execution date. Hardware,
server versions, numerical tolerances, and run logs were not supplied with this
confirmation. Protocols below come from explicit arguments or the CLI default.
The duplicate RT-DETR configuration is counted once.

The server backend is selected by the deployed model's Triton `config.pbtxt`;
TritonIC's `--backend=triton` selects the client connection path. Backend labels
marked “inferred” follow the model-name suffix and need confirmation against the
server configuration. Models without a backend suffix remain unspecified.
The Python entries follow the [ViT deployment guide](../deploy/classifier/vit/README.md).
Coverage applies to each model/backend combination: testing an ONNX Runtime
export does not establish coverage for its TensorRT or Python counterpart.

| Server model | Model type | Triton server backend | Source | Protocol |
| --- | --- | --- | --- | --- |
| `yolov5x-seg_onnx` | `yoloseg` | ONNX Runtime (inferred) | `data/images/person.jpg` | http (default) |
| `yolov8x-seg_onnx` | `yoloseg` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolo11-seg_onnx` | `yoloseg` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolo12-seg_onnx` | `yoloseg` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov5x_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov6l_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov7_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/bus.jpg` | http (default) |
| `yolov7_e2e_trt` | `yolov7e2e` | TensorRT (inferred) | `data/images/bus.jpg` | http (default) |
| `yolov8m_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov8x_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/dog.jpg` | grpc |
| `yolov9e_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov9c_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov10n_onnx` | `yolov10` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `yolov12s_onnx` | `yolo` | ONNX Runtime (inferred) | `data/images/bus.jpg` | http (default) |
| `resnet50_tensorflow` | `tensorflow-classifier` | TensorFlow (inferred) | `data/images/mug.jpg` | http (default) |
| `d-fine_s_coco` | `dfine` | Unspecified | `data/images/person.jpg` | http (default) |
| `deim_dfine_hgnetv2_s_coco` | `deim` | Unspecified | `data/images/person.jpg` | http (default) |
| `deimv2_dinov3_s_coco_onnx` | `rtdetr` | ONNX Runtime (inferred) | `data/images/dog.jpg` | http (default) |
| `rtdetr_r18vd_6x_coco` | `rtdetr` | Unspecified | `data/images/bus.jpg` | http (default) |
| `rtdetrv4_hgnetv2_s_onnx` | `rtdetr` | ONNX Runtime (inferred) | `data/images/bus.jpg` | http (default) |
| `rfdetr_base` | `rfdetr` | Unspecified | `data/images/bus.jpg` | http (default) |
| `rf-detr-seg-preview_onnx` | `rfdetrseg` | ONNX Runtime (inferred) | `data/images/bus.jpg` | http (default) |
| `yolo26m_seg_onnx` | `yolo26seg` | ONNX Runtime (inferred) | `data/images/bus.jpg` | grpc |
| `raft_large_torchscript` | `raft` | PyTorch / LibTorch (inferred) | `/home/oli/Videos/12408393_3840_2160_24fps.mp4` | http (default) |
| `raft_large_onnx` | `raft` | ONNX Runtime (inferred) | `'data/images/raft/frame_001.png,data/images/raft/frame_002.png'` | http (default) |
| `vit_onnx` | `vit-classifier` | ONNX Runtime (inferred) | `data/images/cat.jpeg` | http (default) |
| `vit_standard` | `vit-classifier` | Python | `data/images/cat.jpeg` | http (default) |
| `vit_pipeline` | `vit-classifier` | Python | `data/images/cat.jpeg` | http (default) |

## Inference path coverage

Implementation constraints come from [App.cpp](../src/main/App.cpp) and
[ConfigManager.cpp](../src/main/ConfigManager.cpp).

| Path | Applicable models | Constraints and evidence |
| --- | --- | --- |
| Client CPU preprocessing → Triton → client postprocessing | Registered tensor tasks with a compatible server contract | HTTP and gRPC implemented. Per-model evidence remains as listed above. Server CPU/GPU placement is configured in Triton. |
| Independent-image batching | Classification, detection, instance segmentation, pose, depth, open-vocabulary detection | Requires server `max_batch_size > 1`; request capped by `--batch_size` and server maximum. UNRUN per model; upstream batching helpers do not establish server/export compatibility. |
| Temporal / multi-input processing | RAFT, VideoMAE, ViViT, TimeSformer | Dedicated image-pair or frame-buffer paths; excluded from independent-image batching. Gaussian splatting is also excluded from that batching path. |
| Encoded JPEG → server preprocessing ensemble | `yolo`, `yolo26`, `yoloseg`, `yolo11seg`, `yolo26seg` | Requires `--input_mode=encoded-image`, `--task_model`, batch size 1, no shared memory, and no `--input_sizes`. |
| Server GPU postprocessing | Same encoded-image tags | Requires `--postprocess_mode=gpu` and a matching detection, mask, or polygon output contract. CLI acceptance alone does not supply an ensemble for every export. |
| Client CUDA preprocessing → CUDA shared memory | `yolo`, `yolo26` | Requires CUDA build, `--preprocess_device=cuda`, preprocessed input, CUDA shared memory, batch size 1, no `--input_sizes`, and co-located client/server GPU. A10G evidence below. |
| OpenAI-compatible chat | Models served by the configured endpoint | Single-turn and interactive sessions; image support depends on the endpoint/model. Triton tensor batching and shared memory are OUT-OF-SCOPE. |

Open-vocabulary tasks are eligible for batching by task type, but their prompt
integration GAP still applies. `--text_prompt` used by chat does not populate
`TaskConfig::text_prompts` in the Triton App path.

## Recorded model gates

| Model / environment | Triton server backend | Paths | Status | Evidence and limits |
| --- | --- | --- | --- | --- |
| YOLO26 detection, NVIDIA A10G, Triton 26.08, TensorRT 11.2.1.2, FP32 | TensorRT; DALI or custom CUDA preprocessing backend for ensemble variants | V2 client CUDA; V12 custom CUDA backend with CPU/GPU postprocessing; V13 TensorRT preprocessing with CPU/GPU postprocessing; HTTP and gRPC | PASS | [Comparison and gate results](../specs/2026-10-01-gpu-preprocess-alternatives/evidence/2026-10-09-brev-a10g/comparison.md). Six fixtures, relative to B0/B1 baselines; not an independent framework accuracy gate. Tensor parity was not run for V13; soak was not run for every variant. |
| YOLO26m-seg, RTX 3060 Laptop, FP16 TensorRT 10.14.1 | TensorRT; DALI for ensemble variants | CPU baseline, DALI preprocessing, DALI polygon postprocessing | REPORTED | [Polygon benchmark report](../benchmarks/yolo26-seg/README.md): four fixtures, box IoU ≥ 0.95, polygon IoU ≥ 0.90; identical polygons for GPU vs CPU postprocessing on the same DALI tensors. Raw results directory is not tracked. |
| YOLO26m-seg, same RTX 3060 environment | TensorRT; DALI for ensemble variants | CPU baseline, DALI preprocessing, DALI packed-mask postprocessing | REPORTED | Same report: mask IoU ≥ 0.90; GPU/CPU postprocessing mask IoU 1.0 on the same DALI tensors. Raw results directory is not tracked. |
| Maintainer-confirmed launch models | See launch table | Configurations listed above | TESTED | Manual inference testing confirmed by the maintainer; archived numerical gate results are not available. |
| Remaining inventory entries | Unspecified | Model-specific full inference | UNRUN / GAP | No manual-test confirmation or passing full-model packet identified for these entries. |

The A10G gates initially failed at confidence tolerance 0.03; the accepted
tolerance became 0.04 after baseline decoder differences were measured. Decode
parity also changed its reference from OpenCV to DALI. The original failures
and rationale remain in the [validation record](../specs/2026-10-01-gpu-preprocess-alternatives/validation.md).
The [recommendation](../specs/2026-10-01-gpu-preprocess-alternatives/evidence/2026-10-09-brev-a10g/recommendation.md)
retains DALI as default, offers V2 for co-located clients and V12 as a DALI-free
alternative, and recommends retiring V13 despite its passing detection gates.

## Coverage gaps and CI limits

- Wire prompts and tokenizer assets into the Triton task configuration before
  claiming general open-vocabulary inference coverage.
- Record model/export-specific gates for the UNRUN families, including multiple
  image sizes, nontrivial batch sizes, temporal inputs, and both transports.
- Preserve raw segmentation benchmark packets. The YOLO11 guide names a model
  used in runs but provides no tracked gate results; the YOLO26 reports describe
  an older TensorRT environment and do not establish current-version parity.
- Add persistent GPU validation: [current CI](../.github/workflows/README.md)
  builds and runs unit tests without a GPU runner, deployed ensembles, or the
  `integration-tests/` and benchmark suites.
- Chat integration tests skip unless `CHAT_API_ENDPOINT` and `CHAT_MODEL` are
  set. Their image-inference example is commented out; a green default test run
  does not establish text or vision model coverage against a live endpoint.

## Updating this matrix

When the dependency pin changes, recheck the factory aliases and the App's task
configuration. New model preprocessing and postprocessing belong in
`neuriplo-tasks`; deployment and client integration evidence belong here.

For each new PASS, record the client commit, dependency version, export/engine
hash, server image digest, device, precision, protocol, CLI invocation, fixture
hashes, reference, tolerances, measured results, and exit status in a linked
evidence packet. Separate raw tensor parity from task-level comparisons such as
box IoU, mask IoU, classification agreement, or keypoint error. Record failures,
skips, and approved tolerance changes explicitly. Do not extend a gate to other
model sizes, devices, precisions, protocols, or dependency versions without
evidence for those combinations.
