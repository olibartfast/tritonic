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
| Detection | YOLOv5–v9, YOLO11, YOLOv12 | `yolo` | UNRUN at family scope; [YOLO ensemble tooling](../deploy/object_detection/yolo/ensemble/README.md) and preprocessing checks exist. |
| Detection | YOLOv4 | `yolov4` | UNRUN |
| Detection | YOLOv7 end-to-end | `yolov7e2e` | UNRUN; requires the matching end-to-end export. |
| Detection | YOLOv10 | `yolov10` | UNRUN |
| Detection | YOLO26 | `yolo26` | PASS for the A10G paths in the evidence table below. |
| Detection | YOLO-NAS | `yolonas` | UNRUN |
| Detection | RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2 | `rtdetr`, `rtdetrv2`, `dfine`, `deim`, `deimv2` | UNRUN; use `rtdetr` for the compatible v4 contract, not an assumed `rtdetrv4` alias. |
| Detection | Ultralytics RT-DETR | `rtdetrul`, `rtdetrultralytics` | UNRUN |
| Detection | RF-DETR | `rfdetr` | UNRUN |
| Detection | EdgeCrafter | `ecdet`, `edgecrafter-det` | UNRUN |
| Instance segmentation | YOLOv5/v8/11/v12 | `yoloseg`, `yolo11seg` | UNRUN; [YOLO11 benchmark harness](../benchmarks/yolo11-seg/README.md) exists, but no tracked run packet. |
| Instance segmentation | YOLOv10 | `yolov10seg` | UNRUN |
| Instance segmentation | YOLO26 | `yolo26seg` | REPORTED for YOLO26m-seg on RTX 3060 Laptop; see below. |
| Instance segmentation | RF-DETR | `rfdetrseg` | UNRUN; [export/deployment scripts](../deploy/instance_segmentation/rf-detr/README.md) exist. |
| Instance segmentation | EdgeCrafter | `ecseg`, `edgecrafter-seg` | UNRUN |
| Classification | Torchvision, including ResNet | `torchvision-classifier`, `resnet50` | UNRUN |
| Classification | TensorFlow/Keras | `tensorflow-classifier` | UNRUN |
| Classification | Vision Transformers | `vit-classifier` | UNRUN; [Python backend deployment examples](../deploy/classifier/vit/README.md) exist. |
| Video classification | VideoMAE, ViViT, TimeSformer | `videomae`, `vivit`, `timesformer` | UNRUN; temporal frame-buffer path implemented. |
| Optical flow | RAFT | `raft` | UNRUN; paired-image/consecutive-frame path implemented. |
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

| Model / environment | Paths | Status | Evidence and limits |
| --- | --- | --- | --- |
| YOLO26 detection, NVIDIA A10G, Triton 26.08, TensorRT 11.2.1.2, FP32 | V2 client CUDA; V12 custom CUDA backend with CPU/GPU postprocessing; V13 TensorRT preprocessing with CPU/GPU postprocessing; HTTP and gRPC | PASS | [Comparison and gate results](../specs/2026-10-01-gpu-preprocess-alternatives/evidence/2026-10-09-brev-a10g/comparison.md). Six fixtures, relative to B0/B1 baselines; not an independent framework accuracy gate. Tensor parity was not run for V13; soak was not run for every variant. |
| YOLO26m-seg, RTX 3060 Laptop, FP16 TensorRT 10.14.1 | CPU baseline, DALI preprocessing, DALI polygon postprocessing | REPORTED | [Polygon benchmark report](../benchmarks/yolo26-seg/README.md): four fixtures, box IoU ≥ 0.95, polygon IoU ≥ 0.90; identical polygons for GPU vs CPU postprocessing on the same DALI tensors. Raw results directory is not tracked. |
| YOLO26m-seg, same RTX 3060 environment | CPU baseline, DALI preprocessing, DALI packed-mask postprocessing | REPORTED | Same report: mask IoU ≥ 0.90; GPU/CPU postprocessing mask IoU 1.0 on the same DALI tensors. Raw results directory is not tracked. |
| All other inventory entries | Model-specific full inference | UNRUN / GAP | No passing full-model packet identified in this repository for these entries. Harness presence and the existing supported-model list do not upgrade this status. |

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
