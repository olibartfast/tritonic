# YOLO Ensemble Inference Roadmap

This roadmap adds server-side GPU preprocessing for YOLO inference. Each numbered
step is one reviewable change with its own verification. Option 2 must not start
until the Option 1 exit gate passes.

## Current status

Option 1 is implemented and passed the local GPU acceptance gate on 2026-07-21:
all three Triton models loaded, DALI tensor parity passed, HTTP and gRPC
detection parity passed, invalid input remained controlled, and latency/GPU
evidence was recorded. See the
[workflow and evidence](../deploy/object_detection/yolo/ensemble/README.md).
Promotion into a GPU CI environment remains a deployment follow-up. Phase 2 has
not started.

## Target architecture

```text
Tritonic encoded image bytes
          |
          v
Triton ensemble model
          |
          +-- Option 1: DALI preprocessing model
          |      JPEG decode -> letterbox -> normalize -> HWC-to-CHW
          |
          `-- Option 2: C++/CUDA preprocessing backend
                 nvJPEG decode -> fused CUDA preprocessing kernel
          |
          v
TensorRT YOLO model -> existing neuriplo-tasks postprocessing -> rendering
```

An ensemble is exposed to Tritonic as a normal Triton model. Tritonic does not
need `EnsembleRequest`, `EnsembleResponse`, or `inferEnsemble()` types. The
client-side change is an encoded-image input path plus separate inner-model
metadata for `neuriplo-tasks` output postprocessing.

The [Python backend preprocessing example](https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing)
is the repository-layout and ensemble-wiring reference, not the Option 1
implementation. Option 1 uses the [Triton DALI backend](https://github.com/triton-inference-server/dali_backend)
for GPU preprocessing. The public contract follows NVIDIA's
[ensemble model documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html).

## Fixed first milestone

- One TensorRT YOLO model supported by `--model_type=yolo`: YOLO11s, FP32 NCHW,
  640x640 is the reference.
- JPEG still images, batch size 1, HTTP and gRPC.
- Ensemble input: `IMAGE`, `TYPE_UINT8`, shape `[-1]`.
- Preprocessor output exactly matches the TensorRT input contract.
- TensorRT outputs pass through unchanged. Existing `neuriplo-tasks` NMS,
  coordinate restoration, and rendering remain client-side.
- Video, PNG, batching, segmentation, pose, server NMS, and shared-memory
  optimization are out of scope until the Option 1 exit gate passes.

## Phase 0 - Freeze the contract and baseline

### 0.1 - Record the reference model contract

**Files:** `deploy/object_detection/yolo/ensemble/README.md`

Record TensorRT input/output names, shapes, datatypes, channel order, letterbox
rule, interpolation, normalization, and padding value.

**Verify:** Values match Triton metadata and the corresponding `neuriplo-tasks`
YOLO preprocessor.

### 0.2 - Create a golden CPU-preprocessed result set

**Files:** `tests/data/ensemble/yolo/`, integration-test manifest

Select small, wide, and tall JPEG fixtures. Save the current client-preprocessed
YOLO tensors and final detections with explicit numeric tolerances.

**Verify:** The existing path reproduces the fixtures twice consecutively.

### 0.3 - Add an end-to-end benchmark harness

**Files:** `integration-tests/yolo-ensemble/run.py`

Measure preprocessing, request, server-compute, and end-to-end latency plus
throughput. Keep correctness assertions separate from performance reporting.

**Verify:** Detection mismatches fail; success emits a machine-readable report.

## Phase 1 - Option 1: DALI GPU preprocessing

### 1.1 - Add the model-repository skeleton

**Files:** `deploy/object_detection/yolo/ensemble/dali/model_repository/`

Add configs and version directories for `yolo_dali_preprocess`, `yolo_trt`, and
`yolo_dali_ensemble`. Do not commit a TensorRT engine.

**Verify:** A layout test confirms every configured directory exists.

### 1.2 - Implement the DALI pipeline generator

**Files:** `deploy/object_detection/yolo/ensemble/dali/generate_pipeline.py`

Build an inference-only pipeline with named `external_source`, GPU image decode,
exact YOLO letterbox, channel conversion, scaling/normalization, and CHW output.
Serialize the DALI artifact used by the repository.

**Verify:** Two clean generations produce a loadable pipeline with the declared
input and output names.

### 1.3 - Configure the DALI model explicitly

**Files:** `yolo_dali_preprocess/config.pbtxt`

Declare variable-length `TYPE_UINT8` input and the exact output consumed by
`yolo_trt`. Match the config input name to DALI `external_source`.

**Verify:** Triton strict model configuration loads the model without autocomplete.

### 1.4 - Configure the TensorRT component model

**Files:** `yolo_trt/config.pbtxt`, setup script

Pin tensor names, profiles, `max_batch_size`, and instance placement. Add a setup
command that installs a user-supplied engine.

**Verify:** Direct `yolo_trt` inference accepts the golden preprocessed tensor.

### 1.5 - Wire the DALI ensemble

**Files:** `yolo_dali_ensemble/config.pbtxt`

Map ensemble input to DALI, DALI output to TensorRT, and TensorRT outputs to the
ensemble. Keep the ensemble version directory empty.

**Verify:** All three models are ready and one encoded-image request returns the
declared YOLO tensors.

### 1.6 - Add an explicit encoded-image input mode

**Files:** `include/tritonic/infra/config.hpp`, `src/main/ConfigManager.cpp`, tests

Add `--input_mode=preprocessed|encoded-image`, default `preprocessed`. Reject
encoded mode for chat, temporal tasks, and unsupported multi-input tasks.

**Verify:** Parser tests cover default, valid, and invalid combinations.

### 1.7 - Separate request and task model metadata

**Files:** configuration, `src/main/App.cpp`, tests

Add `--task_model=<name>` for encoded mode. Query `--model` (ensemble) for request
I/O and `--task_model` (inner TensorRT model) to construct the task. Validate
ensemble outputs against the task-model outputs before inference.

**Verify:** Mock tests cover compatible metadata and fail fast on missing or
incompatible metadata.

### 1.8 - Add the encoded still-image request builder

**Files:** focused input builder, unit tests

Read original JPEG bytes without decode/re-encode, set the variable-length
`UINT8` request shape, and use the existing `ITriton::infer()` call.

**Verify:** Tests prove byte preservation and correct shapes for all fixtures.

### 1.9 - Route encoded inference through existing postprocessing

**Files:** `src/main/App.cpp`, App tests

Bypass `task_->preprocess()` only in encoded mode. Convert ensemble outputs and
call the existing YOLO postprocessor with original image dimensions. Preserve
rendering and output files.

**Verify:** Existing path tests remain unchanged; a mock test proves one bypass
per encoded request.

### 1.10 - Prove DALI tensor parity

Compare the DALI component output with the CPU golden tensors. Diagnose letterbox
rounding, interpolation, RGB order, padding, scaling, and layout before changing
tolerances.

**Verify:** Every fixture passes documented absolute and relative tolerances.

### 1.11 - Prove detection parity over HTTP and gRPC

Run both paths and compare count, class, confidence, and restored coordinates.

**Verify:** Both protocols pass; invalid bytes produce controlled errors without
crashing Triton or Tritonic.

### 1.12 - Document a reproducible DALI workflow

**Files:** ensemble `README.md`, root `README.md`

Pin compatible Triton/DALI images and document generation, engine installation,
server startup, Tritonic commands, output, and cleanup. Verify every link.

**Verify:** A clean checkout succeeds without undocumented files or commands.

### Option 1 exit gate

Option 1 is working only when:

- DALI, TensorRT, and ensemble models are ready.
- HTTP and gRPC encoded-JPEG tests pass.
- CPU and DALI preprocessing tensors meet recorded tolerances.
- The original client-preprocessed path still passes.
- Benchmark evidence covers both paths, GPU utilization, and latency. A speedup
  is not required for correctness acceptance, but regressions are recorded.
- The pinned workflow succeeds from a clean checkout.

Do not begin Phase 2 before this gate has CI or reproducible GPU acceptance
evidence.

## Phase 1 follow-ups - Expand the proven DALI path

### 1.13 - Add encoded-image batching

Choose padded uniform input or one request per sample with dynamic/ragged
batching; do not leave the transport implicit.

**Verify:** Batch sizes 1, 2, maximum, and a short final batch match single-image
results for mixed aspect ratios.

### 1.14 - Add video-frame transport

Choose and benchmark client JPEG encoding, raw frame tensors, or a video-aware
decoder. Do not silently JPEG-encode frames.

**Verify:** Frame order, dimensions, colors, and detections match a fixed clip.

### 1.15 - Expand formats and YOLO families

Add PNG after decoder tests, then qualify other YOLO detection exports one at a
time. Segmentation, pose, and end-to-end NMS are separate contracts.

**Verify:** Each addition has metadata and golden-result entries.

## Phase 2 - Option 2: native C++/CUDA preprocessing

Option 2 is a Triton custom C++ backend, not CUDA inside the Tritonic client. It
preserves the DALI preprocessor I/O so the ensemble and client do not fork.

### 2.1 - Freeze the backend ABI

**Files:** `deploy/object_detection/yolo/ensemble/cuda/CONTRACT.md`

Copy the proven DALI names, shapes, datatypes, errors, and tolerances. Specify
JPEG support and CUDA device/stream ownership.

**Verify:** One contract test targets either preprocessor by model name only.

### 2.2 - Scaffold the Triton C++ backend

**Files:** `backends/yolo_preprocess/`, CMake and container files

Implement lifecycle, validation, response allocation, and structured errors
without image processing.

**Verify:** It loads, reports ready, rejects malformed requests, and passes
lifecycle tests.

### 2.3 - Add nvJPEG decode

Decode into device memory on the backend CUDA stream. Reuse per-instance handles
and buffers; reject corrupt or unsupported inputs.

**Verify:** Tests cover grayscale, RGB, wide, tall, truncated, and invalid JPEGs.

### 2.4 - Implement the CUDA preprocessing kernel

Implement letterbox resize, interpolation, channel conversion, padding,
normalization, and HWC-to-CHW. Optimize/fuse only after parity.

**Verify:** Output matches DALI golden tensors and compute-sanitizer memcheck passes.

### 2.5 - Add batching and memory reuse

Process samples on the provided stream, reuse bounded instance buffers, handle
partial failures, and avoid global device synchronization.

**Verify:** Batch and concurrency tests show no leaks or cross-request corruption.

### 2.6 - Package the backend and alternate ensemble

**Files:** `deploy/object_detection/yolo/ensemble/cuda/model_repository/`

Add `yolo_cuda_preprocess` and `yolo_cuda_ensemble` with the same public I/O.
Pin compiler, CUDA, Triton backend SDK, nvJPEG, and server versions.

**Verify:** A clean container build loads every component model.

### 2.7 - Run the unchanged correctness suite

Run Option 1 tensor and detection parity tests against CUDA over HTTP and gRPC.

**Verify:** No CUDA-specific weakening of tests or tolerances.

### 2.8 - Benchmark DALI versus CUDA

Measure preprocessing, allocations, GPU utilization, end-to-end latency, and
throughput across sizes, batches, and concurrency after warmup.

**Verify:** Results include hardware, driver, container, and commit identifiers.

### 2.9 - Harden and document

Add size/overflow limits, cancellation, unload/reload, multi-instance, multi-GPU,
stress, observability, and troubleshooting coverage.

**Verify:** Stress, sanitizer, malformed-input, reload, and clean-checkout gates pass.

### Option 2 exit gate

- CUDA passes the unchanged Option 1 correctness suite.
- Sanitizer and stress tests have no memory, stream, or concurrency failures.
- Container build and deployment are reproducible.
- DALI-versus-CUDA evidence is published with environment identifiers.
- Documentation states the default implementation and measured reason.

## Phase 3 - Optional GPU postprocessing

GPU postprocessing is worthwhile when profiling shows CPU decode/NMS or transfer
of raw YOLO heads is material. Prefer a TensorRT end-to-end export with supported
NMS first; use a Triton C++/CUDA postprocessing backend only when that cannot
implement the required YOLO output contract.

### 3.1 - Measure the current cost

Record raw-output bytes, device-to-host and network transfer, CPU postprocessing
latency, total latency, throughput, batch size, and concurrency.

**Verify:** Publish profiles for representative small and large YOLO outputs.
Proceed only with a recorded target such as latency, throughput, or bandwidth.

### 3.2 - Qualify TensorRT-integrated postprocessing

Export or build an end-to-end TensorRT model that performs YOLO decode and NMS
using APIs/plugins supported by the pinned TensorRT version. Expose only final
boxes, scores, classes, and valid-detection count.

**Verify:** Final detections match the Phase 0 golden results and the engine loads
without unpinned or deprecated plugin dependencies.

### 3.3 - Add a CUDA backend only if TensorRT is insufficient

Reuse the Phase 2 backend infrastructure for decode, thresholding, sorting, and
NMS on the Triton CUDA stream. Bound candidates and workspace explicitly.

**Verify:** Tensor parity, compute-sanitizer, worst-case candidate, batch, and
concurrency tests pass without weakening tolerances.

### 3.4 - Extend the ensemble and Tritonic output contract

Add preprocessing -> TensorRT -> GPU postprocessing ensemble variants. Add an
explicit final-detections output mode in Tritonic that bypasses
`neuriplo-tasks` NMS but preserves rendering and original-image coordinates.

**Verify:** Metadata validation rejects raw/final output mismatches before infer;
HTTP and gRPC results match the existing client-postprocessed path.

### 3.5 - Make the deployment choice from measurements

Benchmark client CPU postprocessing, TensorRT-integrated postprocessing, and the
CUDA backend with identical inputs and server settings.

**Verify:** Retain GPU postprocessing only when its end-to-end benefit justifies
the extra model-specific maintenance. Document the selected default and fallback.

### 3.6 - Qualify YOLO segmentation separately

Profile prototype and mask-coefficient transfer plus CPU mask generation. On GPU,
run detection decode/NMS first, gather coefficients only for retained detections,
then perform prototype multiplication, sigmoid, crop, and resize. Do not
materialize masks for every pre-NMS candidate.

Define a bounded output contract before implementation: boxes/classes/scores plus
one of fixed-size `UINT8` masks, cropped instance masks, contours, or RLE. Include
transfer and client rendering cost when choosing the representation.

**Verify:** Detection and mask parity match the existing
`neuriplo-tasks` segmentation path across empty, single-object, many-object,
overlapping-object, and batch cases. The benchmark must include output bytes and
end-to-end latency, not only GPU kernel time.

## Completion definition

Ensemble support is complete after Option 1 passes its exit gate. Options 2 and 3
are separately gated optimizations requiring real NVIDIA GPU evidence; code
review or compilation alone is not completion.
