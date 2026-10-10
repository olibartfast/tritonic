# Requirements — GPU preprocessing alternatives to DALI

Spec: `specs/2026-10-01-gpu-preprocess-alternatives/requirements.md` ·
Branch: `feature/gpu-preprocess-alternatives` · Roadmap: Phase 1

## Goal
Build three CUDA C++ alternatives to DALI preprocessing for `yolo26det`, prove each
matches DALI numerically and in detections, benchmark them alongside the existing
CPU and DALI paths under one protocol, and publish a measured recommendation for
the default GPU preprocessing path.

## Variants

| ID | Name | Preprocessing runs in | Client → server payload | Client mode |
|---|---|---|---|---|
| B0 | CPU baseline (existing) | client CPU, `neuriplo-tasks` | FP32 `[3,640,640]` | `--input_mode=preprocessed` |
| B1 | DALI baseline (existing) | Triton DALI backend | JPEG bytes | `--input_mode=encoded-image` |
| V12 | Custom C++ backend | Triton C++ backend: nvJPEG + fused CUDA kernel | JPEG bytes | `--input_mode=encoded-image` |
| V13 | TensorRT-graph preprocessing | nvJPEG decode step (V12 backend), then letterbox/normalize inside the TRT engine | JPEG bytes | `--input_mode=encoded-image` |
| V2 | Client-side CUDA | client GPU: CPU decode, CUDA kernel into CUDA shared memory | CUDA IPC handle | `--input_mode=preprocessed --preprocess_device=cuda` |

IDs follow the options numbered in the original discussion. Option 1.1 (Python
backend) is out of scope (D-2).

## In Scope

**Shared contract**
- [R-1] Every server-side preprocessing model (V12, and the decode step of V13 where
  applicable) uses the I/O of `yolo26det_dali_preprocess`: input `IMAGE`
  `TYPE_UINT8 [-1]`; outputs `PREPROCESSED` `TYPE_FP32 [3,640,640]` and
  `ORIGINAL_SIZE` `TYPE_INT64 [2]` as (H, W). It can be swapped into the existing
  ensemble topologies by changing only `model_name`.
- [R-2] Each variant reproduces the DALI letterbox from
  `deploy/object_detection/yolo26/ensemble/generate_pipeline.py`:
  `rw = W*640 // max(W,H)`, `rh = H*640 // max(W,H)`; bilinear interpolation,
  half-pixel centers, no antialiasing; centered paste on a 640×640 canvas with
  fill 0; RGB; `x/255`; CHW FP32. The resized pixels are rounded to 8-bit before
  scaling, as DALI's uint8 resize output is. The pad-offset rounding rule is measured
  against DALI (spike T-3) and then fixed as part of this contract. Until then the
  default is floor, which matches the CPU path (finding:
  `tests/unit/test_yolo_preprocessing_contract.cpp`, OddPaddingRemainderFallsOnBottomAndRight).
  Inputs whose aspect ratio exceeds 640:1 collapse one side to 0 pixels and are
  rejected (finding, 2026-10-02: the ONNX graph fails on 1×1000 and the reference
  cannot represent it). DALI instead emits a one-row tensor for these (finding,
  2026-10-09, spike T-3), so this rejection is a recorded difference from B1.
  Pad rounding is floor, measured against DALI 2.2.0 (spike T-3, 2026-10-09).

**V12 — custom C++ backend**
- [R-3] A Triton C++ backend `yolopreprocess` serves `yolo26det_cuda_preprocess`
  (contract R-1). It decodes JPEG with nvJPEG into device memory on the
  instance's CUDA stream, then runs one fused kernel (resize, pad, scale,
  HWC→CHW) that writes the Triton-allocated GPU output. Per-instance handles and
  buffers are reused across requests, and there is no device-wide
  synchronization.
- [R-4] Ensembles `yolo26det_cudapre_cpu_post` and `yolo26det_cudapre_gpu_post`
  mirror `yolo26det_gpu_pre_cpu_post` / `yolo26det_gpu_pre_gpu_post` with only
  the preprocessing step replaced. Tritonic runs them with no client change.

**V13 — TensorRT-graph preprocessing**
- [R-5] The same `yolopreprocess` backend has a decode-only mode (model config
  parameter), served as `yolo26det_nvjpeg_decode`: `IMAGE` → `DECODED`
  `TYPE_UINT8 [-1,-1,3]` (RGB HWC, original size) and `ORIGINAL_SIZE`.
- [R-6] An ONNX wrapper placed in front of the YOLO26 graph takes `image`
  `UINT8 [H,W,3]` and computes the R-2 letterbox in-graph (Shape → size math →
  Resize → Pad → Cast → Div → Transpose). It is built into the engine
  `yolo26det_trt_pre`, which has one dynamic-shape optimization profile and the
  output `output0` FP32 `[300,6]`.
- [R-7] Ensembles `yolo26det_trtpre_cpu_post` and `yolo26det_trtpre_gpu_post`
  (decode → `yolo26det_trt_pre` → [GPU postprocess]) expose the same outputs as
  their B1 counterparts. Tritonic runs them with `--task_model=yolo26det_trt`
  and no client change.

**V2 — client-side CUDA**
- [R-8] A new flag `--preprocess_device=cpu|cuda` (default `cpu`) is valid only
  with `--input_mode=preprocessed`, `--shared_memory_type=cuda`, and the
  `yolo`/`yolo26` model types. Any other combination fails at config validation
  with a clear message.
- [R-9] With `cuda`, the client decodes on the CPU (OpenCV, as in B0), uploads
  the `UINT8` HWC image, and runs the R-2 kernel, which writes directly into the
  registered CUDA shared-memory input region. The FP32 host tensor and its
  `cudaMemcpy` are not created on this path. The original size used for
  postprocessing is identical to B0.
- [R-10] The kernel source is shared between V12 and V2 under `src/cuda/`
  (one implementation, two consumers).

**Robustness, build, evidence**
- [R-11] V12/V13 return a structured per-request error for zero-length input,
  non-JPEG bytes, a JPEG without decodable scan data, an aspect ratio beyond 640:1,
  and dimensions over the configured `max_side` (8192 for V12; 4096 for V13, which
  is bounded by the TensorRT profile). Grayscale JPEGs succeed. The server stays
  ready and the next valid request succeeds. A JPEG truncated mid-scan is recorded
  for every variant, including B1, but not gated: R-11 does not fix its outcome.
  V2 exits non-zero with a message for frames its kernel contract rejects (aspect
  ratio, side over 8192). Unreadable files keep the existing B0 behavior of logging
  the error and skipping the file (finding, 2026-10-02: `App::processImages`), so
  they are not part of the V2 gate.
- [R-12] A CMake option `TRITONIC_ENABLE_CUDA` (default `OFF`) enables the CUDA
  language, links `CUDA::cudart`, defines `TRITONIC_ENABLE_CUDA` and
  `TRITON_ENABLE_GPU`, and builds `src/cuda/`. With `OFF`, the build, tests, and CI
  behave exactly as before. `TRITON_ENABLE_GPU` is needed because the client's
  `ipc.h` only uses the real `cudaIpcMemHandle_t` with it. The pre-existing
  `RegisterCudaSharedMemory` calls passed a device pointer where the API takes an
  IPC handle (finding, 2026-10-02: r26.08 `grpc_client.h`/`http_client.h`), so
  they would not have compiled with the flag on. They are fixed here.
- [R-13] All variants work over HTTP and gRPC at batch size 1.
- [R-14] The benchmark runner `benchmarks/yolo26-det/run.py` covers B0, B1, V12,
  V13 and V2. It records the environment identifiers and the server per-model
  stats, and writes `summary.json` (schema v4). A script turns that into
  `comparison.md` with no manual editing.
- [R-15] `comparison.md` ends with a recommendation: the default GPU
  preprocessing path for yolo26det and the conditions under which each
  alternative is preferable, citing measured numbers. Variants that failed a
  gate are reported with their failure, not omitted.
- [R-16] B0 and B1 are unchanged: existing unit tests, the B1 tensor-parity gate
  and the B1 detection-parity gate still pass.
- [R-17] *(added 2026-10-10, maintainer)* The client must keep working on CPU-only
  machines. A binary built with `TRITONIC_ENABLE_CUDA=ON` starts and runs the
  default `--preprocess_device=cpu` path on a host with no GPU and no CUDA runtime
  installed; `--preprocess_device=cuda` there fails at startup with an error that
  names the cause and the CPU alternative.

## Out of Scope
- Option 1.1, the Python backend (any library) — dropped by decision D-2;
  recorded in `specs/roadmap.md` Deferred.
- Batch size > 1, video sources, PNG/other codecs, and multi-GPU — revisit after
  Phase 1 (roadmap Deferred).
- Other model families (yolo26seg, yolo11seg, yolo) — rollout follows the Phase 1
  recommendation.
- Changes to GPU postprocessing (`yolo26det_dali_postprocess`) — reused unchanged.
- GPU runners in CI. GPU evidence is produced by manual runs recorded in
  `validation.md`.
- V2 over a network (client and Triton on different hosts) — CUDA IPC requires
  the same host and GPU (A-3).
- A server-side nvJPEG decode feeding V2 — V2 keeps client CPU decode to isolate
  the cost of the kernel versus the CPU letterbox.

## Decisions
- [D-1] V2's kernel lives in tritonic `src/cuda/` behind `TRITONIC_ENABLE_CUDA`,
  not in neuriplo-tasks — Rationale: keeps neuriplo-tasks free of CUDA (user
  decision 2026-10-01; recorded in tech-stack non-choices). This is a scoped
  exception to the AGENTS.md rule that preprocessing belongs in neuriplo-tasks:
  the kernel is a transport/placement optimization of an existing task, not a
  new model type.
- [D-2] No Python backend — Rationale: user decision 2026-10-01 ("only CUDA C++").
- [D-3] V13 keeps JPEG transport through a server-side nvJPEG decode step reusing
  the V12 backend — Rationale: user decision 2026-10-01. This also isolates the
  in-graph letterbox as the only difference from V12.
- [D-4] Reference model is yolo26det — Rationale: it already has a DALI ensemble,
  a TRT config, a GPU postprocess step and a benchmark harness
  (`benchmarks/yolo26-det/run.py`).
- [D-5] DALI is the numerical reference, with the existing tolerances
  (`compare_dali.py`: max 0.15, mean 0.003, p99 0.016;
  `compare_inference.py`: count and class equal, confidence ±0.03, boxes ±12 px).
  Tolerances are not loosened for any variant. **Amended 2026-10-10:** the detection
  confidence tolerance is ±0.04, the same for every variant, because the B1 and B0
  baselines themselves differ by 0.032 on `bus_1080p.jpg` (evidence in
  validation.md, Deviations). The tensor tolerances are unchanged.
- [D-6] V12 and V13 share one backend binary with a `mode` parameter
  (`preprocess` | `decode`) instead of two backends — Rationale: one nvJPEG
  integration and one lifecycle to harden.
- [D-7] Variant model configs live in
  `deploy/object_detection/yolo26/ensemble/variants/`, and
  `setup_model_repository.sh` copies them in only when `--with-cuda-backend` or
  `--trt-pre-onnx` is given — Rationale: Triton loads every model in the repository
  and by default exits if one fails, so unbuilt variant models in the default
  repository would break B1 (R-16).
- [D-8] V2 reaches the GPU through two narrow interfaces,
  `tritonic::triton::IDeviceInputInference` (implemented by `Triton`) and
  `tritonic::core::IDevicePreprocessor` (implemented in `src/cuda/`), rather than
  through new `ITriton` methods — Rationale: existing `ITriton` implementations and
  mocks stay unchanged, and `App` has no CUDA `#ifdef`s, so the device path is unit
  tested with mocks on machines without a GPU.

## Constraints
- Images pinned to `tritonserver:26.08-py3` and `tensorrt:26.08-py3`
  (`specs/tech-stack.md`).
- Defaults unchanged (mission principle). All new behavior is behind new models,
  new flags, or `TRITONIC_ENABLE_CUDA=ON`.
- Performance evidence records the GPU, driver, image digest, TensorRT version
  and commit (`specs/tech-stack.md` constraints).

## Dependencies
- Finding: the DALI reference pipeline is
  `deploy/object_detection/yolo26/ensemble/generate_pipeline.py`, and the model
  configs are in `.../yolo26/ensemble/model_repository/*/config.pbtxt`.
- Finding: tritonic already supports `--input_mode=encoded-image` with
  `--task_model` (`src/main/ConfigManager.cpp:161-216`) and CUDA shared-memory
  regions (`src/triton/Triton.cpp:818-855`, `:1078`).
- Finding (2026-10-02): `--shared_memory_type` is parsed but has no effect.
  `client.cpp` always constructs `Triton` with system shared memory, and `App`
  never calls `inferWithSharedMemory`. V2 therefore wires its own path; the flag
  stays inert for every other mode.
- Finding: `TRITONIC_ENABLE_CUDA` is used in the code but defined by no CMake
  file (grep of `CMakeLists.txt`, 2026-10-01), so the CUDA shared-memory path
  cannot currently be built. R-12 fixes this.
- Finding: the CUDA DALI plugins are built in the `tensorrt:<tag>-py3` image
  (`dali_plugin/build_plugin.sh`). The V12 backend uses the same build-container
  pattern.
- External: Triton backend SDK `triton-inference-server/backend` at tag `r26.08`;
  nvJPEG from the CUDA toolkit in the `tensorrt:26.08-py3` image.
- External: a host with an NVIDIA GPU, driver, and Docker GPU runtime. The
  machine where this spec was written has none (finding, 2026-10-01: AMD iGPU
  only, no `nvidia-smi`), so all GPU validation is run by the maintainer on
  their GPU host.

## Context
- Existing benchmark evidence for B0/B1 is from an RTX 3060 Laptop GPU on Triton
  25.12 (`deploy/object_detection/yolo/ensemble/README.md`, 2026-07-21). It is
  not comparable after the 26.08 bump, so Phase 0 re-baselines.
- `docs/ENSEMBLE_INFERENCE_ROADMAP.md` Phase 2 sketched a C++/CUDA backend for
  `yolo` (YOLO11). This packet supersedes it for yolo26det and adds V13 and V2.

## Assumptions & Open Questions
- [A-1] TensorRT 11.2.1.2 in `tensorrt:26.08-py3` (finding, 2026-10-09: image
  `TRT_VERSION`; the image's CUDA runtime is 13.4.1, and TensorRT is built against
  CUDA 13.3) accepts a `UINT8` network input and a data-dependent Resize/Pad output
  size inside one dynamic-shape profile — Basis: TensorRT 10 behavior (inference).
  The DALI inside `tritonserver:26.08` is 2.2.0 (the pin in
  olibartfast/rf-detr-cpp-inference `versions.env`). **Answered 2026-10-09 (spike T-2):**
  true once `Cast` precedes `Transpose` (TensorRT 11.2 rejects UINT8 `Transpose`).
- [A-2] answered 2026-10-09 (spike T-3): floor. Probed first by spike T-2; if it
  fails, V13 is replanned (e.g. resize in-graph to a fixed size, pad outside).
- [A-2] DALI `fn.paste(paste_x=0.5)` places the image at
  `floor((640 - rw) / 2)` — Basis: none, assumption. Measured by spike T-3.
- [A-3] V2 runs with the client and Triton on the same host and GPU — Basis: the
  CUDA IPC handle semantics of Triton CUDA shared memory.
- [A-4] `tritonserver:26.08-py3` and `tensorrt:26.08-py3` ship the same CUDA and
  TensorRT versions — Basis: NGC same-month release pairing, as already relied
  on by `build_plugin.sh`. Checked in T-1 (V-0).
- [Q-1] Which GPU will produce the evidence? Owner: maintainer. It must be
  recorded before Phase 0's exit; results from different GPUs are not mixed in
  one `summary.json`.

## Definition of Done (requirements level)
- [ ] Every R-n is implemented or explicitly deferred with a tracked location.
- [ ] No in-scope behavior is silently dropped, and no out-of-scope work is added.
