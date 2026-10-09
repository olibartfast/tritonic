# Plan — GPU preprocessing alternatives to DALI

Requirements: [requirements.md](requirements.md) · Proof: [validation.md](validation.md)

Each phase ends with something runnable and its own exit check. Phases are
ordered by risk and dependency. The riskiest unknowns (A-1, the TRT UINT8 dynamic
letterbox, and A-2, the DALI pad rounding) are probed in Phase 0 before any
variant is built. The shared kernel comes before both of its consumers. V2 goes
first because it needs no server-side change.

Legend: **[GPU]** means the exit check needs the maintainer's NVIDIA host.
Tasks without it can be done and checked on a machine without a GPU.

## Phase 0 — Re-baseline and de-risk

### Group 0.1 — Environment and spikes (spike code is throwaway; deliverable is a written decision)
- [T-1] Record the environment probe: a script that prints the GPU, driver, CUDA,
  TRT, image digests and commit, written into each result directory. Confirm A-4.
  — Deliverable: `benchmarks/yolo26-det/env_probe.sh` output.
- [T-2] Spike A-1: generate a minimal ONNX graph (UINT8 `[H,W,3]` → R-2 letterbox
  → FP32 `[1,3,640,640]`) and build it with `trtexec` in `tensorrt:26.08-py3`
  using a dynamic profile (min 32×32, opt 1080×1920, max 4096×4096). Run it on
  the fixtures and compare against the DALI tensors.
  — Deliverable: a note in this file under *Spike results* saying pass, or the
  V13 replan.
- [T-3] Spike A-2: run the DALI pipeline on synthetic sizes with odd and even
  padding (e.g. 641×479, 640×481, 3×5) and infer the paste-offset rule and the
  resize sampling convention. Record them in R-2 and here.
- Exit **[GPU]**: V-0 is recorded, both spike notes are written, and R-2 / A-1 / A-2 are updated
  with the findings.

### Group 0.2 — Benchmark harness v4 (no new variant)
- [T-4] Refactor `benchmarks/yolo26-det/run.py`: replace the hard-coded `PATHS`
  and `rtx3060` label with a variant registry (B0, B1 now; V12, V13, V2 added by
  later phases), a `--label` flag and a `--variants` filter. Snapshot
  `/v2/models/<m>/stats` before and after each run, add HTTP and gRPC loops,
  and set `schema_version: 4`. (R-14)
- [T-5] Add 1920×1080 and 3840×2160 quality-90 JPEG fixtures, generated
  deterministically by a script in `benchmarks/common/fixtures.py`, not
  committed as binaries.
- [T-6] Add a `perf_analyzer` wrapper (SDK image) for concurrency 1/4/8 on the
  1080p fixture, plus GPU memory and utilization sampling from `:8002/metrics`.
- Exit **[GPU]**: `run.py --variants B0,B1` produces `summary.json` v4 for both
  protocols on 26.08 (V-1, V-2). This is the new baseline.

## Phase 1 — Shared CUDA letterbox kernel

### Group 1.1 — Build plumbing (no behavior change)
- [T-7] Add CMake option `TRITONIC_ENABLE_CUDA` (default OFF): `enable_language(CUDA)`,
  `find_package(CUDAToolkit)`, define the macro, link `CUDA::cudart`, and set
  `CMAKE_CUDA_ARCHITECTURES` from a `CUDA_ARCH` cache variable, matching the
  plugin scripts. (R-12)
- Exit: with OFF, the build and `ctest` match the baseline (V-3). With ON, the
  build succeeds in `tensorrt:26.08-py3` (V-4).

### Group 1.2 — Kernel and dump tool
- [T-8] Implement `src/cuda/` with one stream-ordered entry point: input a device
  `UINT8` HWC RGB or BGR image (H, W, pitch, channel order); output a device
  FP32 CHW 640×640 buffer. The R-2 math is fused in one kernel that writes every
  output pixel exactly once, padding included, so no memset is needed. (R-2, R-10)
- [T-9] Add `cuda_preprocess_dump`, a sibling of `tests/tools/yolo_preprocess_baseline.cpp`.
  It decodes with OpenCV, runs the kernel and writes `.fp32` tensors in the
  existing baseline format.
- [T-10] Add `compare_tensors.py`, generalizing `compare_dali.py` to compare any
  two tensor directories with the D-5 thresholds.
- Exit **[GPU]**: kernel tensors versus DALI tensors pass D-5 on all six
  fixtures (V-5), and `compute-sanitizer` memcheck is clean on the dump tool (V-14).

## Phase 2 — V2: client-side CUDA preprocessing

### Group 2.1 — Config surface
- [T-11] Add `--preprocess_device` to `InferenceConfig` / `ConfigManager` with
  the R-8 validation, plus unit tests in `tests/unit/test_config.cpp`.
- Exit: the new config tests pass on a machine without a GPU (V-6).

### Group 2.2 — Pipeline wiring
- [T-12] In the `Triton` CUDA shared-memory path, let a caller provide a
  device-side producer that fills the input region in place. This replaces the
  host-buffer `cudaMemcpy` (`Triton.cpp:1078`) only when
  `preprocess_device=cuda`. Keep `ITriton` mockable: add the capability as a
  narrow interface, and keep `infer()` unchanged for existing callers.
- [T-13] In `App`, when `preprocess_device=cuda`: decode with OpenCV, keep the
  original size for postprocessing, upload the HWC image, invoke the kernel into
  the region, infer, and postprocess as B0. Add a mock test proving
  `task_->preprocess()` is not called and that the region producer is called
  exactly once per frame. (R-9)
- [T-14] Register V2 in the benchmark runner.
- Exit **[GPU]**: V2 detection parity with B1 over HTTP and gRPC (V-7), and the
  V2 benchmark rows exist (V-12).

## Phase 3 — V12: custom C++ Triton backend

### Group 3.1 — Backend skeleton
- [T-15] Scaffold `backends/yolopreprocess/` using the `r26.08` backend SDK:
  model and instance lifecycle, the `mode` parameter, input validation, output
  allocation and structured errors, with no image work yet (it returns a zeroed
  tensor). Build it in `tensorrt:26.08-py3` with a script that mirrors
  `build_plugin.sh`.
- Exit **[GPU]**: Triton loads the model and reports it ready. Malformed requests
  return errors and Triton stays ready (part of V-11).

### Group 3.2 — Decode and preprocess
- [T-16] Add nvJPEG decode on the instance stream, with per-instance reusable
  handles and buffers. Reject non-JPEG, truncated and over-limit inputs, and
  convert grayscale to RGB. (R-3, R-11)
- [T-17] Call the `src/cuda/` kernel into the Triton output buffer, and emit
  `ORIGINAL_SIZE`.
- [T-18] Add the `yolo26det_cuda_preprocess` config plus the
  `yolo26det_cudapre_cpu_post` / `yolo26det_cudapre_gpu_post` ensembles, extend
  `setup_model_repository.sh` behind a `--with-cuda-backend` flag, and register
  V12 in the benchmark runner. (R-4)
- Exit **[GPU]**: V12 tensor parity (V-5) and detection parity on both protocols
  (V-8), and the V12 benchmark rows exist (V-12).

## Phase 4 — V13: TensorRT-graph preprocessing

### Group 4.1 — Decode-only model
- [T-19] Add `mode=decode` to the backend and the `yolo26det_nvjpeg_decode`
  config (R-5).
- Exit **[GPU]**: decoded RGB matches OpenCV `imread` + BGR→RGB, with the
  maximum absolute pixel difference (nvJPEG versus libjpeg IDCT) recorded and
  ≤ 2 (V-9).

### Group 4.2 — Engine and ensembles
- [T-20] Add `generate_trt_pre_onnx.py`: prepend the spike-proven letterbox
  subgraph to the YOLO26 ONNX, set the dynamic profile, and add a build script
  for `yolo26det_trt_pre` (R-6).
- [T-21] Add the `yolo26det_trt_pre` config and the `yolo26det_trtpre_cpu_post` /
  `yolo26det_trtpre_gpu_post` ensembles, and register V13 in the benchmark
  runner (R-7).
- Exit **[GPU]**: V13 detection parity on both protocols (V-10), and the V13
  benchmark rows exist (V-12). Tensor parity for V13 is checked on the spike
  graph's output (V-5) because the engine fuses preprocessing.

## Phase 5 — Hardening, comparison, and integration

### Group 5.1 — Robustness
- [T-22] Add an invalid-input suite (`integration-tests/gpu-preprocess/invalid_inputs.py`)
  covering every R-11 case for V12, V13 and V2, followed by a valid request (V-11).
- [T-23] Add a stability run of 10,000 sequential plus 1,000 at concurrency 8
  per server variant, checking the GPU memory delta (V-13), and run
  `compute-sanitizer` (memcheck and racecheck) on V12 inside Triton and on V2 (V-14).
- Exit **[GPU]**: V-11, V-13 and V-14 recorded.

### Group 5.2 — Comparison and docs
- [T-24] Add `benchmarks/yolo26-det/compare.py`: `summary.json` v4 →
  `comparison.md`, with latency, server breakdown, throughput, memory, request
  bytes, gate status, and the cost-of-ownership table (lines of code, build
  steps, dependencies, version-coupling points). (R-14)
- [T-25] Write the recommendation section from the measured data (R-15), and
  update the YOLO26 ensemble README, the root README ensemble table, CHANGELOG
  `[Unreleased]`, `specs/roadmap.md` status, and the superseded Phase 2 note in
  `docs/ENSEMBLE_INFERENCE_ROADMAP.md`. Verify links per AGENTS.md.
- Exit: V-15, V-16 and the scope-diff check V-17 pass. Every validation row has
  evidence.

## Rollback
Every variant is additive: new models, a new flag defaulting to the current
behavior, and a CMake option defaulting OFF. Reverting a phase removes its models
and code without touching B0/B1. No data or persistence formats change. The only
shared-code edit is the T-12 seam in `Triton.cpp`. It is gated on
`preprocess_device=cuda`, and V-3 proves the default path is unchanged.

## Implementation status (2026-10-02)

Code exists for every task. A group counts as done only when its exit check has run.

| Group | Code | Exit check | Where |
|---|---|---|---|
| 0.1 env + spikes | T-1 `benchmarks/yolo26-det/env_probe.sh`; T-2 `integration-tests/gpu-preprocess/trt_letterbox_spike.py`; T-3 `dali_letterbox_spike.py`; `run_spikes.sh` (Docker) or `colab_validation.ipynb` | spikes **done** on Colab (see Spike results); V-0 env probe on the Docker host pending | validation T-2/T-3 |
| 0.2 harness v4 | T-4 `run.py` + `variants.py`; T-5 `benchmarks/common/fixtures.py`; T-6 `perf.py` | **pending [GPU]** | — |
| 1.1 build plumbing | T-7 `TRITONIC_ENABLE_CUDA` in `CMakeLists.txt` | **done**: OFF unchanged (V-3, V-3b), ON builds (V-4) | validation V-3/V-4 |
| 1.2 kernel | T-8 `src/cuda/letterbox.cu`, `include/tritonic/cuda/letterbox_pixel.hpp`; T-9 `tests/tools/cuda_preprocess_dump.cpp`; T-10 `compare_tensors.py` | **done**: GPU parity vs DALI (V-5) and sanitizers (V-14 kernel) on a T4 | validation V-5/V-14 |
| 2.1 config | T-11 `--preprocess_device` + `test_config.cpp` | **done** (V-6) | validation V-6 |
| 2.2 wiring | T-12 `IDeviceInputInference` on `Triton`; T-13 `App` device path + `test_app_device_preprocess.cpp`; T-14 registry entry `V2` | mock tests pass (V-6); **pending [GPU]** (V-7) | validation V-6 |
| 3.1/3.2 backend | T-15 `backends/yolopreprocess/` (+ `build_backend.sh`); T-16/T-17 nvJPEG + kernel; T-18 `variants/` configs, `setup_model_repository.sh --with-cuda-backend` | builds (V-4), staging dry run (C-4); **pending [GPU]** (load, V-8) | validation V-4/C-4 |
| 4.1/4.2 V13 | T-19 `mode=decode`; T-20 `trt_pre/generate_trt_pre_onnx.py`, `build_trt_pre_engine.sh`; T-21 configs, `--trt-pre-onnx` | merge wiring (C-3); **pending [GPU]** | validation C-3 |
| 5.1 robustness | T-22 `invalid_inputs.py`; T-23 `soak.py`, `sanitize.sh` | **pending [GPU]** | — |
| 5.2 report + docs | T-24 `compare.py`; T-25 docs | determinism (C-6); recommendation **pending data** | validation C-6 |

Run order on the GPU host:

1. `integration-tests/gpu-preprocess/run_spikes.sh` (T-2, T-3), then update R-2,
   A-1 and A-2 and the `pad_rounding` default if DALI measures ceil.
2. Build: default (V-3) and `-DTRITONIC_ENABLE_CUDA=ON -DBUILD_TESTING=ON` into
   `build-cuda` (V-4, V-6).
3. Stage: `setup_model_repository.sh --engine … --trt-pre-onnx …`, then start
   Triton.
4. Gates: V-5, V-7, V-8, V-9, V-10, V-11, V-13, V-14, with reports written to
   `<result>/gates/`.
5. Measure: `run.py` and `perf.py` (V-12), then `compare.py` (V-15), then write
   `recommendation.md` (V-16).

## Spike results

Run on Google Colab on 2026-10-09: Tesla T4 (sm_75), driver 580.82 (CUDA 13.0), CUDA 13.3
toolkit, pip `nvidia-dali-cuda130==2.2.0` and `tensorrt-cu13==11.2.1.2` (the NGC 26.08
pins), via `integration-tests/gpu-preprocess/colab_validation.ipynb`.

- **T-3 (A-2) — answered: floor.** DALI's `fn.paste(paste_x=0.5)` offset matched floor on
  all 7 probe geometries; ceil missed 3 (481×640, 640×481, 333×1000). Resampling is the R-2
  convention: the NumPy reference applied to DALI's own decode reproduces DALI's tensor
  (bus, person, mug bit-exact; horses max 1/255). R-2's `floor` default stands.
- **T-3 finding — degenerate geometry differs from B1.** DALI does *not* reject 1×1000 /
  1000×1; it emits a tensor with a single content row. The variants reject these inputs
  (R-11); this is a deliberate, recorded difference from B1, not a parity failure.
- **T-2 (A-1) — answered: yes, after one graph change.** TensorRT 11.2.1.2 rejected the
  first graph: `Transpose` on a UINT8 tensor is not allowed (UINT8 is accepted only as a
  Cast input). With `Cast` moved ahead of `Transpose` the engine builds with a dynamic
  profile and matches the reference on all 9 cases (fixtures, odd/even padding, 5×3,
  1080p, 2160p; max 1/255) and DALI on all 4 fixtures within D-5. The profile minimum was
  lowered from 32×32 to 1×1 so V13 accepts small images like V12 and B1.
- Harness lesson (not a TensorRT issue): the first Python-API runner uploaded with CuPy on
  the default stream and executed on a separate non-blocking stream, which produced
  partial, order-dependent errors on large inputs. Copies and inference now share one
  stream; with that, every probe is exact regardless of order or content.

## Notes
- Deviations from the task text, recorded on 2026-10-02: the perf wrapper is
  `perf.py` (Python, to share `variants.py`), not `perf.sh`. Detection parity is the
  new `compare_detections.py`, which reuses `compare_inference.compare()`. Variant
  model configs moved to `ensemble/variants/` (D-7). V2 uses the D-8 interfaces
  instead of an `ITriton` change.
- Known risk for V13: `build_trt_pre_engine.sh` builds with `--fp16` by default, like
  the reference engine, so TensorRT may run the letterbox layers in FP16. V-5 (spike,
  FP32) and V-10 (FP16 engine) separate this effect; `PRECISION=fp32` is the fallback.
- File placement follows existing conventions: deploy assets under
  `deploy/object_detection/yolo26/ensemble/`, the benchmark under
  `benchmarks/yolo26-det/`, and integration scripts under `integration-tests/`.
- `backends/` is a new top-level directory, as anticipated by the ensemble
  roadmap Phase 2.2.
