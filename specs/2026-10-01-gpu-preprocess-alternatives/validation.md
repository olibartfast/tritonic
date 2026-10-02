# Validation — GPU preprocessing alternatives to DALI

Written 2026-10-01, before implementation. Criteria change only through a
recorded discovery, with requirements and plan updated in the same branch.
Command lines were updated on 2026-10-02 to the script names chosen during
implementation; the one criterion change is listed under Deviations.

All commands run from the repository root. **[GPU]** checks need the
maintainer's NVIDIA host with the Triton 26.08 server running the
`deploy/object_detection/yolo26/ensemble/model_repository`. Scripts marked
*(new, T-n)* are created by that plan task. Their command lines are fixed here so
that the check is defined before the code exists.

## Baseline (record before Phase 0 code lands)

| ID | Check | Command | Expected |
|---|---|---|---|
| V-0 | Environment identity, including A-4 | `benchmarks/yolo26-det/env_probe.sh` *(new, T-1)* | GPU, driver, CUDA, TRT and image digests recorded; server and TensorRT images report the same CUDA and TRT versions |
| V-0b | Unit-test reference run on the unmodified tree | `cmake -S . -B build -DBUILD_TESTING=ON && cmake --build build -j && ctest --test-dir build --output-on-failure` | Pass count and any pre-existing failures recorded |
| V-0c | B1 tensor parity on 26.08 (regenerated DALI artifact) | `integration-tests/yolo-ensemble/compare_dali.py` (as in the YOLO ensemble README) | Pass; becomes the reference for V-5 |

## Requirements-to-evidence matrix

| ID | Req | Evidence type | Exact command or check | Pass criterion |
|---|---|---|---|---|
| V-1 | R-14 | benchmark run | `python3 benchmarks/yolo26-det/run.py --variants B0,B1 --protocols grpc,http --label <gpu>-2608` | `summary.json` has `schema_version: 4`, environment block, per-model server stats, and both variants on all 6 fixtures |
| V-2 | R-14 | throughput | `python3 benchmarks/yolo26-det/perf.py --variants B0,B1 --concurrency 1,4,8 --label <gpu>-2608` | infer/s and p95 recorded per concurrency, plus GPU memory and utilization samples |
| V-3 | R-12, R-16 | regression | V-0b command with default options | Same pass set as V-0b; `git diff` shows no change to default-path behavior |
| V-4 | R-12 | build | In `nvcr.io/nvidia/tritonserver:26.08-py3-sdk` (it has nvcc, OpenCV and the client libraries; install `cmake ninja-build libcurl4-openssl-dev rapidjson-dev libprotobuf-dev protobuf-compiler`): `TritonClientBuild_DIR=/workspace/install cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DTRITONIC_ENABLE_CUDA=ON && cmake --build build-cuda && ctest --test-dir build-cuda`; backend: `backends/yolopreprocess/build_backend.sh` | Exit 0; tests pass; backend exports only `TRITONBACKEND_*` |
| V-5 | R-2, R-10 | tensor parity | `python3 integration-tests/gpu-preprocess/compare_tensors.py --reference /tmp/tritonic-spikes/dali --actual <variant_dir> --output <result>/gates/tensors_<VARIANT>.json` for the kernel dump (`build-cuda/cuda_preprocess_dump`, V2/V12); the V13 graph is covered by `trt_letterbox_spike.py --dali-dir` (T-2) | Every fixture: `max_abs ≤ 0.15`, `mean_abs ≤ 0.003`, `p99_abs ≤ 0.016`, shape `[3,640,640]` |
| V-6 | R-8, R-9, R-2 | unit tests | `ctest --test-dir build -R "Config\|AppDevicePreprocess\|Letterbox" --output-on-failure` | New cases pass: default `cpu`; `cuda` accepted only with preprocessed + cuda shm + yolo/yolo26; each invalid combination rejected; App calls the device preprocessor and `inferFromDeviceInput` once per frame and never `infer`; shared letterbox math meets D-5 against the neuriplo-tasks CPU preprocessor |
| V-7 | R-9, R-13 | detection parity | `python3 integration-tests/gpu-preprocess/compare_detections.py --candidates V2 --protocols http,grpc --output <result>/gates/detections.json` (one run may list several candidates) | Count and class equal, confidence ±0.03, boxes ±12 px, on all fixtures and both protocols |
| V-8 | R-1, R-3, R-4, R-13 | detection parity | same as V-7 with `--candidates V12,V12-gpupost` (GPU-postprocess candidates are compared against `B1-gpupost`) | Same as V-7 |
| V-9 | R-5 | decode parity | `python3 integration-tests/gpu-preprocess/compare_decode.py --model yolo26det_nvjpeg_decode` *(new, T-19)* | Shape equals the original H×W×3; RGB order (error with R and B swapped > 2× the error); mean abs pixel diff versus OpenCV ≤ 1.0 and p99 ≤ 3, max recorded; `ORIGINAL_SIZE` exact; grayscale fixture included |
| V-10 | R-6, R-7, R-13 | detection parity | same as V-7 with `--candidates V13,V13-gpupost` | Same as V-7 |
| V-11 | R-11 | negative tests | `python3 integration-tests/gpu-preprocess/invalid_inputs.py --variants B1,V12,V13,V2 --protocols http,grpc --output <result>/gates/invalid_inputs.json` | Every gated R-11 case has its expected outcome (V12/V13: Triton error, grayscale succeeds; V2: non-zero exit for degenerate and over-limit frames); `/v2/health/ready` stays true and the next valid request succeeds after every case; B1 and the truncated-mid-scan case are recorded, not gated |
| V-12 | R-14 | benchmark run | V-1 and V-2 commands with `--variants B0,B1,V12,V13,V2` | All five variants present in one `summary.json` from one GPU and one session |
| V-13 | R-11 | stability | `python3 integration-tests/gpu-preprocess/soak.py --variants V12,V13 --sequential 10000 --concurrency 8 --concurrent-requests 1000 --output <result>/gates/soak.json` | 0 errors; server success count grew by exactly the requests sent (a restart resets it); GPU memory after the run ≤ 1.05 × after warmup |
| V-14 | R-3, R-9 | memory safety | `integration-tests/gpu-preprocess/sanitize.sh all build-cuda` (memcheck + racecheck on `cuda_preprocess_dump`, memcheck on the V2 client, memcheck on `tritonserver` with only the yolopreprocess models loaded) | 0 errors reported |
| V-15 | R-14, R-15 | report generation | `python3 benchmarks/yolo26-det/compare.py <result_dir>` *(new, T-24)* | `comparison.md` regenerates byte-identical on rerun; includes the gate status of every variant and the cost-of-ownership table |
| V-16 | R-15 | review | Read `comparison.md` § Recommendation | Names one default and the conditions for each alternative; every claim cites a number from `summary.json` |
| V-17 | R-16, scope | scope diff | `git diff master...feature/gpu-preprocess-alternatives --stat`, reviewed against In/Out of Scope | No Python backend, no batching/video/PNG, no neuriplo-tasks change, no change to `*_dali_*` models; B0/B1 gates (V-0c, V-3) rerun green |

## Manual checks
- [ ] [M-1] **[GPU]** Run the README command for each new ensemble on
  `data/images/bus.jpg`, then view the rendered output image next to the B1
  output. Boxes and labels should be visually identical.
- [ ] [M-2] **[GPU]** Restart Triton with only the V12 models plus the TRT model,
  then unload and reload `yolo26det_cuda_preprocess` through the repository API.
  The model returns to ready and the next request succeeds.
- [ ] [M-3] Follow `deploy/object_detection/yolo26/ensemble/README.md` from a
  clean checkout. Every command must exist and no undocumented step is needed.

## Evidence Log
| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| C-1 | `python3 integration-tests/gpu-preprocess/check_letterbox_onnx.py` and `--pad-rounding ceil` (onnxruntime 1.30 CPU) | pass | 2026-10-02 | 13 image cases within D-5 (max 1/255, 0 on most); 1×1000 and 1000×1 rejected. Proves the V13 graph encodes R-2; TensorRT acceptance is still T-2. |
| C-2 | Host build of `letterbox_pixel.hpp` (g++ 15, scratch harness) against `letterbox_reference.py` | pass | 2026-10-02 | 6 photos + 6 synthetic sizes up to 3840×2160, RGB and BGR input: max 1/255; degenerate input rejected. The in-repo version of this check is `LetterboxHostTest` (V-6), not yet run. |
| C-3 | ONNX merge wiring: `generate_trt_pre_onnx.py` on a stand-in model whose `output0` is the first 1800 input values (onnxruntime) | pass | 2026-10-02 | Merged graph takes `image` UINT8, adopts opset 19, output equals the reference letterbox exactly. |
| C-4 | `setup_model_repository.sh --engine … --trt-pre-onnx …` with `docker` stubbed, into a scratch repository | pass | 2026-10-02 | All 7 variant models staged with configs, backend `.so` and engine; default `model_repository/` unchanged. Docker steps (DALI serialization, builds) not exercised. |
| C-5 | `compare_tensors.py` on synthetic tensors | pass | 2026-10-02 | Reports pass within limits, fail over the mean limit, fail for a missing tensor; exit 1. |
| C-6 | `compare.py` twice on a synthetic result directory | pass | 2026-10-02 | Byte-identical `comparison.md` (md5); a failed variant and a failed protocol are shown, not dropped. |
| V-0b | Untouched `master` (a802184) built against the 26.08 SDK, CI settings (no build type; then `-DTRITONIC_STRICT_WARNINGS=ON`) | **fail** (pre-existing) | 2026-10-02 | Link fails on 4 undefined abseil `CHECK` helpers; the strict build fails with 250 `-Werror` errors in the SDK's abseil headers. The 26.08 bump breaks CI's build regardless of this feature; fixed on this branch (see Deviations). |
| V-3 | Default build, SDK image, Release + `BUILD_TESTING=ON`: `ctest --test-dir build-sdk` | pass | 2026-10-02 | 110/110, including the existing suites unchanged. |
| V-3b | CI job replicas: `cmake -S . -B build-ci -DTRITONIC_STRICT_WARNINGS=ON` (CUDA OFF and ON) and `cmake -S . -B build-ci-test -DBUILD_TESTING=ON` + `ctest` | pass | 2026-10-02 | Both strict builds link; the CI test job passes 110/110. |
| V-4 | `TRITONIC_ENABLE_CUDA=ON` Release build + tests in the SDK image; `build_backend.sh` in `tensorrt:26.08-py3` | pass | 2026-10-02 | `tritonic`, `cuda_preprocess_dump`, `libtritonic_cuda.a` built for sm_86; 110/110 tests; backend built with `-Werror`, exports only the 6 `TRITONBACKEND_*` entry points, links `libcudart.so.13`/`libnvjpeg.so.13` (CUDA 13.4, same as the SDK). Not yet checked: that `libnvjpeg.so.13` is present in `tritonserver:26.08-py3` (image not pulled here). |
| V-6 | `ctest` subsets `Config`, `AppDevicePreprocess`, `Letterbox` within the V-3/V-4 runs | pass | 2026-10-02 | 4 new config tests, 4 App device-path tests (mocked device input and preprocessor: preprocessed once per frame, no host `infer`, failure without a preprocessor or device-capable client, CPU mode ignores an injected preprocessor), 6 letterbox tests including D-5 parity of the shared kernel math against the neuriplo-tasks CPU preprocessor on the 4 fixtures. |
| — | All [GPU] rows | not run | — | No NVIDIA GPU on the implementation machine. |

## Deviations
- V-9 criterion revised on 2026-10-02, before the check existed or ran: "max abs ≤ 2"
  became "mean ≤ 1.0, p99 ≤ 3, max recorded", plus an explicit channel-order check.
  Reason: nvJPEG and libjpeg-turbo differ in IDCT and chroma upsampling, so isolated
  pixels of a correct decode can differ by more than 2 levels. DALI also decodes with
  nvJPEG and passed tensor parity against the CPU path, so a max-only gate would
  measure the decoder library rather than the decode step. Detection parity (V-10)
  remains the end-to-end gate for V13.
- Build fixes needed by the 26.08 SDK, found while running V-0b/V-3 (none changes
  runtime behavior). (1) The SDK include directory is now a `SYSTEM` include, so its
  abseil headers are not held to `TRITONIC_STRICT_WARNINGS`. (2) Builds without
  `CMAKE_BUILD_TYPE` (as in CI) link the SDK's static abseil `CHECK` libraries through
  the `tritonic_sdk_absl` interface target. (3) The unit tests take the SDK headers
  as `-isystem` after googletest. The SDK ships its own `gtest/gmock` headers, which
  shadowed the linked googletest 1.14 and made every gMock expectation segfault
  (latent: no test had used `EXPECT_CALL` before `test_app_device_preprocess.cpp`).
- Environment quirk, recorded on 2026-10-02: in the SDK image `LIBRARY_PATH` ends
  with an empty entry, so a `g++` invoked directly from the repository root reads the
  new `specs/` directory as a GCC specs file and aborts. CMake builds run in the
  build directory and are unaffected.
- V-7/V-8/V-10 use the new `compare_detections.py` instead of extending
  `yolo-ensemble/compare_inference.py`. It imports that script's `compare()`, so
  the tolerances are the same code, not a copy.

## Definition of Done (integration)
- [ ] Every automated and manual check executed, with evidence recorded above
- [ ] Evidence traced to R-n; deviations documented
- [ ] Spec, code, roadmap, and changelog agree — merge as one coherent change
- [ ] Durable discoveries (spike results, chosen default) propagated to
      `specs/tech-stack.md`
