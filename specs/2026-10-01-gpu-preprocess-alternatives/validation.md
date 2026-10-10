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
| V-7 | R-9, R-13 | detection parity | `python3 integration-tests/gpu-preprocess/compare_detections.py --candidates V2 --protocols http,grpc --output <result>/gates/detections.json` (one run may list several candidates) | Count and class equal, confidence ±0.04 (amended 2026-10-10, see Deviations), boxes ±12 px, on all fixtures and both protocols |
| V-8 | R-1, R-3, R-4, R-13 | detection parity | same as V-7 with `--candidates V12,V12-gpupost` (GPU-postprocess candidates are compared against `B1-gpupost`) | Same as V-7 |
| V-9 | R-5 | decode parity | `python3 integration-tests/gpu-preprocess/compare_decode.py --model yolo26det_nvjpeg_decode` *(new, T-19)* | Shape equals the original H×W×3; RGB order (error with R and B swapped > 2× the error); mean abs pixel diff versus DALI's decoder (`fn.decoders.image`, mixed, RGB — the B1 baseline; amended 2026-10-10) ≤ 1.0 and p99 ≤ 3, max recorded; OpenCV differences recorded, not gated; `ORIGINAL_SIZE` exact; grayscale fixture included |
| V-10 | R-6, R-7, R-13 | detection parity | same as V-7 with `--candidates V13,V13-gpupost` | Same as V-7 |
| V-11 | R-11 | negative tests | `python3 integration-tests/gpu-preprocess/invalid_inputs.py --variants B1,V12,V13,V2 --protocols http,grpc --output <result>/gates/invalid_inputs.json` | Every gated R-11 case has its expected outcome (V12/V13: Triton error, grayscale succeeds; V2: non-zero exit for degenerate and over-limit frames); `/v2/health/ready` stays true and the next valid request succeeds after every case; B1 and the truncated-mid-scan case are recorded, not gated |
| V-12 | R-14 | benchmark run | V-1 and V-2 commands with `--variants B0,B1,V12,V13,V2` | All five variants present in one `summary.json` from one GPU and one session |
| V-13 | R-11 | stability | `python3 integration-tests/gpu-preprocess/soak.py --variants V12,V13 --sequential 10000 --concurrency 8 --concurrent-requests 1000 --output <result>/gates/soak.json` | 0 errors; server success count grew by exactly the requests sent (a restart resets it); GPU memory after the run ≤ 1.05 × after warmup |
| V-14 | R-3, R-9 | memory safety | `integration-tests/gpu-preprocess/sanitize.sh all build-cuda` (memcheck + racecheck on `cuda_preprocess_dump`, memcheck on the V2 client, memcheck on `tritonserver` with only the yolopreprocess models loaded) | 0 errors reported |
| V-15 | R-14, R-15 | report generation | `python3 benchmarks/yolo26-det/compare.py <result_dir>` *(new, T-24)* | `comparison.md` regenerates byte-identical on rerun; includes the gate status of every variant and the cost-of-ownership table |
| V-16 | R-15 | review | Read `comparison.md` § Recommendation | Names one default and the conditions for each alternative; every claim cites a number from `summary.json` |
| V-17 | R-16, scope | scope diff | `git diff master...feature/gpu-preprocess-alternatives --stat`, reviewed against In/Out of Scope | No Python backend, no batching/video/PNG, no neuriplo-tasks change, no change to `*_dali_*` models; B0/B1 gates (V-0c, V-3) rerun green |
| V-18 | R-17 | CPU-only client | CUDA build (`TRITONIC_ENABLE_CUDA=ON`) in the SDK container **without** `--gpus`: `ldd build-cuda/tritonic`, `ctest`, then `tritonic` with `--preprocess_device=cpu` and with `--preprocess_device=cuda` | No `libcudart` load dependency (static cudart); ctest passes, GPU tests skip; the CPU run reaches the server connection; the CUDA run exits with the `RequireCudaDevice` message, no crash |

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
| T-3 | `dali_letterbox_spike.py` on Colab T4 (DALI 2.2.0, CUDA 13.3 toolkit, driver 580.82) | pass | 2026-10-09 | Pad rounding floor (7/7 geometries; ceil 4/7). Reference vs DALI on DALI's own decode: max 0 (bus, person, mug), 1/255 (horses). Degenerate 1×1000: DALI emits a tensor (recorded difference). |
| T-2 | `trt_letterbox_spike.py` (Python API runner) on Colab T4, TensorRT 11.2.1.2 | pass (after fix) | 2026-10-09 | First graph failed to parse (UINT8 `Transpose`); Cast-first graph builds; 9/9 cases vs reference (max 1/255, incl. 5×3 and 2160×3840), 4/4 vs DALI within D-5. |
| V-5 | `cuda_preprocess_dump` (nvcc 13.3, sm_75) vs DALI tensors, `compare_tensors.py` | pass | 2026-10-09 | bus max 0.114 / mean 0.0013 / p99 0.0157; horses 0.051 / 0.00034 / 0.0078; person 0.145 / 0.00083 / 0.0118; mug 0.0078 / 0.00006 / 0.0039 (limits 0.15 / 0.003 / 0.016). The residual is OpenCV vs nvJPEG decode; on DALI's own decode the math is exact (T-3). |
| V-14 (kernel) | `compute-sanitizer --tool memcheck` and `--tool racecheck` on `cuda_preprocess_dump` | pass | 2026-10-09 | `ERROR SUMMARY: 0 errors`; `RACECHECK SUMMARY: 0 hazards`. Client (V2) and server parts of V-14 not run. |
| — | **Brev A10G full gate** (`g5.2xlarge`, driver 595.91 with CUDA 13.4 forward compatibility, `tritonserver:26.08-py3` digest `sha256:9185ba5b…`, TensorRT 11.2.1.2, neuriplo-tasks v0.8.2, FP32 engines). Driver `integration-tests/gpu-preprocess/run_gpu_host_gate.sh`; curated evidence in [`evidence/2026-10-09-brev-a10g/`](evidence/2026-10-09-brev-a10g/). Three runs on one instance, 2026-10-09/10 UTC: run 1 all steps; run 2 `bench sanitize report` after fixing the sanitizer copy and `perf.py`; run 5 `gates` after the V-7/V-9 amendments and a `compare_decode.py` fix. The tree was f84bac5 plus the uncommitted changes committed alongside this log (the report's "0 dirty files" reflects the bundle having no `.git`). | — | 2026-10-10 | Total instance time about 1 h 20 min. |
| V-0 | `env_probe.sh` | pass | 2026-10-09 | A10G 23 GB, driver 595.91.07; server CUDA 13.4.1, TensorRT 11.2.1.2+cuda13.3; images consistent (A-4). |
| V-3/V-4 (GPU host) | `build` step: `build-sdk` and `build-cuda` + `ctest` in the SDK image | pass | 2026-10-09 | 111/111 both builds (includes the new YOLO26 pixel-coordinate test). |
| — | `triton` step | pass | 2026-10-09 | All 12 models READY in `tritonserver:26.08-py3`, including the custom backend (so nvJPEG is in the image — closes V-4's open point) and the regenerated DALI ensembles. |
| T-2, T-3 (rerun) | `spikes` step (`run_spikes.sh`) on the A10G | pass | 2026-10-09 | Same verdicts as on Colab. |
| V-5 | `tensors` step: DALI server tensor vs V12 server tensor, and vs the kernel tool (V2) | pass | 2026-10-09 | V12 vs DALI: max 0.0 on bus, mug, person; horses 1 of 1,228,800 elements differs by 1/255. V2 (OpenCV decode) vs DALI: bus 0.114 / 0.0013 / 0.0157, person 0.145 / 0.00083 / 0.0118, mug 0.0078 / 0.000056 / 0.0039 (max / mean / p99; limits 0.15 / 0.003 / 0.016). |
| V-7, V-8, V-10 | `compare_detections.py` (run 5), confidence ±0.04, boxes ±12 px | pass | 2026-10-10 | 60/60: V2, V12, V12-gpupost, V13, V13-gpupost × 6 fixtures × HTTP/gRPC. B1 and B1-gpupost vs B0: 24/24 (absolute check of the YOLO26 postprocessing). Run 1 at ±0.03 failed only bus_1080p (B1 vs B0 0.0323, V2 vs B1 0.0314; [`run1-gates-failures.txt`](evidence/2026-10-09-brev-a10g/run1-gates-failures.txt)); see Deviations. |
| V-9 | `compare_decode.py` vs DALI decoder (run 5) | pass | 2026-10-10 | Bit-exact on all 7 fixtures, including 1080p, 2160p and grayscale (max 0). Versus OpenCV (informational): bus p99 7, max 41, mean 0.60. |
| V-11 | `invalid_inputs.py` (run 5) | pass | 2026-10-10 | 42/42 results pass, 26 of them gated: V12 and V13 12 gated each (empty, header-only, not-JPEG, truncated mid-scan, grayscale, 1×1000), V2 2 gated; B1's 12 results and the over-limit cases (4097×64, 8193×64) are recorded as reference behaviour, not gated. The server was ready and served a valid request after every case. |
| V-13 | `soak.py` 10,000 sequential + 1,000 concurrent per variant (runs 1 and 5) | pass | 2026-10-10 | V12 and V13: 11,000/11,000 served, 0 errors, GPU memory flat (940 MiB after warmup and after, in runs 1 and 5). |
| V-1, V-2, V-12 | `bench` step (run 2): `run.py` 5 warmup + 100 iterations × 6 fixtures × HTTP/gRPC; `perf.py` concurrency 1/4/8 on bus_1080p | pass, one cell failed | 2026-10-10 | All 8 variants in one `summary.json`. perf_analyzer exited without a report for B0 at concurrency 8 (recorded as an error in `perf.json`); all other cells measured. V2 is not measurable by perf_analyzer (client-side preprocessing). |
| V-14 | `sanitize.sh` (run 2): kernel memcheck + racecheck, V2 client memcheck, server memcheck on the custom backend | pass | 2026-10-10 | `ERROR SUMMARY: 0 errors` (kernel, client, server); `RACECHECK SUMMARY: 0 hazards`. Run 1's server check failed because the TensorRT image's `compute-sanitizer` on PATH is a wrapper; fixed in `sanitize.sh`. |
| V-15 | `compare.py` twice on the merged result directory | pass | 2026-10-10 | Byte-identical `comparison.md`. Its gate table shows B0/B1 as "not run" because it does not read `detections_B1_vs_B0.json`; that file passed 24/24. |
| V-16 | Review of `comparison.md` § Recommendation | pass | 2026-10-10 | Default B1 (DALI); V2 for co-located clients; V12 as the DALI-free option; V13 retired. Every claim cites a number from the report. |
| V-18 | CUDA build in the SDK container without `--gpus` | pass | 2026-10-10 | `ldd` shows no `libcudart` (static cudart); 113/113 tests (default build 111/111); output in [`v18-cpu-only-client.txt`](evidence/2026-10-09-brev-a10g/v18-cpu-only-client.txt); `--preprocess_device=cpu` reaches the server connection; `--preprocess_device=cuda` exits with "needs an NVIDIA GPU and driver … Use --preprocess_device=cpu". |
| V-0c | `compare_dali.py` B1 tensor parity on 26.08 | not run | — | Not part of the gate driver. B1's end-to-end correctness on 26.08 is covered by B1 vs B0 detection parity (24/24) and by V12 matching DALI bit for bit. |
| M-1, M-2, M-3 | Manual checks | not run | — | Need a person at a GPU host. |

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
- Colab partial gate (2026-10-09) changed two implementation details, both before any
  GPU gate had passed with the old form: the letterbox ONNX graph casts before
  transposing, and the V13 profile minimum is 1×1 instead of 32×32. Neither changes a
  criterion.
- V-7/V-8/V-10 use the new `compare_detections.py` instead of extending
  `yolo-ensemble/compare_inference.py`. It imports that script's `compare()`, so
  the comparison is the same code, not a copy; the tolerances became keyword
  parameters whose defaults keep the original DALI acceptance at 0.03.
- **Amended after a failing run (Brev A10G, 2026-10-09/10), approved by the
  maintainer on 2026-10-10.** These two criteria were changed *after* the first
  full gate run failed them, so the reasons are recorded with the evidence:
  - V-7/V-8/V-10 confidence tolerance 0.03 → 0.04. On `bus_1080p.jpg` the two
    reference pipelines disagree with each other: B1 (DALI, nvJPEG) vs B0 (OpenCV,
    libjpeg-turbo) differ by 0.0323 in confidence with boxes within 1 px, over HTTP
    and gRPC, for both CPU and GPU postprocessing. V2 (OpenCV decode) vs B1 differs
    by 0.0314 on the same fixture. Every other fixture passed at 0.03. A tolerance
    the baselines themselves fail measures decoder noise, not the candidate.
  - V-9 reference OpenCV → DALI's decoder. Against OpenCV, `bus.jpg` gave p99 7
    (limit 3), max 41, mean 0.60, while the same model's preprocessing (V12) was
    bit-exact with DALI in V-5 (max 0.0). The production baseline B1 decodes with
    DALI, so DALI is the reference that tests "the decode step matches the
    baseline"; the OpenCV numbers stay in the report as information.
- **neuriplo-tasks pin v0.8.0 → v0.8.2 (2026-10-10).** The first Brev run showed
  YOLO26 boxes 640× too large. neuriplo-tasks v0.8.0 multiplied the NMS-free
  `[1,300,6]` output by the input size, which is correct only for exports with
  normalized coordinates (TFLite); Ultralytics ONNX/TensorRT `end2end` exports emit
  input pixels. v0.8.1 detects the convention per tensor; v0.8.2 adds only
  model-input-type handling, which does not affect any model here (all image inputs
  are FP32 or encoded rank-1 UINT8). This changes a dependency pin, not
  neuriplo-tasks (V-17 scope holds). The DALI GPU postprocess step now passes
  `normalized_boxes=False` for the same reason; `test_yolo26_postprocess.cpp` locks
  in the pixel convention.
- **R-17 added (2026-10-10, maintainer):** CUDA-enabled builds must keep CPU-only
  clients working. `tritonic_cuda` links the static cudart, and
  `--preprocess_device=cuda` checks for a device at startup.
- Gate harness fixes found on the GPU host, none changing a criterion: Triton
  26.08 needs `--allow-client-shm=true` for V2; DALI models need all outputs
  requested; detection checks use `--model_type=yolo26`; the sanitizer copies the
  real `compute-sanitizer` directory, not the PATH wrapper; `perf.py` records a
  missing perf_analyzer report as an error.

## Definition of Done (integration)
- [ ] Every automated and manual check executed, with evidence recorded above
- [ ] Evidence traced to R-n; deviations documented
- [ ] Spec, code, roadmap, and changelog agree — merge as one coherent change
- [ ] Durable discoveries (spike results, chosen default) propagated to
      `specs/tech-stack.md`
