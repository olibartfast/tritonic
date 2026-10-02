# Tech Stack — Tritonic

## Fixed (must not be changed per feature)
- Language: C++20 (`CMAKE_CXX_STANDARD 20`); CMake build; OpenCV.
- Triton client libraries (HTTP and gRPC), extracted from the
  `tritonserver:<ver>-py3-sdk` image (`docker/scripts/extract_triton_libs.sh`).
- Reference Triton server image: `nvcr.io/nvidia/tritonserver:26.08-py3`;
  TensorRT image of the same tag for engine builds.
- Model pre/postprocessing and `TaskFactory`: `neuriplo-tasks` (FetchContent).
  New model types go there, not in tritonic.
- Testing: GoogleTest/GMock unit tests (`BUILD_TESTING=ON`); Python
  integration/benchmark scripts under `integration-tests/` and `benchmarks/`.
- Formatting: clang-format via `scripts/format_all.sh` / git hook.

## Preferred
- Backend abstraction via `IInferenceBackend` / `ITriton` (Strategy pattern) so
  that new paths are mockable.
- Server-side GPU work is exposed as ordinary Triton models (ensembles), so that
  the client needs no ensemble-specific types.
- Opt-in CLI flags with the current behavior as the default.

## Open (decide per feature, record the decision)
- GPU preprocessing implementation for ensembles: DALI is current. CUDA C++
  alternatives are under evaluation in `specs/2026-10-01-gpu-preprocess-alternatives/`.

## Optional build features
- `TRITONIC_ENABLE_CUDA` (default OFF): CUDA toolkit, `src/cuda/` (the shared letterbox
  kernel), and CUDA shared memory. It also defines `TRITON_ENABLE_GPU` so the Triton
  client's `cudaIpcMemHandle_t` matches the GPU-enabled client libraries.

## Explicit Non-Choices
- No `nlohmann/json` or `cpp-base64` in `ChatBackend` — keeps the chat backend
  dependency-free beyond CURL.
- No CUDA dependency in `neuriplo-tasks` — CUDA-specific client code lives in
  tritonic under `src/cuda/` behind a build option (decided 2026-10-01).
- No Python backends for GPU preprocessing — server-side preprocessing is DALI
  or CUDA C++ (decided 2026-10-01).

## Constraints (cross-feature)
- Tolerances in `integration-tests/yolo-ensemble/compare_dali.py` and
  `compare_inference.py` are the parity bar; do not loosen them per feature.
- Serialized DALI pipelines, DALI plugins, and TensorRT engines are coupled to
  the Triton/TensorRT release that builds them; rebuild them on every image bump.
- Performance evidence must record GPU, driver, image tag, and commit.

## Open Questions
- [Q-1] Resolved 2026-10-02: `TRITONIC_ENABLE_CUDA` is now a CMake option (R-12).
  `--shared_memory_type` alone still has no effect outside `--preprocess_device=cuda`
  (finding). Wiring it for the other paths is not planned.

_Revision: 2026-10-02 — added the TRITONIC_ENABLE_CUDA build feature and resolved Q-1._
_Revision: 2026-10-01 — initial version; records the 26.08 image baseline and the
CUDA/Python non-choices from the GPU preprocessing interview._
