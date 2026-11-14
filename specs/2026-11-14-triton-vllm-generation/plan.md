# Plan — Native Triton vLLM text generation

Requirements: [requirements.md](requirements.md) · Design:
[design.md](design.md) · Proof: [validation.md](validation.md)

The work is additive and ordered by dependency. CPU tests establish the request
contract and CLI behavior before an opt-in GPU smoke. **[GPU]** marks work that
needs a host able to run the specified Triton/vLLM image and model.

## Phase 0 — Confirm external contract and repository baseline

- **T-1** Record the official r26.10 request/model contract and the current
  build/test baseline for both `WITH_CHAT_BACKEND` values. Keep observations
  separate from proposed behavior. Resolve existing chat-OFF failures before
  using that configuration as a feature regression gate. Do not change version
  pins.
- Exit: V-1 has source references and recorded baseline results; the
  requirements/design agree on endpoint, fields, limits, prompt handling, and
  option policy.

## Phase 1 — Add the HTTP generation strategy

- **T-2** Add `GenerateRequest` and `GenerateResponse` to
  `include/tritonic/core/types.hpp` and extend `BackendRequest` /
  `BackendResponse`. Update existing adapter diagnostics for unexpected
  variants so they are no longer chat-specific.
- **T-3** Add `tritonic::triton::TritonGenerateBackend` in a focused public
  header/source pair. Implement host/model validation, RFC 3986 path-segment
  encoding, strict RapidJSON UTF-8 serialization/parsing, CURL POST, bounded
  body callback, timeout, error handling, and no-retry/no-redirect behavior.
  Use the exact request/response contract in R-8–R-13.
- **T-4** Add focused CPU tests for URL/path validation, JSON escaping and
  parsing, numeric values, empty output, wrong schema, embedded NUL/Unicode,
  errors, status handling, diagnostic cap, and exact response-size boundary.
- Exit: V-2 passes without a running Triton server or GPU.

## Phase 2 — Build option and application routing

- **T-5** Add `WITH_TRITON_GENERATE_BACKEND` (default `ON`), independent of
  `WITH_CHAT_BACKEND`, and conditionally wire sources and tests into the
  executable and applicable test targets. Make the minimum source/test guard
  corrections needed to configure/build/test all four option combinations.
  With the new feature disabled, the CLI reports the unavailable feature.
- **T-6** Extend `InferenceConfig` only where needed. Add backend-specific
  preflight validation in `ConfigManager`: backend allowlist, HTTP-only target,
  host/port/model/prompt validation, numeric bounds, and explicit unsupported
  option detection including parser aliases. Preserve default Triton and chat
  routing and prevent unknown backends from falling through to vision.
- **T-7** In `src/main/client.cpp`, construct the generation request/backend,
  handle the typed response, route mode-specific logs and errors to stderr, and
  print only successful output plus a newline. Failure returns nonzero without
  partial stdout.
- **T-8** Extend config/CLI tests for required values, invalid targets,
  numerical boundaries, supported defaults, explicit unsupported options,
  aliases, feature-off selection, backend dispatch, and no-network rejection.
- Exit: V-3 and V-5 pass for all supported feature/chat option combinations;
  default vision and chat tests continue to pass.

## Phase 3 — Exercise the HTTP boundary

- **T-9** Add a deterministic loopback HTTP fixture bound to an ephemeral port.
  Capture request method, path, headers, and JSON body. Return configured JSON,
  status, delay, disconnect, and body-size scenarios. Avoid dependence on an
  external network service.
- **T-10** Verify successful generation, escaping/UTF-8, repeated invocation,
  empty output, non-200, Triton error object, malformed/trailing JSON,
  missing/wrong-type output, connection failure, timeout, exact size limit and
  overflow, and successful recovery after a failed call.
- Exit: V-4 passes deterministically on CPU; invalid CLI configuration reaches
  no server.

## Phase 4 — Documentation and deployed-server smoke

- **T-11** Update the user-facing CLI documentation and build-option reference
  with a generation example, image and endpoint distinction, raw prompt
  boundary, accepted parameters, and unsupported modes. Verify every edited
  relative and GitHub hyperlink.
- **T-12 [GPU]** Run the opt-in smoke in V-6 with
  `nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3` and a model repository
  derived from the upstream r26.10 vLLM sample. Pin and record an immutable
  model revision chosen for the available hardware. Confirm readiness, direct
  `curl`, equivalent TritonIC request, model-not-found failure, and a valid
  request after the failure. Record image digest and host/model metadata.
- Exit: V-7 passes; V-6 remains explicitly pending until the GPU run is
  recorded. No compatibility claim is made from source inspection alone.

## Rollback

The backend, its CLI selection, and its source list are additive and guarded by
`WITH_TRITON_GENERATE_BACKEND`. Turning the option off removes the generation
backend at build time and leaves the existing default Triton and chat choices
available. No server model files, dependency pins, or persistent formats
change. The only shared changes are the core request/response variants, generic
adapter diagnostics, and explicit backend routing; V-3/V-5 guard these paths.
