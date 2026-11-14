# Requirements — Native Triton vLLM text generation

Spec: `specs/2026-11-14-triton-vllm-generation/` · Target server image:
`nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3`

## Goal

Let TritonIC submit one raw text prompt to a vLLM model loaded in Triton using
Triton’s HTTP `/generate` endpoint, then print the generated text. This is an
additive CLI/backend path for the plain Triton server in the vLLM container; it
does not use the optional OpenAI-compatible frontend.

## Scope

- Single request / single response over HTTP `POST
  /v2/models/{model}/generate` for an already-loaded model.
- Raw prompt text in `text_input`; no client-side chat-template rendering.
- Synchronous, non-streaming output; server-side vLLM batching remains internal.
- Existing Triton vision inference (`--backend=triton`) and OpenAI-compatible
  chat (`--backend=chat`) remain available with their existing defaults.
- No model repository management, model loading, or `TaskFactory` integration.

## Requirements

### Command and routing

- **R-1** Add `--backend=triton-generate`. Its target is derived from the
  existing `--serverAddress` and `--port`; scheme and route are fixed to HTTP
  and Triton’s generate route. Defaults remain `localhost:8000`.
- **R-2** Require `--protocol=http`, a non-empty `--model`, and a non-empty
  `--text_prompt`. Return generated text on stdout with one presentation
  newline. Route informational logs and errors to stderr (or the configured
  log file) in this mode. A request or config error exits nonzero and does not
  print partial output.
- **R-3** Accept a DNS host, IPv4 literal, or bracketed IPv6 literal in
  `--serverAddress`, without scheme, path, query, fragment, user info, or an
  embedded port. Require `--port` in 1–65535. Invalid host/port/model values
  fail before network access. Encode the model as exactly one URL path segment:
  reject empty, `.` and `..`, slash, and backslash; percent-encode UTF-8 bytes
  except RFC 3986 unreserved bytes.
- **R-4** Unknown backend names fail with an actionable diagnostic; they must
  not fall through to the vision inference path. Selecting
  `triton-generate` when `WITH_TRITON_GENERATE_BACKEND=OFF` fails explicitly.
- **R-5** Reject `--api_endpoint`, `--api_service`, and `--api_key_env` when
  explicitly supplied for this backend. This path does not expose TLS,
  authentication, or a proxy endpoint. Accept ordinary defaults for unrelated
  options without creating image/video output (including `write_frame=true`).
- **R-6** Reject explicitly supplied options that select another input or
  inference path: `--source`, `--model_type`, `--labelsFile`, `--input_sizes`,
  `--task_model`, `--postprocess_mode`, `--segmentation_output`,
  `--show_frame`, `--benchmark_warmup`,
  `--benchmark_iterations`, `--benchmark_output`, and `--target_image_size`.
  Also reject `--text_input`, `--audio_input`, `--enable_multimodal=true`,
  `--interactive=true`, `--input_mode=encoded-image`,
  `--preprocess_device=cuda`, non-`none` `--shared_memory_type`,
  non-1 `--batch_size`, non-default CUDA device,
  `--stop_words`, and `--repetition_penalty`. Rejecting `stop_words` and
  `repetition_penalty` is based on explicit presence, even when the supplied
  value is empty or equals its current default; unsupported options must not be
  silently ignored. Accept defaults that are inert for this backend, including
  `write_frame=true`, `input_mode=preprocessed`, `batch_size=1`,
  `preprocess_device=cpu`, `shared_memory_type=none`, default thresholds, and default
  modality values while multimodal mode is off. Reject non-default confidence
  or NMS thresholds and non-default modality values while multimodal mode is
  off. Tests cover long and short aliases and explicit default-valued
  unsupported options.
- **R-7** Validate `max_tokens > 0`, finite `temperature >= 0`, finite
  `0 < top_p <= 1`, and `inference_timeout >= 0`. A zero timeout means no
  client-side deadline; positive values set a total request deadline in
  milliseconds. Timeouts do not promise server-side generation cancellation.

### Backend contract and HTTP exchange

- **R-8** Add dedicated `tritonic::core::GenerateRequest` and
  `GenerateResponse` value types and include them in `BackendRequest` and
  `BackendResponse`. A request contains model, raw prompt, max tokens,
  temperature, and top-p. A response contains text, success, error, and HTTP
  status (0 when no HTTP response was received). Preserve existing alternatives
  in both variants; dispatch uses `std::get_if`/`std::visit`, and unsupported
  request alternatives fail clearly. Preserve an observed HTTP status even if
  a transfer subsequently fails; use status 0 only when no status was
  observed. Validate host, model, prompt, and generation values at the backend
  boundary too, so callers that bypass CLI validation cannot send invalid
  sampling parameters or route segments.
- **R-9** Implement an independently selectable
  `tritonic::triton::TritonGenerateBackend`. Use CURL and the already-required
  RapidJSON dependency. Do not add `nlohmann/json` or `cpp-base64`, and do not
  introduce those dependencies into `ChatBackend`.
- **R-10** POST JSON to `/v2/models/{encoded-model}/generate` with
  `Content-Type: application/json` and `Accept: application/json`. Preserve the
  prompt as UTF-8 JSON string content without trimming or rewriting. The body
  carries top-level `text_input`, `stream: false`, and
  `exclude_input_in_output: true`; its `parameters` object carries
  `max_tokens`, `temperature`, and `top_p`. The explicit exclude flag prevents
  Triton’s vLLM backend from prepending the input prompt to generated output.
  Do not retry, follow redirects, or send a streaming request.
- **R-11** Treat only HTTP 200 with a valid JSON object containing a string
  `text_output` and no `error` member as success. Empty `text_output` is valid
  and is preserved exactly. Any HTTP status other than 200, any `error`
  member, malformed JSON, invalid UTF-8/JSON, missing `text_output`, or a
  non-string `text_output` is a failure. Do not extract text from error bodies
  as a successful response.
- **R-12** Bound the response body at 16 MiB, accepting exactly that limit and
  failing once a byte beyond it is received. Bound error diagnostics to 1024
  bytes; never log or return an unbounded raw response body. Handle CURL
  initialization, DNS/connect, disconnect, HTTP, timeout, and response-limit
  failures without crashing. Disable signals in CURL so timeout behavior is
  safe in a multithreaded client.
- **R-13** Print only `text_output` plus the CLI presentation newline on success.
  Do not strip whitespace or add the input prompt; prompt exclusion is set in
  the request itself.

### Build, compatibility, and tests

- **R-14** Add `WITH_TRITON_GENERATE_BACKEND` (default `ON`), independent from
  `WITH_CHAT_BACKEND`. When disabled, omit generation backend sources and tests
  and retain a clear runtime selection diagnostic. Keep existing Triton and
  chat behavior unchanged. Repair only the source/test guards needed for all
  four combinations of the two options to configure, build, and test.
- **R-15** Add CPU-only unit/config/CLI tests and a deterministic loopback HTTP
  test. The loopback test checks the actual path, method, headers, JSON fields,
  escaping, response parsing, error handling, response bound, timeout, and a
  successful call after a failed call. Invalid configuration is rejected
  before connecting.
- **R-16** Provide an opt-in smoke procedure against a deployed vLLM model in
  the exact `26.10-vllm-python-py3` image, with readiness check, equivalent
  `curl` and TritonIC calls, and a failed-model/recovery check. Record image
  digest, GPU/driver, model ID and immutable model revision, and repository
  configuration. GPU smoke remains pending until run on a suitable host.
- **R-17** Document the command, server route, raw-prompt/chat-template boundary,
  build option, unsupported features, and the distinction from both
  standalone-vLLM OpenAI API and Triton’s optional OpenAI frontend. Verify
  edited documentation links.

## Out of scope

- `/generate_stream`, SSE, token streaming, cancellation, or gRPC.
- OpenAI `/v1/chat/completions` through Triton’s optional frontend; that route
  is already available through `--backend=chat` when that frontend is running.
- Conversation roles/history, system prompts, chat templates, tool calls,
  multimodal image/audio input, embeddings, and raw tensor inference.
- Client-side batching, retry policy, model management, or model-specific
  generation options beyond max tokens, temperature, and top-p.
- Updating the repository’s Triton SDK or default server image pins. The 26.10
  image is a compatibility target for this feature, not a global baseline bump.

## Verified contract and open validation

The r26.10 vLLM backend source declares `text_input` as a string input,
`text_output` as a string output, and optional `stream` and
`exclude_input_in_output` inputs. Its request implementation defaults to
prepending the prompt for a non-streaming request when the exclude input is
omitted, so the client sends `exclude_input_in_output: true` explicitly. The
Triton Generate extension documents one JSON response from `/generate`, while
`/generate_stream` is SSE; the extension is provisional and HTTP-specific.
These source checks establish the intended wire contract. They do not replace
the live 26.10 image smoke test in R-16.

Sources:

- [Triton Generate extension](https://github.com/triton-inference-server/server/blob/r26.10/docs/protocol/extension_generate.md)
- [vLLM request handling, r26.10](https://github.com/triton-inference-server/vllm_backend/blob/r26.10/src/utils/request.py)
- [vLLM model I/O contract, r26.10](https://github.com/triton-inference-server/vllm_backend/blob/r26.10/src/model.py)
- [vLLM sample model config, r26.10](https://github.com/triton-inference-server/vllm_backend/blob/r26.10/samples/model_repository/vllm_model/config.pbtxt)
