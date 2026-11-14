# Design — Native Triton vLLM text generation

Requirements: [requirements.md](requirements.md) · Plan:
[plan.md](plan.md) · Validation: [validation.md](validation.md)

## Decision summary

Add a small HTTP generation strategy alongside the image-oriented Triton and
OpenAI-compatible chat strategies. The selected route is Triton’s HTTP generate
extension on the ordinary Triton server port. It requires no change to the
Triton vLLM model repository and makes no assumption about a tokenizer or chat
template.

`--backend=triton-generate` chooses this strategy. The caller provides the
server host and HTTP port using the existing `--serverAddress` and `--port`
options. The route is fixed to `/v2/models/{model}/generate`; the caller does
not provide a full endpoint URL.

## Scope boundary

The ordinary Triton vLLM backend accepts a Triton Generate request at
`/v2/models/{model}/generate`. Triton’s optional OpenAI frontend instead serves
`/v1/chat/completions` (commonly on port 9000), and standalone vLLM typically
serves its own OpenAI-compatible API. Those are distinct routes and startup
configurations. Existing `--backend=chat` remains the path for either
OpenAI-compatible API; this backend speaks only to Triton’s Generate route.

The client sends one raw prompt string. The caller owns prompt rendering. A
user who needs roles, a system message, or a model-specific chat template must
render that into the prompt before passing it to TritonIC. The stock vLLM
backend receives the resulting raw text; no conversation state is inferred or
retained.

## Architecture and data flow

```text
ConfigManager
  --backend=triton-generate, host, port, model, text_prompt
      ↓ validate before network access
client.cpp
  GenerateRequest
      ↓
tritonic::triton::TritonGenerateBackend
  CURL POST /v2/models/{encoded-model}/generate
      ↓
plain Triton HTTP frontend → loaded vLLM model → JSON text_output
      ↓
GenerateResponse → stdout or nonzero exit
```

The backend is a focused HTTP facade and does not use the binary Triton client
library, `App`, model metadata lookup, or `TaskFactory`. It implements the
existing `IInferenceBackend` strategy boundary. Extend the core request/response
variants with `GenerateRequest` and `GenerateResponse`; preserve the current
`TritonInferRequest` and `ChatRequest` alternatives. Existing adapters must
report an unexpected request alternative generically rather than claim that
every wrong request is a chat request.

The request value contains:

```cpp
struct GenerateRequest {
    std::string model;
    std::string prompt;
    int max_tokens{256};
    float temperature{1.0f};
    float top_p{1.0f};
};

struct GenerateResponse {
    std::string text;
    bool success{false};
    std::string error;
    long http_status{0};
};
```

Names and exact placement follow the existing public type conventions in
`include/tritonic/core/types.hpp`. The backend is declared in
`include/tritonic/triton/triton_generate_backend.hpp` and implemented in
`src/triton/TritonGenerateBackend.cpp` under `tritonic::triton`.
Its public construction contract takes `(std::string host, int port,
int timeout_ms = 0)`, implements `infer(const BackendRequest&)` and reports
`backendName() == "triton-generate"`. Validate constructor target fields and
all request fields again in `infer`; the strategy interface can be called
without going through `ConfigManager`.

## Wire contract

The client builds the URL from a validated host, numeric port, and one encoded
model path segment. It sends UTF-8 JSON with `Content-Type` and `Accept` set to
`application/json`. The model name is in the URL, as required by Triton’s
Generate extension, rather than in the JSON body.

Request body:

```json
{
  "text_input": "raw prompt goes here",
  "stream": false,
  "exclude_input_in_output": true,
  "parameters": {
    "max_tokens": 256,
    "temperature": 1.0,
    "top_p": 1.0
  }
}
```

The `stream` and `exclude_input_in_output` fields correspond to optional vLLM
backend inputs. Numeric sampling values are forwarded as model parameters.
The explicit exclude flag matters: r26.10 vLLM request handling otherwise
prepends the prompt for a non-streaming request. `text_input` is not trimmed,
templated, or concatenated with another role. JSON encoding must preserve UTF-8
and escape quotes, backslashes, and control bytes correctly.

The response is successful only when the HTTP status is 200, the body is valid
UTF-8 JSON with no trailing non-whitespace data, the top-level value is an
object, no `error` member is present, and `text_output` is a string. An empty
string is a valid model output. The parser returns that string exactly,
including leading/trailing whitespace and embedded NUL bytes represented by
JSON escapes. The CLI adds one newline when printing.

All other outcomes become `GenerateResponse{success=false}` with a bounded
diagnostic and the observed status when available. Preserve a received HTTP
status if the transfer later fails; status 0 means no status was observed.
CURL, DNS, connection, timeout, disconnect, oversized-body, non-200, JSON
parse, schema, and Triton error-object cases are distinct enough to diagnose.
The final diagnostic, including any prefix and escaped server detail, is at
most 1024 bytes. It must not include an unbounded response body or accidentally
print a failed body as generated text.

## URL and transport rules

- Accept a plain DNS name, IPv4 literal, or bracketed IPv6 literal. Do not
  accept a URL scheme, user information, path, query, fragment, or an embedded
  port in the host option.
- Use port range 1–65535. `--protocol` must be `http`.
- Validate the model as one UTF-8 path segment. Reject empty, dot segments,
  slash, and backslash before percent-encoding all bytes except RFC 3986
  unreserved bytes. This prevents a model value from changing the route.
- Use a total CURL timeout only when `inference_timeout > 0`; zero leaves the
  timeout unset. Set `CURLOPT_NOSIGNAL`, disable redirect following, and do not
  retry a request. Do not claim that a client timeout cancels work already
  running in Triton/vLLM.
- The write callback enforces a 16 MiB body cap with checked size arithmetic.
  It accepts a body of exactly that size and aborts when more bytes arrive.

## CLI validation and compatibility

`ConfigManager` owns all preflight checks so invalid values fail before CURL is
initialized. New validation is scoped to `triton-generate`; it must not change
the meaning of existing vision or chat options. The backend needs
`--model` and `--text_prompt`; aliases `m`/`tp` behave identically.

Explicitly incompatible options are rejected as described by R-5/R-6. Detect
whether a value was supplied, including short aliases and explicit values equal
to defaults, rather than inferring explicitness from the parsed value. The
specific policy is:

| Option group | Policy for `triton-generate` |
|---|---|
| `backend`, `protocol`, `serverAddress`, `port`, `model`, `text_prompt`, `max_tokens`, `temperature`, `top_p`, `inference_timeout` | Supported; validate required values and ranges. |
| `api_endpoint`, `api_service`, `api_key_env` | Reject if explicitly supplied. |
| source/model-type/labels/input sizes/task model, postprocess or segmentation selection, show-frame, benchmark settings, target-image-size | Reject if explicitly supplied. |
| text/audio input, multimodal true, interactive true, encoded-image input, CUDA preprocessing, non-none shared memory, batch size other than 1, nondefault CUDA device | Reject when the unsupported mode is selected. |
| stop words, repetition penalty | Reject if explicitly supplied, even empty/default-valued. |
| write-frame, preprocessed input mode, batch size 1, CPU preprocessing, no shared memory, default thresholds/modality settings, verbosity and logging settings | Accept the inert/default settings; do not produce frame files. Nondefault confidence/NMS thresholds and nondefault modality values while multimodal is off are rejected. |

Aliases are covered by the same policy. The global `write_frame=true` default
must not create files or block generation.

The source currently routes only exact `chat` to chat and otherwise enters
vision inference. Change dispatch to an explicit selection over supported
backends so a typo cannot start the vision pipeline. When the generation
feature is compiled out, the CLI must identify the missing build option before
attempting network access.

## Build and dependency decisions

Add `WITH_TRITON_GENERATE_BACKEND=ON`, independent from
`WITH_CHAT_BACKEND`. CURL and RapidJSON are already required by the project, so
this feature adds no package. Use RapidJSON for strict parse/serialize behavior
and length-aware strings; keep the existing no-`nlohmann/json` and
no-`cpp-base64` rule for `ChatBackend`.

Keep the generation implementation in a dedicated source list and conditionally
include it in the application and applicable test targets. The existing CMake
build unconditionally references chat code in some targets; make only the
guards necessary to support and verify the independent feature/chat option
matrix. Do not change Triton version pins. The repository’s server and SDK
baseline remains 26.08; 26.10 is exercised as an opt-in compatibility image.

## Failure and output behavior

The executable returns nonzero for invalid configuration, unavailable feature,
request construction failure, transport failure, non-200 status, Triton error
JSON, invalid JSON, missing/wrong-type `text_output`, and response overflow.
Errors and informational logs for this mode go to stderr (or the configured
log file). Stdout contains only a successful `text_output` followed by a
newline. Do not retry: generation is stateful work at the server and a retry
could duplicate compute.

## Risks and deferred choices

The Generate extension is provisional and limited to HTTP/REST. The 26.10
backend sources establish the expected model inputs, but the exact request
mapping and model initialization must be confirmed against the pinned image by
V-6 before declaring compatibility. GPU memory and model availability depend
on the operator’s selected model and hardware. A successful raw-prompt call
does not establish that the prompt follows a model’s chat template.

Streaming, gRPC, conversation history, template selection, model repository
management, TLS/authentication, additional sampling options, and multimodal
payloads remain separate future decisions.
